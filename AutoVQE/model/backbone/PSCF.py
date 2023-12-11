from Peach.modeling import BACKBONE_REGISTRY
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    '''
    input:
    B x Cin x H x W
    output:
    B x Cout x H x W
    '''
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
    
class invPixelShuffle(nn.Module):
    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)
        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b,-1,y // ratio,x // ratio)

#Learnable scale parameter for residual connection in AdaCNN  
class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input.mul(self.scale)

class AdaCNN(nn.Module):
    #Reference: Adaptive CNN for Image Super-Resolution (CVPR 2019)
    #Little modification in the residual connection
    def __init__(
        self, n_feats, kernel_size,wn=lambda x: torch.nn.utils.weight_norm(x), act=nn.ReLU(True), expand=4, conv =default_conv):
        super(AdaCNN, self).__init__()
        self.res_scale = Scale(0.2)
        self.x_scale = Scale(1)
        self.body = nn.Sequential(
            wn(conv(n_feats, n_feats * expand, kernel_size)),
            nn.ReLU(True),
            wn(conv(n_feats * expand, n_feats, kernel_size))
        )

    def forward(self, x):
        res = self.res_scale(self.body(x)) + self.x_scale(x)
        return res
  
class WMSA(nn.Module):
    """ 
    Self-attention module in Swin Transformer
    input:
    BxHxWxC
    
    output:
    BxHxWxC
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]
        
class SWINBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(SWINBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        # print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class SWINLayer(nn.Module):    
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path=0.0, input_resolution=1024):
        """ SwinTransformer Layer
        Input: [b, c, h, w]
        Output: [b, c, h, w]
        """
        super(SWINLayer, self).__init__()
        self.window_size = window_size
        self.swblock = SWINBlock(input_dim, output_dim, head_dim, window_size, drop_path, type='SW', input_resolution=input_resolution)
        self.wblock = SWINBlock(input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=input_resolution)
    def forward(self, x):
        h, w = x.size()[-2:]
        
        if h % self.window_size or w % self.window_size:
            paddingBottom = int(np.ceil(h/self.window_size)*self.window_size-h)
            paddingRight = int(np.ceil(w/self.window_size)*self.window_size-w)
            x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)


        x = self.wblock(x.permute(0, 2, 3, 1)) # b c h w -> b w h c
        x = self.swblock(x)[:, :h, :w, :]
        
        return x.permute(0, 3, 1, 2) # b w h c -> b c h w

class PSCFLocalBlock(nn.Module):
    def __init__(self, n_feat):
        super(PSCFLocalBlock, self).__init__()
        self.ada_cnn = AdaCNN(n_feat, kernel_size=3)
    def forward(self, x):
        x = self.ada_cnn(x)
        return x

class PSCFGlobalBlock(nn.Module):
    def __init__(self, n_feat, head_nc, window_size, drop_path=0.0):
        super(PSCFGlobalBlock, self).__init__()
        self.swin_layer = SWINLayer(n_feat, n_feat, head_nc, window_size, drop_path)
    def forward(self, x):
        x = self.swin_layer(x)
        return x

class PSCFFusionBlock(nn.Module):
    def __init__(self, n_feat, head_nc, window_size, drop_path=0.0):
        super(PSCFFusionBlock, self).__init__()
        self.swin_layer = SWINLayer(n_feat, n_feat, head_nc, window_size, drop_path)
        self.ada_cnn = AdaCNN(n_feat, kernel_size=3)
        self.fusion = nn.Conv2d(n_feat * 2, n_feat, 1, 1, 0)
        
    def forward(self, input):
        xl = self.ada_cnn(input)
        xg = self.swin_layer(input)
        x = torch.cat([xl, xg], dim=1)
        x = self.fusion(x)
        return x
           
class PSCFAddBlock(nn.Module):
    def __init__(self, n_feat, head_nc, window_size, drop_path=0.0):
        super(PSCFAddBlock, self).__init__()
        self.swin_layer = SWINLayer(n_feat, n_feat, head_nc, window_size, drop_path)
        self.ada_cnn = AdaCNN(n_feat, kernel_size=3)
        self.fusion = nn.Conv2d(n_feat * 2, n_feat, 1, 1, 0)
        
    def forward(self, input):
        xl = self.ada_cnn(input)
        xg = self.swin_layer(input)
        x = xl + xg
        return x

class PSCF_QE(nn.Module):    
    def __init__(self, in_nc, nb, nf, out_nc, win_size, drop_path=0.0, mode=None, wn=lambda x: torch.nn.utils.weight_norm(x) ,conv=default_conv):
        super(PSCF_QE, self).__init__()
        kernel_size = 3
        n_feats = nf // 2
        self.head = wn(conv(in_nc, n_feats, kernel_size))
        self.down=nn.Sequential(*[invPixelShuffle(2),wn(conv(4 * n_feats,  n_feats, kernel_size, True))])
        if mode == 'fusion':
            self.body = nn.Sequential(*[PSCFFusionBlock(n_feats, n_feats//4, win_size, drop_path) for _ in range(nb)])
        elif mode == 'add':
            self.body = nn.Sequential(*[PSCFAddBlock(n_feats, n_feats//4, win_size, drop_path) for _ in range(nb)])
        elif mode == 'local':
            self.body = nn.Sequential(*[PSCFLocalBlock(n_feats) for _ in range(nb)])
        elif mode == 'global':
            self.body = nn.Sequential(*[PSCFGlobalBlock(n_feats, n_feats//4, win_size, drop_path) for _ in range(nb)])
        else:
            raise NotImplementedError('wrong mode, the mode [%s] is not implemented' % mode)
        
        self.up=nn.Sequential(*[wn(conv(n_feats, 4 * n_feats, kernel_size, True)),nn.PixelShuffle(2)])
        self.tail=wn(conv(n_feats, out_nc, kernel_size))
        
    def forward(self, x):
        x = self.head(x)
        x1=x
        x=self.down(x)
        res = self.body(x).mul(0.2)
        res += x
        x=self.up(res)
        
        x+=x1
        x = self.tail(x)
        return x
    
    
@BACKBONE_REGISTRY.register()
def build_pscf_qe_backbone(cfg):
    in_nc = cfg.MODEL.PSCF.IN_NC
    nb = cfg.MODEL.PSCF.NB
    nf = cfg.MODEL.PSCF.NF
    out_nc = cfg.MODEL.PSCF.OUT_NC
    win_size = cfg.MODEL.PSCF.WIN_SIZE
    drop_path = cfg.MODEL.PSCF.DROP_PATH
    mode = cfg.MODEL.PSCF.MODE
    
    return PSCF_QE(
        in_nc,
        nb,
        nf,
        out_nc,
        win_size,
        drop_path,
        mode
    )





