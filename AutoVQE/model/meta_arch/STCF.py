from Peach.modeling import META_ARCH_REGISTRY
from Peach.config import configurable
from Peach.utils.comm import get_local_rank
from Peach.utils.events import get_event_storage

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..layer import deform_conv, PSNR, CharbonnierLoss, calculate_ssim_pt, FFTLoss
from ..backbone import build_backbone
from AutoVQE.datasets import yuv2rgb

@META_ARCH_REGISTRY.register()
class STCF(nn.Module):
    
    @configurable
    def __init__(
        self, 
        *,
        radius      : int,
        in_nc       : int,
        st_out_nc   : int,
        st_nf       : int,
        st_nb       : int,
        deform_ks   : int,
        base_ks     : int,
        qe_net      : nn.Module,
        enable_fft  : bool,
        ):
        super().__init__()
        self.in_nc = in_nc
        self.radius = radius
        self.in_len = 2 * radius + 1
        self.enable_fft = enable_fft
        
        self.st_align = STFF(
            in_nc = self.in_nc * self.in_len, 
            out_nc = st_out_nc, 
            nf = st_nf, 
            nb= st_nb, 
            deform_ks= deform_ks,
            base_ks= base_ks,
            )
        
        self.qenet = qe_net
        self.charloss = CharbonnierLoss()
        self.fftloss = FFTLoss()
        self.psnr = PSNR()
        

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            'radius'       : cfg.DATASETS.MFQEV2.RADIUS,
            'in_nc'        : cfg.MODEL.STCFNET.IN_NC,
            'st_out_nc'    : cfg.MODEL.STCFNET.ST_OUT_NC,
            'st_nf'        : cfg.MODEL.STCFNET.ST_NF,
            'st_nb'        : cfg.MODEL.STCFNET.ST_NB,
            'deform_ks'    : cfg.MODEL.STCFNET.DEFORM_KS,
            'base_ks'      : cfg.MODEL.STCFNET.BASE_KS,      
            'qe_net'       : backbone,
            'enable_fft'   : cfg.MODEL.STCFNET.ENABLE_FFT,
        }
    
    @property
    def device(self):
        return get_local_rank()
        
        
    def forward(self, batched_inputs):
        '''
        batch input_dict
        {
            'lq': img_lqs,      # (B x T  x C x H x W)
            'hq': img_gts,      # (B x C x H x W)
        }
        '''
        if not self.training:
            return self.inference_test(batched_inputs)
        else:
            return self.inference_train(batched_inputs)
        
    def denoise(self, batched_inputs):
        lq_data = batched_inputs['lq'].to(self.device)
        hq_data = batched_inputs['hq'].to(self.device)
        if not self.training:
            lq_data = lq_data[0]
            hq_data = hq_data[0]
        b, _, c, _, _  = lq_data.shape
        if c>1: 
            input_data = torch.cat(
                [lq_data[:, :, i, ...] for i in range(c)], 
                    dim=1
            )  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W
        else:
            input_data = lq_data[:, :, 0, ...] # B x LC x H x W (new c)  
        
        out = self.st_align(input_data) # B x nf x H x W
        out = self.qenet(out) # B x out_nc x H x W
        frm_lst = [self.radius + idx_c * self.in_len for idx_c in range(self.in_nc)]
        out += input_data[:, frm_lst, ...]  # res: add middle frame
        return out, hq_data

    def inference_train(self, batched_inputs):
        out, hq_data = self.denoise(batched_inputs)
        b = hq_data.shape[0]
        char_loss = torch.mean(torch.stack(
                            [self.charloss(out[i], hq_data[i]) for i in range(b)]
                            ))  # cal loss
        
        if self.enable_fft:
            fft_loss = torch.mean(torch.stack(
                [self.fftloss(out[i], hq_data[i]) for i in range(b)]
            )) # fft loss
            loss_dict = {'CharbonnierLoss': char_loss, 'FFTLoss': fft_loss}
            
        else:
            loss_dict = {'CharbonnierLoss': char_loss}
            
        return loss_dict

    def inference_test(self, batched_inputs):
        #Pre eval
        his = get_event_storage()
        if his.iter == 0:
            lq_data = batched_inputs['lq'][0].to(self.device)
            hq_data = batched_inputs['hq'][0].to(self.device)
            name_id_list = batched_inputs['name_id']
            frame_id_list = batched_inputs['frame_id']
            
            b, t, c, _, _  = lq_data.shape
            lq_data = lq_data[:, t//2, ...] # B x C x H x W (new c)
            
            psnr_log = []
            ssim_log = []
            for o, g in zip(lq_data, hq_data):
                psnr_log.append(self.psnr(o, g))   
            ssim_log.extend(calculate_ssim_pt(lq_data, hq_data))       
        
        #Inference    
        else:
            name_id_list = batched_inputs['name_id']
            frame_id_list = batched_inputs['frame_id']
            test_iter = batched_inputs['iters'][0]

            out, hq_data = self.denoise(batched_inputs)
            out = torch.clip(out, 0.0, 1.0)# B x out_nc x H x W
            
            psnr_log = []
            ssim_log = []
    
            for o, g in zip(out, hq_data):
                psnr_log.append(self.psnr(o, g))  
            ssim_log.extend(calculate_ssim_pt(out, hq_data))
            
            if test_iter % 50 == 0:
                # only dispaly the first batch
                his = get_event_storage()
                u = batched_inputs['hqu'][0][0].cpu().detach().numpy().transpose(1,2,0) * 255
                v = batched_inputs['hqv'][0][0].cpu().detach().numpy().transpose(1,2,0) * 255
                y = out[0].cpu().detach().numpy().transpose(1,2,0) * 255
                y_gt = hq_data[0].cpu().detach().numpy().transpose(1,2,0) * 255
                h, w = y.shape[:2]
                out_rgb = yuv2rgb(y, u, v, h, w)    # H x W x 3
                gt_rgb = yuv2rgb(y_gt, u, v, h, w)  # H x W x 3
                show_rgb = np.concatenate([out_rgb, gt_rgb], axis=1) # H x 2W x 3
                # show image must in C x H x W
                his.put_image('visualize_test/test_case_'+str(int(test_iter)), show_rgb.transpose(2,0,1).astype('uint8'))
         
        return {
                'name_id':   name_id_list,
                'psnr':      psnr_log,
                'ssim':      ssim_log,
                'frame_id':  frame_id_list
            }
    

class STFF(nn.Module):
    '''
    Fuse the feature with unet and dcn
    
    @Param:
            in_nc   : int -> the channel of the input feature
            nf      : int -> the mid channel of the u shape net
            base_ks : int -> the ks of the conv in u shape net
            nb      : int -> the num of blocks in u shape net
            deform_ks: int -> dcn's ks
            out_nc  : int -> the outputs's channel of stff
    @Input:
        [B x C x H x W]
    
    @Output:
        [B x out_nc x H x W]
    '''
    
    def __init__(
            self,
            in_nc       : int,
            nf          : int,
            base_ks     : int,
            nb          : int,
            deform_ks   : int,
            out_nc      : int,):
        super(STFF, self).__init__()
        
        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        
        
        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)
                    )
                )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                    )
                )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
            )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
            )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = deform_conv.ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
            )
        

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks
    
        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
                )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        out = self.out_conv(out)
        off_msk = self.offset_mask(out)
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
            )

        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs.float(), off.float(), msk.float()), 
            inplace=True
            )
        
        return fused_feat