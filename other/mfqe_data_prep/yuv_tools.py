import torch
import numpy as np


#Import YUV420P frame by frame
def import_yuv(seq_path, h, w, tot_frm, yuv_type='420p', start_frm=0, only_y=True):
    """Load Y, U, and V channels separately from a 8bit yuv420p video.
    
    Args:
        seq_path (str): .yuv (imgs) path.
        h (int): Height.
        w (int): Width.
        tot_frm (int): Total frames to be imported.
        yuv_type: 420p or 444p
        start_frm (int): The first frame to be imported. Default 0.
        only_y (bool): Only import Y channels.

    Return:
        y_seq, u_seq, v_seq (3 channels in 3 ndarrays): Y channels, U channels, 
        V channels.

    Note:
        YUV传统上是模拟信号格式, 而YCbCr才是数字信号格式.YUV格式通常实指YCbCr文件.
        参见: https://en.wikipedia.org/wiki/YUV
    """
    # setup params
    if yuv_type == '420p':
        hh, ww = h // 2, w // 2
    elif yuv_type == '444p':
        hh, ww = h, w
    else:
        raise Exception('yuv_type not supported.')

    y_size, u_size, v_size = h * w, hh * ww, hh * ww
    blk_size = y_size + u_size + v_size
    
    # init
    y_seq = np.zeros((tot_frm, h, w), dtype=np.uint8)
    if not only_y:
        u_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)
        v_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)

    # read data
    with open(seq_path, 'rb') as fp:
        for i in range(tot_frm):
            fp.seek(int(blk_size * (start_frm + i)), 0)  # skip frames
            y_frm = np.fromfile(fp, dtype=np.uint8, count=y_size).reshape(h, w)
            if only_y:
                y_seq[i, ...] = y_frm
            else:
                u_frm = np.fromfile(fp, dtype=np.uint8, \
                    count=u_size).reshape(hh, ww)
                v_frm = np.fromfile(fp, dtype=np.uint8, \
                    count=v_size).reshape(hh, ww)
                y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm

    if only_y:
        return y_seq
    else:
        return y_seq, u_seq, v_seq


def yuv2rgb(y, u, v, h, w):
    rgb_data = np.zeros((h,w,3)).astype(np.uint8)
    #upsample u/v channel
    u = np.repeat(u, 2, 0)
    u = np.repeat(u, 2, 1)
    v = np.repeat(v, 2, 0)
    v = np.repeat(v, 2, 1)
    
    c = (y-np.array([16])) * 298
    d = u - np.array([128])
    e = v - np.array([128])

    r = (c + 409 * e + 128) // 256
    g = (c - 100 * d - 208 * e + 128) // 256
    b = (c + 516 * d + 128) // 256

    r = np.where(r < 0, 0, r)
    r = np.where(r > 255,255,r)

    g = np.where(g < 0, 0, g)
    g = np.where(g > 255,255,g)

    b = np.where(b < 0, 0, b)
    b = np.where(b > 255,255,b)

    rgb_data[:, :, 2:] = b
    rgb_data[:, :, 1:2] = g
    rgb_data[:, :, :1] = r
    
    return rgb_data


def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).
    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img