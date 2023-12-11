import torch
import random
import cv2
import numpy as np
# ==========
# Data augmentation
# ==========
def bgr2rgb(img):
    code = getattr(cv2, 'COLOR_BGR2RGB')
    img = cv2.cvtColor(img, code)
    return img


def rgb2bgr(img):
    code = getattr(cv2, 'COLOR_RGB2BGR')
    img = cv2.cvtColor(img, code)
    return img

def paired_random_crop(img_gts, img_lqs, gt_patch_size, gt_path, scale=1):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    
    
    # if len(img_gts) == 1:
    #     img_gts = img_gts[0]
    # if len(img_lqs) == 1:
    #     img_lqs = img_lqs[0]
    
    return img_gts, img_lqs



def augment(imgs, hflip=True, rotation=True, flows=None):
    """Augment: horizontal flips or rotate (0, 90, 180, 270 degrees).

    Use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray]: Image list to be augmented.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation or not. Default: True.
        flows (list[ndarray]: Flow list to be augmented.
            Dimension is (h, w, 2). Default: None.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.
    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _imflip_(img, direction='horizontal'):
        """Inplace flip an image horizontally or vertically.

        Args:
            img (ndarray): Image to be flipped.
            direction (str): The flip direction, either "horizontal" or
                "vertical" or "diagonal".

        Returns:
            ndarray: The flipped image (inplace).
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        if direction == 'horizontal':
            return cv2.flip(img, 1, img)
        elif direction == 'vertical':
            return cv2.flip(img, 0, img)
        else:
            return cv2.flip(img, -1, img)

    def _augment(img):
        if hflip:
            _imflip_(img, 'horizontal')
        if vflip:
            _imflip_(img, 'vertical')
        if rot90:  # for (H W 3) image, H <-> W
            img = img.transpose(1, 0, 2)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    def _augment_flow(flow):
        if hflip:
            _imflip_(flow, 'horizontal')
            flow[:, :, 0] *= -1
        if vflip:
            _imflip_(flow, 'vertical')
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        return imgs


def totensor(imgs, opt_bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        opt_bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, opt_bgr2rgb, float32):
        if img.shape[2] == 3 and opt_bgr2rgb:
            img = bgr2rgb(img)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, opt_bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, opt_bgr2rgb, float32)


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