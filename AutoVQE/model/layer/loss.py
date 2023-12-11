import torch
import torch.nn as nn

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = X - Y
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class FFTLoss(torch.nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.L1 = nn.L1Loss(reduction='mean')
    def forward(self, img1, img2):
        img1=torch.stack([torch.fft.fft2(img1, dim=(-2, -1)).real, torch.fft.fft2(img1, dim=(-2, -1)).imag], -1)
        img2=torch.stack([torch.fft.fft2(img2, dim=(-2, -1)).real, torch.fft.fft2(img2, dim=(-2, -1)).imag], -1)
        return self.L1(img1,img2)