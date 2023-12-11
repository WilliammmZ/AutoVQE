import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Peach.utils.file_io import PathManager
from tqdm import tqdm
from yuv_tools import yuv2rgb

'''
do: turn the yuv file in our dataste to rgb data
complete!
'''

# ==== config ==== #
data_root = '/workspace/datasets/mfqe_frames/test/QP27' #'/workspace/datasets/mfqe_frames/train/raw'
video_name_list = sorted(os.listdir(data_root)) # video name
save_root = '/workspace/datasets/mfqe_rgb'
mode = 'test' #train
v_type = 'QP27'



# ==== run ==== #
video_size_h = []
video_size_w = []
video_nfs = []
#get video info
for v_name in video_name_list:
    
    w,h = v_name.split('_')[1].split('x')
    nfs = v_name.split('_')[-1]
    video_size_h.append(h)
    video_size_w.append(w)
    video_nfs.append(nfs)

for video_id in tqdm(range(len(video_name_list)), desc='convert_to_rgb'):
    path = os.path.join(data_root, video_name_list[video_id])
    print(path)
    save_path = os.path.join(save_root, mode, v_type, video_name_list[video_id]) # rgbroot/train/video_name
    PathManager.mkdirs(save_path)
    H = int(video_size_h[video_id])
    W = int(video_size_w[video_id])
    nfs = int(video_nfs[video_id])
    
    for i in range(nfs):
        y = np.load(os.path.join(path, 'y_%04d.npy' % i))
        u = np.load(os.path.join(path, 'u_%04d.npy' % i))    
        v = np.load(os.path.join(path, 'v_%04d.npy' % i))
        
        rgb_img = yuv2rgb(y, u, v, H, W)
        img_name = str('%04d.png' % i)
        img_save_path = os.path.join(save_path, img_name)
        plt.imsave(img_save_path, rgb_img)
        
        

        
        
    
