from Peach.data.datasets import DATASETS_REGISTRY
import torch
from torch.utils import data
import numpy as np
import os.path as op
import random

from .augmentation import totensor, paired_random_crop, augment

@DATASETS_REGISTRY.register()
class MFQEV2_TRAIN(data.Dataset):
    '''
    @Func: get data from mfqev2
    
    @Return: {
            'lq': img_lqs,      # (T C H W)
            'hq': img_gts,      # (1 C H W)
            }
    '''
    def __init__(self, cfg):
        super().__init__()
        self.radius = cfg.DATASETS.MFQEV2.RADIUS
        self.QP = cfg.DATASETS.MFQEV2.QP
        self.data_root =cfg.DATASETS.MFQEV2.DATA_ROOT

        self.crop_size = cfg.DATASETS.MFQEV2.PAIRED_CROP_SIZE
        self.use_flip = cfg.DATASETS.MFQEV2.USE_FLIP
        self.use_rot = cfg.DATASETS.MFQEV2.USE_ROT
        
        self.key = np.load(op.join(self.data_root, 'train', 'train_dict.npy'), allow_pickle=True).item()
        # dataset paths
        self.gt_root = op.join(self.data_root, 'train', 'raw')
        self.lq_root = op.join(self.data_root, 'train', str(self.QP))

        self.train_key = {
            'index_vid':[], # the clip id from 0
            'name_vid':[], #the clip name
            'index_fid':[], #the train data frame id:from 0
            'w':[],
            'h':[]
        }
        for idx_vid in range(len(self.key['name_id'])):
            name_vid = self.key['name_id'][idx_vid]
            nfs = self.key['nfs'][idx_vid]
            w = self.key['w'][idx_vid]
            h = self.key['h'][idx_vid]
            
            for f_idx in range(self.radius, nfs-self.radius):
                self.train_key['index_vid'].append(idx_vid)
                self.train_key['name_vid'].append(name_vid)
                self.train_key['index_fid'].append(f_idx)
                self.train_key['w'].append(w)
                self.train_key['h'].append(h)
                
        print('dataset_len:',len(self.train_key['index_fid']))
        
    def __getitem__(self, index):
        frame_id = self.train_key['index_fid'][index]
        video_name = self.train_key['name_vid'][index]
        
        #get neighbor frames idx list
        neighbor_list = [
            i for i in range(
            int(frame_id) - self.radius, int(frame_id) + self.radius + 1
            )
        ]
        
        # load gt images
        img_gts = []
        img_gt_path = op.join(self.gt_root, video_name, str('y_%04d.npy' % frame_id))
        img_gts.append(np.load(img_gt_path))  # ! H x W x C
        
        # load lqs images    
        img_lqs = []
        for neighbor in neighbor_list:
            img_lq_path = op.join(self.lq_root, video_name, str('y_%04d.npy' % neighbor))
            img_lqs.append(
                np.load(img_lq_path)  # ! H x W x C
            )
        
        # * ==========
        # * data augmentation
        # * ==========
        # * randomly crop
        gt_size = self.crop_size
        img_gts, img_lqs = paired_random_crop(
                img_gts, img_lqs, gt_size, video_name
            )
            
        # * flip, rotate
        for igt in img_gts:
            img_lqs.append(igt)
            
        img_results = augment(
            img_lqs, self.use_flip, self.use_rot
            )
        # * To Tensor
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:len(neighbor_list)], dim=0) / 255.
        img_gts = torch.stack(img_results[len(neighbor_list):], dim=0) / 255.

        return {
            'lq': img_lqs,      # (T C H W)
            'hq': img_gts[0],   # (C H W)
        }

    def __len__(self):
        return len(self.train_key['index_fid'])
    

def get_list_index(lst=None, item=None):
    return [index for (index, value) in enumerate(lst) if value==item]


@DATASETS_REGISTRY.register()
class MFQEV2_TEST(data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.radius = cfg.DATASETS.MFQEV2.RADIUS
        self.QP = cfg.DATASETS.MFQEV2.QP
        self.data_root =cfg.DATASETS.MFQEV2.DATA_ROOT
        self.float_batch = cfg.DATASETS.MFQEV2.TEST_FLOAT_BATCH

        self.key = np.load(op.join(self.data_root, 'test', 'test_dict.npy'), allow_pickle=True).item()
        # dataset paths
        self.gt_root = op.join(self.data_root, 'test', 'raw')
        self.lq_root = op.join(self.data_root, 'test', str(self.QP))
        
        self.pack_meta= {
            '416':  self.float_batch[0],   #1900
            '832':  self.float_batch[1],   #1900
            '1280': self.float_batch[2],   #1800
            '1920': self.float_batch[3],   #2080
            '2560': self.float_batch[4]    #300
        }

        # pack data to floating batchsize
        # self.pack_meta= {
        #     '416':  19,  #1900
        #     '832':  10,  #1900
        #     '1280': 4,   #1800
        #     '1920': 2,   #2080
        #     '2560': 1    #300
        # }
        
        # self.pack_meta= {
        #     '416':  1,  #1900
        #     '832':  1,  #1900
        #     '1280': 1,   #1800
        #     '1920': 1,   #2080
        #     '2560': 1    #300
        # }

        self.test_key = {
            'gt_path': [],
            'gtu_path': [],
            'gtv_path': [],
            'lq_paths': [],
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [],
            'frame_id': [],
        }
        
        self.vid_num = len(self.key['w'])

        for idx_vid in range(len(self.key['name_id'])):
            name_vid = self.key['name_id'][idx_vid]
            nfs = self.key['nfs'][idx_vid]
            w = self.key['w'][idx_vid]
            h = self.key['h'][idx_vid]

            for iter_frm in range(nfs):
                
                lq_indexes = list(range(iter_frm - self.radius, iter_frm + self.radius + 1))
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))

                img_gt_path = op.join(self.gt_root, name_vid, str('y_%04d.npy'% iter_frm))
                self.test_key['gt_path'].append(img_gt_path)

                img_gtu_path = op.join(self.gt_root, name_vid, str('u_%04d.npy'% iter_frm))
                self.test_key['gtu_path'].append(img_gtu_path)

                img_gtv_path = op.join(self.gt_root, name_vid, str('v_%04d.npy'% iter_frm))
                self.test_key['gtv_path'].append(img_gtv_path)
                
                self.test_key['frame_id'].append(iter_frm)

                img_lq_paths = []
                for lq_index in lq_indexes:
                    img_lq_path = op.join(self.lq_root, name_vid, str('y_%04d.npy' % lq_index))
                    img_lq_paths.append(img_lq_path)                   
                    
                self.test_key['lq_paths'].append(img_lq_paths)
                self.test_key['index_vid'].append(idx_vid)
                self.test_key['name_vid'].append(name_vid)
                self.test_key['w'].append(w)
                self.test_key['h'].append(h)

        self.num_data = 0
        self.final_key = {}

        index_416 = get_list_index(self.test_key['w'], 416) 
        index_832 = get_list_index(self.test_key['w'], 832) 
        index_1280 = get_list_index(self.test_key['w'], 1280) 
        index_1920 = get_list_index(self.test_key['w'], 1920) 
        index_2560 = get_list_index(self.test_key['w'], 2560) 
        # print(len(index_416))
        # print(len(index_832))
        # print(len(index_1280))
        # print(len(index_1920))
        # print(len(index_2560))


        for n_b in range(len(index_416)//self.pack_meta['416']):
            tmp_key = {
            'gt_path': [],
            'gtu_path': [],
            'gtv_path': [],
            'lq_paths': [],
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            'frame_id': [],
            }
            for b_id in range(n_b * self.pack_meta['416'], (n_b + 1) * self.pack_meta['416']):
                w_id = index_416[b_id]
                tmp_key['gt_path'].append(self.test_key['gt_path'][w_id])
                tmp_key['gtu_path'].append(self.test_key['gtu_path'][w_id])
                tmp_key['gtv_path'].append(self.test_key['gtv_path'][w_id])
                tmp_key['lq_paths'].append(self.test_key['lq_paths'][w_id])
                tmp_key['w'].append(self.test_key['w'][w_id])
                tmp_key['h'].append(self.test_key['h'][w_id])
                tmp_key['index_vid'].append(self.test_key['index_vid'][w_id])
                tmp_key['name_vid'].append(self.test_key['name_vid'][w_id])
                tmp_key['frame_id'].append(self.test_key['frame_id'][w_id])
            
            self.final_key[self.num_data] = tmp_key
            self.num_data = self.num_data + 1
        
        for n_b in range(len(index_832)//self.pack_meta['832']):
            tmp_key = {
            'gt_path': [],
            'gtu_path': [],
            'gtv_path': [],
            'lq_paths': [],
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [],
            'frame_id': [],
            }
            for b_id in range(n_b * self.pack_meta['832'], (n_b + 1) * self.pack_meta['832']):
                w_id = index_832[b_id]
                tmp_key['gt_path'].append(self.test_key['gt_path'][w_id])
                tmp_key['gtu_path'].append(self.test_key['gtu_path'][w_id])
                tmp_key['gtv_path'].append(self.test_key['gtv_path'][w_id])
                tmp_key['lq_paths'].append(self.test_key['lq_paths'][w_id])
                tmp_key['w'].append(self.test_key['w'][w_id])
                tmp_key['h'].append(self.test_key['h'][w_id])
                tmp_key['index_vid'].append(self.test_key['index_vid'][w_id])
                tmp_key['name_vid'].append(self.test_key['name_vid'][w_id])
                tmp_key['frame_id'].append(self.test_key['frame_id'][w_id])
            
            self.final_key[self.num_data] = tmp_key
            self.num_data = self.num_data + 1

        for n_b in range(len(index_1280)//self.pack_meta['1280']):
            tmp_key = {
            'gt_path': [],
            'gtu_path': [],
            'gtv_path': [],
            'lq_paths': [],
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [],
            'frame_id': [], 
            }
            for b_id in range(n_b * self.pack_meta['1280'], (n_b + 1) * self.pack_meta['1280']):
                w_id = index_1280[b_id]
                tmp_key['gt_path'].append(self.test_key['gt_path'][w_id])
                tmp_key['gtu_path'].append(self.test_key['gtu_path'][w_id])
                tmp_key['gtv_path'].append(self.test_key['gtv_path'][w_id])
                tmp_key['lq_paths'].append(self.test_key['lq_paths'][w_id])
                tmp_key['w'].append(self.test_key['w'][w_id])
                tmp_key['h'].append(self.test_key['h'][w_id])
                tmp_key['index_vid'].append(self.test_key['index_vid'][w_id])
                tmp_key['name_vid'].append(self.test_key['name_vid'][w_id])
                tmp_key['frame_id'].append(self.test_key['frame_id'][w_id])
            
            self.final_key[self.num_data] = tmp_key
            self.num_data = self.num_data + 1

        for n_b in range(len(index_1920)//self.pack_meta['1920']):
            tmp_key = {
            'gt_path': [],
            'gtu_path': [],
            'gtv_path': [],
            'lq_paths': [], 
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [],
            'frame_id': [],
            }
            for b_id in range(n_b * self.pack_meta['1920'], (n_b + 1) * self.pack_meta['1920']):
                w_id = index_1920[b_id]
                tmp_key['gt_path'].append(self.test_key['gt_path'][w_id])
                tmp_key['gtu_path'].append(self.test_key['gtu_path'][w_id])
                tmp_key['gtv_path'].append(self.test_key['gtv_path'][w_id])
                tmp_key['lq_paths'].append(self.test_key['lq_paths'][w_id])
                tmp_key['w'].append(self.test_key['w'][w_id])
                tmp_key['h'].append(self.test_key['h'][w_id])
                tmp_key['index_vid'].append(self.test_key['index_vid'][w_id])
                tmp_key['name_vid'].append(self.test_key['name_vid'][w_id])
                tmp_key['frame_id'].append(self.test_key['frame_id'][w_id])
            
            self.final_key[self.num_data] = tmp_key
            self.num_data = self.num_data + 1
        
        for n_b in range(len(index_2560)//self.pack_meta['2560']):
            tmp_key = {
            'gt_path': [],
            'gtu_path': [],
            'gtv_path': [],
            'lq_paths': [],
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            'frame_id': [],
            }
            for b_id in range(n_b * self.pack_meta['2560'], (n_b + 1) * self.pack_meta['2560']):
                w_id = index_2560[b_id]
                tmp_key['gt_path'].append(self.test_key['gt_path'][w_id])
                tmp_key['gtu_path'].append(self.test_key['gtu_path'][w_id])
                tmp_key['gtv_path'].append(self.test_key['gtv_path'][w_id])
                tmp_key['lq_paths'].append(self.test_key['lq_paths'][w_id])
                tmp_key['w'].append(self.test_key['w'][w_id])
                tmp_key['h'].append(self.test_key['h'][w_id])
                tmp_key['index_vid'].append(self.test_key['index_vid'][w_id])
                tmp_key['name_vid'].append(self.test_key['name_vid'][w_id])
                tmp_key['frame_id'].append(self.test_key['frame_id'][w_id])
            
            self.final_key[self.num_data] = tmp_key
            self.num_data = self.num_data + 1
        
        print('test_data_len:', self.num_data)

    def get_video_num(self):
        return self.vid_num
    
    def get_video_name_list(self):
        return self.key['name_id']
    
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, index):
        # index_key = self.test_index_list[index]
        # key = self.final_key[index_key]
        key = self.final_key[index]

        img_gt_path_list = key['gt_path']
        img_gtu_path_list = key['gtu_path']
        img_gtv_path_list = key['gtv_path']
        img_lq_path_list = key['lq_paths']
        name_vid_list = key['name_vid']
        frame_id_list = key['frame_id']
        
 

        img_gt_list = []
        img_gtu_list = []
        img_gtv_list = []

        for i in range(len(img_gt_path_list)):
            img_gt_list.append(
                np.load(img_gt_path_list[i])
            )
    
            img_gtu_list.append(
                np.load(img_gtu_path_list[i])
            )
            img_gtv_list.append(
                np.load(img_gtv_path_list[i])
            )

        img_gt_t = totensor(img_gt_list)
        img_gt = torch.stack(img_gt_t, dim=0) / 255.
        img_gtu_t = totensor(img_gtu_list)
        img_gtu = torch.stack(img_gtu_t, dim=0) / 255.
        img_gtv_t = totensor(img_gtv_list)
        img_gtv = torch.stack(img_gtv_t, dim=0) / 255. # B * C * H * W

        
        img_lq_list = []
        for lq_paths in img_lq_path_list:
            img_lq_segs = []
            for lq_path in lq_paths:
                img_lq_segs.append(
                    np.load(lq_path)
                )
            img_lq_segs_t = totensor(img_lq_segs)
            img_lq_list.append(torch.stack(img_lq_segs_t, dim=0) / 255.)

        img_lqs = torch.stack(img_lq_list, dim=0) # B * T * C * H * W
        

        return {
            'lq'     :    img_lqs,  # B * T * C * H * W (0 to 1)
            'hq'     :    img_gt,   # B * C * H * W
            'hqu'    :    img_gtu,  # B * C * H * W
            'hqv'    :    img_gtv,  # B * C * H * W
            'name_id':    name_vid_list, # B
            'frame_id':   frame_id_list, # B
            'iters'   :   index
            }