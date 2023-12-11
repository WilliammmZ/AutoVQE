# use to split the yuv file in mfqev2 into frames
import numpy as np
import os.path as op
import glob
from tqdm import tqdm
from Peach.utils.file_io import PathManager
from yuv_tools import import_yuv

# extend dims of the y seqs (hxwx1)
def extend_dims(y_seq):
    new_seq = []
    for y in y_seq:
        img = np.expand_dims(
                np.squeeze(y), 2
                )
        new_seq.append(img)
    return new_seq

def store_mfqev2_frames(
    data_root,
    save_root,
    is_train,
    qp,
    only_y = False,
    save_raw = False,
    save_dict = False,
    dict_path = None,
    ):
    '''
    Save yuv data in (hxwxc)
    After the mfqev2 official video compress script, the data must follow its file structure.
    Then this func will store the yuv video in the form of frames.
    
    Args:
        data_root: The path of MFQEv2 data
        save_root: The path of MFQEv2 frames data
        is_train: True for train data, otherwise is test data
            qp: The quality of video e.g.,'QP37'
        only_y: Save only Y channel or Y,U,V channels
        save_raw: Save raw data
        save_dict: Save a dict to record the video info, such as name, h, w, nfs (all qp only need one)
        dict_path: If you already have dict, please use the saved dict
    '''
    
    if save_dict:
        print('---------------- start train data store ---------------')
        data_dict = {
            'name_id':[],
            'w':[],
            'h':[],
            'nfs':[]
        }
    else:
        print('---------------- start train data store with saved dict ---------------')
        if dict_path is not None:
            data_dict = np.load(dict_path, allow_pickle=True).item()

        else:
            print('[error]: need dict for load data.....')
    
    
    if is_train:  
        save_lq_root = op.join(save_root, 'train', qp)
        PathManager.mkdirs(save_lq_root)
        lq_root = op.join(data_root, 'train_108', 'HM16.5_LDP', qp)
        gt_root = op.join(data_root, 'train_108', 'raw')
        if save_raw:
            save_gt_root = op.join(save_root, 'train', 'raw')
            PathManager.mkdirs(save_gt_root)
            
    else:
        save_lq_root = op.join(save_root, 'test', qp)
        PathManager.mkdirs(save_lq_root)
        lq_root = op.join(data_root, 'test_18', 'HM16.5_LDP', qp)
        gt_root = op.join(data_root, 'test_18', 'raw')
        if save_raw:
            save_gt_root = op.join(save_root, 'test', 'raw')
            PathManager.mkdirs(save_gt_root)
            

    gt_path_list = sorted(glob.glob(op.join(gt_root, '*.yuv')))
    print('all video count:', len(gt_path_list))
    total_frame = 0
    for idx_vid, gt_vid_path in tqdm(enumerate(gt_path_list), desc='save frames'):
        name_vid = gt_vid_path.split('/')[-1] #video file name 'videoSRC183_640x360_120.yuv'
        w, h = map(int, name_vid.split('_')[-2].split('x'))
        nfs = int(name_vid.split('.')[-2].split('_')[-1])
        lq_vid_path = op.join(
            lq_root,
            name_vid
        )
        
        name = name_vid.split('.')[0]
        lq_root_video_path = op.join(save_lq_root, name)
        PathManager.mkdirs(lq_root_video_path)
        
        if save_raw:
            gt_root_video_path = op.join(save_gt_root, name)
            PathManager.mkdirs(gt_root_video_path)
            
        if save_dict:
            data_dict['name_id'].append(name)
            data_dict['h'].append(h)
            data_dict['w'].append(w)
            data_dict['nfs'].append(nfs)
        else:
            check_name = data_dict['name_id'][idx_vid]
            assert name == check_name, 'wrong name !!!!!!!!!'

        if only_y:
            lq_y_seq = import_yuv(lq_vid_path, h, w, nfs, only_y=True)
            lq_y_seq = extend_dims(lq_y_seq)
            if save_raw:
                gt_y_seq = import_yuv(gt_vid_path, h, w, nfs, only_y=True)
                gt_y_seq = extend_dims(gt_y_seq)
        else:
            lq_y_seq, lq_u_seq, lq_v_seq = import_yuv(lq_vid_path, h, w, nfs, only_y=False)
            lq_y_seq = extend_dims(lq_y_seq)
            lq_u_seq = extend_dims(lq_u_seq)
            lq_v_seq = extend_dims(lq_v_seq)
            if save_raw:
                gt_y_seq, gt_u_seq, gt_v_seq= import_yuv(gt_vid_path, h, w, nfs, only_y=False)
                gt_y_seq = extend_dims(gt_y_seq)      
                gt_u_seq = extend_dims(gt_u_seq)
                gt_v_seq = extend_dims(gt_v_seq)
        
        total_frame = total_frame + nfs
        for f_id in range(nfs):
            if only_y:     
                lq_save_y_path = op.join(lq_root_video_path, 'y_%04d.npy'%f_id)
                np.save(lq_save_y_path, lq_y_seq[f_id])
                if save_raw:
                    gt_save_y_path = op.join(gt_root_video_path, 'y_%04d.npy'%f_id)
                    np.save(gt_save_y_path, gt_y_seq[f_id])
            else:
                lq_save_y_path = op.join(lq_root_video_path, 'y_%04d.npy'%f_id)
                np.save(lq_save_y_path, lq_y_seq[f_id])
                lq_save_u_path = op.join(lq_root_video_path, 'u_%04d.npy'%f_id)
                np.save(lq_save_u_path, lq_u_seq[f_id])
                lq_save_v_path = op.join(lq_root_video_path, 'v_%04d.npy'%f_id)
                np.save(lq_save_v_path, lq_v_seq[f_id])
                if save_raw:
                    gt_save_y_path = op.join(gt_root_video_path, 'y_%04d.npy'%f_id)
                    np.save(gt_save_y_path, gt_y_seq[f_id])
                    gt_save_u_path = op.join(gt_root_video_path, 'u_%04d.npy'%f_id)
                    np.save(gt_save_u_path, gt_u_seq[f_id])
                    gt_save_v_path = op.join(gt_root_video_path, 'v_%04d.npy'%f_id)
                    np.save(gt_save_v_path, gt_v_seq[f_id])

    if save_dict:
        if is_train:
            np.save(op.join(save_root, 'train', 'train_dict.npy'),data_dict)
        else:
            np.save(op.join(save_root, 'test', 'test_dict.npy'),data_dict)
            
    print('all frames:', total_frame)
    print('Finshed !')
    

if __name__ == '__main__':
    store_mfqev2_frames(
        data_root   =   '/workspace/datasets/MFQE',
        save_root   =   '/workspace/datasets/mfqe_frames',
        is_train    =   False,
        qp          =   'QP37',
        only_y      =   False,
        save_raw    =   True,
        save_dict   =   True,
        dict_path   =   None
        # /workspace/datasets/mfqe_frames/train/train_dict.npy
    )