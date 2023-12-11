import logging
import numpy as np
import torch
import itertools
from collections import OrderedDict

import Peach.utils.comm as comm
from Peach.evaluation import DatasetEvaluator

class VQE_Evaluator(DatasetEvaluator):
    def __init__(self, distributed=True, save_path = None):
        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device("cpu")
        self._distributed = distributed
        self.save_path = save_path
    
    def reset(self):
        self._predictions = []
  
    def process(self, inputs, outputs):
        name_id_list = outputs['name_id']
        psnr_list = outputs['psnr']
        ssim_list = outputs['ssim']
        frame_id_list = outputs['frame_id']

        for name_id, psnr, ssim, f_id in zip(name_id_list, psnr_list, ssim_list, frame_id_list): #name id is a tuple
            prediction = {'name_id': name_id[0], 'psnr': psnr.to(self._cpu_device), 'ssim': ssim.to(self._cpu_device), 'f_id': f_id.to(self._cpu_device)}
            self._predictions.append(prediction)
        
    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return
        else:
            predictions = self._predictions    
        
        if len(predictions) == 0:
            self._logger.warning("Did not receive valid predictions.")
            return {}
        
        if comm.is_main_process() and self.save_path is not None:
            np.save(self.save_path, predictions)
        
        name_id_list = []
        tmp_result = {}
        for test_dict in predictions:
            name_id = test_dict['name_id']
            if name_id not in name_id_list:
                name_id_list.append(name_id)
                tmp_result[name_id] = {}
                tmp_result[name_id]['psnr'] = []
                tmp_result[name_id]['ssim'] = []
                
                tmp_psnr = test_dict['psnr']
                tmp_result[name_id]['psnr'].append(tmp_psnr)
                
                tmp_ssim = test_dict['ssim']
                tmp_result[name_id]['ssim'].append(tmp_ssim)    
            else:
                tmp_psnr = test_dict['psnr']
                tmp_result[name_id]['psnr'].append(tmp_psnr)
                
                tmp_ssim = test_dict['ssim']
                tmp_result[name_id]['ssim'].append(tmp_ssim)
                   
        self._results = OrderedDict()
        all_psnr = []
        all_ssim = []
        for name_id in name_id_list:
            avg_psnr = np.mean(tmp_result[name_id]['psnr'])
            avg_ssim = np.mean(tmp_result[name_id]['ssim'])
            
            all_psnr.append(avg_psnr)
            all_ssim.append(avg_ssim)
            
            self._results['test_psnr/' + str(name_id)] = avg_psnr
            self._results['test_ssim/' + str(name_id)] = avg_ssim
        
        self._results['test_psnr/ALL_AVG_PSNR'] = np.mean(all_psnr)  
        self._results['test_ssim/ALL_AVG_SSIM'] = np.mean(all_ssim)      
        
        return self._results