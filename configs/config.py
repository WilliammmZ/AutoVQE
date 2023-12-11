from Peach.config import CfgNode as CN

def add_vqe_config(cfg):
    add_dataset_mfqev2_config(cfg)
    add_model_stcf_config(cfg)
    add_backbone_pscf_config(cfg)

def add_dataset_mfqev2_config(cfg):
    cfg.DATASETS.MFQEV2 = CN()
    cfg.DATASETS.MFQEV2.RADIUS = 5
    cfg.DATASETS.MFQEV2.QP = 'QP37'
    cfg.DATASETS.MFQEV2.DATA_ROOT = '/workspace/datasets/mfqe_frames'
    cfg.DATASETS.MFQEV2.PAIRED_CROP_SIZE = 96
    cfg.DATASETS.MFQEV2.USE_FLIP = True
    cfg.DATASETS.MFQEV2.USE_ROT = True
    # {
    #         '416':  19,  #1900
    #         '832':  10,  #1900
    #         '1280': 4,   #1800
    #         '1920': 2,   #2080
    #         '2560': 1    #300
    # }
    cfg.DATASETS.MFQEV2.TEST_FLOAT_BATCH = [19, 10, 4, 2, 1]


def add_backbone_pscf_config(cfg):
    cfg.MODEL.PSCF = CN()
    cfg.MODEL.PSCF.IN_NC = 64
    cfg.MODEL.PSCF.NB = 8
    cfg.MODEL.PSCF.NF = 128
    cfg.MODEL.PSCF.OUT_NC = 1
    cfg.MODEL.PSCF.WIN_SIZE = 8
    cfg.MODEL.PSCF.DROP_PATH = 0.0
    cfg.MODEL.PSCF.MODE = 'fusion' #add/local/global/fusion

def add_model_stcf_config(cfg):
    cfg.MODEL.STCFNET = CN()
    cfg.MODEL.STCFNET.IN_NC = 1
    cfg.MODEL.STCFNET.ST_OUT_NC = 64 
    cfg.MODEL.STCFNET.ST_NF = 32
    cfg.MODEL.STCFNET.ST_NB = 3
    cfg.MODEL.STCFNET.DEFORM_KS = 3
    cfg.MODEL.STCFNET.BASE_KS = 3
    cfg.MODEL.STCFNET.ENABLE_FFT = False
    