from Peach.kernel import default_argument_parser, launch, default_setup, DefaultTrainer
from Peach.config import get_cfg, CfgNode
from Peach.solver import maybe_add_gradient_clipping
from Peach.checkpoint import DefaultCheckpointer


from AutoVQE.vqe_evaluator import VQE_Evaluator
from AutoVQE.datasets import build_train_loader, build_test_loader
from AutoVQE.model import build_model

from configs.config import add_vqe_config

import torch
import os

#default adam optim func
def build_adam_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config. -> user design
    """
    params = model.parameters()
    return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
        params,
        lr=cfg.SOLVER.BASE_LR,
    )

def setup(args):
    cfg = get_cfg()
    add_vqe_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

class Trainer(DefaultTrainer):
    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_adam_optimizer(cfg, model)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_test_loader(cfg, dataset_name)
    
    @classmethod
    def build_model(cls, cfg):
        return build_model(cfg)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        if dataset_name == 'MFQEV2_TEST':
            return VQE_Evaluator(True, os.path.join(cfg.LOG_DIR, cfg.EXP_NAME, cfg.TEST.SAVE_PATH))
        else:
            return

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DefaultCheckpointer(model, save_dir=os.path.join(cfg.LOG_DIR, cfg.EXP_NAME)).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )
