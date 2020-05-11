import torch

from core import cfg

def make_optimizer(cfg, model):
    params = model.parameters()
    if cfg.SOLVER.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR)
    elif cfg.SOLVER.OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=cfg.SOLVER.BASE_LR)
    else:
        raise NotImplementedError

    if cfg.SOLVER.SCHEDULER == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                milestones=list(cfg.SOLVER.STEPS), 
                gamma=cfg.SOLVER.GAMMA)
    else:
        raise NotImplementedError
    return optimizer, scheduler
