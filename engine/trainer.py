import datetime
import time
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm

import torch

from data.build import make_data_loader
from modeling.model import Modelbuilder
from engine.solver import make_optimizer
from engine.tester import test
from utils.checkpoint import Checkpointer
from utils.metric_logger import MetricLogger
from utils.logger import setup_logger
from utils.timer import time_for_file

def train(cfg):
    device = torch.device(cfg.DEVICE)    
    arguments = {}
    arguments["epoch"] = 0
    if not cfg.DATALOADER.BENCHMARK:
        model = Modelbuilder(cfg)
        print(model)
        model.to(device)
        model.float()
        optimizer, scheduler = make_optimizer(cfg, model)
        checkpointer = Checkpointer(
                model=model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                save_dir=cfg.OUTPUT_DIR
        )
        extra_checkpoint_data = checkpointer.load(cfg.WEIGHTS, prefix=cfg.WEIGHTS_PREFIX, prefix_replace=cfg.WEIGHTS_PREFIX_REPLACE, loadoptimizer=cfg.WEIGHTS_LOAD_OPT)
        arguments.update(extra_checkpoint_data)
        model.train()

    logger = setup_logger("trainer", cfg.FOLDER_NAME)
    if cfg.TENSORBOARD.USE:
        writer = SummaryWriter(cfg.FOLDER_NAME)
    else:
        writer = None
    meters = MetricLogger(writer=writer)
    start_training_time = time.time()
    end = time.time()
    start_epoch = arguments["epoch"]
    max_epoch = cfg.SOLVER.MAX_EPOCHS

    if start_epoch == max_epoch:
        logger.info("Final model exists! No need to train!")
        test(cfg, model)
        return 

    data_loader = make_data_loader(
        cfg,
        is_train=True,
    )
    size_epoch = len(data_loader)
    max_iter = size_epoch * max_epoch
    logger.info("Start training {} batches/epoch".format(size_epoch))

    for epoch in range(start_epoch, max_epoch):
        arguments["epoch"] = epoch
        #batchcnt = 0
        for iteration, batchdata in enumerate(data_loader):
            cur_iter =  size_epoch * epoch + iteration
            data_time = time.time() - end

            batchdata = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batchdata.items()}

            if not cfg.DATALOADER.BENCHMARK:
                loss_dict, metric_dict = model(batchdata)
                # print(loss_dict, metric_dict)
                optimizer.zero_grad()
                loss_dict['loss'].backward()
                optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            meters.update(time=batch_time, data=data_time, iteration=cur_iter)
            
            if cfg.DATALOADER.BENCHMARK:
                logger.info(
                    meters.delimiter.join(
                        [
                            "iter: {iter}",
                            "{meters}",
                        ]
                    ).format(
                        iter=iteration,
                        meters=str(meters),
                    )
                )
                continue

            eta_seconds = meters.time.global_avg * (max_iter - cur_iter)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % cfg.LOG_FREQ == 0:
                meters.update(iteration=cur_iter, **loss_dict)
                meters.update(iteration=cur_iter, **metric_dict)
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            # "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        # memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        #UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule.See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step()
                
        if (epoch + 1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            arguments["epoch"] += 1
            checkpointer.save("model_{:03d}".format(epoch), **arguments)
        if epoch == max_epoch - 1:
            arguments['epoch'] = max_epoch
            checkpointer.save("model_final", **arguments)

            total_training_time = time.time() - start_training_time
            total_time_str = str(datetime.timedelta(seconds=total_training_time))
            logger.info(
                "Total training time: {} ({:.4f} s / epoch)".format(
                    total_time_str, total_training_time / (max_epoch - start_epoch)
                )
            )
        if epoch == max_epoch - 1 or ((epoch + 1) % cfg.EVAL_FREQ == 0):
            results = test(cfg, model)
            meters.update(is_train=False, iteration=cur_iter, **results)

