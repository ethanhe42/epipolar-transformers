"""
Basic training script for PyTorch
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import numpy as np
import random

import torch

from core import cfg
from engine.trainer import train
from engine.tester import test
from utils.misc import *
from utils.logger import *
from utils.timer import time_for_file

def main():
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description="PyTorch Keypoints Training")
    parser.add_argument(
        "-c", "--cfg",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    if cfg.TENSORBOARD.COMMENT != "":
        cfg.FOLDER_NAME = os.path.join(cfg.OUTPUT_DIR, cfg.TENSORBOARD.COMMENT+'-'+time_for_file())
    else:
        cfg.FOLDER_NAME = os.path.join(cfg.OUTPUT_DIR, time_for_file())
    cfg.freeze()

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    output_dir = cfg.FOLDER_NAME
    if output_dir:
        mkdir(output_dir)
    

    logger = setup_logger("kp", cfg.FOLDER_NAME)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.cfg))
    with open(args.cfg, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.DOTRAIN:
        train(cfg)
    elif cfg.DOTEST:
        test(cfg)

    if cfg.VIS.DOVIS:
        from vision.visualization import visualization
        visualization(cfg)

if __name__ == "__main__":
    main()
