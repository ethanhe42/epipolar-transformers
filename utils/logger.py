import logging
import os
import sys


def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    if len(logger.handlers):
        return logger
    logger.propagate = False
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
