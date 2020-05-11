import bisect
import copy
import logging
import numpy as np

import torch.utils.data
from torch.utils.data.dataset import ConcatDataset

from data.datasets.RHD import RHDDataset
from data.datasets.h36m import H36MDataset
from data.datasets.multiview_h36m import MultiViewH36M 
from data.samplers.iteration_based_batch_sampler import IterationBasedBatchSampler
from core.paths_catalog import DatasetCatalog
from core import cfg

def build_dataset(dataset_list, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        args = data["args"]
        args["is_train"] = is_train
        factory = globals()[data['factory']]
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]

def make_data_loader(cfg, is_train=True, is_distributed=False, force_shuffle=False, dataset_list=None):
    if is_train or force_shuffle:
        shuffle = True
    else:
        shuffle = False

    if dataset_list is None:
        dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(dataset_list, DatasetCatalog, is_train)

    data_loaders = []
    for dataset in datasets:
        #sampler = make_data_sampler(dataset, shuffle, is_distributed)
        #batch_sampler = make_batch_data_sampler(
        #    dataset, sampler, 
        #    cfg.SOLVER.IMS_PER_BATCH if is_train else cfg.TEST.IMS_PER_BATCH, 
        #    cfg.SOLVER.MAX_EPOCHS * len(dataset) if is_train else None, 
        #    0 #start_iter
        #)
        #data_loader = torch.utils.data.DataLoader(
        #    dataset,
        #    num_workers=cfg.DATALOADER.NUM_WORKERS,
        #    batch_sampler=batch_sampler,
        #    pin_memory=False, #cfg.DATALOADER.PIN_MEMORY,
        #)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.SOLVER.IMS_PER_BATCH if is_train else cfg.TEST.IMS_PER_BATCH,
            shuffle=shuffle,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=cfg.DATALOADER.PIN_MEMORY,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        if False: debug_dataset(dataset)
        return data_loaders[0]
    return data_loaders

def make_batch_data_sampler(
    dataset, sampler, images_per_batch, num_iters=None, start_iter=0
):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False
    )
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

def make_data_sampler(dataset, shuffle, distributed=False):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def debug_dataset(dataset):
    import matplotlib.pyplot as plt
    for i, j in zip(dataset, dataset.image_path):
        img = plt.imread(j)
        f, axarr = plt.subplots(2)
        axarr = axarr.flatten()
        axarr[0].imshow(img)
        heatmap = i['heatmap'].numpy().sum(0)
        axarr[1].imshow(heatmap)
        plt.show()
