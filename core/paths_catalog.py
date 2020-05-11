import os 

class BackboneCatalog(object):
    BACKBONES = {
        "HG11": {
            "dir": 'outs/keypoint_HG11'
        },
        "HG1": {
            "dir": 'outs/keypoint_HG1'
        },
        "HG": {
            "dir": 'outs/keypoint'
        },
        'poseR-50': {
            "dir": 'outs/benchmark/keypoint_h36m'
        },
    }
    @staticmethod
    def get(name):
        for i in ['HG11', 'HG1', 'HG', 'poseR-50']:
            if i in name:
                return i, BackboneCatalog.BACKBONES[i]['dir']
        return None

class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "h36m_train": {
            "image_set": "train"
        },
        "h36m_val": {
            "image_set": "validation"
        },
        "multiview_h36m_train": {
            "image_set": "train"
        },
        "multiview_h36m_val": {
            "image_set": "validation"
        },
        "RHD_train": {
            "img_dir": "RHD_published_v2/training",
            "ann_file": "RHD_published_v2/training/anno_training.pickle"
        },
        "RHD_val": {
            "img_dir": "RHD_published_v2/evaluation",
            "ann_file": "RHD_published_v2/evaluation/anno_evaluation.pickle"
        },
        "STB": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]      
        is_train = ~ ('test' in name)  
        if "RHD" in name:
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="RHDDataset",
                args=args,
            )
        elif "h36m" in name:
            args = dict(
                root=data_dir,
                image_set=attrs["image_set"],
            )
            return dict(
                factory="MultiViewH36M" if 'multiview' in name else "H36MDataset",
                args=args,
            )
        else:
            raise RuntimeError("Dataset not available: {}".format(name))
