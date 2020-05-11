### Human 3.6M Dataset (for epipolar transformer training)
- download h36m from the official website http://vision.imar.ro/human3.6m/description.php
- process https://github.com/CHUNYUWANG/H36M-Toolbox
- undistort
```bash
python scripts/undistort_h36m.py
python scripts/undistort_h36m.py --anno ~/datasets/h36m/annot/h36m_train.pkl
~/datasets/h36m/undistortedimages
```

### RHD Dataset (for 2D->3D lifting training)
download using bash scripts
```bash
bash get_RHD.sh
```
create soft link of datasets here
```bash
ln -s ~/hand3d/RHD_published_v2/ .
```

### Custom Dataset
If you are going create your own dataset:
- `data/datasets/`: dataloader
- `data/build.py`: if clause in `build_dataset`
- `core/paths_catalog.py`: `DatasetCatalog.DATASETS` and corresponding if clause in `DatasetCatalog.get()`
