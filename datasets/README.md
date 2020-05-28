### Human 3.6M Dataset (for epipolar transformer training)
1. Download the H36M from the official website http://vision.imar.ro/human3.6m/description.php
2. Process using https://github.com/CHUNYUWANG/H36M-Toolbox
3. Undistort
```bash
python scripts/undistort_h36m.py
python scripts/undistort_h36m.py --anno ~/datasets/h36m/annot/h36m_train.pkl
~/datasets/h36m/undistortedimages
```

**We can't share the dataset according to their [license]( http://vision.imar.ro/human3.6m/eula.php). Please visit http://vision.imar.ro/human3.6m/ to request access and contact the maintainer of the dataset.**

To test without, please feed the model with a pair of images, the intrinsic and extrinsic matrices. Our visualization Ipython notebook provides a good example. https://github.com/yihui-he/epipolar-transformers/blob/master/READMD.md#1-epipolar-transformers-visualization

### RHD Dataset (for 2D->3D lifting training)
Download using bash scripts
```bash
bash get_RHD.sh
```
Create soft link of datasets
```bash
ln -s ~/hand3d/RHD_published_v2/ .
```

### Custom Dataset
If you are going to create your own dataset, please change the following places:
- `data/datasets/`: dataloader
- `data/build.py`: `if` clause in `build_dataset`
- `core/paths_catalog.py`: `DatasetCatalog.DATASETS` and corresponding `if` clause in `DatasetCatalog.get()`
