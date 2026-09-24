# GAM-UNet


## Environment

We recommend setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

Create and activate a new conda environment
```text
conda create -n GAM-UNet
conda activate GAM-UNet
```
Install Dependencies

```text
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```

## Datasets

Arrange RGB images, segmentation masks, and depth maps as follows:

```text
data/
├── TrainDataset/
│   ├── images/
│   ├── masks/
│   └── depth_mask/
├── ValDataset/
│   ├── images/
│   ├── masks/
│   └── depth_mask/
└── TestDataset/
    ├── CVC-300/
    ├── CVC-ClinicDB/
    ├── Kvasir/
    ├── CVC-ColonDB/
    └── ETIS-LaribPolypDB/
```

Each test dataset uses the same `images/`, `masks/`, and `depth_mask/` subdirectories. Corresponding files should align when sorted by filename.

Prepare the data from the project root:

```bash
python dataprepare/prepare_polyp_npy.py
```

The script resizes images, masks, and depth maps to 256 × 256 and writes `data_train.npy`, `mask_train.npy`, `depth_train.npy`, the corresponding validation files, and `*_test_<dataset>.npy` files under `data/`.

## Training

Review the settings in `configs/config_setting.py`, then run:

```bash
python train.py
```

Training outputs are saved under a timestamped `results/GAMM_UNet_polyp_*/` directory. `checkpoints/latest.pth` contains a training checkpoint.

## Testing

Before running the existing test script, set `resume_model` in `test.py` to the checkpoint you want to evaluate. Its current value is an environment-specific absolute path.

```bash
python test.py
```

The script evaluates CVC-300, CVC-ClinicDB, Kvasir, CVC-ColonDB, and ETIS-LaribPolypDB. Metrics are printed and logged; saved prediction masks are written under the run's `outputs/` directory.
