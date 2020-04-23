# Splits for MNIST-USPS domain adaptation experiments

This repository contains the revised split protocol for creating splits for few shot domain adaptation on the MNIST-USPS datasets.

Contrary to often seen splits, we define an independent test split here and only let the train-val split vary according to a user-defined random seed.

## Installation
```bash
pip install mnist-usps
```

## Usage
Getting the splits is a simple as:

```python
from mnistusps import mnistusps

train, val, test = mnistusps(
    source_name = "mnist",
    target_name = "usps",
    seed=1,
    num_source_per_class=200,
    num_target_per_class=3,
    same_to_diff_class_ratio=3,
    image_resize=(240, 240),
    group_in_out=True, # groups data: ((img_s, img_t), (lbl_s, _lbl_t))
    framework_conversion="tensorflow",
    data_path = None, # downloads to "~/data" per default
)
```

The function automatically downloads and unpacks the data using Torchvision internally. It then creates the splits using the [Dataset Ops library](https://github.com/LukasHedegaard/datasetops). 
Depending on your choice of machine learning library, the dataset can be converted to Tensorflow or PyTorch (assuming either is pre-installed) using Dataset Ops.


