"""Data layer: series + windows + classification datasets."""
from .series import SeriesDataset, train_windows, windows
from .classify import (ClassificationDataset, direction_dataset, extreme_dataset,
                       volreg_dataset, label_dataset, ucr_dataset, UCR_SMALL)

__all__ = ["SeriesDataset", "windows", "train_windows", "ClassificationDataset",
           "direction_dataset", "extreme_dataset", "volreg_dataset", "label_dataset",
           "ucr_dataset", "UCR_SMALL"]
