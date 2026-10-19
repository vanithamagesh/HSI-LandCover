"""HSI-LandCover: spectral-spatial land-cover classification of hyperspectral and multispectral imagery."""
from .data import (SCENES, S2_BANDS, RGB_BANDS, L8_BANDS, SENSORS, Scene, load_scene, load_custom, preprocess,
                   read_array, gaussian_srf, sensor_bands, spectral_response, split_random, split_blocks, make_split)
from .model import SpectralSpatialFCN, SpectralMLP, build_model, count_parameters, load_checkpoint
from .engine import train, predict, evaluate, confusion, scores
from .baselines import svm

__version__ = "0.4.0"
