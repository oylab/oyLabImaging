# __init__.py
import oyLabImaging.Processing as Processing
from oyLabImaging.Metadata import Metadata

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
os.environ["HDF5_USE_FILE_LOCKING"] = "False"
os.environ["OMP_NUM_THREADS"] = "2"

__version__ = "0.2.6"