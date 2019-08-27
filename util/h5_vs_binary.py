import matplotlib
matplotlib.use('Agg')
import h5py
import random
import os, timeit, struct
import numpy as np
from matplotlib import pyplot as plt
import glob


# Utility functions
def h5_to_binary(in_filename, out_filename):
  f = h5py.File(in_filename, 'r')
  dataset = f['data'][:]
  open(out_filename, 'w').write(dataset.tobytes())


def binary_read(f, d, indexes):
  data = np.zeros((len(indexes), d))
  for i, idx in enumerate(indexes):
    f.seek(4*d*idx)
    data[i] = np.fromfile(f, dtype=np.float32, count=d)
  return data


def binary_read_all(f, d):
  data = np.fromfile(f, dtype=np.float32)
  data = data.reshape(-1, d)
  return data
