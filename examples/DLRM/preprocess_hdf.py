import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Path to input numpy file", required=True)
parser.add_argument("-o", "--output", help="Path to output HDF file", required=True)

args = parser.parse_args()

file = np.load(args.input)
hdf = h5py.File(args.output, 'w')

X_cat = file['X_cat']
X_cat = X_cat.astype(np.long)
hdf.create_dataset("X_cat", data=X_cat)

X_int = file['X_int']
X_int = np.log(X_int.astype(np.float32) + 1)
hdf.create_dataset("X_int", data=X_int)

y = file['y']
y = y.astype(np.float32)
hdf.create_dataset("y", data=y)
