# Import major 'libraries' to do our work for us
import numpy as np
import scipy.io as sio

# Import the methods for 2-pt stats
from pymks import PrimitiveBasis
from pymks.stats import correlate
from pymks.tools import draw_correlations
from pymks.tools import draw_microstructures

# Just loads the data from the .mat file in dropbox
def load_data(filename):
  return sio.loadmat(filename)

# The pymks takes a different format of data than our .mat data
def shuffle_ndarray(data):
  n,m,k = data.shape
  new_data = np.zeros((k,n,m))
  for i in xrange(k):
    new_data[i, :, :] = data[:, :, i]
  return new_data


def analyze_data_slice(al_data_slice):
  # prim basis
  prim_basis = PrimitiveBasis(n_states=3, domain=[0,2])
  disc_basis = prim_basis.discretize(al_data_slice)
  correlations = correlate(disc_basis)
  return correlations

if __name__ == '__main__':
  al_data_blob = load_data('/Users/Astraeus/Dropbox/Project 8883/800_1_pp1.mat')
  al_data = shuffle_ndarray(al_data_blob['phase_field_solid'])

  # draw_microstructures(al_data[10:11, :,:])
  
  # Compute correlations for a single time
  corrs = analyze_data_slice(al_data[10:11, :, :])  
  draw_correlations(corrs[0].real)
