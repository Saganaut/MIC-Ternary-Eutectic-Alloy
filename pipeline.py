# Scroll down the the section marked MAIN

# Imports for loading Data
import numpy as np
import scipy.io as sio
import csv
import h5py

# Import the methods for 2-pt stats
from pymks import PrimitiveBasis
from pymks.stats import correlate
from pymks.tools import draw_correlations
from pymks.tools import draw_microstructures

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Just loads the data from the .mat file in dropbox
def load_data(filename):    
  data = sio.loadmat(filename)
  # The line below is because the matlab format has to be "unstructed"
  return shuffle_ndarray(data[data.keys()[0]])

# The pymks takes a different format of data than our .mat data
def shuffle_ndarray(data):
  n,m,k = data.shape
  new_data = np.zeros((k,n,m))
  for i in xrange(k):
    new_data[i, :, :] = data[:, :, i]
  return new_data

# Opens the metadata csv in data/metadata.csv, necessary for reading %Ag, %Cu, v etc..
def load_metadata(filename):
  metadata = []
  with open(filename, 'rb') as csvfile:
    metareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    next(metareader, None)
    for row in metareader:
      print row
      metadata.append({'ag':row[0],'cu':row[1],'sv':row[2], 'x':row[3], 'y':row[4], 'filename':row[5]})
  return metadata

# Gets the periodic DFT correlations for a single 2D face of the solid.
def get_correlations_for_slice(al_data_slice):
  # prim basis tellis it to use 0,1,2 as the 3 Stats
  prim_basis = PrimitiveBasis(n_states=3, domain=[0,2])
  disc_basis = prim_basis.discretize(al_data_slice)
  # get the correlations
  correlations = correlate(disc_basis, periodic_axes=(0, 1))
  return correlations


#~~~~~~~MAIN~~~~~~~
if __name__ == '__main__':
  # Load the metadata
  metadata = load_metadata('data/metadata.csv')
  
  # Set up the inputs and output containers 
  samples = len(metadata)
  x=np.ndarray(shape=(samples, metadata[0]['x']**2))
  y=np.ndarray(shape=(samples, 3)) 

  # For each sample sim in our dataset:
  for metadatum in metadata:
    # Load data frames
    metadatum['data'] = load_data('data/'+metadatum['filename'])
    # Get a representative slice from the block (or ave or whatever we decide on)
    best_slice = get_best_slice(metadatum['data'])
    # Get 2-pt Stats for the best slice
    metadatum['stats'] = get_correlations_for_slice(best_slice)
    
  # Reduce all 2-pt Stats via PCA
  # Try linear reg on inputs and outputs
  reducer = PCA(n_components=2)
  linker = LinearRegression() 
  model = MKSHomogenizationModel(dimension_reducer=reducer,
                                 property_linker=linker,
                                 compute_correlations=False)
  model.fit(metadatum['stats']) 

