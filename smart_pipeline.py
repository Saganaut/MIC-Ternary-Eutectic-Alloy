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
from pymks import MKSHomogenizationModel
from pymks.tools import draw_component_variance

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Just loads the data from the .mat file in dropbox
def load_data(filename):    
  data = sio.loadmat(filename)
  # The line below is because the matlab format has to be "unstructed"
  #print data['phase_field_solid']
  return shuffle_ndarray(data['phase_field_model'])


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
    metareader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    next(metareader, None)
    for row in metareader:
      metadata.append({'ag':float(row[0]),
                       'cu':float(row[1]),
                       'sv':float(row[2]), 
                       'x':int(row[3]), 
                       'y':int(row[4]), 
                       'filename':row[5]+'.mat'})
  return metadata


# Extracts the "best slice" of the block,
# Currently takes the slice 10th from the end
def get_best_slice(al_data_slice):
  # TODO come up with a better way to summarize the block
  return al_data_slice[-10:-9]


#~~~~~~~MAIN~~~~~~~
if __name__ == '__main__':
  # Load the metadata
  print "-->Loading Metadata"
  metadata = load_metadata('data/metadata_all.tsv')
  
  # Set up the inputs and output containers 
  samples = len(metadata)
  x=np.ndarray(shape=(samples, metadata[0]['x'], metadata[0]['y']))
  y=np.ndarray(shape=(samples, 3)) 

  # For each sample sim in our dataset:
  print "-->Constructing X"
  i = 0
  for metadatum in metadata:
    # Load data frames
    print "--->Loading: " + metadatum['filename']
    metadatum['data'] = load_data('data/test/'+metadatum['filename'])
    # Get a representative slice from the block (or ave or whatever we decide on)
    best_slice = get_best_slice(metadatum['data'])
    x[i] = best_slice 
    y[i,0] = metadatum['ag']
    y[i,1] = metadatum['cu']
    y[i,2] = metadatum['sv']
    i += 1
    

  prim_basis = PrimitiveBasis(n_states=3, domain=[0,2])
  x_ = prim_basis.discretize(x)
  x_corr = correlate(x_, periodic_axes=[0, 1])
  #uncomment to view one containers
  #draw_correlations(x_corr[0].real)

  # Reduce all 2-pt Stats via PCA
  # Try linear reg on inputs and outputs
  reducer = PCA(n_components=3)
  linker = LinearRegression() 
  model = MKSHomogenizationModel(dimension_reducer=reducer,
                                 property_linker=linker,
                                 compute_correlations=False)
  model.n_components = 40
  model.fit(x_corr, y, periodic_axes=[0, 1]) 
  print model.reduced_fit_data
  draw_component_variance(model.dimension_reducer.explained_variance_ratio_)
