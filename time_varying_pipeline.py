import smart_pipeline as sp

import numpy as np
import cPickle as pickle
import gzip
import bz2
import os
import csv

from pymks.tools import draw_component_variance

from sklearn.decomposition import PCA
# Global Variables
end_of_transience = 100


if __name__ == '__main__':
  print '->Time Varying Analysis'
  print "-->Loading Metadata"
  metadata = sp.load_metadata('data/metadata_all.tsv')

  if os.path.isfile('data/pca_scores_transient.pgz'):
    with gzip.GzipFile('data/pca_scores_transient.pgz', 'r') as f:
      y_pca = pickle.load(f)
  else:
    samples = len(metadata)
    x=np.ndarray(shape=(samples, 2))
    y=np.ndarray(shape=(samples*end_of_transience, metadata[0]['x']*metadata[0]['y']*2))
    y_ind = 0
    x_ind = 0

    for metadatum in metadata:
      # Load data frames
      print "--->Loading: " + metadatum['filename']
      al_chunk = sp.load_data('data/test/'+metadatum['filename'])
      corrs = sp.compute_correlations(al_chunk[0:end_of_transience, :, :],
                                                    correlations=[(0,0), (1,1)],
                                                    compute_flat=False)
      for row in corrs:
        y[y_ind] = row.flatten()
        y_ind+=1
      x[x_ind,0] = metadatum['ag']
      x[x_ind,1] = metadatum['sv']
      x_ind+=1

    # Get some PCA components!
    pca = PCA(n_components=12)
    y_pca = pca.fit_transform(y)
    draw_component_variance(pca.explained_variance_ratio_)
    # Zip and pickle PCA comps
    with gzip.GzipFile('data/pca_scores_transient.pgz', 'w') as f:
      pickle.dump(y_pca, f)
    sp.write_pca_to_csv(y_pca, '_transients')
