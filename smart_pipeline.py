# Scroll down the the section marked MAIN

# Imports for loading Data
import numpy as np
import scipy.io as sio
import csv
import os
import h5py
import cPickle as pickle
import gzip

from itertools import combinations

import matplotlib.pyplot as plt

# Import the methods for 2-pt stats
from pymks import PrimitiveBasis
from pymks.stats import correlate
from pymks.tools import draw_correlations
from pymks.tools import draw_microstructures
from pymks import MKSHomogenizationModel
from pymks.tools import draw_component_variance
from pymks.tools import draw_components
from pymks.tools import draw_gridscores_matrix

from sklearn import linear_model, svm, tree
from sklearn.cross_validation import train_test_split, cross_val_score, LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, make_scorer, mean_squared_error

# Just loads the data from the .mat file in dropbox
def load_data(filename):
  data = sio.loadmat(filename)
  # The line below is because the matlab format has to be "unstructed"
  #print data['phase_field_solid']
  return shuffle_ndarray(data['phase_field_model'])


# The pymks takes a different format of data than our .mat data
def shuffle_ndarray(data):
  n,m,k = data.shape
  new_data = np.zeros((k,n,m), dtype=np.uint8)
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
      if float(row[2]) > .2: continue
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


def plot_component_variance(y):
  pca = PCA(n_components=12)
  y_pca = pca.fit_transform(y)
  # Draw the plot containing the PCA variance accumulation
  draw_component_variance(pca.explained_variance_ratio_)

def run_conventional_linkage(x, y, n_comps, linker_model, verbose=0, k_folds=3):
  print "---->Cross validating"
  r2_scorer = make_scorer(r2_score, multioutput = 'variance_weighted')
#  mse_scorer = make_scorer(mean_squared_error, multioutput = 'variance_weighted')
  cvs = cross_val_score(linker_model, x, y, cv=k_folds, scoring=r2_scorer, verbose=verbose)
  mse = cross_val_score(linker_model, x, y, cv=k_folds, scoring='mean_squared_error', verbose=verbose)
  print '---->R2: ', np.mean(cvs)
  print '---->MSE: ', np.mean(mse)
  return np.mean(cvs), np.std(cvs), np.mean(mse), np.std(mse)

def plot_sample_time_variation(block, n_comps=3):
  import matplotlib as mpl
  import matplotlib.cm as cm
  print '-->Plotting Time Variation'
  if os.path.isfile('cache/corr_example.pgz'):
    print "-->Pickle found, loading corr directly"
    with gzip.GzipFile('cache/corr_example.pgz', 'r') as f:
      x_pca = pickle.load(f)
  else:
    n = block.shape[0]
    sample_ind = range(0,301,5)
    corrs, corrs_flat = compute_correlations(block[sample_ind,:,:])
    pca = PCA(n_components=n_comps)
    x_pca = pca.fit(corrs_flat).transform(corrs_flat)
    with gzip.GzipFile('cache/corr_example.pgz', 'w') as f:
      pickle.dump(x_pca, f)
  t = range(0,301,5)
  x = [xi[0] for xi in x_pca]
  y = [xi[1] for xi in x_pca]
  norm = mpl.colors.Normalize(vmin=min(t), vmax=max(t))
  mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)
  colors = [mapper.to_rgba(ti) for ti in t]

  f, (ax1, ax2, ax3) = plt.subplots(3, 1)
  for ax in [ax1, ax2, ax3]:
    for target in ['x', 'y']:
      ax.tick_params(
        axis=target,       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',
        right='off',
        labelbottom='on')  # labels along the bottom edge are off
  ax1.scatter(x, y, color=colors,alpha=0.5, edgecolors='black')
  ax1.set_xlabel('PCA Score 1')
  ax1.set_ylabel('PCA Score 2')
  ax2.plot(t, x)
  ax2.set_xlabel('Time')
  ax2.set_ylabel('PC1')
  ax3.plot(t, y)
  ax3.set_xlabel('Time')
  ax3.set_ylabel('PC2')

  plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
  plt.savefig('PCA_over_block.png')
  quit()

def plot_transient_time_variation(block, n_comps=3):
  import matplotlib as mpl
  import matplotlib.cm as cm
  print '-->Plotting Time Variation'
  if os.path.isfile('cache/corr_transient_example.pgz'):
    print "-->Pickle found, loading corr directly"
    with gzip.GzipFile('cache/corr_transient_example.pgz', 'r') as f:
      x_pca = pickle.load(f)
  else:
    n = block.shape[0]
    sample_ind = range(0,100)
    corrs, corrs_flat = compute_correlations(block[sample_ind,:,:], correlations=[(0,0),(1,1)] )
    pca = PCA(n_components=n_comps)
    x_pca = pca.fit(corrs_flat).transform(corrs_flat)
    with gzip.GzipFile('cache/corr_transient_example.pgz', 'w') as f:
      pickle.dump(x_pca, f)
  t = range(0,100)
  x = [xi[0] for xi in x_pca]
  y = [xi[1] for xi in x_pca]
  norm = mpl.colors.Normalize(vmin=min(t), vmax=max(t))
  mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)
  colors = [mapper.to_rgba(ti) for ti in t]

  f, (ax1, ax2, ax3) = plt.subplots(3, 1)
  for ax in [ax1, ax2, ax3]:
    for target in ['x', 'y']:
      ax.tick_params(
        axis=target,       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',
        right='off',
        labelbottom='on')  # labels along the bottom edge are off
  #ax1.scatter(x, y, color=colors,alpha=0.5, edgecolors='black')
  ax1.scatter(range(0,800, 8), np.ones((100,1)), color=colors,alpha=0.5, edgecolors='black')
  #ax1.set_xlim([-150,150])
  ax1.set_xlabel('PCA Score 1')
  ax1.set_ylabel('PCA Score 2')
  ax2.plot(t, x)
  ax2.set_xlabel('Time')
  ax2.set_ylabel('PC1')
  ax3.plot(t, y)
  ax3.set_xlabel('Time')
  ax3.set_ylabel('PC2')

  plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
  plt.savefig('PCA_over_transient.png')
  quit()

def plot_regression_function(model, x, y, label=None):
  s_plot = np.linspace(np.min(x[:,1]), np.max(x[:,1]), 10)
  ag_plot = np.linspace(np.min(x[:,0]), np.max(x[:,0]), 10)
  synth_data = np.zeros((10,2))
  synth_data[:, 0] = ag_plot
  synth_data[:, 1] = s_plot
  f, (ax, ax1) = plt.subplots(2,1)
  ax.set_title('PCA 1 vs Inputs')
  ax.scatter(x[:,1], y[:,0],  color='black')
  preds = model.predict(synth_data)
  ax.plot(s_plot, preds[:,0], color='red', label=label)
  ax.set_ylabel('PCA 1')
  ax.set_xlabel('v_s')

  ax1.scatter(x[:,0], y[:,0],  color='black')
  ax1.plot(ag_plot, preds[:,1], color='red', label=label)
  ax1.set_ylabel('PCA 1')
  ax1.set_xlabel('% Ag')

  plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
  plt.show()


def plot_components(x):
  draw_components([x[0:3, :2],
                   x[3:6, :2],
                   x[6:9, :2],
                   x[9:12, :2],
                   x[12:13, :2],
                   x[13:14, :2],
                   x[14:17, :2],
                   x[17:18, :2],
                   x[18:19, :2],
                   x[19:20, :2],
                   x[20:21, :2]],
                   ['Ag:0.237 v:0.0525',
                    'Ag:0.237 v:0.0593',
                    'Ag:0.237 v:0.0773',
                    'Ag:0.237 v:0.0844',
                    'Ag:0.239 v:0.0659',
                    'Ag:0.243 v:0.0659',
                    'Ag:0.239 v:0.0791',
                    'Ag:0.239 v:0.0525',
                    'Ag:0.243 v:0.0525',
                    'Ag:0.237 v:0.0914',
                    'Ag:0.237 v:0.0512'])

def plot_a_tree(clf):
  from sklearn.externals.six import StringIO
  try:
    import pydot
  except:
    print 'Error: Install Pydot if you want a tree drawn.'
    return 'No Pydot installed'
  dot_data = StringIO()
  tree.export_graphviz(clf, out_file=dot_data)
  graph = pydot.graph_from_dot_data(dot_data.getvalue())
  graph.write_png("tree_example.png")

def run_gridcv_linkage(x, y, model, params_to_tune, k_folds=3):
  # fit_params = {'periodic_axes':[0, 1]}
  gs = GridSearchCV(model, params_to_tune, cv=k_folds, n_jobs=2, scoring='mean_squared_error').fit(x, y)
  return gs
  # print('Order of Polynomial'), (gs.best_estimator_.degree)
  # print('Number of Components'), (gs.best_estimator_.n_components)
  # print('R-squared Value'), (gs.score(X_test, y_test))

def compute_correlations(x, correlations=None, compute_flat=True):
  print "-->Constructing Correlations"
  prim_basis = PrimitiveBasis(n_states=3, domain=[0,2])
  x_ = prim_basis.discretize(x)
  if correlations == None:
    x_corr = correlate(x_, periodic_axes=[0, 1])
  else:
    x_corr = correlate(x_, periodic_axes=[0, 1], correlations=correlations)
  if compute_flat:
    x_corr_flat = np.ndarray(shape=(x.shape[0],  x_corr.shape[1]*x_corr.shape[2]*x_corr.shape[3]))
    row_ctr = 0
    for row in x_corr:
      x_corr_flat[row_ctr] = row.flatten()
      row_ctr += 1
    return x_corr, x_corr_flat
  return x_corr

def test_polynomial_fits(x, y, n_comps, model, k_folds=3, csvwriter=None):
  for i in range(2,6):
    poly = PolynomialFeatures(degree=i)
    poly_x = poly.fit_transform(x)
    r2_mean, r2_std, mse_mean, mse_std = run_conventional_linkage(poly_x, y, n_comps, model)
    print r2_mean, r2_std, mse_mean, mse_std
    if csvwriter != None:
      if i == 2: name = 'QuadraticReg'
      elif i == 3: name = 'CubicReg'
      elif i == 4: name = 'QuarticReg'      
      elif i == 5: name = 'QuinticReg'      
      csvwriter.writerow([name, mse_mean, mse_std])
    print

def compute_pca_scores(x_flat, n_comps=5):
  pca = PCA(n_components=n_comps)
  x_pca = pca.fit(x_flat).transform(x_flat)
  return x_pca

def test_correlation_combos(x,y):
  print '-->Testing Correlations'
  x_corr, x_corr_flat = compute_correlations(x)
  subsets = [lam_x for lam_x in combinations(range(x_corr.shape[3]), 2)]
  temp_x_corr_flat = np.ndarray(shape=(x.shape[0],  x_corr.shape[1]*x_corr.shape[2]*2))
  results_mat = np.ndarray(shape=(x_corr.shape[3], x_corr.shape[3]))
  n = x.shape[0]
  for subset in subsets:
    print '--->Testing correlations: '+str(subset)
    for i in xrange(n):
      # set the temporary flattened matrix to use only a pair of correlations
      temp_x_corr_flat[i] = x_corr[i, :, :, subset].flatten()
    x_pca = compute_pca_scores(temp_x_corr_flat)
    # Train the model
    linker = linear_model.LinearRegression()
    params = {'fit_intercept':(True, False)}
    opt_model = run_gridcv_linkage(y,x_pca,linker,params)
    r2_mean, r2_std, mse_mean, mse_std = run_conventional_linkage(y,x_pca,5,opt_model)
    results_mat[subset[0], subset[1]] = abs(mse_mean)
    results_mat[subset[1], subset[0]] = abs(mse_mean)

  # Plot the results
  fig, ax = plt.subplots()
  heatmap = ax.pcolor(results_mat, cmap=plt.cm.Blues)
  fig.colorbar(heatmap)
  ax.set_yticks(np.arange(results_mat.shape[0]) + 0.5, minor=False)
  ax.set_xticks(np.arange(results_mat.shape[1]) + 0.5, minor=False)
  labels = ['AlAl', 'AgAg', 'CuCu', 'AlAg', 'AlCu', 'AgCu']
  ax.set_xticklabels(labels, minor=False)
  ax.set_yticklabels(labels, minor=False)
  ax.set_title('R^2 for pairs of correlations')
  plt.show()

def write_pca_to_csv(pcas, title=""):
  with open('data/pca_scores'+title+'.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for row in pcas:
      spamwriter.writerow(row)

def train_test_all_models(x_pca, y, plot_regression=False, filetag=''):
  # LinearModel
  print "-->Optimizing Linkers"
  with open('data/linker_results_r2_data'+filetag+'.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')

    with open('data/linker_results_mse_data'+filetag+'.csv', 'w') as csv_file_mse:
      csv_writer_mse = csv.writer(csv_file_mse, delimiter=',')
      print "--->LinearRegression"
      linker = linear_model.LinearRegression()
      params = {'fit_intercept':(True, False)}
      opt_model = run_gridcv_linkage(y,x_pca,linker,params)
      print('---->Use Intercept: '), (opt_model.best_estimator_.fit_intercept)
      print('---->Normalize: '), (opt_model.best_estimator_.normalize)
      r2_mean, r2_std, mse_mean, mse_std = run_conventional_linkage(y,x_pca,3,opt_model)
      csv_writer.writerow(['LinearRegression', r2_mean, r2_std])
      csv_writer_mse.writerow(['LinearRegression', mse_mean, mse_std])
      print opt_model.best_estimator_.coef_
      if plot_regression:
        plot_regression_function(opt_model.best_estimator_, y, x_pca[:, :2], 'LinearModel')
      print "--->Polynomial fits"
      test_polynomial_fits(y,x_pca,5,opt_model, csvwriter=csv_writer_mse)
      print
      print "--->Lasso"
      linker = linear_model.Lasso()
      params = {'alpha':(.0001,.001,.01,.1,1,10,100)}
      opt_model = run_gridcv_linkage(y,x_pca,linker,params)
      print('---->Best Alpha: '), (opt_model.best_estimator_.alpha)
      r2_mean, r2_std, mse_mean, mse_std = run_conventional_linkage(y,x_pca,5,opt_model)
      csv_writer.writerow(['Lasso', r2_mean, r2_std])
      csv_writer_mse.writerow(['Lasso', mse_mean, mse_std])
      print
      print "--->Ridge"
      linker = linear_model.Ridge()
      params = {'alpha':(.0001,.001,.01,.1,1,10,100)}
      opt_model = run_gridcv_linkage(y,x_pca,linker,params)
      print('---->Best Alpha: '), (opt_model.best_estimator_.alpha)
      r2_mean, r2_std, mse_mean, mse_std = run_conventional_linkage(y,x_pca,5,opt_model)
      csv_writer.writerow(['Ridge', r2_mean, r2_std])
      csv_writer_mse.writerow(['Ridge', mse_mean, mse_std])
      print

      # NON linear
#      print "--->LinearSVR"
#      linker = svm.LinearSVR()
#      r2_mean, r2_std, mse_mean, mse_std = run_conventional_linkage(y,x_pca,5,linker)
#      csv_writer.writerow(['LinearSVR', r2_mean, r2_std])
#      csv_writer_mse.writerow(['LinearSVR', mse_mean, mse_std])

#      print "--->nuSVR"
#      linker = svm.NuSVR()
#      r2_mean, r2_std, mse_mean, mse_std = run_conventional_linkage(y,x_pca,5,linker)
#      csv_writer.writerow(['nuSVR', r2_mean, r2_std])
#      csv_writer_mse.writerow(['nuSVR', mse_mean, mse_std])

      print "--->TreeRegressor"
      linker = tree.DecisionTreeRegressor()
      params = {'max_depth':np.arange(2,50), 'splitter':('best', 'random')}
      opt_model = run_gridcv_linkage(y,x_pca,linker,params)
      print('---->Best max_depth,splitter:'), (opt_model.best_estimator_.max_depth, opt_model.best_estimator_.splitter)
      plot_a_tree(opt_model.best_estimator_)
      r2_mean, r2_std, mse_mean, mse_std = run_conventional_linkage(y,x_pca,5,opt_model)
      csv_writer.writerow(['TreeRegressor', r2_mean, r2_std])
      csv_writer_mse.writerow(['TreeRegressor', mse_mean, mse_std])

#      print "--->RandomForest"
#      linker = RandomForestClassifier()
#      params = {'n_estimators':range(1,100,10)}
#      opt_model = run_gridcv_linkage(y,x_pca,linker,params)
#      print('---->n_est:'), (opt_model.best_estimator_.n_estimators)
#      r2_mean, r2_std, mse_mean, mse_std = run_conventional_linkage(y,x_pca,5,opt_model)

def trim_corrs(x_corr, radius=200):
  print "--->Trimming Corrs"
  x_corr_flat = np.ndarray(shape=(x.shape[0],  radius*radius*4*x_corr.shape[3]))
  row_ctr = 0
  start = x_corr.shape[1]/2-radius
  end = x_corr.shape[1]/2+radius
  for row in x_corr:
    x_corr_flat[row_ctr] = row[start:end, start:end, :].flatten()
    row_ctr += 1
  print x_corr_flat
  return x_corr_flat

def legendre_expansion(y, terms=3): 
  y_leg = np.zeros((y.shape[0], y.shape[1]*terms))
  row_ind = 0
  for row in y:
    feat_ind = 0
    for feat in row:
      for i in range(terms):
        legendre = np.polynomial.legendre.Legendre.basis(i).convert(kind=np.polynomial.Polynomial)
        y_leg[row_ind, feat_ind+i] = legendre(feat)/(2*i+1)
#        print feat, i, legendre(feat)/(2*i+1)
      feat_ind += 3
    row_ind += 1
  return y_leg 

  #~~~~~~~MAIN~~~~~~~
if __name__ == '__main__':
  # Load the metadata
  print "-->Loading Metadata"
  metadata = load_metadata('data/metadata_all.tsv')
  # metadata = load_metadata('data/metadata_all_no_funny_bizness.tsv')

  if os.path.isfile('cache/x_y_data.pgz'):
    print "-->Pickle found, loading x and y directly"
    with gzip.GzipFile('cache/x_y_data.pgz', 'r') as f:
      x, y = pickle.load(f)
  else:
    if not os.path.isdir('cache'):
      os.mkdir('cache')
    # Set up the inputs and output containers
    samples = len(metadata)
    x=np.ndarray(shape=(samples, metadata[0]['x'], metadata[0]['y']))
    y=np.ndarray(shape=(samples, 2))
    solid_vel=np.ndarray(shape=(samples, 3))

    # For each sample sim in our dataset:
    print "-->Constructing X"
    i = 0
    for metadatum in metadata:
      # Load data frames
      print "--->Loading: " + metadatum['filename']
      al_chunk = load_data('data/test/'+metadatum['filename'])
      # Get a representative slice from the block (or ave or whatever we decide on)
      best_slice = get_best_slice(al_chunk)
      x[i] = best_slice
      y[i,0] = metadatum['ag']
      #y[i,1] = metadatum['cu']
      y[i,1] = metadatum['sv']
      solid_vel[i] = metadatum['sv']
      i += 1
    print "-->Pickling x and y"
    with gzip.GzipFile('cache/x_y_data.pgz', 'w') as f:
      pickle.dump((x,y), f)



  # Test correlation params
#  test_correlation_combos(x,y)

  # Plot blocks time varying behavior in PCA space.
#  plot_sample_time_variation(load_data('data/test/'+metadata[0]['filename']))

  # Plot blocks time varying behavior in PCA space.
#  plot_transient_time_variation(load_data('data/test/'+metadata[0]['filename']))


  # Plot components in pca space
#  plot_components(x,y, 5, linear_model.LinearRegression())

  x_corr, x_corr_flat = compute_correlations(x, correlations=[(0,0),(0,2)])

  
  # PCA component variance plot
  #plot_component_variance(x_corr_flat)

 # Use PCA on flattened correlations
  x_pca = compute_pca_scores(x_corr_flat, n_comps=2)
  write_pca_to_csv(x_pca, '_steady_state')
  # Test and Train our models
  train_test_all_models(x_pca, y)

  
  # Trimming Test
  print "========= TESTING TRIMMING ========="
  x_corr_trimmed = trim_corrs(x_corr, 250)
  x_pca_trimmed = compute_pca_scores(x_corr_trimmed, n_comps=2)
  write_pca_to_csv(x_pca_trimmed, '_steady_state_trimmed')
  train_test_all_models(x_pca_trimmed, y, filetag = '_trimmed')
  plot_component_variance(x_corr_trimmed)
  plot_components(x_pca_trimmed)

  
  # Normalization Test
#  print "========= TESTING NORMALIZATION =========" 
  #y_norm = StandardScaler().fit_transform(y)
  #train_test_all_models(x_pca, y_norm)


  # Create Legendre y variance
#  y_leg = legendre_expansion(y,terms=3)
#  train_test_all_models(x_pca, y_leg)
