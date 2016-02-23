import smart_pipeline as sp

import numpy as np
import cPickle as pickle
import gzip
import bz2
import os
import csv

import matplotlib.pyplot as plt
from statsmodels.graphics.api import qqplot

from pymks.tools import draw_component_variance

from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.decomposition import PCA
import statsmodels.api as sm

# Global Variables
end_of_transience = 250

def plot_qqplot(arma_mod):
  fig = plt.figure(figsize=(12,8))
  ax = fig.add_subplot(111)
  fig = qqplot(arma_mod.resid, line='q', ax=ax, fit=True)
  plt.show()

def plot_prediction(arma_mod, sample_block, points_to_est=20, filename=None):
  predict_pca = arma_mod.predict(end_of_transience-points_to_est,end_of_transience, dynamic=True)
  plt.hold(True)
  plt.plot(range(end_of_transience), sample_block)
  plt.plot(range(end_of_transience-points_to_est,end_of_transience+1), predict_pca)
  plt.legend();
  if filename != None:
    plt.savefig(filename)
    plt.clf()
  else:
    plt.show()

def leave_one_out(full_x, full_time_y):
  print "-->Leave one out CV"
  loo = LeaveOneOut(21)
  mse = []
  names = []
  for train,test in loo:
    test_name = 'Ag='+str(full_x[test,0][0])+' v='+str(full_x[test,1][0])
    names.append(test_name)
    print "--->Testing " + test_name
    x = full_x[train, :]
    time_y = full_time_y[train, :] 
    regressors = []
    for t in range(end_of_transience):
      linker = linear_model.LinearRegression()
      params = {'fit_intercept':(True, False), 'normalize':(True, False)}
      opt_model = sp.run_gridcv_linkage(x, time_y[:, t], linker, params)
      regressors.append(opt_model.best_estimator_)

    prediction = []
    nu_x = full_x[test]
    for reg in regressors:
      prediction.append(reg.predict(nu_x)[0])
    plt.clf() 
    plt.hold(True)
    plt.title(test_name + ' Prediction vs real') 
    plt.xlabel('Time')
    plt.ylabel('PCA 1')
    plt.plot(range(end_of_transience), full_time_y[test, :].T, label='Actual')
    plt.plot(range(end_of_transience), prediction, '--', label='Predction')
    plt.legend()
    plt.savefig('data/plots/'+test_name.replace(' ', '_')+'_'+str(test[0])+'.png')
    mse.append(mean_squared_error(prediction,full_time_y[test,:].T))
  
  with open('data/mse_transient.csv', 'wb') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    for i in xrange(len(mse)):
      csv_writer.writerow([names[i], mse[i]])
  return mse, names

if __name__ == '__main__':
  print '->Time Varying Analysis'
  print "-->Loading Metadata"
  metadata = sp.load_metadata('data/metadata_all.tsv')
  samples = len(metadata)
  # Set up x
  x=np.ndarray(shape=(samples, 2))
  x_ind = 0
  for metadatum in metadata:
    x[x_ind,0] = metadatum['ag']
    x[x_ind,1] = metadatum['sv']
    x_ind+=1

  # load y
  if os.path.isfile('data/pca_scores_transient.pgz'):
    print '-->Found pickle, Loading!'
    with gzip.GzipFile('data/pca_scores_transient.pgz', 'r') as f:
      y_pca = pickle.load(f)
  elif os.path.isfile('data/pca_scores_transients.csv'):
    print '-->Found csv, Loading!'
    with open('data/pca_scores_transients.csv', 'rb') as csvfile:
      y_pca = np.ndarray(shape=(5250,5))
      csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      y_ind = 0
      for row in csv_reader:
        y_pca[y_ind, :] = np.asarray(map(lambda x: float(x), row))
        y_ind += 1
  else:
    y=np.ndarray(shape=(samples*end_of_transience, metadata[0]['x']*metadata[0]['y']*2))
    y_ind = 0
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
    # Get some PCA components!
    pca = PCA(n_components=12)
    y_pca = pca.fit_transform(y)
    draw_component_variance(pca.explained_variance_ratio_)
    # Zip and pickle PCA comps
    with gzip.GzipFile('data/pca_scores_transient.pgz', 'w') as f:
      pickle.dump(y_pca, f)
    sp.write_pca_to_csv(y_pca, '_transients')
 

  time_y = np.ndarray((samples, end_of_transience))
  row_ind = 0;
  for i in range(0, y_pca.shape[0], end_of_transience):
    time_y[row_ind, :] = y_pca[i:(i+end_of_transience), 0]
    row_ind += 1
  
  mse = leave_one_out(x, time_y)
  print "MSE AVE: " + str(np.mean(mse))

  regressors = []
  for t in range(end_of_transience):
    linker = linear_model.LinearRegression()
    params = {'fit_intercept':(True, False), 'normalize':(True, False)}
    opt_model = sp.run_gridcv_linkage(x, time_y[:, t], linker, params)
    regressors.append(opt_model.best_estimator_)

  prediction = []
  nu_x = np.asarray([0.237, 0.056])
  for reg in regressors:
    prediction.append(reg.predict(nu_x)[0])
  
  plt.hold(True)
  plt.title('Const. %Ag:0.237 With Varying v')
  plt.xlabel('Time')
  plt.ylabel('PCA 1')
  plt.plot(range(end_of_transience), time_y[0, :], label='v:0.053')
  plt.plot(range(end_of_transience), time_y[4, :], label='v:0.059')
  plt.plot(range(end_of_transience), prediction, '--', label='pred v:0.056')
  plt.legend()
  plt.show() 

  #print y_pca.shape
  #print y_pca[0]
#  block_num = 0
#  block_start = block_num*end_of_transience
#  block_end = block_start+end_of_transience
#  sample_block = y_pca[block_start:block_end, 0]
#  print sample_block.shape
#  print sample_block
#  for ar_param in xrange(2,5):
#    arma_mod = sm.tsa.ARMA(sample_block, (ar_param, 0)).fit()
#    print arma_mod.params
#    print 'Durbin-Watson for AR('+str(ar_param)+',0): ', sm.stats.durbin_watson(arma_mod.resid)

 # arma_mod = sm.tsa.ARMA(sample_block, (, 0)).fit() 
#  plot_qqplot(arma_mod)
#  for q in range(1):
#    for p in range(2,30):
#      print '-->AR(',str(p),',',str(q),')'
#      arma_mod = sm.tsa.ARMA(sample_block, (p, q)).fit()
 #     plot_prediction(arma_mod, sample_block, 30, filename='data/plots/arma'+str(p)+'_'+ str(q))

