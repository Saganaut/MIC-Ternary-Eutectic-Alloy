---
layout:     post
title:      Second Progress Report Presentation
date:       2015-11-01 12:00:00
author:     Materials Innovation
tags: 		result
---
<!-- Start Writing Below in Markdown -->

##Quick Recap

* Our pipeline is almost complete
  * We are now working on the final linking (steady state microstructures)
  * We have created a series of optimization and cross-validation (CV)

![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/pca.png)

* Previously we had 11 datasets

* Now we have 18 datasets
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone3_pres/new_pca_space.png)

* Number of PCs (for 90% variance) increased to 5

![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone3_pres/pca.png)

##Cross-Validation (CV)
* To avoid overfitting we use leave-one-out CV 
* For each Leave-One-Out test, we measure
  * Goodness of fit: $$r^2$$
  * Mean Squared Error (MSE)

##Parameter Optimization 
* k-dimensional-grid parameter estimation using CV
  * Requires training each model hundreds of times
  * Finds approximately optimized parameters per model

##Linear P-S Linkage
* Linear Regression
* Ridge (L2 Regression)
* Lasso (L1 Regression)

##Non-Linear P-S Linkage 
* Our nonlinear modeling will be covered in subsequent blog posts
  * Linear SVR
  * $$\nu$$-SVR
  * Forest
  * Random Forest

## Example Linear Model Coeffs
For the solidication velocity case:
$$ y_v = -6.1112E-04x_1+  -6.4253E-04x_2  -3.4660E-06x_3  $$
$$ -2.4926E-04x_4  -2.0120E-04x_5 + 8.51442E-06 $$ 

##General Performance
![MSE](/MIC-Ternary-Eutectic-Alloy/img/milestone3_pres/mse.png)

![r2](/MIC-Ternary-Eutectic-Alloy/img/milestone3_pres/r2.png)

##Ongoing Work
* Explore transient datasets and create a time series model
  * Increasing the number of samples from a single simulation
* Running an optimized version of each linkage model
* Nonlinear models 
* 2 Suspicious datasets

