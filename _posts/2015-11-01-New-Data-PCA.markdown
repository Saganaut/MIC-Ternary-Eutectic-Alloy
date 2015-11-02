---
layout:     post
title:      Additional Data
date:       2015-11-01 12:00:00
author:     Almambet Iskakov
tags: 		result
---
<!-- Start Writing Below in Markdown -->

{{page.json}}

## We received more data
We received new data from Yuksel, our project mentor. To remind, our data comes in format of microstructures representations generated during phase field simulations. The process parameters that are attached to the simulations are the concentration of the solid, by wt% of Al, Ag, Cu, and solidfication velocity, ***v***, of the alloy. Our initial data consisted of constant ratios of Al, Ag, and Cu, and varying ***v***. The new data has variations both in ***v*** and wt% of the elements.

### Initial data
Initially, we had 14 datasets of data, each containing 800x800x301 microstructures (first microstructure is a random Voronoi structure). We performed our analysis on the first dataset as outlined by the pipline. Our PCA results looked very good, with 2 PC components capturing about 93% of variance in our data. Note that we didn't use two suspect datasets in which the solidification velocity parameter is an order of magniture higher than the rest of the data. 

![pca](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/pca.png) 

<!--$$#$$ | *wt% Ag* | *wt% Cu* | *Solid. v*
|---------|:----------|:----------:|---------:|
1  |0.237|  0.141  |0.0525|
2  | 0.237 | 0.141 | 0.0.0525
3  | 0.237 | 0.141 | 0.0.0525
4  | 0.237 | 0.141 | 0.05934375
5  | 0.237 |0.141 | 0.05934375
6  | 0.237 | 0.141 | 0.05934375
7  | 0.237 | 0.141 | 0.077367
8  | 0.237 | 0.141 | 0.077367
9  | 0.237 | 0.141 | 0.077367
10 | 0.237 | 0.141 | 0.0844
11 | 0.237 | 0.141 | 0.0844
12 | 0.237 | 0.141 | 0.0844
-->

### New data
As mentioned before, new data consists of evolving microstructure images with attached process parameters to them: solidification velocity and wt% of Al, Ag, and Cu.

$$ # $$    | *wt% Ag* | *wt% Cu* | *Solid. v*
|----------|:---------|:--------:|---------:|
1          |0.237     |  0.141   |0.09143|
2          | 0.237    | 0.141    | 0.051285
3          | 0.2391   | 0.1389   | 0.0525
4          | 0.2433   | 0.1347   | 0.0525
5          | 0.2391   |0.1389    | 0.079125
6          | 0.2433   | 0.1347   | 0.079125
7          | 0.2391   | 0.1389   | 0.079125

Looking at the percent variance captured by principle components, we see that we now need more PCs to do an effective dimensionality reduction. We see that to capture about 93%, like with initial dataset, we need 7 PCs. It seems that varying both wt% and solidification velocities, ***v***, has additional effects on final microstructures than by varying ***v*** alone. 

![tradeoff](/MIC-Ternary-Eutectic-Alloy/img/milestone3_pres/pca.png)

Looking at the new PC plot of PC1 vs PC2, we see that the different classes, based on process parameters, are still more or less grouped together, but not as clearly as in our initial dataset. 

![pca](/MIC-Ternary-Eutectic-Alloy/img//milestone3_pres/new_pca_space.png) 


So far, the work has been done on steady state results of the simulation. We are also exploring ways to incorporate  transient to our project. Stay updated for these developments.



