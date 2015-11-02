---
layout:     post
title:      First Progress Report Presentation
date:       2015-11-01 12:00:00
author:     Almambet Iskakov, Robert Pienta
tags: 		
---
<!-- Start Writing Below in Markdown -->

## Quick Recap
* Directional Solidification of Al, Ag, Cu Eutectic Alloy
  * Made from state-of-the-art simulations
  * Multiple solidification velocities and volume-fractions
  * Plenty of data, plenty of challenges to overcome
  
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/directional-solid-3d.png)
  
## Our Data (so far) (12x) 800x800x300
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/sim_params.png)

### We can Characterize these points using: 
* Two volume fractions (the third is dependent) 
* Solidification velocity

##Curious Parameters 

* We have more than 12 simulations at this resolution
  * 2 have solidification velocities around 0.6
  * All other experimental velocities are 0.09 or less
  * Is this suspicious?   
  
  * Possibly more data in future

##Our Workflow

![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/workflow/dataflow.png)

Steps with unanticipated challenges:
![dataflow2](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/dataflow1.jpg)

##2-Points Everywhere

* 3 phases (0 - Al, 1 - Ag-Al, 2 - Al-Cu)
* Assume periodic microstructure (based on simulation)
* Finding a representative microstructure

![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/4_microstructures.png)
* Which microstructure is representative?


Example two-point statistics (autocorrelation)
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/auto.png)
*(axes will be fixed)

##Steady-State Solidification

Finding representative microstructure (preliminary method)
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/delta_method.png)
* Compute the difference between spatial correlations to identify convergence/trends

Comparing autocorrelation
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/compare.png)
* High initial variation; steady in full simulation result? 
* Explore other methods (RVE, etc.)

##PCA
Simulated Volumes in PCA-Component-Space
![pca](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/pca.png) 
(grouped by solidification vel.)

Cumulative Variance by PCA Component
![tradeoff](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/decay.png)
Encouraging singular-value fall-off characteristics

##Linkage Overview
Multivariate regression problem:
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/workflow/overview.png)

##Linkage and its application
![linksteps](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/link_steps.jpg)
From a wt% - fraction and solidification vel. to "a microstructure".

##Ongoing Work
* We are currently working on cross-validation for our pipeline.
  * Originally wanted k-fold cross validation, but...

  * That's leave-one-out for ~10 data points.

* We have not completed the reconstruction code, but can produce everything up to it.

##Challenges
* Representing each volume with a microstructure
  * Choosing an RVE
  * Doing an expensive 3D 2-pt statistic
* Choosing which correlations to use as PCA inputs
* No control over simulation data



