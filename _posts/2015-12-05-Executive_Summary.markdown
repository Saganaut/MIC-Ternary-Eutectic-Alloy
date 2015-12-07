---
layout:     post
title:     	Executive Summary (In Progress)
date:      	2015-12-05
author:     Almambet Iskakov, Robert Pienta
---
<section markdown="1" data-background="http://matin-hub.github.io/project-pages/img/slidebackground.png"><section markdown="1">
## {{ page.title }}

<hr>

#### {{ page.author }}

#### {{ page.date | | date: "%b %d %Y"}}

{% raw  %}{% endraw %}
<!-- Start Writing Below in Markdown -->

##Project Definition

###Description
Our data is a product of a phase field simulations on the microstructure evolution in directional solidification of a aluminum-silver-copper ternary eutectoid alloy. The data consists of 21 datasets, while each dataset contains the microstructure infomation through time, from beginning of simulation to steady state. The simulations include varied concentrations and solidification velocities to simulate effect of process parameters of the final microstructure.

##Dataset
The data consists of 21 simulation results datasets, each dataset is 301 microstructure images with 800x800 pixel resolution. For each simulation, the concentration of Al, Ag, and Cu, and solidification velocities is specified. The microstructure image data is can be characterize in the following way 21x301x800x800 in terms of pixel information. Below is a tabulated display of the process parameters.


The microstructure consists of 3 phases, Al, Al-Ag, and Al-Cu. The following colors correspond to each phase.


##Collaboration

##Challenges
There were fewer process (input) variables into the simulation than we expected, which would be helpful in creating a process-structure model. There are only two process parameters that were available to us: concentration and solidification velocity. Since concentration of Al was always constant, concentration of either Ag or Cu would be sufficient to know the concentration of the whole material.

Other challenges.


<iframe width="436" height="344" src="http://www.youtube.com/embed/ZlDdydWGbA4" frameborder="0" allowfullscreen>
</iframe>
* Al = Green, Ag<sub>2</sub>Al = Orange, and Al<sub>2</sub>Cu = Blue

##2 Point Statistics
We extracted 2-point spatial statistics for our data. We assumed a periodic boundary condition for both the x- and y-axis.

##2 Point Statistics Optimization
We investigated which resolutions of 2-pt statistics offered a good balance between computational speed and accuracy.

We also investigated which pairs of correlations perform best with our entire pipeline.
Since we have only three phases, we chose to run the pipeline for all pairs and use the pair with best final model performance (minimal MSE).

![steady](/MIC-Ternary-Eutectic-Alloy/img/exec_summary/correlations_mse.png)


##Principal Component Analysis (PCA)
We used PCA to reduce the large 2-pt statistics to a low rank representation.
Our data exhibit reasonable variance falloff.

![steady](/MIC-Ternary-Eutectic-Alloy/img/exec_summary/decay.png)

##PCA Optimization


##Final PCA Results - Steady State
PCA components of a single simulation over time
![transient](/MIC-Ternary-Eutectic-Alloy/img/transience/PCA_over_block_allstats.png)

* Wild oscillations until the early 120s

##Process-Property Linkage Model

##Exploring Transient Data

##Summary

##Future Work


##Acknowledgements

