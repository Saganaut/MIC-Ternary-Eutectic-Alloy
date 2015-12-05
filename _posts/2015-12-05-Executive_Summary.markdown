---
layout:     post	
title:     	Executive Summary (In Progress)	
date:      	2015-12-05 
author:     Robert Pienta, Almambet Iskakov
---
<section markdown="1" data-background="http://matin-hub.github.io/project-pages/img/slidebackground.png"><section markdown="1">
## {{ page.title }}

<hr>

#### {{ page.author }}

#### {{ page.date | | date: "%I %M %p ,%a, %b %d %Y"}}

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
There were less input variables into the simulation than we expected, which would be helpful in creating a process-structure model. There are only two process parameters that were available to us: concentration and solidification velocity. Since concentration of Al was always constant, concentration of either Ag or Cu would be sufficient to know the concentration of the whole material. 

Other challenges.

##2 Point Statistics

##2 Point Statistics Optimization

##Principal Component Analysis

##PCA Optimization

##Final PCA Results - Steady State

##Pricess-Property Linkage Model

##Exploring Transient Data

##Future Work

##Acknowledgements

