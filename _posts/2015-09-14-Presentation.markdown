---
layout:     	slide
title:     		Phase Field Models of Ternary Eutectoid Alloys
date:      	2015-09-14 03:00 
author:     	Almambet Iskakov, Robert Pienta

theme:		night # default/beige/blood/moon/night/serif/simple/sky/solarized
trans:		default # default/cube/page/concave/zoom/linear/fade/none

horizontal:	</section></section><section markdown="1" data-background="http://matin-hub.github.io/project-pages/img/slidebackground.png"><section markdown="1">
vertical:		</section><section markdown="1">
---
<section markdown="1" data-background="http://matin-hub.github.io/project-pages/img/slidebackground.png"><section markdown="1">
## {{ page.title }}

<hr>

#### {{ page.author }}

#### {{ page.date | | date: "%I %M %p ,%a, %b %d %Y"}}

{{ page.horizontal }}
<!-- Start Writing Below in Markdown -->

## The Data

* More than 20 simulations (we have only a subset now)
  * Resolution from 200x200 to over 2000x2000 (with depth ranging from 1000 to 4000)
  * Simulated with varied solidification velocities
  * Varied volume-fractions of Al, Ag<sub>2</sub>Al, and Al<sub>2</sub>Cu
  * Plenty of measurements!
    * An 800x800x4256 sample has 2.72 billion data points
    * It took 16 hours on 13700 cores to compute

{{ page.horizontal }}

## Simulation in action

<iframe width="436" height="344" src="http://www.youtube.com/embed/ZlDdydWGbA4" frameborder="0" allowfullscreen>
</iframe>
Each slice was taken at equilibrium 

{{ page.horizontal }}

## Extracting Features

* We have spatial statistics (correlations) for our data

![Correlations](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/correlations.png)

0 = Al, 1 = Ag<sub>2</sub>Al, 2 = Al<sub>2</sub>Cu

{{ page.horizontal }}
  
## Distance-varying distributions

* Our simulated microstructures model directional solidification
* We have the steady-state, solidified structure at numerous solidification-front heights, using the solidification velocity we can calculate the times of each slice  
* With numerous different slices we have several varied microstructures  
  * analyze the differences between successive spatial statistics 
  * find an approach to aggregate over several time steps.
* We may later consider the changes in spatial statistics over time/distance to help mine the Process-Structure linkages

{{ page.horizontal }}

## Computational Plans 
__Next Steps__

* Use dimensionality reduction (DR) over our large space of spatial statistics:
  * Conventional PCA 
  * Newer low-rank approximation DR techniques
  * (possibly) Attempt to use locality sensitive hashing

{{ page.vertical }}
* Model the relationship between our simulated solidification processes
  * linear model (regression)
    * interpretable
    * simple model, unlikely that the features will be linear 
  * nonlinear model (kernel methods)
    * can model complex relationships


<!-- End Here -->
{{ page.horizontal }}

## Questions & Comments

#[Print]({{ site.url }}{{ site.baseurl }}{{ page.url }}/?print-pdf#)

#[Back]({{ site.url }}{{ site.baseurl }})

</section></section>
