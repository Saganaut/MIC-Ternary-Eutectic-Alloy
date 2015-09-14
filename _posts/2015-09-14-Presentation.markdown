---
layout:     	slide
title:     		Phase Field Models of Ternary Eutectoid Alloys
date:      	2015-09-14 
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

{% raw  %}{% endraw %} {{ page.horizontal }}
<!-- Start Writing Below in Markdown -->


##Background

* Directional Solidification of Al, Ag, Cu Eutectic Alloy
 * Control structure during solidification process
 * Align phases/structures parallel to growth direction 
 * Achieve desired material properties

![2D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/directional-solidification.png)



{% raw  %} {% endraw %}{{ page.vertical }}

![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/directional-solid-3d.png)



{{ page.horizontal }}

## The Data

* More than 20 simulations
  * Resolution from 200x200 to over 2000x2000
  * Simulated with varied solidification velocities
  * Varied volume-fractions of Al, Ag<sub>2</sub>Al, and Al<sub>2</sub>Cu
  * Plenty of measurements!
    * An 800x800 sample with 400 time steps has 256 million data points

{{ page.horizontal }}

## Simulation in action

<!-- <iframe width="420" height="315" src="http://www.youtube.com/embed/dQw4w9WgXcQ" frameborder="0" allowfullscreen></iframe> -->
<iframe width="436" height="344" src="http://www.youtube.com/embed/ZlDdydWGbA4" frameborder="0" allowfullscreen>
</iframe>


{{ page.horizontal }}

## Extracting Features

* we have already started generating spatial statistics (correlations) for our data

![Correlations](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/correlations.png)

{{ page.horizontal }}
  
## Time-varying distributions

* Our simulated microstructures vary through the course of each simulation
* We can sample the approximate steady-state, solidified structure, 
  * using the difference between successive spatial statistics 
  * or just using one of the last time steps (assuming that the simulation was run until a steady state was reached)
* We may later consider the changes in spatial statistics over time to help mine the Process-Structure linkages

{{ page.horizontal }}

## Computational Plans

* Use dimensionality reduction (DR) over our large space of spatial statistics:
  * Conventional PCA 
  * Newer low-rank approximation DR techniques
  * (possibly) Attempt to use locality sensitive hashing  

{{ page.horizontal }}

<!-- End Here -->
{{ page.horizontal }}

## Questions & Comments

#[Print]({{ site.url }}{{ site.baseurl }}{{ page.url }}/?print-pdf#)

#[Back]({{ site.url }}{{ site.baseurl }})

</section></section>
