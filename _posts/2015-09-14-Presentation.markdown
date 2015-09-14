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

{{ page.vertical }}

![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/directional-solid-3d.png)

{{ page.horizontal }}

##Phase-Field Simulation: Solidification

* Solidification Model:
  * Simulate growth of phases 
  * Thermodynamic model
  * Concentration model
  * Three phase metal
  * Varied Parameters: Temperature Gradient and Gradient Velocity

{{ page.vertical }}

![Boundary-Conditions](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/boundary-conditions.png)

{{ page.horizontal }}

##Directional Solidification Visual

![simulation-slices](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/simulation-slices.png)

{{ page.vertical }}

![lamellae](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/lamellae.png)

Phases evolve parallel to growth direction.

{{ page.horizontal }}

##Binary Phase Diagram

![Binary Phases](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/binary-diagram.png)

* Eutectic mixture:
  * Liquid transforms into two different phases
  * Lowest melting/freezing temperature


{% raw  %}{% endraw %}{{ page.vertical }}

![ternary-diagram](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/ternary-diagram.png)

* In this Al, Ag, Cu alloy, the eutectic ratios by mole fraction at 773.6K:
  * 18% Ag   (25% experimental)
  * 69% Al    (62% experimental)
  * 13% Cu   (14% experimental)

{{ page.horizontal }}

##Simulated Microtstructure

![colored-phases](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/colored-phases.png)

* A cross-section of solidified material close to solidification front
* Most continuous phase: Al
* Chained brick-like structure: Al<sub>2</sub>Cu and Ag<sub>2</sub>Al

{{ page.horizontal }}
##Common Microstructure Patterns

![ms-patterns](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/ms-patterns.png)

{{ page.horizontal }}

## The Data

* More than 20 simulations
  * Resolution from 200x200 to over 2000x2000
  * Simulated with varied solidification velocities
  * Varied volume-fractions of Al, Ag<sub>2</sub>Al, and Al<sub>2</sub>Cu
  * Plenty of measurements!
    * An 800x800x4256 sample has 2.72 billion data points
      * This simulation took 16 hours on 13700 cores! 

{{ page.horizontal }}

## Simulation in action

<!-- <iframe width="420" height="315" src="http://www.youtube.com/embed/dQw4w9WgXcQ" frameborder="0" allowfullscreen></iframe> -->
<iframe width="436" height="344" src="http://www.youtube.com/embed/ZlDdydWGbA4" frameborder="0" allowfullscreen>
</iframe>
Al = Green, Ag<sub>2</sub>Al = Orange, and Al<sub>2</sub>Cu = Blue

{{ page.horizontal }}

## Extracting Features

We have already started generating spatial statistics (correlations) for our data

![Correlations](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/correlations.png)

0 = Al, 1 = Ag<sub>2</sub>Al, and 3 = Al<sub>2</sub>Cu

{{ page.horizontal }}
  
## Distance-varying Distributions

* The simulated microstructures vary through the course of each simulation
* We can sample the approximate steady-state, solidified structure at various times, 
  * using the difference between successive spatial statistics 
  * or just using one of the last time/height steps (assuming that the simulation was run until a steady state was reached)
* We may later consider the changes in spatial statistics over time to help mine the Process-Structure linkages

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
    * simple model, unlikely that Eutectoid Al has linear structural relationships  
  * nonlinear model (kernel methods)
    * less interpretable
    * complex model, can model complex relationships


<!-- End Here -->
{{ page.horizontal }}

## Questions & Comments

#[Print]({{ site.url }}{{ site.baseurl }}{{ page.url }}/?print-pdf#)

#[Back]({{ site.url }}{{ site.baseurl }})

</section></section>
