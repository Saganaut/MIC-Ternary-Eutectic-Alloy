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

*Solidification Model:
 *Simulate growth of phases 
 *Thermodynamic model
 *Concentration model
 *Three phase metal
 *Varied Parameters: Temperature Gradient and Gradient Velocity

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

*Eutectic mixture:
 *Liquid transforms into two different phases
 *Lowest melting/freezing temperature


{% raw  %}{% endraw %}{{ page.vertical }}

![ternary-diagram](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/ternary-diagram.png)

*In this Al, Ag, Cu alloy, the eutectic ratios by mole fraction at  at 773.6K are 
 *18% Ag   (25% experimental)
 *69% Al    (62% experimental)
 *13% Cu   (14% experimental)

{{ page.horizontal }}

##Simulated Microtstructure

![colored-phases](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/colored-phases.png)

*Represents cross-section through solidified material close to solidification front
*Most continuous phase: Al
*Chained brick-like structure: Al2Cu and Ag2Al

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
