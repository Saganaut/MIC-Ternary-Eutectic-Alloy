---
layout:     	slide
title:     	Ternary Eutectoid Alloy
date:      	2015-12-01
author:     	Almambet Iskakov, Robert Pienta

theme:		solarized # default/beige/blood/moon/night/serif/simple/sky/solarized
trans:		default # default/cube/page/concave/zoom/linear/fade/none

horizontal:	</section></section><section markdown="1" data-background="http://matin-hub.github.io/project-pages/img/slidebackground.png"><section markdown="1">
vertical:		</section><section markdown="1">
---
<section markdown="1" data-background="http://matin-hub.github.io/project-pages/img/slidebackground.png"><section markdown="1">
## {{ page.title }}

<hr>

#### {{ page.author }}

#### {{ page.date | | date: "%b %d %Y"}}

{% raw  %}{% endraw %} {{ page.horizontal }}
<!-- Start Writing Below in Markdown -->

**Preface**

1. Quantifying Microstructures 
2. Dimensionality Reduction

<br>
**Ternary Eutectoid Aluminum Alloys**

3. Introduction to Our Project
4. Objectives
5. Data
6. Approach and Methodology
7. Results

{{ page.horizontal}}

## Quantifying Microstructures
2-point Statistics (Preface)

{{ page.horizontal}}

## Dimensionality Reduction



## Ternary Eutectoid Al Alloys
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
  * Three phase metal: (Al, Ag, Cu)
  * Varied Parameters: Temperature Gradient and Gradient Velocity

{{ page.vertical }}

![Boundary-Conditions](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/boundary-conditions.png)

{{ page.horizontal }}

##Directional Solidification Visual

![simulation-slices](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/simulation-slices.png)

{{ page.horizontal }}
## Motivation

{{ page.horizontal }}
## Objectives 

{{ page.horizontal }}
## Data 
* 21 simulations
  * Resolution: 800x800x300 
  * Simulated with varied solidification velocities
  * Varied volume-fractions of Al, Ag<sub>2</sub>Al, and Al<sub>2</sub>Cu
  * Plenty of measurements!
    * An 800x800x4000 sample has over 2.7 billion data points
      * This simulation took 16 hours on 13,700 cores! 


{{ page.horizontal }}
## Workflow

{{ page.horizontal }}
## Workflow

{{ page.horizontal }}
## Conclusions

{{ page.horizontal }}
## Acknowledgements and References 

{{ page.horizontal }}

#[Print]({{ site.url }}{{ site.baseurl }}{{ page.url }}/?print-pdf#)

#[Back]({{ site.url }}{{ site.baseurl }})

</section></section>
