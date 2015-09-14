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

* More than 20 simulations
  * Resolution from 200x200 to over 2000x2000
  * Simulated with varied solidification velocities
  * Plenty of measurements!
    * An 800x800 sample with 400 time steps has 256 million data points

{{ page.horizontal }}

## Extracting Features

* we have already started generating spatial statistics (correlations) for our data

![Correlations](/project-pages/img/milestone1_pres/correlations.png)

{{ page.horizontal }}
  
## Time-varying distributions

* Our simulated microstructures vary through the course of each simulation
* we cannot expect the spatial statistics to perform this alone

{{ page.horizontal }}

## Computational Plans

* Consider 

{{ page.horizontal }}

## Simulation in action

<video width="435" height="343" controls>
  <source src="/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/800_1_pp1_film_slo_mo.avi" type="video/avi">
Your browser does not support the video tag.
</video>

{{ page.horizontal }}

<!-- End Here -->
{{ page.horizontal }}

## Questions & Comments

#[Print]({{ site.url }}{{ site.baseurl }}{{ page.url }}/?print-pdf#)

#[Back]({{ site.url }}{{ site.baseurl }})

</section></section>
