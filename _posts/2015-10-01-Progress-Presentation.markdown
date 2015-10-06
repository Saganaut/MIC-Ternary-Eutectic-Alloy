---
layout:     	slide
title:     	Progress Report (Pres)	
date:      	2015-10-01 
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

#### {{ page.date | | date: "%I %M %p ,%a, %b %d %Y"}}

{% raw  %}{% endraw %} {{ page.horizontal }}
<!-- Start Writing Below in Markdown -->

##Quick Recap

* Directional Solidification of Al, Ag, Cu Eutectic Alloy
  * Made from state-of-the-art simulations
  * Multiple solidification velocities and volume-fractions
  * Plenty of data, plenty of challenges to overcome



{{ page.vertical }}
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/directional-solid-3d.png)



{{ page.horizontal }}
##Data: (12x) 800x800x300 

![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/sim_params.png)

We can characterize these points using:

* two volume fractions (the third is dependent) 
* solidification velocity



{{ page.vertical }}
##Curious Parameters 

* We have more than 12 simulations at this resolution
  * 2 have solidification velocities around 0.6
  * all other experimental velocities are 0.09 or less
  * Is this suspicious?   



{{ page.horizontal }}
##Our Workflow

![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/workflow/dataflow.png)



{{ page.vertical }}
Steps with unanticipated challenges:
![dataflow2](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/dataflow1.jpg)



{{ page.horizontal }}
##2-Points Everywhere

* we are using 2-pt stats, duh...
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/2pt-file-here.png)



{{ page.horizontal }}
##Steady-State Solidification
* we must find the steady-state in 2pt stat space..



{{ page.horizontal }}
##PCA
Simulated Volumes in PCA-Component-Space
![pca](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/pca.png) 
(grouped by solidification vel.)



{{ page.vertical }} 
Cumulative Variance by PCA Component
![tradeoff](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/decay.png)
Encouraging singular-value fall-off characteristics



{{ page.horizontal }}
##Linkage Overview
Multivariate regression problem:
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/workflow/overview.png)



{{ page.vertical }} 
##Linkage and its application
![linksteps](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/link_steps.jpg)

From a vol-fraction and solidification vel. to "a microstructure".


{{ page.vertical }} 
##Ongoing Work
* We are currently working on cross-validation for our pipeline.
  * Originally wanted k-fold cross validation, but...

  * That's leave-one-out for ~10 data points.

* We have not completed the reconstruction code, but can produce everything up to it.

{{ page.horizontal }}
##Challenges



{{ page.horizontal }}



<!-- End Here -->
{{ page.horizontal }}

## Questions & Comments

#[Print]({{ site.url }}{{ site.baseurl }}{{ page.url }}/?print-pdf#)

#[Back]({{ site.url }}{{ site.baseurl }})

</section></section>
