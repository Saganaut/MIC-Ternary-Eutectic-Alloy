---
layout:     	slide
title:     	Progress Report (Pres)	
date:      	2015-10-07 
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

* Two volume fractions (the third is dependent) 
* Solidification velocity



{{ page.vertical }}
##Curious Parameters 

* We have more than 12 simulations at this resolution
  * 2 have solidification velocities around 0.6
  * All other experimental velocities are 0.09 or less
  * Is this suspicious?   
  
  * Possibly more data in future



{{ page.horizontal }}
##Our Workflow

![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/workflow/dataflow.png)



{{ page.vertical }}
Steps with unanticipated challenges:
![dataflow2](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/dataflow1.jpg)



{{ page.horizontal }}
##2-Points Everywhere

* 3 phases (0 - Al, 1 - Ag-Al, 2 - Al-Cu)
* Assume periodic microstructure (based on simulation)
* Challenge in choosing a representative microstructure (RVE or average)

![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/4_microstructures.png)

{{ page.vertical }}
Example two-point statistics (autocorrelation)
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/auto.png)



{{ page.horizontal }}
##Steady-State Solidification

Finding representative microstructure (preliminary method)
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/delta_method.png)
* Compute the difference between spatial correlations to identify convergence/trends

{{ page.vertical }}
Comparing autocorrelation
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone1_pres/compare.png)
* High initial variation; steady in 
* Full simulation result? 
* Explore other methods (RVE, etc.)

{{ page.horizontal }}
##PCA

* Sample Result:
![pca](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/pca.png)



{{ page.vertical }} 
Cumulative Variance by PCA Component
![tradeoff](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/decay.png)
Encouraging singular-value fall-off characteristics

{{ page.horizontal }}
##Linkage Overview
Multivariate regression problem:
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/workflow/overview.png)



{{ page.horizontal }}
##Challenges
* Choosing an RVE
* PCA inputs
* No control over simulation data


{{ page.horizontal }}



<!-- End Here -->
{{ page.horizontal }}

## Questions & Comments

#[Print]({{ site.url }}{{ site.baseurl }}{{ page.url }}/?print-pdf#)

#[Back]({{ site.url }}{{ site.baseurl }})

</section></section>
