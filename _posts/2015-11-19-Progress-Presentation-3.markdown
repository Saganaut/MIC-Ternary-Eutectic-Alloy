---
layout:     	slide
title:     	Progress Report III (Pres)	
date:      	2015-11-19 
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
* Reduce our correlations to only the essential

* Truncate the 2-pt statistics 

* Study our transient data

* Use time-varying regression to model our simulation data

{{ page.vertical }} 

*  Reduce our correlations to only the essential ✓ 

* Truncate the 2-pt statistics ✓ 

* Study our transient data ✓ 

* Use time-varying regression to model our simulation data

{{ page.horizontal }}
## Reducing Correlations

{{ page.horizontal }}
## Truncating statistics
* Truncation based of average 2-pt statistics in each sample in steady state
<iframe width="436" height="344" src="http://www.youtube.com/embed/ZlDdydWGbA4" frameborder="0" allowfullscreen>
</iframe>
*Al = Green, Ag<sub>2</sub>Al = Orange, and Al<sub>2</sub>Cu = Blue

{{ page.vertical }}
### Choosing a vector size
<!-- Placeholder -->
![vector_size](/MIC-Ternary-Eutectic-Alloy/img/truncation/truncation_schematic.png) 

{{ page.vertical }}
### Example for autocorrelation
![horizontal_auto](/MIC-Ternary-Eutectic-Alloy/img/truncation/horizontal_auto.png)

{{ page.vertical }}
### All steady state data
![combined_violation](/MIC-Ternary-Eutectic-Alloy/img/truncation/combined_violations.png)

{{ page.horizontal }}

## A New Pipeline
{{ page.horizontal }}

## Expectations
{{ page.vertical}}

## Challenges

{{ page.horizontal }}

![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/pca.png)

* Previously we had 11 datasets

{{ page.vertical }}

#[Print]({{ site.url }}{{ site.baseurl }}{{ page.url }}/?print-pdf#)

#[Back]({{ site.url }}{{ site.baseurl }})

</section></section>
