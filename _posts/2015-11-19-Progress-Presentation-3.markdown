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
* Only two sets of correlations are dependent, clearly the model doesn't need all six
* This doesn't mean all pairs of correlations will work equally!
* We ran our entire pipeline on combinations of two correlations to see which perform the best

{{ page.vertical }}
* This was computationally expensive, but still feasible to do with just 6 choose 2 combinations.
![r2](/MIC-Ternary-Eutectic-Alloy/img/correlations/correlations_r2.png)
{{ page.vertical }}
* Ag-Ag and Al-Al performed the best with $$r^2$$ around 0.74
* Al-Al and Ag-Cu performed very close with $$r^2$$ around 0.72 
![transient](/MIC-Ternary-Eutectic-Alloy/img/correlations/overall.png)
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
* We created a whole new pipeline to perform our transient data linkages.
* Its more than 100x more expensive than the previous pipeline
{{ page.vertical }}
![workwork](/MIC-Ternary-Eutectic-Alloy/img/milestone4_pres/transient_workflow.jpg)
{{ page.horizontal }}
## Transient Data
PCA components of a single simulation over time
![transient](/MIC-Ternary-Eutectic-Alloy/img/transience/PCA_over_block_allstats.png)

* Wild oscillations until the early 100s 

{{ page.vertical}}
Here is just the first 100 points plotted out:
![transient](/MIC-Ternary-Eutectic-Alloy/img/transience/PCA_over_transient.png)

{{ page.vertical}}
A sanity check of our correlation pair from earlier
![transient](/MIC-Ternary-Eutectic-Alloy/img/transience/full_corr_vs_2_corr.png)


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
