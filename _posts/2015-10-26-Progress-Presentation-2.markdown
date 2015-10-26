---
layout:     	slide
title:     	Progress Report II (Pres)	
date:      	2015-10-26 
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

* Our pipeline is almost complete
  * we are now working on the final linking 
  * we have created a series of optimization and cross-validation (CV)



{{ page.horizontal }}
##Cross-Validation (CV)
* To avoid overfitting we use leave-one-out CV 
* For each Leave-One-Out test
  * $$r^2$$
  * Mean Squared Error (MSE)




{{ page.horizontal }}
##Parameter Optimization 
* k-dimensional grid estimation using CV
*


{{ page.horizontal }}
##Linear P-S Linkage

* Linear Regression
* Ridge (L2 Regression)
* Lasso (L1 Regression)


{{ page.horizontal }}
##Non-Linear P-S Linkage
* Linear SVR
* $$\nu$$-SVR
* Forest
* Random Forest

{{ page.horizontal }}
##Ongoing Work
* We are currently working on cross-validation for our pipeline.
  * Originally wanted k-fold cross validation, but...

  * That's leave-one-out for ~10 data points.

* We have not completed the reconstruction code, but can produce everything up to it.

{{ page.horizontal }}
##Challenges
* Representing each volume with a microstructure
  * Choosing an RVE
  * Doing an expensive 3D 2-pt statistic
* Choosing which correlations to use as PCA inputs
* No control over simulation data


{{ page.horizontal }}



<!-- End Here -->
{{ page.horizontal }}

## Questions & Comments

#[Print]({{ site.url }}{{ site.baseurl }}{{ page.url }}/?print-pdf#)

#[Back]({{ site.url }}{{ site.baseurl }})

</section></section>
