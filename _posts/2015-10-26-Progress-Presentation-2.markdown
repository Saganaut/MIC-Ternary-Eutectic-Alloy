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
  * We are now working on the final linking (steady state microstructures)
  * We have created a series of optimization and cross-validation (CV)

{{ page.horizontal }}
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone2_pres/pca.png)

* Previously we had 11 datasets

{{ page.vertical }}

* Now we have 18 datasets
![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone3_pres/new_pca_space.png)

{{ page.vertical }}

* Number of PCs (for 90% variance) increased to 5

![3D-DS](/MIC-Ternary-Eutectic-Alloy/img/milestone3_pres/pca.png)



{{ page.horizontal }}
##Cross-Validation (CV)
* To avoid overfitting we use leave-one-out CV 
* For each Leave-One-Out test, we measure
  * Goodness of fit: $$r^2$$
  * Mean Squared Error (MSE)


{{ page.horizontal }}
##Parameter Optimization 
* k-dimensional-grid parameter estimation using CV
  * Requires training each model hundreds of times
  * Finds approximately optimized parameters per model


{{ page.horizontal }}
##Linear P-S Linkage

* Linear Regression
* Ridge (L2 Regression)
* Lasso (L1 Regression)

{{ page.vertical }}
## Example Linear Model Coeffs
For the solidication velocity case:
$$ y_v = -6.1112E-04x_1+  -6.4253E-04x_2  -3.4660E-06x_3  $$
$$ -2.4926E-04x_4  -2.0120E-04x_5 + 8.51442E-06 $$ 



{{ page.horizontal }}


##General Performance
![MSE](/MIC-Ternary-Eutectic-Alloy/img/milestone3_pres/mse.png)

{{ page.vertical }}

![r2](/MIC-Ternary-Eutectic-Alloy/img/milestone3_pres/r2.png)

{{ page.horizontal }}
##Ongoing Work
* Explore transient datasets and create a time series model
  * Increasing the number of samples from a single simulation
* Running an optimized version of each linkage model
* Nonlinear models 
* 2 Suspicious datasets

<!-- End Here -->
{{ page.horizontal }}

{{ page.horizontal }}
##Non-Linear P-S Linkage 
* Our nonlinear modeling will be covered in subsequent blog posts
  * Linear SVR
  * $$\nu$$-SVR
  * Forest
  * Random Forest

## Questions & Comments

#[Print]({{ site.url }}{{ site.baseurl }}{{ page.url }}/?print-pdf#)

#[Back]({{ site.url }}{{ site.baseurl }})

</section></section>
