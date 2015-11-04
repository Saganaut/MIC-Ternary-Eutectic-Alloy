---
layout:     post
title:      Plan of Attack
date:       2015-11-03
author:    Almambet Iskakov, Robert Pienta
---
<!-- Start Writing Below in Markdown -->

# Main Objectives

* Reduce our correlations to only the essential
  1. we only need 2 of the correlations to represent the whole space of correlations in our 2-pt stats
  2. we will try different combinations to see how they perform 

* Truncate the 2-pt statistics 
  1. find the point at which we can truncate our 2-pt stats
  2. truncate all of them there to save significantly on the PCA computation

* Study our transient data
	0. We have some time-varying data before a steady state has been reached
	1. Typically the simulations appear to have reached steady state by about simulation frame 150, we want to better quantify this for our simulations
	2. Measure the impact of different initial seeds on the final results
	3. Do more analysis (both high level and numerical) of the strange 0.65 solidification velocity case.

* Use time-varying regression to model our simulation data
	1. Implement a version of the time-dynamic regression that David Brough lectured on.
	2. Test the quality of our models under this dynamic behavior
	3. optimize our models under the time-varying regime

