---
layout:     post
title:      Plan of Attack
date:       2015-11-03
author:    Almambet Iskakov, Robert Pienta
---
<!-- Start Writing Below in Markdown -->

# Main Objectives

* Study our transient data
	0. We have some time-varying data before a steady state has been reached
	1. Typically the simulations appear to have reached steady state by about simulation frame 150, we want to better quantify this for our simulations
	2. Measure the impact of different initial seeds on the final results
	3. Do more analysis (both high level and numerical) of the strange 0.65 solidification velocity case.

* Use time-varying regression to model our simulation data
	1. Implement a version of the time-dynamic regression that David Brough lectured on.
	2. Test the quality of our models under this dynamic behavior
	3. optimize our models under the time-varying regime

