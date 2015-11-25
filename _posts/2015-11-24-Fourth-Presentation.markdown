---
layout:     	post
title:     	Progress Report III (Post)
date:      	2015-11-24
author:     	Almambet Iskakov, Robert Pienta

---

##Quick Recap

*  Reduce our correlations to only the essential ✓

* Truncate the 2-pt statistics ✓

* Study our transient data ✓

* Use time-varying regression to model our simulation data

![steady](/MIC-Ternary-Eutectic-Alloy/img/milestone4_pres/regplot.png)

## Reducing Correlations
* Only two sets of correlations are dependent, clearly the model doesn't need all six
* This doesn't mean all pairs of correlations will work equally!
* We ran our entire pipeline on combinations of two correlations to see which perform the best

* This was computationally expensive, but still feasible to do with just 6 choose 2 combinations.
![r2](/MIC-Ternary-Eutectic-Alloy/img/correlations/correlations_r2.png)

* Ag-Ag and Al-Al performed the best with $$r^2$$ around 0.74
* Al-Al and Ag-Cu performed very close with $$r^2$$ around 0.72
![transient](/MIC-Ternary-Eutectic-Alloy/img/correlations/overall.png)

## Truncating statistics
* Truncation based of average 2-pt statistics in each sample in steady state

<iframe width="436" height="344" src="http://www.youtube.com/embed/ZlDdydWGbA4" frameborder="0" allowfullscreen>
</iframe>
* Al = Green, Ag<sub>2</sub>Al = Orange, and Al<sub>2</sub>Cu = Blue

### Choosing a vector size

![vector_size](/MIC-Ternary-Eutectic-Alloy/img/milestone4_pres/truncation_vector.png)


### Example for autocorrelation
![horizontal_auto](/MIC-Ternary-Eutectic-Alloy/img/truncation/horizontal_auto.png)

### All steady state data
![combined_violation](/MIC-Ternary-Eutectic-Alloy/img/milestone4_pres/truncation_loss.png)

## A New Pipeline
* We created a whole new pipeline to perform our transient data linkages.
* Its more than 100x more expensive than the previous pipeline

![workwork](/MIC-Ternary-Eutectic-Alloy/img/milestone4_pres/transient_workflow.jpg)

### Transient Data
PCA components of a single simulation over time
![transient](/MIC-Ternary-Eutectic-Alloy/img/transience/PCA_over_block_allstats.png)

* Wild oscillations until the early 100s

Here are just the first 100 points plotted out:
![transient](/MIC-Ternary-Eutectic-Alloy/img/transience/PCA_over_transient.png)

A sanity check of our correlation pair from earlier
![transient](/MIC-Ternary-Eutectic-Alloy/img/transience/full_corr_vs_2_corr.png)

## Future Work
* (Nov) modeling the time-varying behavior of our system (we are close!)
* (Nov) post about transience
* (Nov) post about steady state performance
* (Dec) Final "In Summa" Post
* (Dec) Final Presentation
