---
layout:     post
title:      Plan of Attack
date:       2015-09-09 12:00:00
author:     Robert Pienta
tags: 		PoA, plans, plan of attack
---
<!-- Start Writing Below in Markdown -->


### Modeling Process-Structure Linkages Overview
Given the large and varied amount of simulation data we have, there will be several computational challenges involved with the project.  To model the mapping between the process and structure we will have to find a way to correlate the simulated process inputs:

 * n x m x k, initial condition microstructure tensor;
 * volume-fraction of Al, Ag<sub>2</sub>Al, and Al<sub>2</sub>Cu;
 * solidification velocity;

 and the outputs:
 * final n x m x k, microstructure tensor;

We will both build on the successful prior work of others, as well as investigate alternative approaches at different points during the mapping of linkages between the available data.

### Computational Challenges
The initial work using this data tackled many computational issues:
![Plan Overview](/../img/poa_overview.png "Overview of workflow")

1. Finding sub-image invariant features (intra-sample features that remain constant). This problem is slightly simplified in our case since we only have data from one length-scale.
	* n-point statistics (like n=3 in our case) have been successfully leveraged for this purpose by [Niezgoda et al.](http://www.sciencedirect.com/science/article/pii/S1359645408004886#)
	* computer-vision inspired local-feature extraction like crimp, blob, edge, and corner extraction. [This slide deck](http://www.cs.toronto.edu/~urtasun/courses/CV/lecture04.pdf) has good coverage of potential local-feature detection techniques (as well as pointers to the relevant literature)

2. Tracking the time-varying behavior of the features will produce a computationally infeasible number of random variables for many of the simulation sizes. Typically dimensionality reduction through low-rank approximation is used to ameliorate this issue. PCA has a well respected place both in data mining, but also importantly in recent material informatics done by the MINED group ([here](http://www.sciencedirect.com/science/article/pii/S1359645411004654) and [here](http://link.springer.com/article/10.1186%2F2193-9772-2-3)). We will start with PCA, but likely try several other approaches to compare them. We are also interested in trying [locality sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) as well.

3. Once the number of random variables has been reduced to a subset of (hopefully!) representative dimensions, we will need to model the complex relationships among them.  PSP linkages are essential in improving the understanding of the materials pipeline. Im curious to see the performance of various contemporary machine learning models on our reduced dataset (2. above). The literature seems to focus on linear approaches to regression, which we will try along-side some non-linear techniques. I am interested in trying some [kernel methods](https://en.wikipedia.org/wiki/Kernel_method) (for example an SVM is a kernel method) to model the relationships.

### Risks
Major:
* We still have yet to receive the dataset; however, we should have access to it shortly through Yuksel.
* We may not have enough variation in the intial conditions to be able to say anything really inciteful about process-structure linkages in eutectoid Al solidification.

Minor:
* I (Robert) am likely to abuse all this new domain jargon until corrected by members of the class.  Feel free to comment, since comments offer a way for both other students to get their required quota done and for me to learn.