---
layout:     post
title:      Additional Data
date:       2015-11-01 12:00:00
author:     Almambet Iskakov
tags: 		result
---
<!-- Start Writing Below in Markdown -->

{{page.json}}

## We received more data
We received new data from Yuksel, our project mentor. To remind, our data comes in format of microstructures representations generated during phase field simulations. The process parameters that are attached to the simulations are the concentration of the solid, by wt% of Al, Ag, Cu, and solidfication velocity, ***v***, of the alloy. Our initial data consisted of constant ratios of Al, Ag, and Cu, and varying ***v***. The new data has variations both in ***v*** and wt% of the elements.

### Initial data
Initially, we had 14 datasets of data, each containing 800x800x301 microstructures (first microstructure is a random Voronoi structure). We performed our analysis on the first dataset as outlined by the pipline. Our PCA results looked very good, with 2 PC components capturing about 93% of variance in our data. Note that we didn't use two suspect datasets in which the solidification velocity parameter is an order of magniture higher than the rest of the data. 

### New data
As mentioned before, new data consists of evolving microstructure images with attached process parameters to them: solidification velocity and wt% of Al, Ag, and Cu.

# | *wt% Ag* | *wt%Cu | *Solid. v*
|---------|:----------|:----------:|
is   |Left|  Center  |
a    | cut | it 
column  | short | B

$$#$$ | *wt% Ag* | *wt% Cu* | *Solid. v*
|---------|:----------|:----------:|---------:|
1  |0.237|  0.141  |0.09143|
2  | 0.237 | 0.141 | 0.051285
3  | 0.2391 | 0.1389 | 0.0525
4  | 0.2433 | 0.1347 | 0.0525
5  | 0.2391 |0.1389 | 0.079125
6  | 0.2433 | 0.1347 | 0.079125
7  | 0.2391 | 0.1389 | 0.079125

### Header 3

#Styling:

**Bold**

*Italics*

***Bold and Italics***

#Lists:

1. Item 1

2. Item 2

* Unordered Item

  * Sub Item 1

    1. **Bold** Sub Sub Ordered Item

#Links:

[In-Line](https://www.google.com)

[I'm a reference-style link 1][1]

[I'm a reference-style link 1][2]

[1]:https://www.mozilla.org
[2]:http://www.reddit.com

#Images:

Hover your pointer over the image to expand the view.

![Description](/project-pages/img/Logo_Fairy_Tail_right.png)

#Code:

Inline `code`.

{% highlight python %}
import numpy as np
def hello_world():
    print('Hello World!'')
{% endhighlight %}

#MathJax

Use MathJax for Math.
$$ A = \pi r^2 $$

#Tables:

Here | is | a | row!
|---------|:----------|:----------:|---------:|
is   |Left|  Center  |Right|
a    | cut | it | A
column  | short | B | C

#Quotes

> War does not decide who is *right*, only who is **left**.

# Rule

---

# HTML

Can write the whole post or sections in HTML for very specific needs. [For the advanced user or the code savvy.]

# Customized and Advanced Functionality

Head over to the [documentation page](http://matin-hub.github.io/ppguide/) for tutorials on some basic html formatting and some extensions you can use for cool stuff like interactive 3D visualizations.

# Some HTML Functionality

## Color and Alignment

<p align="center">This text is centered.</p>

<p style="color:red">This is a red text with <span style="color:blue">blue</span> and <span style="color:green">green</span> inline text.</p>

# Some Advanced Features

## Data Projector

<embed src="/project-pages/projectors/projector0001/" height="500px" width="100%">

## STL

<div align="center"><script src="https://embed.github.com/view/3d/matin-hub/project-pages/gh-pages/img/stl/test.stl"></script></div>


