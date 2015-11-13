---
layout:     post
title:      Post Headline
date:       2015-11-12 12:00:00
author:     Almambet Iskakov
tags: 		
---
<!-- Start Writing Below in Markdown -->

{{page.json}}

## Truncation of Spatial Statistics
We realized that we can reduce our microstrucure images to a certain 'truncated' size while still retaining all the necessary structure information to represent the spatial statistics. This is one of the recent optimizations will that enable us to save computing time and memory in our project pipeline.

Our current microstructures are 800x800 pixels, and therefore out spatial correlations are also 800x800, per correlation. Truncation will be performed on the microstructures, for which spatial correlations will be calculated.

## Vector Size Trucation
Above a certain vector size, the two point statistics oscillate around a value for which the probability between two local states becomes independent of each other. 

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


