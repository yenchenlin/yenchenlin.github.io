---
layout: post
title:  "Hello Google Summer of Code!"
date:   2016-05-02 18:52:21 -0500
categories: GSoc
comments: true
---
![](https://musescore.org/sites/musescore.org/files/Capture%20d'e%CC%81cran%202016-03-01%2009.48.11.png)

In this summer, I will participate in [Google Summer of Code](https://summerofcode.withgoogle.com/) (GSoC for short), a program offers student developers stipends to write code for various open source projects. My proposal for GSoC, [Adding fused types to Cython files](https://summerofcode.withgoogle.com/projects/#5336489219063808), which aims to enhance the popular machine learning library [scikit-learn](http://scikit-learn.org/) has been accepted and will be supervised by two mentors from scikit-learn community: [Jnothman](https://github.com/jnothman) and [Mechcoder](https://github.com/MechCoder).

Below, I'll briefly describe the work I'd like to achieve during GSoC.

## Proposal Abstract

The current implementation of many algorithms in scikit-learn, such as Stochastic Gradient Descent, Coordinate Descent, etc. only allow input with `np.float64` and `np.int64` dtypes due to the adoption of [Cython fused types](http://docs.cython.org/src/userguide/fusedtypes.html) may result in explosion of the generated C code. However, since scikit-learn has removed Cython files from the repo and re-generate them from every build, it provides a good chance to refactor some of the ".pyx" files by introducing Cython fused types. This will allow those algorithms to support `np.float32` and `np.int32` dtypes data, which is currently casted into `np.float64` and `np.int64` respectively, and therefore reduce the waste of memory space.

You can find the detailed version of my proposal [here](https://github.com/scikit-learn/scikit-learn/wiki/GSoC-2016-Proposal:-Adding-fused-types-to-Cython-files)!

## Example

Here, I'll use an example to illustrate how Cython fused types can benefit the whole project.

`mean_variance` function in scikit-learn, like some algorithms I mentioned in my proposal abstract above, will explicitly cast `np.float32` data into `np.float64` before this [pull request](https://github.com/scikit-learn/scikit-learn/pull/6593), which yields waste of memory. However, after we introduce Cython fused types into this function's implementation, it can now accept `np.float32` data directly.

Results of this enhancement can be visualized via memory profiling figures showed below:

- Memory usage **before** using fused types

![](https://cloud.githubusercontent.com/assets/7057863/14371491/dc5a77fc-fd6a-11e5-9d5c-42d17799667f.png)

- Memory usage **after** using fused types

![](https://cloud.githubusercontent.com/assets/7057863/14371492/dc5e585e-fd6a-11e5-8633-5fb68a3d02f9.png)

As one can see, memory usage surrounded by the bracket drastically decrease.

## Summary

I believe that scikit-learn's memory efficiency can be hugely improved after I add fused types into existing Cython files in the project.

On the other hand, great thanks to scikit-learn community for giving me this golden opportunity to work on an open source projects I use every day.

Really Looking forward to this productive summer!
