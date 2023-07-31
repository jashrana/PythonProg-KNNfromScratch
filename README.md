# PythonProg KNNfromScratch
 Designing a KNN from Scratch

 Code File: 
1. 22222806_22229629__Assignment_2.ipynb

## Assignment 2
In this assignment, we'll put together a clustering method specialised to sequence mining, in the style of Scikit-Learn. We'll use components from Scikit-Learn itself, Numpy, and Matplotlib.

Usually, when we do machine learning, we fit with numerical, rectangular data, so each sample is a feature vector. But the sequences are a bit different. For starters, each sequence might be of a different length. Also, commonalities between sequences might be misaligned, eg these two sequences are pretty similar:

$(0, 0, 1, 4, 7, 0, 0, 1)$

$(0, 1, 4, 7, 0, 1)$

The approach we'll use is to count common subsequences of length $n$, ie $n$-grams. 

Many machine learning algorithms work fine even if we don't have feature vectors, but we have distances between points. 

We will use the $n$-gram counts to compute a measure of dissimilarity between any pair of sequences.

In Scikit-Learn, several algorithms accept a keyword such as `metric="precomputed"` or `affinity="precomputed"`, which allows us to pass in the square matrix of distances instead of passing in the points themselves.

This notebook will walk us through all the steps, with doctests and examples for each. Most of the code needed is based on something we've seen in class.
