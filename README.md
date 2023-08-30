# Linear Regression and Gradient Descent in Python

## Overview

This project aims to provide an easy-to-understand, hands-on example of linear regression and gradient descent in Python. It includes implementations for both least squares linear regression and gradient descent optimizer.

## Features

**Least Squares Linear Regression**

**Mean Squared Error Calculation**

**Gradient Descent Optimization**

**Data visualization using Matplotlib**


## Description

### Subject

Getting and analyzing existing data is the very first task of a data scientist.
The next step is to find tendencies and to generalize.

For example, let's say we want to know what a cat is. We can learn by heart some pictures of cats and then classify them as cat animals that are similar to the pictures.
We then need a way to "measure" similarity. This is called instance-based learning.

Another way of generalizing is by creating a model from the existing examples and making predictions based on that model.

For instance, let's say we want to analyze the relation between two attributes and plot one against the other:

![image](https://storage.googleapis.com/qwasar-public/track-ds/linear_points_nude.png)

We see a trend here, even though the data is quite noisy, it looks like feature 2 goes up linearly as feature 1 increases.
So in the model selection step, we can decide to go for a linear model.

feature_2 = θ0 + θ1 . feature_1

This model has two parameters, θ0 and θ1. After choosing the right values for them, we can make our model represent a linear function matching the data:

![image](https://storage.googleapis.com/qwasar-public/track-ds/linear_points_regressed.png)

Everything stands in "choosing the right values". The "right values" are those for which our model performs "best".
We then need to define a performance measure (how well the model performs) or a cost function (how badly the model performs).

These kinds of problems and models are called **Linear Regression**.

The goal of this journey is to explore linear and **logistic regressions**.


## Introduction

A linear model makes predictions by computing a weighted sum of the features (plus a constant term called the bias term):

y = hθ(x) = θ^T x = θ_n * x_n + ... + θ_2 * x_2 + θ_1 * x_1 + θ_0

• y is the predicted value.

• n is the number of features.

• xi is the ith feature value (with x0 always equals 1).

• θj is the jth model feature weight (including the bias term θ0).

• · is the dot product.

• hθ is called the hypothesis function indexed by θ.

Now that we have our linear regression model, we need to define a cost function to train it, i.e measure how well the model performs and fits the data.
One of the most commonly used functions is the Root Mean Squared Error (RMSE). As it is a cost function, we will need to optimize it and find the value
of theta which minimizes it.

Since the sqrt function is monotonous and increasing, we can minimize the square of RMSE, the Mean Square Error (MSE) and it will lead to the same result.


         m
MSE(X, hθ) =   1⁄m ∑ (θT·x(i) - y(i))2

         k=1

• X is a matrix that contains all the feature values. There is one row per instance.

• m is the number of instances.

• xi is the feature values vector of the ith instance

• yi is the label (desired value) of the ith instance.

## Closed-Form Solution

To find the value of θ that minimizes the cost function, we can differentiate the MSE with respect to θ.
It directly gives us the correct θ in what we call the Normal Equation:

θ = (XT·X)-1·XT·y

(NB: This requires XTX to be reversible).

## Gradient Descent


Reminder about Gradient Descent
As you may have noticed, our MSE cost function is a convex function. This means that to find the minimum, a strategy based on a gradient descent
will always lead us to a global optimum. Remember that the gradient descent moves toward the direction of the steepest slope.

We will write a class to perform the gradient descent optimization.


### Limitations

This is a basic example and is not suitable for large datasets. Future versions may include:

**Batch Gradient Descent**

**Mini-batch Gradient Descent**

**Regularization techniques**


