# Linear Regression from Scratch Using Gradient Descent

This code block demonstrates how to implement Linear Regression from scratch using Gradient Descent optimization. It's designed to provide a hands-on understanding of how Linear Regression works under the hood.

---

## Introduction

Linear Regression is a supervised learning algorithm used to model the relationship between one or more input features and a continuous target variable. It assumes a linear relationship and fits a straight line that best approximates the target values.

---

## Mathematical Intuition

Given training data points $(x_i, y_i)$, Linear Regression tries to find parameters $\theta = [\theta_0, \theta_1, \dots, \theta_n]$ (where $\theta_0$ is the intercept) to model the target as:

$$
\hat{y}_i = \theta_0 + \theta_1 x_{i1} + \theta_2 x_{i2} + \cdots + \theta_n x_{in}
$$

The goal is to minimize the **Mean Squared Error (MSE)** cost function:

$$
J(\theta) = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)^2
$$

where $m$ is the number of training samples.

---

## Gradient Descent Optimization

Gradient Descent is used to minimize the cost function $J(\theta)$ by iteratively updating the parameters $\theta$ in the direction that reduces the error:

$$
\theta := \theta - \alpha \nabla_\theta J(\theta)
$$

where:

- $\alpha$ is the learning rate (controls step size),
- $\nabla_\theta J(\theta)$ is the gradient (vector of partial derivatives) with respect to each parameter.

For Linear Regression, the gradient for each parameter $\theta_j$ is computed as:

$$
\nabla_{\theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)\, x_{ij}
$$

Where:

- $m$ is the number of training samples  
- $\hat{y}_i = \theta_0 + \theta_1 x_{i1} + \cdots + \theta_n x_{in}$ is the predicted value for the $i^{\text{th}}$ sample  
- $x_{ij}$ is the $j^{\text{th}}$ feature of the $i^{\text{th}}$ sample (with $x_{i0} = 1$ for the intercept)

Substituting this into the update rule gives:

$$
\theta_j := \theta_j - \alpha \cdot \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)\, x_{ij}
$$

This update is applied simultaneously for all parameters $\theta_j$ during each epoch.

---

## Methodology

Gradient Descent works by:

1. **Initializing** parameters (coefficients and intercept) to zero.  
2. **Calculating** the gradient of the cost function w.r.t. each parameter.  
3. **Updating** each parameter using the gradient and the learning rate.  
4. **Repeating** the process for a fixed number epochs or until convergence.

This process gradually adjusts the parameters to minimize prediction error and fit the best line through the data.

---

## Requirements

- Python 3.x  
- `numpy` (for numerical computations)

