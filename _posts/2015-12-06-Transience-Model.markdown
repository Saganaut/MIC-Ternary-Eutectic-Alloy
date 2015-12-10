---
layout:     post
title:     Modeling Transient Behavior 
date:       2015-12-03
author:    Robert Pienta & Almambet Iskakov

---
<!-- Start Writing Below in Markdown -->

## Transience

We have begun working on time-varying models to predict the entire system behavior. 
Thus far we have tried an ensemble approach, VAR and MA.

## MA
When we tested using a moving average we often got poor and unrealistically fit results.  This was probably because the error at different points are dependent.  We performed the Durbin-Watson statiscal test which measures the potential autocorrelation in the predictions. Most of the predictions scored in the range: [0.7, 0.95], which suggests dependent error terms--the cause of the poor results.
 
## VAR
We trained a multivariate autoregression model on our data. We used the statsmodels package, a potential source of error.
We were unable to produce realistic transient behavior with this model.

## Ensemble Model 
We tried an intuitive approach of training a model at each time independently. 
The ensemble uses a single linear regression for each time step.
This allows us to put in a pair of process inputs and get out the entire transient behavior of the PCA values.
This does not use the dependencies between time points, but still performs surprisingly well. 

## Testing the Model
We use leave-one-out cross validation and present a few of the test sets.
![behavior_all](/MIC-Ternary-Eutectic-Alloy/img/time/Ag=0.2433_v=0.0659375_13.png)
This is an example of a good fit with our model.

![behavior_all](/MIC-Ternary-Eutectic-Alloy/img/time/Ag=0.2391_v=0.079125_16.png)
This is an example of the poorest fit of the ensemble.

Here are the MSE values for each tested model in our dataset. 
![behavior_all](/MIC-Ternary-Eutectic-Alloy/img/time/mse.png)
What's interesting is that one of the runs using different intial conditions, for each dataset,  ends up getting consistently poor results.   We discovered that one of the initial conditions (which is repeated for different process parameters) is very different from the others.
Given more time we would investigate the initial conditions affect on steady-state behavior. 

## Using Imagined Inputs
We have tried several different process inputs. To compare we have selected two constant %Ag with varied solidification velocity (v = 0.053 and v = 0.059) and plotted them with the predicted behavior using our model with the same %Ag and v = 0.056.
![behavior_all](/MIC-Ternary-Eutectic-Alloy/img/time/imaginary.png)
The model gets very believeable results!

