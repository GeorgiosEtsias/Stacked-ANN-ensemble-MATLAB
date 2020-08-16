# Stacked-ANN-ensemble

The current project introduces a [script](https://github.com/GeorgiosEtsias/Stacked-ANN-ensemble/blob/master/StackedANNensemble2.m)  generating a stacked learning ensemble, containing a single multilayered ANN (Level-1 learner or meta-learner) trained using the predictions (meta-data) of a number of ANNs (Level-0 learners).

The [Vinyl bromide dataset](https://www.mathworks.com/help/deeplearning/gs/sample-data-sets-for-shallow-neural-networks.html), that is available with the [Deep Learning Toolbox™](https://www.mathworks.com/products/deep-learning.html) was used for training and testing the ensemble. The dataset contains 16 input and 1 output variable and a total of 68308 observations. 

Multiple script executions indicated that the stacked ensemble performed 20% better than the best individual ANN. 
