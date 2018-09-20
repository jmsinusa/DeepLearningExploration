# DeepLearningExploration
## 1. Learn and use ML
1.1 Basic classification: Use of a small fully connected network in Keras to classify fashion NMIST.
[tensorflow_tutorials/1-LearnAndUseMachineLearning/1-1-BasicClassification.py](tensorflow_tutorials/1-LearnAndUseMachineLearning/1-1-BasicClassification.py)

1.2 Text classification with movie reviews: Classifies movie reviews as positive or negative using the text of the review.
[tensorflow_tutorials/1-LearnAndUseMachineLearning/1-2-TextClassificationWithMovieReviews.py](tensorflow_tutorials/1-LearnAndUseMachineLearning/1-2-TextClassificationWithMovieReviews.py)

1.3 House price regression
[tensorflow_tutorials/1-LearnAndUseMachineLearning/1-3-HousePriceRegression.py](tensorflow_tutorials/1-LearnAndUseMachineLearning/1-3-HousePriceRegression.py)

1.4 Explore overfitting and underfitting.
[tensorflow_tutorials/1-LearnAndUseMachineLearning/1-4-ExploreOverUnderFitting.py](tensorflow_tutorials/1-LearnAndUseMachineLearning/1-4-ExploreOverUnderFitting.py)

1.5 Saving and Restoring models
[tensorflow_tutorials/1-LearnAndUseMachineLearning/1-5-SavingAndRestoringModels.py](tensorflow_tutorials/1-LearnAndUseMachineLearning/1-5-SavingAndRestoringModels.py)

## 2. Research and experimentation

Eager execution provides an imperative, define-by-run interface for advanced operations. Write custom layers, forward passes, and training loops with auto differentiation. Start with these notebooks, then read the eager execution guide.

2.1 Eager execution
[tensorflow_tutorials/2-ResearchAndExperimentation_EagerExec/2-1-EagerExecIntro.py](tensorflow_tutorials/2-ResearchAndExperimentation_EagerExec/2-1-EagerExecIntro.py)

2.2 Automatic differentiation and gradient tape
[tensorflow_tutorials/2-ResearchAndExperimentation_EagerExec/2-2-AutoDiffGradTape.py](tensorflow_tutorials/2-ResearchAndExperimentation_EagerExec/2-2-AutoDiffGradTape.py)

2.3 Custom training: basics
[tensorflow_tutorials/2-ResearchAndExperimentation_EagerExec/2-3-CustomTrainingBasics.py](tensorflow_tutorials/2-ResearchAndExperimentation_EagerExec/2-3-CustomTrainingBasics.py)

2.4 Custom layers
[tensorflow_tutorials/2-ResearchAndExperimentation_EagerExec/2-4-CustomLayers.py](tensorflow_tutorials/2-ResearchAndExperimentation_EagerExec/2-4-CustomLayers.py)

2.5 Custom training: walkthrough
[tensorflow_tutorials/2-ResearchAndExperimentation_EagerExec/2-5-CustomTrainWalkthrough.py](tensorflow_tutorials/2-ResearchAndExperimentation_EagerExec/2-5-CustomTrainWalkthrough.py)

## 3. ML at production scale

3.1. Build a linear model with Estimators

This tutorial uses the tf.estimator API in TensorFlow to solve a benchmark binary classification problem. 
Estimators are TensorFlow's most scalable and production-oriented model type.
[tensorflow_tutorials/3-MLatProductionScale/3-1-LinearModel.py](tensorflow_tutorials/3-MLatProductionScale/3-1-LinearModel.py)

3.2. How to build a simple text classifier with TF-Hub 

TF-Hub is a platform to share machine learning expertise packaged in reusable resources, notably pre-trained modules. This tutorial is organized into two main parts.

*1. Introduction: Training a text classifier with TF-Hub:* We will use a TF-Hub text embedding module to train a simple sentiment classifier with a reasonable baseline accuracy. We will then analyze the predictions to make sure our model is reasonable and propose improvements to increase the accuracy.
*2. Advanced: Transfer learning analysis:* In this section, we will use various TF-Hub modules to compare their effect on the accuracy of the estimator and demonstrate advantages and pitfalls of transfer learning.

[tensorflow_tutorials/3-MLatProductionScale/3-2-TextClassifier.py](tensorflow_tutorials/3-MLatProductionScale/3-2-TextClassifier.py)