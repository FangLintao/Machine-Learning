# Boosting
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Boosting/images/cover.png)  
###### Reference: ["Understanding AdaBoost for Decision - An implementation with R"](https://towardsdatascience.com/understanding-adaboost-for-decision-tree-ff8f07d2851), Valentina Alto, Jan 11
## 1. Brief Introduction
Boosting is an emsemble model techniques that establishes upon weak models. For each model added into boosting, it is prone to correct errors generating from former sequence of models, in this case, boosting tends to be overfitting, and in order to avoid overfitting case, hyperparameters play an important role in boosting.  
Only when errors are small enough or the number of models reaches its maximum, then boosting algorithm will stop.
## 2. Dataset
In this project, we split it into two sections, the boosting and gradient boosting.  
##### For Boosting: [mushroom.csv](https://datahub.io/machine-learning/mushroom)
Since mushroom.csv contains alphabets as values to represent each column, we use the following codes to convert alphabet into numbers    

    from sklearn.preprocessing import LabelEncoder  
    for label in dataset.columns:
      dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])


##### For Gradient Boosting: [Titanic Dataset](https://www.kaggle.com/c/titanic/data)
## 3. Core
##### Basic idea: iteratively construct submodels Gm that fix previous errors  
### 3.1 Relationship to Bagging
##### Difference

    Ⅰ. Boosting trains submodels sequentially, with the m-th submodel trained to fix the mistakes of the first m − 1 submodels   
       Bagging, it is not sequential because all of training on subsampling dataset is independent  
    Ⅱ. Boosting can optionally give different weights to its submodels

##### Same

    Ⅰ. Boosting can optionally bootstrap samples and use random feature subsets  

### 3.2 Adaboost
##### mistakes are identified by weightings on more “difficult” data points, Each submodel also gets a different weight, based on how “good” it is  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Boosting/images/1.png)
#### Pesudcode Algorithm
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Boosting/images/2.png)
###### Reference: "Chapter 10: Boosting and Additive Trees", Hastie, Tibshirani and Friedman
#### Drawback
if the data is very noisy, it can overfit badly, it's very sensitive to outliers  
### 3.2 Gradient Boost

      Ⅰ。Gradient Boosting Classifier depends on a loss function  
      Ⅱ. Custom loss functions and standardized loss functions are supported by gradient boosting classifiers   
      Ⅲ. Loss function has to be differentiable

##### two necessary component: a weak learner and an additive component
Weak Learner  

    Gradient boosting systems use decision trees - Regression Trees, as their weak learners.  
      Ⅰ. Regression Tree output is real values  
      Ⅱ. As new learners are added into the model the output of the regression trees can be added together to correct for errors in the predictions

Additive Component

    Ⅰ。trees are added to the model over time
    Ⅱ. When this occurs, the existing trees aren't manipulated, so their values remain fixed

![image](https://github.com/FangLintao/Machine-Learning/blob/master/Boosting/images/3.png)
### 3.3 Connection between Adaboost and Gradient Boost
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Boosting/images/4.png)
### 3.4 Regularisation for Tree Complexity
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Boosting/images/5.png)
## 4. Project
when selecting training model as DecisionTreeClassifier, setting criterion as entropy

    Tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=1)

the accuracy is around 73%, but after using boosting with 50 base model, the accuracy is up to 99%, which proves that prediction error gets smaller when sequently adding weak models into boosting algorithm   
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Boosting/images/6.png)

