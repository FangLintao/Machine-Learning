# Linear Regression
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Linear%20Regression/Images/linear.png)  
Linear Regression belongs to supervised learning in machine learning. Unlike classification, Linear Regression focuses on real number computation, which makes itself widely used.
# Background
In this sction, Linear Regression is based on the project of Boston Housing. By going through this projects, knowledge in Linear Regression will mix with this project
# the Core
Assumption

    1. The expected value of the residual errors is zero  
    2. The residual errors are uncorrelated and share the same variance  
    3. The residual errors follow a normal distribution
Influences on Regression

    1. Outlier  
    2. Heteroscedasctic data  
    3. Offset

Linear Regression can be consider into the field of polynominal Linear Regression, which means linear regression when polynominal=1 while non-linear regression when polynominal>1.  
### 1. Linear Regression:  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Linear%20Regression/Images/1.png)    
Linear Regression has its own limitation because of its properties of linear function. In real project, most of data distribution cannot be reresented by just a straight line. However, It is the calssic and fundamental regression model in machine learning.  In Boston Housing regresison, Linear function has inferior performance in regression. 

    wi -> the sensitivity of variables  
    w0 -> the offset of linear model  

weight wi shows how much weight variables could contibution or affect the final value. 
From last function, weights is defined by datasets and labels, in this case, we can optimize weights by using large amount of datasets.
### 2. Non-linear regression
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Linear%20Regression/Images/2.png)  
non-linear regression has similar basic properties to linear regression but it has much more powerful performance than linear function because polynominal factors make it able to regulate itself to fit data distribution. However, here is some drawbacks:  

    1. within certain scales of variables, non-linear function has perfect performance to represent datadistribution, but if beyong this scale, the performance drops greatly because a little bit changes could generate super large prediction value which is way beyong the real value.  
    2. non-linear regression could preprocess datasets. 

Basic Properties  

    1. w.r.t weights, non-linear regression is the linear function  
    2. w.r.t variables, non-linear regression is the polynominal function  

### 3.Optimizaton  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Linear%20Regression/Images/optimization.png)  
since we have optimizaion parameter w, then what bridges both is the loss function between prediction values and actual values. Here sre two loss functions widely used, that are L1 and L2 and by combining with punishment methods, Loss functions can penalty for large weight vectors w.  
Two punishment methods

    1. Ridge regression: quadratic loss on residuals with L2 norm penalty on the weights. Analytic formulation for w. Strong influence of outliers -> very hard to punish  
    2. Lasso: quadratic loss on residuals with L1 norm penalty on the weights. No analytic solution, but sparse in w. Reduced influence by outliers -> not hard to punish  
# Boston Housing  
In this project, we explore data distribution to get basic information about each pair of features, and by changes polynominal factor, we would like to discover how polinominal fucntions affect regression performance. 
### 1. Data
in this project, Boston Housing dataset is adapted.

    from sklearn.datasets import load_boston

It contains the following items
    
    CRIM - per capita crime rate by town
    ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS - proportion of non-retail business acres per town.
    CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    NOX - nitric oxides concentration (parts per 10 million)
    RM - average number of rooms per dwelling
    AGE - proportion of owner-occupied units built prior to 1940
    DIS - weighted distances to five Boston employment centres
    RAD - index of accessibility to radial highways
    TAX - full-value property-tax rate per $10,000
    PTRATIO - pupil-teacher ratio by town
    B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT - % lower status of the population
    MEDV - Median value of owner-occupied homes in $1000's

### 2. Data Feature Exploration
the following feature plots offers vivid relations between each apir of features  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Linear%20Regression/Images/features.png)  
### 3. Results 
#### 3.1 prediction and labels distribution
Below image showing prediction and actual label distibution. The ideal case that prediction and actual label values are exacly equal, then it has shows a straight line on graph.  By changing polynominal factorform [1,4], we discover that when polynominal factor=2 , distrbution is tighter. When polynominal factor=1, it is underfitting, when polynominal fatcor>2, it is overfitting.   
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Linear%20Regression/Images/overall.png)  
#### 3.2 the distribution of difference between prediction and labels
we change this distribution into Gaussian distribution, which is much more clear, and this also prove that polynominal function has better performance than linear function.  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Linear%20Regression/Images/Polinomial.png)  
#### 3.3 Loss  
the below image shows three loss functions, before polynominal factor beyond 2, it has slightly decreasing tendency. In this case, it proves that when polynominal factor=2, performance is better.  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Linear%20Regression/Images/loss.png)  
### 4. Conclusion
polynominal function shouldn't be large too much if we would like to have better performance in prediction. Besides, we have to consider that variables would be within reasonable range for better prediction results.
