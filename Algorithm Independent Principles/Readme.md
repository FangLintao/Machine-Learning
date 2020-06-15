# Algorithm Independent Principles
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Algorithm%20Independent%20Principles/images/cover.png)
## 1. Underfitting, Fitting & Overfitting
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Algorithm%20Independent%20Principles/images/9.png)
## 2. Bias-Variance Analysis & Validation Protocols
### 2.1 Bias-Variance Analysis
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Algorithm%20Independent%20Principles/images/1.png)
Forexample, by given a dataset with numbers of points on plots, and we would like like to use polynominal linear regression to express. However, we basically don't know in which highest polynominal could express well. In this case, what we need to consider is Approximation-Generalization trade-off, in other words, underfitting, fitting and overfitting.  

    * More complex hypothesis class -> more flexibility to approximate target function -> low bias, high variance
    * Less complex hypothesis class -> more likely to generalize well on new data -> high bias, low variance


#### 2.1.1 Risk Minimization
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Algorithm%20Independent%20Principles/images/2.png)
#### 2.1.2 Decompose Risk
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Algorithm%20Independent%20Principles/images/3.png)  
by given different datasets, we obtain different prediction functions ![image](https://github.com/FangLintao/Machine-Learning/blob/master/Algorithm%20Independent%20Principles/images/4.png), and accordingly, different loss ![image](https://github.com/FangLintao/Machine-Learning/blob/master/Algorithm%20Independent%20Principles/images/5.png). In this case, we would like to average overall performance and loss over all datasets.   

![image](https://github.com/FangLintao/Machine-Learning/blob/master/Algorithm%20Independent%20Principles/images/7.png)  
Not practical in real case, Because  

    * requires the ground truth for the bias and noise terms
    * splitting the data into many datasets to estimate the variance

### 2.2 Validation Protocols
#### 2.2.1 train_test_split
in real case, one dataset includes training dataset and testing dataset. However, by considering the model fitting situations, we would like to offer perfect results on testing dataset with just one-time prediction. In this case, how to use dataset is more important.  
training dataset splits itself into two sub datasets, training dataset and validation datasets.    

    1. training dataset -> train model -> pursue low loss and high accuracy  
    2. validation dataset -> evaluate model  
    3. using train_test_split in sklearn.model_selection, and the spliting ratio is generally 8:2  

Advantages  

    . Run K times faster than K-fold cross-validation   
    . Simpler to examine the detailed results of testing process

#### 2.2.2 k-fold Cross-Validation  
-> K can be any number, but k=10 is greatly recommended  
-> For Classification problem, stratified sampling is recommended for creating the folds  
    => Each response class should be represented with equal proportions in each of the k folds  
    => Scikit-learn's cross_val_score fucntion does this by default

    from sklearn.model_selection import KFold

In sklearn, "cross_val_score()" cover K-fold Cross-Validation, "cross_val_score()" automatically send data to model knn,for example, and get scores as results  

    from sklearn.model_selection import cross_val_score  
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

Advantages & Disadvantages  

    Advantage:
    * More accurate estimate of out of sample accuracy  
    * More "efficient" use of data -- [ every observation is used for both training and validation stage ]  
    Disadvantage:
    * validation set and training set are dependent

![image](https://github.com/FangLintao/Machine-Learning/blob/master/Algorithm%20Independent%20Principles/images/8.png)   
#### 2.2.3 Bootstrap
Bootstrap is randomly sampling data from dataset without any order, which is different from K-fold Cross Validation.
Advantage & Disadvantages

    Advantages
    * independent estimates of the loss  
    Disadvantages
    * sample with replacement, we change the training distribution


