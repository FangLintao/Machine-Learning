# Linear Discriminant Analysis
![image](https://github.com/FangLintao/Machine-Learning/blob/master/LDA/images/cover.png)  
###### Reference: ["Linear Discriminant Analysis for Starters"](https://app.slack.com/client/T011DL7B6A1/D0143NRS5G8), George Ho  
Linear Discriminant Analysis, short for LDA, is a classic method in supervised learning. It belongs to the linear classification family and usually it is most fundamental.
## the Core
##### fucntions

    Dimension Reduction & Feature Selection  

##### assumptions

    1. Gaussian data distributions 
    2. all of class covariance are equal

![image](https://github.com/FangLintao/Machine-Learning/blob/master/LDA/images/9.png)  
Linear Discriminant Analysis is a method linearly characterizing objects or features. Usually, one obejects could have D features that describe itself and not all of them are important or, in other words, only some of them contribute highly to this obejct. In this case, lInear discriminant analysis can lower the dimensions from D to d dimensions [D>>d], which simplify complexity of analyse object classification. After this, datasets could be transfrom form D dimensions down to d dimensions, in other words, from original space to subspace, which collects the most highly informative features.   
In short, LDA simplify the analysis on object classification by removing less important features.  
To do so,LDA involves the following things:  

    1. mean -> class average value   
    2. Sk -> class covariance  
    3. Sw -> within-class covariance  
    4. Sb -> between-class covariance
 
 ## Data
 In this partion, Iris datasets is used to accomplish this task.
 
    1. name => "setosa", "versicolor", "virginica"  
    2. features => "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"
 
 ## Analysis
 ### Feature Extraction
 it is helpfule to plot each feature from three objects in order to discover and get familiar with Iris datasets  
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/LDA/images/1.png)   
 Discovering:  above image offers certain valuable infrormation to LDA. for these four fetaures, only "petal length" & "petal width" show less ovelaps in these three classes, in this case, LDA can select two features, that are "petal length" & "petal width"; Meanwhile, based on these two features with highly informative characters, LDA is allowed to reduce original four dimensions down to two dimensions.   
 ### class Features
 Discovering: in this section, this image just offers feature combination in each class.  
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/LDA/images/2.png)  
 ### Covariance
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/LDA/images/class%20mean%20value.png)
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/LDA/images/overall%20mean%20value.png)
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/LDA/images/class%20covariance.png)
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/LDA/images/within-class%20covariance.png)  
 
     within-class covariance is to determine the level of features distribution. In object classification, we want features inside one object can gather tightly, so that chances of overlap can be smaller. So, within-class covariance should be as SMALL as possible  
 
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/LDA/images/between-class%20covariance.png)
 
    between class covariance is to determine the distribution between classes. In object classification, we want classes to be seperate as much as possible in order to easily classify them. In this case, between-class covariance should be large  

 ### eigenvalues & eigenvectors
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/LDA/images/8.png)
 
 LDA is to analysis on relationship between within-class covariance and between-class covariance by setting a parameter W. In this equation, thefor the value, nomiator should be larger while the denominator should be small, so that the totall value J(W) can reach maxmimum. This is relevant to the properties of within-class covariance and between-class covariance; for the matrix, we use eigen value and eigen vectors to represent this equation, and meanwhile, the most highly informative vectors should be selected in order to achieve dimension reduction, so that original space is transfered into subspace.
 ## Result
 this image shows that datasets in new subspace distribution. The classification is clear  
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/LDA/images/3.png)
