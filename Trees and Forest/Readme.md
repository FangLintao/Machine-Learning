# Trees and Forest
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Trees%20and%20Forest/images/cover.png)
## 1. Introduction
Tree and Forest is non linear method in Machine Learning, including classification and regression. Such a method has tree structure with the internal node representing features, the branch representing the decision rule, and each leaf node representing the outcome. Tree and Forest shares internal decision-making logic, which makes itself the white box in MachineLearning Algorithm and which is unlike black box such as Neural Network. 
## 2. Core

    * Decision Trees + Bagging = Bagged trees;  
    * Decision Trees + random feature subsets + Bagging = Random forest;  
    * Decision Trees with almost random splits + Bagging = ExtraTrees;  

### 2.1 CART algorithm

    Ⅰ. Use Attribute Selection Measures, ASM, to split records;  
    Ⅱ. Make that attribute a decision node and breaks the dataset into smaller subsets;  
    Ⅲ. Repeat spliting process recursively until each child with same attribute value;  

Attribute selection measure:   

    * A heuristic for selecting the splitting criterion that partition data into the best possible manner
    * To determine breakpoints for tuples on a given node  

![image](https://github.com/FangLintao/Machine-Learning/blob/master/Trees%20and%20Forest/images/3.png)
### 2.2 Classification&Regression Tree
When using Desicion Tree in sklearn  

    from sklearn.tree import DecisionTreeClassifier  
    clf = DecisionTreeClassifier()  
    defalut criterion = gini

#### Split Criterion
Split Criterion: Information Gain, Gain Ratio, and Gini Index  
Ⅰ. Information Entropy  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Trees%20and%20Forest/images/4.png)  

    *  information entropy : to determine information uncertainty  
    *  higher uncertainty, the higher entropy  
    *  larger P(x) is, the smaller uncertainty

Ⅱ. Information Gain  
-> Information gain computes the difference between entropy before split and average entropy after split of the dataset based on given attribute values  
Maximizing information gain
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Trees%20and%20Forest/images/5.png)   

    -> NH(V): the average amount of information needed to identify the class label of a tuple in D;  
    -> NlH(Vl)+HrN(Vr) : the expected information required to classify a tuple from D;  
    -> For each children node, we want to reduce uncertainty, which means lower uncertainty  

#### Hyperparameters  

    * Minimum number of samples in a leaf (min leaf)
    * Maximal depth of the tree (max depth)
    * Total number of nodes
    * Leaf model (weak learner; here constant)
    * Split criterion

#### Bias and Variance of Trees
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Trees%20and%20Forest/images/6.png)   
Ⅰ. Trees are a high-variance model -> slightly change the data, might get a very different tree  
Ⅱ. Trees are a low-bias model -> fits all kinds of functions
#### Time Complexity
Background: N data points of dimensionality D  
Ⅰ. Finding the best split value for a given feature (pre-sorted): O(N)  
Ⅱ. Finding best split point at root: O(DN)  
Comparion:

    * by comparing with kernel SVM, if SVM has more than 10 thousand datapoints, then kernel svm will be really slow
    * by comparing with linear SVM, Linear SVM doesn't care about the number of data points

Best case: balanced trees  

    -> Fitting: T (N ) = O(DN ) + 2T (N/2)  
    -> This leads to O(DNlogN) since we pay O(D·N) at each of logN levels  
    -> Prediction: O(log N )  

Worst case: splitting off one data point at a time  

    -> Fitting: T (N) = O(DN) + T (N − 1); this leads to O(DN^2)  
    -> Prediction: O(N)

#### Pros&Cons

    Pros:  
    * Flexible framework with exchangeable components:splitting criterion, leaf model, type of split;  
    * Interpretability  
    * Handle categorical input values natively  
    * Handle unimportant features well  
    * Scalable for large datasets  
    Cons:  
    * Tend to overfit
    * Deterministic, i.e. not suitable for some ensemble methods

### 2.3 Bagging

    Bagging = Boostrap

Ⅰ. Train N models on bootstrap samples of the training data;  
Ⅱ. For each model, data is drawn randomly with replacements;  
Ⅲ. Average output of all models (bagging = bootstrap aggregation);  

    * lower variance
      -> because in last boostrap step, the predictions have been averaged;  
      -> according to the formular, there is averaging prediction minus expected prediction;  
    * higher bias  
      ->  lossing representation calpacity because of drawing sample data from overall dataset;  
      ->  be more constrained, be more conservetive, be more open averaged;  

### 2.4 Ramdom Forest

    * Best split using a random subset of m ≤ D features;  
    * Splitting using best out of a fixed number of random splits;  
    * Training on bootstrap-samples of the data;  

Best case: balanced trees  

    * Fitting: O(BmNlogN)  
    * Prediction: O(BlogN)  

Worst case: splitting off one data point at a time  

    * Fitting: O(BmN^2)  
    * Prediction: O(BN)

## Dataset
In this project, [Pima Indian Diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database) dataset is used to exploit Decision tree properties. 
## Result
by using the following libraries, tree structure coming out of sklearn.DesicionTree can be shown

    * from sklearn.tree import export_graphviz;  
    * from sklearn.externals.six import StringIO;  
    * import pydotplus;  

after running the first training and testing stage, without optimization, the tree structure is not clear to underunstood and accuracy only 69%   
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Trees%20and%20Forest/images/2.png)  
after optimizing by changing criterion to "entropy", the tree structure is clear and accuracy is improved up to 77%  

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

![image](https://github.com/FangLintao/Machine-Learning/blob/master/Trees%20and%20Forest/images/1.png)  
