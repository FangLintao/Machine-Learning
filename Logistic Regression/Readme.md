# Logistic Regression
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Logistic%20Regression/Images/1.png)  
Logistic Regression belongs to supervised learning in machine learning. Logistic Regression is one of classification, even through it is evolutionized from Linear Regression. Commonly, Logistic Regression is binary classification in machine learning, but Logistic Regression can be used to classify multi-class problems, which is called multinomial logistic regression.
# Background
In this sction, Logistic Regression is based on the project of [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). By going through this projects, knowledge in Logistic Regression will mix with this project
# the Core
Assumption

    1. Binary logistic regression requires binary dependent variable  
    2. Only the meaningful variables should be included【LDA,PCA】  
    3. The independent variables should be independent of each other  
    4. The independent variables are linearly related to the log odds

Influences on Regression

    1. an outlier would biase linear model, which would affect the original threshold value 
    2. making incorrect descision in linear regression model

Features

    middle area around zero, it is similar with linear regression

### 1. Logistic Regression:  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Logistic%20Regression/Images/6.png)    
Logistic Regression is limited to the value [0,1], and within this scale, Logistic Regression categorizes objects by probability coming out of Logistic Regression. Since Logistic Regression comes from Linear Regression, then it shares the same parameter, weights, with Linear Regression.  

### 2.Optimizaton  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Logistic%20Regression/Images/7.png)  

    1. Nice Property: J is convex, gradient descent works well  
    2. Not so nice property: There is no analytic solution

### 3. Data
in this project, Bank Marketing dataset is adapted. This type of dataset includes numerical items and word items, so it is hard to compare connections between each of two features because so far, we don't know importance on these numerical features. If all of them are numercial, then we can directly plot relationships between each pair of features.     
Input variables:   

    ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

Desired variables:

    y: term deposit

#### 3.1. Data Feature Exploration
the following feature plots offers vivid distributions in each word feature  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Logistic%20Regression/Images/2.png)  
#### 3.2. Data Feature Independence
the following feature plot offers independence relationships amonge all of analysed features. From this praph, the assumption " variables should be independent" in Logistic Regression is proved.  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Logistic%20Regression/Images/3.png)  
#### 3.3 Data Extraction
Becuase not all of features are numerical, it is hard to contribute its information to model prediction if these features are very important. In this case, we use the following codes to extract important features from intuitive and do data processing

    pd.get_dummies(dataset, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
    dataset.drop(dataset.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)

### 4. Results 
#### 4.1 Confusion Matrix
Accuracy of logistic regression classifier is 0.89. From this image, we have 9852+218 corrections and 126+1107 incorrect prediction.   
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Logistic%20Regression/Images/4.png)
#### 4.2 Classification Report
![image](https://github.com/FangLintao/Machine-Learning/blob/master/Logistic%20Regression/Images/5.png)  

    1. The precision:  true positive / (true positive + false positive) 
        The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.
    2. The recall: true positive / (true positive + false negative) 
        The recall is intuitively the ability of the classifier to find all the positive samples.
    3. The F1 score: [0,1], F1 = 1.0 means recall and precision are equally important.
        Interpreted as a weighted harmonic mean of the precision and recall, weights the recall more than the precision by a factor of beta. 
    4. The support: the number of occurrences of each class in y_test.

### 5. Conclusion
1. 88% of term deposit are customers like while 89% of term deposit customers like are promoted.  
2. 86% of f1 score means almost equality on recall and precision
