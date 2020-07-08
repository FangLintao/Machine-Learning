# DeepLearning
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/1.png)  
###### Reference: "Machine Learning, Deep Learning, and AI: What’s the Difference?", Alex Woodie, May 10, 2017  
## 1. Definition
representation learning methods with multiple levels of representation, obtained by composing simple but nonlinear modules that each transform the representation at one level into a [...] higher, slightly more abstract (one)  
## 2. Activation Functions
The activation function does the non-linear transformation to the input, making it capable to learn and perform more complex tasks
##### General Features  
    Ⅰ。Activation Functions introduce non-linearity to the output of neurons  
        -> The outcome does not change in proportion to a change in any of the inputs - linear regression model
        -> result: limit model learning power
    Ⅱ。Activation function should be differentiable or the concept of updating weight (backpropagation)fails  
### 2.1 Logistic sigmoid activation function  
#### Features
Ⅰ. output range: [0, 1] - Advantage   

    when encountered with (- infinite, + infinite) as in the linear function, values will be limited in range [0,1]  

Ⅱ. derivative range: [0, 0.25]
Ⅲ. value Y reacts very little to variable X. so small changes in X generate small changes in Y. - Drawback  

    1. the vanishing gradient -> The derivative values in these regions are very small and converge to 0  
    2. slow learning -> optimization algorithm minimizing error can be attached to local minimum values  
    3. slow learning -> cannot get maximum performance from the artificial neural network model
 
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/2.png)
### 2.2 Logistic hyperbolic tangent activation function  
#### Features
Ⅰ. output range: [-1, 1] - Advantage   
    
    when encountered with (- infinite, + infinite) as in the linear function, values will be limited in range [-1,1]  

Ⅱ. derivative range: [0, 1]
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/3.png)
### 2.3 Rectified Linear (ReLU) activation function  
#### Features
Ⅰ. output range: [0, inf]  
Ⅱ. derivative range: 1
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/4.png)
### Comparison
#### Sigmoid & tanh
-> Same
    monotonically increasing function  
    
-> Difference  
