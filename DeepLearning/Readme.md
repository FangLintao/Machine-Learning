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
Ⅱ. dead neural situation: whenever neural values below zero will be zero,which is dead neurons and it will loss learning features in the future  
Ⅲ. derivative range: 1
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/4.png)
### 2.4 Softmax activation function
#### Features
Ⅰ. Using probability to express likehood of each class  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/16.png)
#### Why tanh activation fucntion is always better than sigmoid activation function
##### the outputs using tanh
centre around 0 rather than sigmoid's 0.5, and this makes learning for the next layer a little bit easier.   
Analysis:  
Ⅰ. Convergence is usually faster if the average of each input variable over the training set is close to zero  
Ⅱ. Outputs close to zero are best: during optimization, they produce the least weight swings, and hence let your model converge faster
#### the potential problem in gradient descent
##### if the convolution layers are very deep, then data will split into two polar sides in these two activation function, because of the small derivate in activation functions, the learning process is quite slow, generating gradiet vanish.
## 3. Loss Function
In the context of an optimization algorithm, loss function used to evaluate a candidate solution (i.e. a set of weights) is referred to as the objective function. The cost or loss function has an important job in that it must faithfully distill all aspects of the model down into a single number in such a way that improvements in that number are a sign of a better model
### 3.1 Cross-Entropy Loss
Ⅰ. Application Field: Classification
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/7.png)
### 3.2 MSE Loss & RMSE
Ⅰ. Application Field: Regression
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/8.png)
## 4. [Optimization](https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/)
when initializing weights in CNN, the loss position is at somewhere of loss function,so what we have to check is that from all possible directions in the x-y plane if assuming in 3-dimensional space, moving along which direction would have the steepest decline in the value of the loss function.   
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/9.png)
###### reference: ["Intro to optimization in deep learning: Gradient Descent"](https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/), Ayoosh Kathuria, 1 Jun 2018  
in all of possible direction, only one of them is the steepest descent direction like showing in image above. After locking at the direction, the step size [leraning rate] to the minimum value along this steepest descent direction should be considered.  
Two potential cases:  
Ⅰ. large learning rate: overshooting in loss function and never gets to the minimum position    
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/10.png)
###### reference: ["Intro to optimization in deep learning: Gradient Descent"](https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/), Ayoosh Kathuria, 1 Jun 2018
Ⅱ. small learning rate: time comsuming and stuck at local minimum in non-convex situation   
### 4.1 Challenge
#### 4.1.1 Local Minimum
In real case, loss function is non-convex, because neural networks are complicated functions, with lots of non-linear transformations thrown in our hypothesis function  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/11.png)
###### reference: ["Intro to optimization in deep learning: Gradient Descent"](https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/), Ayoosh Kathuria, 1 Jun 2018  
#### 4.1.2 Saddle Point
Saddle Point is similar with local minimum at zero the gradient position, but it is not local minimum.   
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/12.png)
###### reference: ["Intro to optimization in deep learning: Gradient Descent"](https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/), Ayoosh Kathuria, 1 Jun 2018
#### 4.1.3 Zig-Zag Situation
Ⅰ. ill-conditioning causes the typical but undesired zig-zag behavior  
Ⅱ. Usually, optimization methods involve zig-zag situation.   
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/18.png)
#### Methods to solve zig-zag situation
Ⅰ. [Momentum](https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/)   
Instead of using normal gradient descent formular, momentum accumulates the gradient of the past steps to determine the direction.  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/19.png)  
If we sum up values along the direction w1 and w2, then we discover that total value along direction w1 is zero while the w2 value is accumlated. In momentum, for each update iteration, we add value into w2 direction, which speeds up converage rates, moving quickly to minima.   
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/20.png)  
##### reference : [Intro to optimization in deep learning: Momentum, RMSProp and Adam](https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/)
Ⅱ. Preconditioning  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/21.png)  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/22.png)  
### 4.2  Gradient descent
#### 4.2.1 Batch Gradient Descent
Features: slow; intractable for datasets that don't fit in memory
Ⅰ. computes the gradient of the cost function w.r.t. to the parameters θ for the entire training dataset. 
Ⅱ. calculate the gradients for the whole dataset to perform just one update

        Convex Function: converge to the global minimum  
        Non-Convex Function: converge to a local minimum

![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/13.png)  
#### 4.2.2 Stochastic Gradient Descent
##### Randomness: in order to escape local minima and saddle points, while trying to converge to a global minima.
In stochastic gradient descent, instead of taking a step by computing the gradient of the loss function by summing all the loss functions, we take a step by computing the gradient of the loss only one randomly sampled example without replacement. This means, at every step, we are taking the gradient of a loss function, which is different from our actual loss function. The gradient of this "one-example-loss" at a particular may actually point in a direction slighly different to the gradient of "all-example-loss".  
##### SGD is useful to solve Saddle Point situation
#### 4.2.3 Mini-batch gradient descent
reduces the variance of the parameter updates, which can lead to more stable convergence  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/14.png)  
### 4.3 Optimization algorithm
#### Adam
computes adaptive learning rates for each parameter  

        Ⅰ. storing an exponentially decaying average of past squared gradients vt  
        Ⅱ. keeps an exponentially decaying average of past gradients mt  

![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/15.png)  
### 4.4 Gradient Vanishing & Gradient Exploding
deep neural network is stacked by numerous non-linear layers because for activation functions in neural networks, each layer can be viewed as non-linear function.
#### 4.4.1 Gradient Vanishing
-> Reasons  
Ⅰ. because the depth of neural networks, while propergating, values will have polar distribution in activation function, which is hard to learn even through large changes in input variables;  
Ⅱ. For those values below one, after certain depth of neural networks, these values have exponentially decreasing tendency to zero, which is hard to learn;  
-> Solutions  
Ⅰ. Batch normalization after each convolution layer;  
Ⅱ. Using different activation functions;  
Ⅲ. Resnet structure;  
Ⅳ. In RCNN, using LSTM;
#### 4.4.2 Gradient Exploding
-> Reasons  
Ⅰ. For initial weight values above one, values have exponentially increasing tendency  
-> Solutions  
Ⅰ. Setting threshold values to cut off values once weight parameters are above certain large value;  
Ⅱ. weigts regularization: L1 or L2 penalities to lower polynominal parameter values;
## 5. Multilayers Perceptrons  
### 5.1 Computation Complexity
#### the computational complexity (in big-Oh notation) of computing the cross-entropy loss J(w) for logistic regression on a data set of N data points with d dimensions

        O(N*d)

#### memory complexity (in big-Oh notation) of a forward pass in an MLP with two hidden layers (of size k1 and k2,respectively) for a batch size B and input dimensionality d

        O(max(B*d,d*K1,B*K1,K1*K2,B*K2))  
        -> B*d: store the activation of what we need to store the input  
        -> d*K1: input dimension d multiply with wieght matrix between the input dimension and the first layer with k1 size  
        -> B*K1: store the activation from the first layer  
        -> K1*K2: store the computation between the first layer and the second layer  
        -> B*K2: store the activation from the second layer  

####  the memory complexity (in big-Oh notation) of a forward pass in a perceptron, depending on the batch size B and input dimensionality d

        O(B*d)

### 5.2 Advantages and Disadvantages
#### Advantages
A fully connected layer offers features learned from all the combinations of the features of the previous layer;
#### Disadvantages
Ⅰ. Expensive Computation  
Ⅱ. values in weight matrix are different and numerous
### 7. Overfitting
Training loss performs well while validation loss is not when epoches get larger  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/DeepLearning/images/17.png)  
##### reference: [Memorizing is not learning! — 6 tricks to prevent overfitting in machine learning](https://hackernoon.com/memorizing-is-not-learning-6-tricks-to-prevent-overfitting-in-machine-learning-820b091dc42)
#### 7.1 Tackling Overfitting
Ⅰ. Data Augmentation: adding more gerenal transforms to data sources, like mirroring, flipping, rotation...  
Ⅱ. Dropout: uniformaly drop out some neurals in fully connection layers  
Ⅲ. Early Stopping: according to the graph above, it is better to stop early   
Ⅳ. Regulations: adding penality methods to loss functions, like L1 & L2 penalities to reduce the complexity of the model  

