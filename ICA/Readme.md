# Independent Component Analysis
![image](https://github.com/FangLintao/Machine-Learning/blob/master/ICA/Images/9.png)   
In real world, It is impossible to use one micro to record sounds just from one object. Actually, sounds from all objects involving in an area, will be recorded by micros, which generates mixing signals. However, it is possible to decompose mixing signals into several single signals by Independnt Component Analysis.
## 1 the Core
* labels are not needed -> unsupervised  
#### 1.1 Goal

    Recover signals from mixing signals without lossing information   
 
#### 1.2 Assumptions  

    1. The sources mix linearly into the observations   
    2. At least n-1 of the sources have a non-Gaussian distribution  
      -> in real world, signals are non-Gaussian Distribution  
    3. any mixture of sources is more Gaussian than (n-1-many of the) the original sources
    4. The sources are statistically independent at each time point t

#### 1.3 Non-Gaussian Distributions

    1. Distributions can be described by their moments (mean, variance, skewness, kurtosis)  
    2. Gaussian distributions have kurtosis of zero, while non-Gaussian distributions have non-zero kurtosis  

![image](https://github.com/FangLintao/Machine-Learning/blob/master/ICA/Images/7.png)  
#### 1.4 Comparison with [PCA](https://github.com/FangLintao/Machine-Learning/tree/master/PCA)

    Difference:
    PCA；can not recover the original sources  
    ICA: able to separate and recover original sources

#### 1.5 ICA steps
![image](https://github.com/FangLintao/Machine-Learning/blob/master/ICA/Images/8.png)  
A -> mixing matrix [unknown]  
W -> unmixing matrix [unknown, need to be pre-defined]  
with unmixing matrix, mixing signals can be decomposed into original signals  
  * Initialize weight vector w    
  * Determine direction, in which kurtosis S  
    -> grows most strongly (for positive kurtosis)  
    -> decreases most strongly (for negative kurtosis)
  * Run a step with a gradient descent method to get improved vector w  

Unfortunately kurtosis is hard to estimate in a robust way, thus other measures of non-Gaussianity are preferred in practical implementations:  

    -> negentropy [FastICA] (to be maximized)  
    -> mutual information (to be minimized)  
    -> likelihood (to be maximized)  

#### 1.6 Advantages & Disadvantages   
1.6.1 Advantages
* ICA is a linear method - once trained, it can be applied extremely fast for online systems
* Being a linear method, independent components can be visualized
* Many variants of ICA exist. They use different assumptions on what ”statistical independence” means. Thus for most unmixing
problems, you will probably find a variant, which can deal with your data  

1.6.2 Disadvantages
* Components have arbitrary sign, arbitrary order and amplitude  
## 2 Data
In this partion, we create three types of signals by simply using numpy
 
    1. Sinusoidal Signal  
    2. Square Signal
    3. Sawtooth Signal
 
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/ICA/Images/2.png)
 ## 3 Analysis
 #### 3.1 Gaussian vs. Non-Gaussian signals
 according to ICA assumption, signals in ICA are non-Gaussian and independedt, so it means that signal one doesn't contain information of signal two. Then these two signals are uncorrelated. [Important: independent -> uncorrelated; uncorrelated !-> independent ]  
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/ICA/Images/3.png)   
 Discovering:  above plots, showing that non-Gaussian independent signals can uniformly distribute around space before mixing, then it turns into correlected after mixing matrix but showing a uniform distribution on a parallelogram; for Gaussian Distribution, signals are symmetric then it does not contain any information about the mixing matrix, the inverse of which we want to calculate.  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/ICA/Images/4.png)  
 #### 3.2 ICA and PCA Processing 
 Discovering: ICA recovers original signals from mixing signals but it loss signals' order and magnititudes, while PCA losses information when recovering original signals, not only on signal's order and magnititudes, but also signal shapes. In this case, it reflects PCA main property that trying to saving signals with the most important information.      
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/ICA/Images/1.png)  
 #### 3.3 Kurtosis
 Kurtosis is the fourth moment of the data and measures the "tailedness" of a distribution. Reminding of Kurtosis properties, when Kurtosis=0, then gaussian distribution exists. From below image, kurtosis values in unmixing image are = abs(1.5) which is larger than mixing signal. In this case, it shows that, mixing signal is more Gaussian then un mixing signal because ICA works on maximization on non-Gaussianity.    
 ![image](https://github.com/FangLintao/Machine-Learning/blob/master/ICA/Images/6.png)  
