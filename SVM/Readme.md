# Support Vector Machine
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/1.png)
## 1. Introduction
Support Vector Machine is one of supervised classification, based on max margin in feature space. It is simple but powerful supervised machine learning algorithm that performs really well on both linearly separable and non-linearly separable datasets.

    1. dot product as similarity measure  
    2. mapping to high-dimensional feature space  
    3. statistical learning theory  
    4. large margin principle  
    5. quadratic optimization problem  
    6. instance-based learning  
    7. kernels & kernel trick

## 2. Core
### 2.1 Similarity
If pattern xi is from an arbitrary set X, then simiarity measure in vector space should be considered  
-> compare two patterns x,x'  
-> deliver a real number describing their similarity  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/3.png)  
-> dot product => similarity measure
### 2.2 Large Margin Principle

    1. when dataset is linearly separable, then no need to map to high-dimensional space  
    2. when dataset is nonlinearly, sepaprable, then need to map to high-dimensional space

![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/4.png)  

    * The optimal hyperplane has the largest margin  
    * The largest margin is defined by the shortest normal vector w  

However,the optimal hyperplane classifier leads to the following constrained optimization problem and dual optimization problem. As showing below, If we see one axis X belongs to constrain optimization problem and another axis Y belongs to dual optimization problem, then teh saddle point exists at the minimum point of constrained optimization problem and the maximum point of dual optimization problem at the same time.   
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/17.png)
#### 2.2.1 Solving Constrained Optimization Problem  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/5.png)  

    * Function𝜏 the objective function
    * 𝑦𝑖(⟨𝑤,𝑥𝑖⟩+𝑏≥1 gaurantee correct label By combination these two equations 

![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/6.png)  

    * 𝑚 means data points in high dimension
    * partial derivatives on 𝐿, minimize the Lagrangian 𝐿 w.r.t the primal variables 𝑤 and 𝑏
    * partial derivatives on 𝐿, maximize the Lagrangian 𝐿 w.r.t dual variables 𝛼𝑖

#### 2.2.2 Solving Dual Optimization Problem  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/7.png)  
Once solving these optimization problem, the hyperplane decision function for a novel data point 𝑥 in the feature space 𝐻 can be shown  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/8.png)  

    * ‖𝑥,𝑥𝑖‖ dot products of training data points 𝑥𝑖⇒ instance based learning
    * in practical problems, only a subset of the Lagrange multipliers 𝛼𝑖 are active and influence the decision hyperplane
    * Other training data points have 𝛼𝑖=0, don't define the shape of the hyperplane

### 2.3 Mapping & VC-Dimension
Not all of datasets should be mapped to feature space. It mainly depends on data distribution. If dataset is linearly separable, then we can consider optimization problem directly; If dataset is nonlinear separable, then we have to map to high-dimensional feature space.  
#### 2.3.1 Mapping  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/9.png)  

    * mapping function 𝜙 opens a wide choice of similarity measures and learning algorithms;
    * 𝜙 define a similarity measure using dot product in 𝐻
      𝑘(𝑥,𝑥′):=⟨𝑥,𝑥′⟩=⟨𝜙(𝑥),𝜙(𝑥′)⟩

#### 2.3.2 VC-Dimension  

    * In supervised learning, the choice of the function class / hypothesis class is relevant for good eneralization of the trained model to novel data  
    * Very powerful function classes (high ”capacity”) tend to overfit the training data  
    * Overfitting: powerful function class; underfitting: poor function class
    * ℜ2 -> 3 function class in VC-Dimension
    * ℜ3 -> 4 function class in VC-Dimension

empirical risk/ average training error with zero-one loss  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/10.png)  
vc-bound  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/11.png)  

    * the capacity term 𝜙 describing the capacity of the function class
    * large VC-dimension ℎ increases 𝜙

![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/12.png)

    * Large VC-dimension h of a function class implies a small empirical risk, but enlarges overall risk -> overfitting  
    * Prefer function classes of lower capacity, if they perform equally well on the training data, or restrict its capacity

### 2.4 Hard Margin & Soft Margin
Not all of cases can be seperate completely by hyperplane in practical, which means hard margin.

        1. high noise level may cause an overlap between classes  
        2. training data set contains mislabelled data points

1. soft margin: allow some data existing at boarder or in margin space  
2. hard margin: only allow data at boarder  
3. hard margin is easy to overfit while soft margin belongs to good generation
4. The hardness of the margin: controlled by a tuning parameter C. 

    * large C -> the margin is hard  
    * smaller C -> the margin is softer

![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/13.png)  
#### 2.4.1 Hard Margin  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/5.png)
#### 2.4.2 Soft Margin  
Soft Margin allows some data points existin margin area, which means, the gap of points to line function is less than one   
-> Chacteristics  

        * get a solution with a function class of lower capacity
        * enlarge the margin (→ smoother hyperplane)
        * reduce R[f]


![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/14.png)  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/19.png)

    * C determines a trade-off between enlarging the margin and minimizing the training error  
    * C limits the influence of a single αi , thus not allowing a single SV to ”push” upon the decision hyperplane too strongly
    * large C, punishment of misclassification will increase  

-> Two parameters b & ν
->-> scalar b 

        * shifts the decision hyperplane such that SVs with zero slack lie on ±1 lines

->-> parametrization ν

        * 0 < ν ≤ 1  
        * replace penalty parameter C  
        * lower-bounds (relative to the number of training data points) the fraction of training patterns which can become SVs and thus can have non-zero slacks  
        * upper-bounds the fraction of margin errors

The larger v, the harder margin SVM would be  
For kernel parameters, larger the 𝜎, harger margin is  
, ### 2.5 Kernel Trick  
1. All computations can be formulated in a dot product space  
2. All computations can be executed as dot product operations in H  
3. To express formulas in terms of the input patterns in X , we can make use of a kernel function k  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/15.png)

        The kernel trick replaces the mapping φ and following dot product operations by a (simple) calculation in the input space!

![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/16.png)  
![image](https://github.com/FangLintao/Machine-Learning/blob/master/SVM/images/18.png)  
Advantages
1. Computations in the input space usually are preferred compared to an explicit mapping into a high dimensional  
2. Every linear algorithm, which can be expressed by dot product operations, can be ”kernelized”, thus leading to a non-linear version of the algorithm
