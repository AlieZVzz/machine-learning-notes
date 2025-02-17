# 机器学习模型组成与分类

* [返回上层目录](../machine-learning-introduction.md)
* [机器学习三要素](#机器学习三要素)
  * [模型](#模型)
  * [策略](#策略)
    * [损失函数和风险函数](#损失函数和风险函数)
    * [经验风险最小化与结构风险最小化](#经验风险最小化与结构风险最小化)
  * [算法](#算法)
* [监督学习](#监督学习)
  * [监督学习分类](#监督学习分类)
  * [生成模型与判别模型](#生成模型与判别模型)
    * [简述](#简述)
    * [第一种定义](#第一种定义)
      * [生成模型](#生成模型)
      * [判别模型](#判别模型)
      * [非概率模型（也称为判别模型）](#非概率模型（也称为判别模型）)
    * [第二种定义](#第二种定义)
    * [生成模型与判别模型的特点](#生成模型与判别模型的特点)
* [非监督学习](#非监督学习)
  * [非监督学习分类](#非监督学习分类)
* [半监督学习](#半监督学习)
  * [半监督学习分类](#半监督学习分类)
* [强化学习](#强化学习)
  * [强化学习分类](#强化学习分类)

注意：分类没有把神经网络体系加进来。因为NNs的范式很灵活，不太适用这套分法。

# 机器学习三要素

统计学习方法都是由模型、策略和算法构成的，即统计学习方法由三要素构成，可以简单地表示为

**方法 = 模型 + 策略 + 算法**

下面论述机器学习三要素。可以说构建一种机器学习方法就是确定具体的统计学习三要素。

## 模型

统计学习首要考虑的问题是**学习什么样的模型**。在监督学习过程中，模型就是所要学习的条件概率分布或决策函数。

模型的假设空间包含所有可能的条件概率分布或决策函数。例如，假设决策函数是输入变量的线性函数，那么模型的假设空间就是所有这些线性函数构成的函数集合。假设空间中的模型一般有无穷多个。

假设空间用$F$表示。

假设空间可以定义为**决策函数的集合**
$$
F=\{ f|Y=f(X) \}
$$
其中，$X$和$Y$是定义在输入空间$X$和输出空间$Y$上的变量。这时通常是由一个参数向量决定的函数族：
$$
F=\{ f|Y=f_{\theta}(X),\theta\in R^n \}
$$
参数向量$\theta$取值于$n$维欧氏空间$R^n$，称为参数空间。

假设空间也可以定义为**条件概率的集合**
$$
F=\{ P|P(y|X)\}
$$
其中，$X$和$Y$是定义在输入空间$X$和输出空间$Y$上的随机变量。这时$F$通常是由一个参数向量决定的条件概率分布族：
$$
F=\{ P|P_{\theta}(y|X) ,\theta\in R^n \}
$$
参数向量$\theta$取值于$n$维欧氏空间$R^n$，称为参数空间。

本书中称由决策函数表示的模型为非概率模型，由条件概率表示的模型为概率模型。为了简便起见，当论及模型时，有时只用其中一种模型。

## 策略

有了模型的假设空间，统计学习接着需要考虑的是**按照什么样的准则学习或选择最优的模别**。机器学习的**目标在于从假设空间中选取最优模型**。

首先引入损失函数与风险函数的概念。损失函数度量模型一次预测的好坏，风险函数度量平均意义下模型预测的好坏。

### 损失函数和风险函数

监督学习问题是在假设空间$F$中选取模型$f$作为决策函数，对于给定的输入$X$，由$f(X)$给出相应的输出$y$，这个输出的预测值$f(X)$与真实值$y$可能一致也可能不一致，用一个损失函数或代价函数来度量预测错误的程度。损失函数记作$L(Y, f(x))$。

损失函数值越小，模型就越好。由于模型的输入、输出$(X, Y)$是随机变量，遵循联合分布$P(X, Y)$，所以损失函数的期望是
$$
R_{\text{exp}}(f)=E_P[L(Y, f(X))]=\int_{X\times Y}L(y,f(x))P(x,y)dxdy
$$
这是理论上模型$f(X)$关于联合分布$P(X, Y)$的平均意义下的损失，称为期望风险或**期望损失**。

**学习的目标就是选择期望风险最小的模型**。由于联合分布$P(X, Y)$是未知的，$R_{exp}(f)$不能直接计算。实际上，如果知道联合分布$P(X, Y)$，可以从联合分布直接求出条件概率分布$P(Y|X)$，也就不需要学习了。正因为不知道联合概率分布，所以才需要进行学习。这样一来，一方面根据期望风险最小学习模型要用到联合分布，另一方面联合分布又是未知的，所以监督学习就成为一个病态问题。

给定一个训练数据集
$$
T=\{ (x_1,y_1), (x_2,y_2), ... , (x_N,y_N) \}
$$
模型$f(X)$关于训练数据集的平均损失称为经验风险（empirical risk）或**经验损失**，记作$R_{emp}$：
$$
R_{emp}(f)=\frac{1}{N}\sum_{i=1}^NL(y_i,f(x_i))
$$
期望风险$R_{exp}(f)$是模型关于联合分布的期望损失，经验风险$R_{emp}(f)$是模型关于训练样本集的平均损失。根据大数定律，当样本容量N趋于无穷时，经验风险$R_{emp}(f)$趋于期望风险$R_{exp}(f)$。所以—个很自然的想法是**用经验风险估计期望风险**。但是，由于现实中训练样本数目有限，甚至很小，所以用经验风险估计期望风险常常**并不理想**，要对经验风险进行一定的矫正。这就关系到监督学习的两个基本策略：经验风险最小化和结构风险最小化。

### 经验风险最小化与结构风险最小化

在假设空间、损失函数以及训练数据集确定的情况下，经验风险$R_{emp}(f)$就可以确定。经验风险最小化的策略认为，经验风险最小的模型是最优的模型。根据这一策略，按照经验风险最小化求最优模型就是求解最优化问题：
$$
\mathop{\text{min}}_{f\in F}\frac{1}{N}L(y_i,f(x_i))
$$
其中，$F$是假设空间。

**当样本容量足够大时，经验风险最小化能保证有很好的学习效果**，在现实中被广泛采用。比如，极大似然估计就是经验风险最小化的一个例子。当模型是条件概率分布，损失函数是对数损失函数时，经验风险最小化就等价于极大似然估计。

但是，**当样本容量很小时**，经验风险最小化学习的效果就未必很好，**会产生过拟合现象**。

**结构风险最小化**是为了**防止过拟合**而提出来的策略。**结构风险最小化等价于正则化**。**结构风险在经验风险上加上表示模型复杂度的正则化项或罚项**。在假设空间、损失函数以及训练数据集确定的情况下，结构风险的定义是
$$
R_{\text{srm}}(f)=\frac{1}{N}L(y_i,f(x_i))+\lambda J(f)
$$
其中$J(f)$为模型的复杂度，是定义在假设空间$F$上的泛函。模型$f$越复杂，复杂度$J(f)$就越大；反之，模型$f$越简单，复杂度$J(f)$就越小。也就是说，**复杂度表示了对复杂模型的惩罚**。**$\lambda \geqslant 0$是系数，用以权衡经验风险和模型复杂度**。结构风险小需要经验风险与模型复杂度同时小。结构风险小的模型往往对训练数据以及**未知的**测试数据都有较好的预测。
比如，贝叶斯估计中的最大后验概率估计（maximum posterior probability estimatkm，MAP）就是结构风险最小化的一个例子。当模型是条件概率分布、损失函数是对数损失函数、**模型复杂度由模型的先验概率表示时**，结构风险最小化就等价于最大后验概率估计。

结构风险最小化的策略认为结构风险最小的模型是最优的模型。所以求最优模型，就是求解最优化问题：
$$
\mathop{\text{min}}_{f\in F}\frac{1}{N}L(y_i,f(x_i))+\lambda J(f)
$$
这样，机器学习学习问题就变成了**经验风险或结构风险函数的最优化问题**。这时经验或结构风险函数是**最优化的目标函数**。

## 算法

算法是指机器学习模型的具体计算方法。

机器学习基于训练数据集，根据学习策略，从假设空间中选择最优模型，最后需要考虑用什么样的计算方法求解最优模型。

这时，**机器学习问题归结为最优化问题**，机器学习的算法成为求解最优化问题的算法。如果最优化问题有显式的解析解，这个最优化问题就比较简单。但通常解析解不存在，这就需要用数值计算的方法求解。如何保证找到全局最优解，并使求解的过程非常高效，就成为一个重要问题。机器学习可以利用已有的最优化算法，有时也需要开发独自的最优化算法。

机器学习方法之间的不同，主要来自其**模型、策略、算法**的不同。确定了模型、策略、算法，机器学习的方法也就确定了。这也就是将其称为机器学习三要素的原因。

# 监督学习

## 监督学习分类

* **分类算法(线性和非线性)**

  * 感知机

  * KNN

  * 概率

    * 朴素贝叶斯（NB）
    * Logistic Regression（LR）
    * 最大熵MEM（与LR同属于对数线性分类模型）

  * 支持向量机(SVM)

  * 决策树(ID3、CART、C4.5)

  * assembly learning

    * Boosting

      * Gradient Boosting

        * GBDT

        * XGBoost

          传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）；

          XGBoost是Gradient Boosting的一种高效系统实现，并不是一种单一算法。

      * AdaBoost

    * Bagging

      * 随机森林

    * Stacking

* ……

* 概率图模型（标注）
  * HMM
  * MEMM（最大熵马尔科夫）
  * CRF
  * ……
* 回归预测
  * 线性回归
  * 树回归
  * Ridge岭回归
  * Lasso回归
  * ……
* ……  

## 生成模型与判别模型

### 简述

监督学习的任务就是学习一个模型，应用这一模型，对给定的输入预测相应的输出。这个模型的一般形式为决策函数：
$$
Y=f(X)
$$
或者条件概率分布：
$$
P(Y|X)
$$
监督学习方法又可以分为生成方法（generative approach）和判别方法（discriminative approach）。所学到的模型分别称为**生成模型（generative model）**和**判别模型（discriminative model）**。

（1）**生成模型**

**生成方法**由数据学习联合概率分布$P(X, Y)$，然后求出条件概率分布$P(Y|X)$作为预测的模型，即生成模型：
$$
P(Y|X)=\frac{P(X,Y)}{P(X)}
$$
这样的方法之所以称为生成方法，是因为**模型表示了给定输入X产生输出Y的生成关系**。典型的生成模型有：**朴素贝叶斯法和隐马尔可夫模型**，将在后面章节进行相关讲述。

（2）**判别模型**

**判别方法**由数据直接学习决策函数$f(X)$或者条件概率分布$P(Y|X)$作为预测的模型，即判别模型。判别方法关心的是对给定的输入X，应该预测什么样的输出Y。典型的判别模塑包括：k近邻法、感知机、决策树、逻辑斯谛回归模型、最大熵模型、支持向量机、提升方法和条件随机场等，将在后面章节讲述。

------

我们都知道，对于有监督的机器学习中的分类问题，求解问题的算法可以分为生成模型与判别模型两种类型。但是，究竟什么是生成模型，什么是判别模型？不少书籍和技术文章对这对概念的解释是含糊的。在今天这篇文章中，我们将准确、清晰的解释这一组概念。

### 第一种定义

对于判别模型和生成模型有两种定义，第一种定义针对的是有监督学习的分类问题。

该问题的目标是给定一个样本的向量x（可以是原始数据如图像，声音，也可以是提取出来的特征向量），在这里我们将它称为输入变量，目标是预测出这个样本的类别y即标签值，一般是一个离散的标量，即类别编号。因此算法要解决的核心问题是根据训练样本建立下面这样的映射函数：
$$
y=f(x)
$$
对于这个问题，有3种求解思路，下面我们分别介绍。

#### 生成模型

第一种做法称为生成模型。已知输入变量x和目标变量y，先对它们的联合概率分布$p(x, y)$建模，然后计算样本属于每一个类的条件概率$p(y|x)$即类后验概率，按照这个值来完成分类，如将样本分到概率p(y|x)最大的那个类。根据概率论的知识，有：
$$
p(y|x)=\frac{p(x,y)}{p(x)}
$$
在这里，$p(x, y)$为联合概率密度函数，$p(x)$为样本输入向量x的边缘密度函数。对上面这种做法的直观解释是：**我们已知某一个样本具有某种特征x，现在要确定它输入哪个类，而自然的因果关系是，样本之所以具有这种特征x，是因为它属于某一类**。例如，我们要根据体重，脚的尺寸这两个特征x来判断一个人是男性还是女性，我们都知道，男性的体重总体来说比女性大，脚的尺寸也更大，因此从逻辑上来说，是因为一个人是男性，因此才有这种大的体重和脚尺寸。而在分类任务中要做的却相反，是给了你这样个特征的样本，让你反推这人是男性还是女性。

联合概率密度函数等于类概率$p(y)$与类条件概率$p(x|y)$即先验概率的乘积，即：
$$
p(x,y)=p(x|y)p(y)
$$
将上面两个公式合并起来，有：
$$
p(y|x)=\frac{p(x|y)p(y)}{p(x)}
$$
这就是贝叶斯公式。它完成了**因果转换**，我们要完成的是**由果推断因**，而在训练时我们建立的是**因到果**的模型及p(x|y)，即男性和女性的体重、脚尺寸分别服从的概率分布。

总结起来，生成模型对联合概率$p(x, y)$建模，根据它，我们可以得到类后验概率$p(y|x)$。事实上，这种做法不仅仅局限于分类问题，如果将x看做可以观测的变量，y看做不可观测到的变量，只要具有这种特征的问题，我们都可以建立生成模型。

生成模型的典型代表是贝叶斯分类器，它对类条件概率$p(x|y)$建模，而$p(x|y)p(y)$就是联合概率$p(x, y)$。通过贝叶斯公式，根据联合概率又可以得到类后验概率：
$$
p(y|x)=\frac{p(x|y)p(y)}{p(x)}
$$
如果只用于分类而不需要给出具体的概率值，则分母p(x)对所有类型y都是一样的，只需要计算分子p(x|y)p(y)。如果我们假设每个类y的样本的特征向量x的每个分量相互独立，则可以得到朴素贝叶斯分类器，如果假设每个类的样本服从正态分布，则p(x|y)为正态分布，此时为正态贝叶斯分类器。

生成模型的另外一个典型代表是受限玻尔兹曼机（RBM），这是一种随机性的神经网络，由两类神经元组成（每一类为一个层），两个层之间有连接关系，第一种神经元为可见变量，即可以直接观测到的值v，如图像的每个像素。第二类为隐含变量h，是不能直接观测到的变量，如图像的特征。v和h的联合概率服从玻尔兹曼分布：
$$
p(v,h)=\frac{1}{Z_{\theta}}\text{exp}\left(-E_{\theta}(v,h)\right)\frac{1}{Z_{\theta}}\text{exp}\left( v^TWh+b^Tv+d^Th \right)
$$
根据这个联合概率，我们可以很容易得到条件概率$p(x|y)$和$p(y|x)$。例如为了得到$p(y|x)$，可以先求边缘概率$p(x)$，对于离散型随机变量，对y的概率求和，对于连续型随机变量，则为求积分，然后有：
$$
p(y|x)=\frac{p(x,y)}{p(x)}
$$
生成模型最显著的一个特征是假设样本向量x服从何种概率分布，如正态分布，均匀分布。

#### 判别模型

第二种做法称为判别模型。已知输入变量x，它直接对目标变量y的条件概率$p(y|x)$建模。即计算样本x属于每一类的概率。注意，这里和生成模型有一个本质的区别，那就是每一假设x服从何种概率分布，而是直接估计出条件概率$p(y|x)$。

这类模型的典型代表是logistic回归和softmax回归，它们直接对$p(y|x)$建模，而不对$p(x, y)$建模，即每一假设x服从何种概率分布。logistic回归用于二分类问题，它直接根据样本x估计出它是正样本的概率：
$$
p(y=1|x)=\frac{1}{1+\text{exp}\left( -w^Tx+b \right)}
$$
注意，这里只是直接猜测出了这个概率，而没有假设每个类的样本服从何种概率分布，即没有对$p(x|y)$或者$p(x, y)$建模。

softmax回归是logistic回归的多分类版本，它直接估计出一个样本向量x属于k个类中每一个类的概率：
$$
\begin{aligned}
h_{\theta}(x)=\frac{1}{\sum_{i=1}^ke^{\theta_i^Tx}}
\begin{bmatrix}
e^{\theta_1^Tx}\\ 
...\\
e^{\theta_k^Tx}
\end{bmatrix}
\end{aligned}
$$
这里预测出的是一个向量，每个分量为样本属于每个类的概率。和logistic回归一样，它是直接预测出了这个条件概率，而没有假设每个类的样本x所服从的概率分布。

#### 非概率模型（也称为判别模型）

第三种做法最直接，分类器根本就不建立概率模型，而是直接得到分类结果，这种是非概率模型，也称为判别模型。它直接根据样本向量x预测出类别编号y：
$$
y=f(x)
$$
这类模型的典型代表是决策树，支持向量机，随机森林，kNN算法，AdaBoost算法，XGBoost，标准的人工神经网络（包括全连接神经网络，卷积神经网络，循环神经网络等）。如果神经网络的最后一层是softmax变换，即softmax回归，则可以归到第二种情况里，如果没有使用，则是这第三种情况。

支持向量机的预测函数是：
$$
\text{sign}\left( \sum_{i=1}^l\alpha_iy_iK(x_i^Tx)+b \right)
$$
它自始至终没有假设样本向量x服从何种分布，也没有估计类后验概率$p(y|x)$。这可以看成是一种几何划分的思想，把空间划分成多个部分。

类似的，决策树的预测函数时分段常数函数，直接实现从向量x到类别标签y的映射，没有计算任何概率值。其他的算法如随机森林，kNN，也是如此。

这类模型没有使用概率的观点进行建模，而是用几何或者分析（函数）的手段建模，如找出分类超平面或者曲面，直接得到映射函数。

一般来说，我们把使用第一种模型的分类器称为生成式分类器，把使用第二种和第三种模型的分类器称为判别式分类器。

### 第二种定义

除此之外，对生成模型和判别模型还有另外一种定义。生成模型是已知样本的标签值y，对样本的特征向量x的条件概率进行建模，即对条件概率p(x|y)建模，它研究的是每种样本服从何种概率分布。判别模型则刚好相反，已知样本的特征向量x，对样本的标签值y的概率进行建模，即对条件概率p(y|x)建模，这种一般用于分量，即给定样本x，计算它属于每个类的概率。

根据这种定义，生成模型可以用来根据标签值y生成随机的样本数据x。生成对抗网络（GAN）就是典型的例子，它可以生成服从某种概率分布的随机变量，即拟合类条件概率密度函数p(x|y)，而此时它的目的不是分类，而是生成样本。事实上，如果我们知道了p(x|y)或者p(x, y)，无论是用来做分类，还是用来做数据生成，都是可以的。

而判别模型以及不使用概率模型的判别型分类器则根据样本特征向量x的值判断它的标签值y，即用于判断样本的标签值y。

### 生成模型与判别模型的特点

在监督学习中，生成方法和判别方法各有优缺点，适合于不同条件下的学习问题。

**生成方法的特点**：

- 生成方法可以还原出联合概率分布$P(X, Y)$，而判别方法则不能
- 生成方法的学习收敛速度更快，即当样本容量增加的时候，学到的模型可以更快地收敛于真实模型
- 当存在隐变量时，仍可以用生成方法学习，此时判别方法就不能用

**判别方法的特点**：

- 判别方法直接学习的是条件概率P(Y|X)或决策函数f(X)，直接面对预测，往往学习的准确率更高；
- 由于直接学习P(Y|X)或f(X)，可以对数据进行各种程度上的抽象、定义特征并使用特征，因此可以简化学习问题



# 非监督学习

## 非监督学习分类

* 聚类
  * 基础聚类
    * Kmean
    * 二分kmean
    * K中值聚类
    * GMM聚类
  * 层次聚类
  * 密度聚类
  * 谱聚类
* 主题模型
  * pLSA
  * LDA隐含狄利克雷分析
* 关联分析
  * Apriori算法
  * FPgrowth算法
* 降维
  * PCA算法
  * SVD算法
  * LDA线性判别分析
  * LLE局部线性嵌入
* 异常检测：
* ……

# 半监督学习

## 半监督学习分类



# 强化学习

## 强化学习分类



# 参考资料

- [如何用简单易懂的例子解释条件随机场（CRF）模型？它和HMM有什么区别？](https://www.zhihu.com/question/35866596/answer/236886066)

"机器学习模型分类"一节参照了此知乎回答。

- [理解生成模型与判别模型](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247486873&idx=1&sn=f7701e13b29cd8db3dc4de15bfadd0ff&chksm=fdb6900ecac1191855b5f43ebc3c12f3eff808b4fa7d4c48275ff8c8819a0e7f8567466c7d8c&mpshare=1&scene=1&srcid=1010HYNCE9Q9XvHOPgIo8J6N#rd)

"生成模型与判别模型"一节参考此微信文章。

