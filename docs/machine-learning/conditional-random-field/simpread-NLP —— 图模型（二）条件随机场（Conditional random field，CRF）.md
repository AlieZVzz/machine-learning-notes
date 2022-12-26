> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [www.cnblogs.com](https://www.cnblogs.com/Determined22/p/6915730.html)

       本文简单整理了以下内容：

      （一）马尔可夫随机场（Markov random field，无向图模型）简单回顾

      （二）条件随机场（Conditional random field，CRF）

       这篇写的非常浅，基于 [1] 和 [5] 梳理。感觉 [1] 的讲解很适合完全不知道什么是 CRF 的人来入门。如果有需要深入理解 CRF 的需求的话，还是应该仔细读一下几个英文的 tutorial，比如 [4] 。

**（一）马尔可夫随机场简单回顾**
==================

      概率图模型（Probabilistic graphical model，PGM）是由图表示的概率分布。**概率无向图模型（Probabilistic undirected graphical model）**又称**马尔可夫随机场（Markov random field）**，表示一个**联合概率分布**，其标准定义为：

      设有联合概率分布 P(V) 由无向图 G=(V, E) 表示，图 G 中的节点表示随机变量，边表示随机变量间的依赖关系。如果联合概率分布 P(V) 满足成对、局部或全局马尔可夫性，就称此联合概率分布为概率无向图模型或马尔可夫随机场。

      设有**一组**随机变量 Y ，其联合分布为 P(Y) 由无向图 G=(V, E) 表示。图 G 的一个节点 $v\in V$ 表示**一个**随机变量 $Y_v$ ，一条边 $e\in E$ 就表示两个随机变量间的依赖关系。

      **1. 成对马尔可夫性（pairwise Markov property）**

      设无向图 G 中的任意两个没有边连接的节点 u 、v ，其他所有节点为 O ，成对马尔可夫性指：给定 $Y_O$ 的条件下，$Y_u$ 和 $Y_v$ 条件独立

![](https://images2015.cnblogs.com/blog/1008922/201705/1008922-20170528162455422-1812780274.png)

$$P(Y_u,Y_v|Y_O)=P(Y_u|Y_O)P(Y_v|Y_O)$$

      **2. 局部马尔可夫性（local）**

      设无向图 G 的任一节点 v ，W 是与 v 有边相连的所有节点，O 是 v 、W 外的其他所有节点，局部马尔可夫性指：给定 $Y_W$ 的条件下，$Y_v$ 和 $Y_O$ 条件独立

![](https://images2015.cnblogs.com/blog/1008922/201705/1008922-20170528160105610-150035802.png)

$$P(Y_v,Y_O|Y_W)=P(Y_v|Y_W)P(Y_O|Y_W)$$

当 $P(Y_O|Y_W)>0$ 时，等价于

$$P(Y_v|Y_W)=P(Y_v|Y_W,Y_O)$$

如果把等式两边的条件里的 $Y_W$ 遮住，$P(Y_v)=P(Y_v|Y_O)$ 这个式子表示 $Y_v$ 和 $Y_O$ 独立，进而可以理解这个等式为给定条件 $Y_W$ 下的独立。

      **3. 全局马尔可夫性（global）**

      设节点集合 A 、B 是在无向图 G 中被节点集合 C 分开的任意节点集合，全局马尔可夫性指：给定 $Y_C$ 的条件下，$Y_A$ 和 $Y_B$ 条件独立

![](https://images2015.cnblogs.com/blog/1008922/201705/1008922-20170528162514578-985158227.png)

$$P(Y_A,Y_B|Y_C)=P(Y_A|Y_C)P(Y_B|Y_C)$$

      这几个定义是等价的。

      **4. 概率无向图模型**

 无向图模型的优点在于其没有隐马尔可夫模型那样严格的独立性假设，同时克服了最大熵马尔可夫模型等判别式模型的标记偏置问题。

      **（1）有向图的联合概率分布**

      考虑一个有向图 $G^d=(V^d,E^d)$ ，随机变量间的联合概率分布可以利用条件概率来表示为

$$P(v_1^d,...,v_n^d)=\prod_{i=1}^nP(v_i^d|v_{\pi i}^d)$$

其中 $v_{\pi i}^d$ 表示节点 $v_i^d$ 的父节点的集合。

      **（2）无向图的**因子分解**（Factorization）******

      不同于有向图模型，无向图模型的无向性很难确保每个节点在给定它的邻节点的条件下的条件概率和以图中其他节点为条件的条件概率一致。由于这个原因，无向图模型的联合概率并不是用条件概率参数化表示的，而是定义为由一组条件独立的局部函数的乘积形式。因子分解就是说将无向图所描述的联合概率分布表达为若干个子联合概率的乘积，从而便于模型的学习和计算。

      实现这个分解要求的方法就是使得每个局部函数所作用的那部分节点可以在 G 中形成一个最大团（maximal clique）。这就确保了没有一个局部函数是作用在任何一对没有边直接连接的节点上的；反过来说，如果两个节点同时出现在一个团中，则在这两个节点所在的团上定义一个局部函数来建立这样的依赖。

      无向图模型最大的特点就是易于因子分解，标准定义为：

      将无向图模型的联合概率分布表示为其最大团（maximal clique，可能不唯一）上的随机变量的函数的乘积形式。

      给定无向图 G ，其最大团为 C ，那么联合概率分布 P(Y) 可以写作图中所有最大团 C 上的势函数（potential function） $\psi_C(Y_C)$ 的乘积形式：

$$P(Y)=\frac1Z\prod_C\psi_C(Y_C)$$

$$Z=\sum_Y\prod_C\psi_C(Y_C)$$

其中 Z 称为规范化因子，对 Y 的所有可能取值求和，从而保证了 P(Y) 是一个概率分布。要求势函数严格正，通常定义为指数函数

$$\psi_C(Y_C)=\exp(-\mathbb E[Y_C])$$

      上面的因子分解过程就是 Hammersley-Clifford 定理。

**（二）条件随机场**
============

      **条件随机场（Conditional random field，CRF）**是条件概率分布模型 P(Y|X) ，表示的是给定一组输入随机变量 X 的条件下另**一组输出**随机变量 Y 的马尔可夫随机场，也就是说 CRF 的特点是假设输出随机变量构成马尔可夫随机场。

      条件随机场可被看作是最大熵马尔可夫模型在标注问题上的推广。

      这里介绍的是用于序列标注问题的线性链条件随机场（linear chain conditional CRF），是由输入序列来预测输出序列的判别式模型。

![](https://images2015.cnblogs.com/blog/1008922/201705/1008922-20170530155800805-2112012644.png)

图片来源：[3]

![](https://images2015.cnblogs.com/blog/1008922/201706/1008922-20170611212955684-1763220149.png)

图片来源：[2]

![](https://images2015.cnblogs.com/blog/1008922/201705/1008922-20170528172136235-591980719.png)

图片来源：[4]

      从问题描述上看，对于序列标注问题，X 是需要标注的观测序列，Y 是标记序列（状态序列）。在学习过程时，通过 MLE 或带正则的 MLE 来训练出模型参数；在测试过程，对于给定的观测序列，模型需要求出条件概率最大的输出序列。

      如果随机变量 Y 构成一个由无向图 G=(V, E) 表示的马尔可夫随机场，对任意节点 $v\in V$ 都成立，即

$$P(Y_v|X,Y_w,w\not=v)=P(Y_v|X,Y_w,w\sim v)$$

对任意节点 $v$ 都成立，则称 P(Y|X) 是条件随机场。式中 $w\not=v$ 表示 w 是除 v 以外的所有节点，$w\sim v$ 表示 w 是与 v 相连接的所有节点。不妨把等式两遍的相同的条件 X 都遮住，那么式子可以用下图示意：

![](https://images2015.cnblogs.com/blog/1008922/201705/1008922-20170528171904703-929128343.png)

很明显，这就是马尔可夫随机场的定义。

      **线性链条件随机场**

      在定义中并没有要求 X 和 Y 具有相同的结构，而在现实中，一般假设 X 和 Y 有相同的图结构。对于**线性链**条件随机场来说，图 G 的每条边都存在于状态序列 Y 的相邻两个节点，最大团 C 是相邻两个节点的集合，X 和 Y 有相同的图结构意味着每个 $X_i$ 都与 $Y_i$ 一一对应。

$$V=\{1,2,...,n\},\quad E=\{(i, i+1)\},\quad i=1,2,...,n-1$$

      设两组随机变量 $X=(X_1,...,X_n),Y=(Y_1,...,Y_n)$ ，那么线性链条件随机场的定义为

$$P(Y_i|X,Y_1,...,Y_{i-1},Y_{i+1},...,Y_n)=P(Y_i|X,Y_{i-1},Y_{i+1}),\quad i=1,...,n$$

其中当 i 取 1 或 n 时只考虑单边。

      **一、线性链条件随机场的数学表达式**

      **1. 线性链条件随机场的参数化形式：特征函数及例子**

      此前我们知道，马尔可夫随机场可以利用最大团的函数来做因子分解。给定一个线性链条件随机场 P(Y|X) ，当观测序列为 $x=x_1x_2\cdots$ 时，状态序列为 $y=y_1y_2\cdots$ 的概率可写为（实际上应该写为 $P(Y=y|x;\theta)$ ，参数被省略了）

$$P(Y=y|x)=\frac{1}{Z(x)}\exp\biggl(\sum_k\lambda_k\sum_it_k(y_{i-1},y_i,x,i)+\sum_l\mu_l\sum_is_l(y_i,x,i)\biggr)$$

$$Z(x)=\sum_y\exp\biggl(\sum_k\lambda_k\sum_it_k(y_{i-1},y_i,x,i)+\sum_l\mu_l\sum_is_l(y_i,x,i)\biggr)$$

 $Z(x)$ 作为规范化因子，是对 y 的所有可能取值求和。

      **序列标注 vs 分类**

      是不是和 Softmax 回归挺像的？它们都属于对数线性模型（log linear model），线性链 CRF 用来解决序列标注问题，Softmax 回归、最大熵模型都是用来解决分类问题。但需要注意，这两类问题存在非常大的区别：

      （1）如果把序列标注问题看作分类问题，也就是为每一个待标注的位置都当作一个样本然后进行分类，那么将会有很大的信息损失，因为一个序列的不同位置之间存在联系：比如说有一系列连续拍摄的照片，现在想在照片上打上表示照片里的活动内容的标记，当然可以将每张照片单独做分类，但是会损失信息，例如当有一张照片上是一张嘴，应该分类到 “吃饭” 还是分类到 “唱 K” 呢？如果这张照片的上一张照片内容是吃饭或者做饭，那么这张照片表示 “吃饭” 的可能性就大一些，如果上一张照片的内容是跳舞，那这张照片就更有可能在讲唱 K 的事情。（这个例子来自 [5] 的开头。）

      （2）不同的序列有不同的长度，不便于表示成同一维度的向量。

      （3）状态序列的解集随着序列长度指数级增长，穷举法是不可行的。

      **特征函数**

      对于线性链 CRF，特征函数是个非常重要的概念：

      转移特征 $t_k(y_{i-1},y_i,x,i)$ 是定义在边上的特征函数（transition），依赖于当前位置 i 和前一位置 i-1 ；对应的权值为 $\lambda_k$ 。

      状态特征 $s_l(y_i,x,i)$ 是定义在节点上的特征函数（state），依赖于当前位置 i ；对应的权值为 $\mu_l$ 。

      一般来说，特征函数的取值为 1 或 0 ，当满足规定好的特征条件时取值为 1 ，否则为 0 。

      **以词性标注为例**

      下面给出一些特征函数的例子，参考自 [5] 。词性标注（Part-of-Speech Tagging，POS）任务是指 the goal is to label a sentence (a sequence of words or tokens) with tags like ADJECTIVE, NOUN, PREPOSITION, VERB, ADVERB, ARTICLE. 在对英文序列进行词性标注时可以使用以下特征：

      （1）$s_1(y_i,x,i)=1$ ，如果 $y_i=ADVERB$ 且 $x_i$ 以 “-ly” 结尾；否则为 0 。如果该特征函数有一个较大的正权重，就表明倾向于将 “-ly” 结尾的单词标注为副词。

      （2）$s_2(y_i,x,i)=1$，如果 $i=1$ 、$y_i＝VERB$ 且 x 以 “?” 结尾；否则为 0 。如果该特征函数有一个较大的正权重，就表明倾向于将问句的首词标注为动词，例如“Is this a sentence beginning with a verb?”

      （3）$t_3(y_{i-1},y_i,x,i)=1$，如果 $y_{i-1}=ADJECTIVE$ 且 $y_{i}=NOUN$；否则为 0 。 如果该特征函数有一个较大的正权重，就表明倾向于认为形容词后面跟着名词。

      （4）$t_4(y_{i-1},y_i,x,i)=1$，如果 $y_{i-1}=PREPOSITION$ 且 $y_{i}=PREPOSITION$；否则为 0 。 如果该特征函数有一个较大的负权重，就表明倾向于认为介词不会连用。

      **CRF vs HMM**

      [5] 中还比较了 HMM 和 CRF 在序列标注的异同，作者认为 CRF 更加强大，理由如下：

      （1）可以为每个 HMM 都建立一个等价的 CRF（记号中的 s、l 就是本文的 x、y ）：

![](https://images2015.cnblogs.com/blog/1008922/201705/1008922-20170531215200836-1169133956.png)

图片来源：[5]

      （2）CRF 的特征可以囊括更加广泛的信息：HMM 基于 “上一状态 to 当前状态” 的转移概率以及 “当前状态 to 当前观测” 的释放概率，使得当前位置的词（观测）只可以利用当前的状态（词性）、当前位置的状态又只能利用上一位置的状态。但 CRF 的特征函数中，输入包含 $(y_{i-1},y_i,x,i)$ ，对于当前位置 i 来说可以利用完整的 x 信息。

      （3）CRF 的参数的取值没有限制，而 HMM 的参数（转移概率矩阵、释放概率矩阵、初始概率向量）都需要满足一些限制。

      **2. 线性链条件随机场的简化形式**

      需要注意的是，以 $\sum_k\lambda_k\sum_it_k(y_{i-1},y_i,x,i)$ 这项为例，可以看出外面那个求和号是套着里面的求和号的，这种双重求和就表明了对于同一个特征（k），在各个位置（i）上都有定义。

      基于此，很直觉的想法就是把同一个特征在各个位置 i 求和，形成一个全局的特征函数，也就是说让里面那一层求和号消失。在此之前，为了把加号的两项合并成一项，首先将各个特征函数 t（设其共有 $K_1$ 个）、s（设共 ${K_2}$ 个）都换成统一的记号 f ：

$$t_1=f_1,t_2=f_2,\cdots,t_{K_1}=f_{K_1},\quad s_1=f_{K_1+1},s_2=f_{K_1+2},\cdots,s_{K_2}=f_{K_1+K_2}$$

相应的权重同理：

$$\lambda_1=w_1,\lambda_2=w_2,\cdots,\lambda_{K_1}=w_{K_1},\quad \mu_1=w_{K_1+1},\mu_2=w_{K_1+2},\cdots,\mu_{K_2}=w_{K_1+K_2}$$

那么就可以记为

$$f_k(y_{i-1},y_i,x,i)=\begin{cases}t_k(y_{i-1},y_i,x,i), & k=1,2,...,K_1 \\s_l(y_i,x,i), & k=K_1+l;l=1,2,...,K_2\end{cases}$$

$$w_k=\begin{cases}\lambda_k, & k=1,2,...,K_1 \\\mu_l, & k=K_1+l;l=1,2,...,K_2\end{cases}$$

然后就可以把特征在各个位置 i 求和，即

$$f_k(y,x)=\sum_{i=1}^n f_k(y_{i-1},y_i,x,i), \quad k=1,2,...,K$$

其中 $K=K_1+K_2$ 。进而可以得到简化表示形式

$$P(Y=y|x)=\frac{1}{Z(x)}\exp\sum_{k=1}^Kw_kf_k(y,x)$$

$$Z(x)=\sum_y\exp\sum_{k=1}^Kw_kf_k(y,x)$$

      如果进一步，记 $\textbf w=(w_1,w_2,...,w_K)^{\top}$ ，$F(y,x)=(f_1(y,x),...,f_K(y,x))^{\top}$ ，那么可得内积形式：

$$P_{\textbf w}(Y=y|x)=\frac{1}{Z_{\textbf w}(x)}\exp(\textbf w^{\top}F(y,x))$$

$$Z_{\textbf w}(x)=\sum_y\exp(\textbf w^{\top}F(y,x))$$

      **3. 线性链条件随机场的矩阵形式**

      这种形式依托于线性链条件随机场对应的图模型仅在两个相邻节点之间存在边。在状态序列的两侧添加两个新的状态 $y_0 = start$ 、$y_{n+1}=stop$ 。

      这里，引入一个新的量 $M_i(y_{i-1},y_i|x)$ ：

$$M_i(y_{i-1},y_i|x)=\exp\sum_{k=1}^Kw_kf_k(y_{i-1},y_i,x,i),\quad i=1,2,...,n+1$$

首先，这个量融合了参数和特征，是一个描述模型的比较简洁的量；其次，不难发现，这个量相比于原来的非规范化概率 $P(Y=y|x)\propto\exp\displaystyle\sum_{k=1}^Kw_kf_k(y,x)$ ，少了对位置的内层求和，换句话说这个量是针对于某个位置 i （及其前一个位置 i-1 ）的。那么，假设状态序列的状态存在 m 个可能的取值，对于任一位置 i = 1,2,...,n+1 ，定义一个 m 阶方阵：

$$\begin{aligned}M_i(x)&=[\exp\sum_{k=1}^Kf_k(y_{i-1},y_i,x,i)]_{m\times m}\\&=[M_i(y_{i-1},y_i|x)]_{m\times m}\end{aligned}$$

      因为有等式 $\displaystyle\prod_i\biggl[\exp\sum_{k=1}^Kw_kf_k(y_{i-1},y_i,x,i)\biggr]=\exp\biggl(\sum_{k=1}^Kw_k\sum_i f_k(y_{i-1},y_i,x,i)\biggr)$ 成立，所以线性链条件随机场可以表述为如下的矩阵形式：

$$P_{\textbf w}(Y=y|x)=\frac{1}{Z_{\textbf w}(x)}\prod_{i=1}^{n+1}M_i(y_{i-1},y_i|x)$$

$$Z_{\textbf w}(x)=(M_1(x)M_2(x)\cdots M_{n+1}(x))_{(start,stop)}$$

其中规范化因子 $Z_{\textbf w}(x)$ 是这 n+1 个矩阵的乘积矩阵的索引为 $(start,stop)$ 的元素。 $Z_{\textbf w}(x)$ 它就等于以 start 为起点、以 stop 为终点的所有状态路径的非规范化概率 $\prod_{i=1}^{n+1}M_i(y_{i-1},y_i|x)$ 之和（证明略）。

      上面的描述或多或少有些抽象，[1] 中给出了一个具体的例子：给定一个线性链条件随机场，n = 3 ，状态的可能取值为 5 和 7 。设 $y_0 = start = 5$ 、$y_{n+1}=stop=5$ ，且 M 矩阵在 i = 1,2,...,n+1 的值已知，求状态序列以 start 为起点、以 stop 为终点的所有状态路径的非规范化及规范化概率。

$$M_1(x)=\begin{pmatrix}a_{01} & a_{01}\\0&0\end{pmatrix},\quad M_2(x)=\begin{pmatrix}b_{11} & b_{12}\\b_{21} & b_{22}\end{pmatrix}$$

$$M_3(x)=\begin{pmatrix}c_{11} & c_{12}\\c_{21} & c_{22}\end{pmatrix},\quad M_4(x)=\begin{pmatrix}1 & 0\\1 & 0\end{pmatrix}$$

      所有可能的状态路径，共 8 条（没有刻意区分 Y 和 y 这两种记号）：

![](https://images2015.cnblogs.com/blog/1008922/201705/1008922-20170529180319258-1113258467.png)

      先看一下 M 矩阵的含义。以 $M_3(x)$ 为例：行索引就是当前位置（此处为 3）的上一位置（此处为 2）的状态可能取值，列索引就是当前位置的状态可能取值。

![](https://images2015.cnblogs.com/blog/1008922/201705/1008922-20170530001654539-81518938.png) 

每个 M 矩阵的行／列索引都是一致的，对应于状态的可能取值。因此，M 矩阵的每个元素值就有点 Markov chain 里的 “转移概率” 的意思：以 $M_3(x)$ 的 $c_{12}$ 为例，它的行索引是 5，列索引是 7，可以 “看作” 是上一位置（2）的状态是 5 且当前位置（3）的状态是 7 的 “非规范化转移概率”。

      那么根据公式 $P_{\textbf w}(Y=y|x)\propto\displaystyle\prod_{i=1}^{n+1}M_i(y_{i-1},y_i|x)$ ，可知状态序列 $y_0y_1\cdots y_{4}$ 为 (5, 5, 5, 7, 5) 的非规范化概率为 $a_{01}\times b_{11}\times c_{12}\times 1$ ，其中 $a_{01}$ 是位置 0 的状态为 5 且位置 1 的状态为 7 的 “转移概率”，其他三项亦可以看作 “转移概率”。同理，可求得其他七条路径的非规范化概率。

![](https://images2015.cnblogs.com/blog/1008922/201705/1008922-20170530004035664-545058566.png)

      规范化因子就等于 $M_1(x)M_2(x)M_3(x)M_4(x)$ 的行索引为 5、列索引为 5 的值，经计算，等于所有 8 条路径的非规范化概率之和。

      **二、线性链条件随机场的计算问题**

      与隐马尔可夫模型类似，条件随机场也有三个基本问题：计算问题、解码问题和学习问题，其中前两个问题属于 inference，第三个问题当然是 learning。下面简单介绍。

      CRF 的计算问题是指，给定一个条件随机场 P(Y|X) 、观测序列 x 和状态序列 y ，计算 $P(Y_i=y_i|x)$ 、$P(Y_{i-1}=y_{i-1},Y_i=y_i|x)$ 以及特征函数关于分布的期望。

      回顾一下 HMM，当时解决这个问题使用的是前向算法／后向算法。这里类似，对每个位置 i =0,1,...,n+1 ，定义前向向量 $\boldsymbol\alpha_i(x)$ ：

$$\alpha_0(y|x)=\begin{cases}1, & y=start\\0, & otherwise\end{cases}$$

$$\alpha_i(y_i|x)=\sum_{y_{i-1}}\alpha_{i-1}(y_{i-1}|x)M_i(y_{i-1},y_i|x),\quad i=1,2,...,n+1$$

 $\alpha_i(y_i|x)$ 的含义是在位置 i 的标记 $Y_i=y_i$ 且从起始位置到位置 i 的局部标记序列的非规范化概率，这个递推式子可以直观地把 $M_i(y_{i-1},y_i|x)$ 理解为 “转移概率”，求和号表示对 $y_{i-1}$ 的所有可能取值求和。写成矩阵的形式就是下式

$$\boldsymbol\alpha_i^{\top}(x)=\boldsymbol\alpha_{i-1}^{\top}(x)M_i(x)$$

这里的 $\boldsymbol\alpha_i(x)$ 是 m 维列向量，因为每个位置的标记都有 m 种可能取值，每一个维度都对应一个 $\alpha_i(y_i|x)$ 。

      类似地，可以定义后向向量 $\boldsymbol\beta_i(x)$ ：

$$\beta_{n+1}(y_{n+1}|x)=\begin{cases}1, & y_{n+1}=stop\\0, & otherwise\end{cases}$$

$$\beta_i(y_i|x)=\sum_{y_{i+1}}M_{i+1}(y_{i},y_{i+1}|x)\beta_{i+1}(y_{i+1}|x),\quad i=0,1,...,n$$

 $\beta_i(y_i|x)$ 的含义是在位置 i 的标记 $Y_i=y_i$ 且从位置 i+1 到位置 n 的局部标记序列的非规范化概率。写成矩阵的形式就是

$$\boldsymbol\beta_i^{\top}(x)=M_{i+1}(x)\boldsymbol\beta_{i+1}(x)$$

      另外，规范化因子 $Z(x)=\boldsymbol\alpha^{\top}_n(x)\boldsymbol 1=\boldsymbol 1^{\top}\boldsymbol\beta_1(x)$ 。

      **1. 概率值的计算**

      给定一个 CRF 模型，那么 $P(Y_i=y_i|x)$ 、$P(Y_{i-1}=y_{i-1},Y_i=y_i|x)$ 可以利用前向向量和后向向量计算为

$$P(Y_i=y_i|x)=\frac{\alpha_i(y_i|x)\beta_i(y_i|x)}{Z(x)}$$

$$P(Y_{i-1}=y_{i-1},Y_i=y_i|x)=\frac{\alpha_{i-1}(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{Z(x)}$$

      **2. 期望值的计算**

      （1）特征函数 $f_k$ 关于条件分布 P(Y|X) 的期望：

$$\begin{aligned}\mathbb E_{P(Y|x)}[f_k]&=\sum_{y}P(Y=y|x)f_k(y,x)\\&=\sum_{y}P(Y=y|x)\sum_{i=1}^{n+1} f_k(y_{i-1},y_i,x,i)\\&=\sum_{i=1}^{n+1}\sum_{y_{i-1}y_i}f_k(y_{i-1},y_i,x,i)P(Y_{i-1}=y_{i-1},Y_i=y_i|x)\\&=\sum_{i=1}^{n+1}\sum_{y_{i-1}y_i}f_k(y_{i-1},y_i,x,i)\frac{\alpha_{i-1}(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{Z(x)}\end{aligned}$$

第一个等号，可以看出计算代价非常大，但转化为第二个等号后，便可利用前向向量和后向向量来高效计算。

      （2）特征函数 $f_k$ 关于联合分布 P(X,Y) 的期望：

      这里假设已知边缘分布 P(X) 的经验分布为 $\widetilde P(X)$ ，经验分布就是根据训练数据，用频数估计的方式得到 $\widetilde P(X=x)=\dfrac{\#x}{N}$。

$$\begin{aligned}\mathbb E_{P(X,Y)}[f_k]&=\sum_{x,y}P(x,y)f_k(y,x)\\&=\sum_x\widetilde P(x)\sum_{y}P(Y=y|x)\sum_{i=1}^{n+1} f_k(y_{i-1},y_i,x,i)\\&=\sum_x\widetilde P(x)\sum_{i=1}^{n+1}\sum_{y_{i-1}y_i}f_k(y_{i-1},y_i,x,i)\frac{\alpha_{i-1}(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{Z(x)}\end{aligned}$$

第二个等号那里类似于最大熵模型的条件熵的定义。

      对于给定的观测序列 x 和标记序列 y ，通过一次前向扫描计算 $\boldsymbol\alpha_i$ 及 $Z(x)$ ，一次后向扫描计算 $\boldsymbol\beta_i$ ，进而计算所有的概率值，以及特征的期望。

      **三、线性链条件随机场的解码问题**

      解码问题即预测问题，给定条件随机场 P(Y|X) 和观测序列 x ，求最有可能的状态序列 y* 。与 HMM 类似，使用维特比算法求解。

      **四、线性链条件随机场的学习问题**

      CRF 是定义在时序数据上的对数线性模型，使用 MLE 和带正则的 MLE 来训练。类似于最大熵模型，可以用改进的迭代尺度法（IIS）和拟牛顿法（如 BFGS 算法）来训练。

      训练数据 $\{(x^{(j)},y^{(j)})\}_{j=1}^N$ 的对数似然函数为

$$\begin{aligned}L(\textbf w)=L_{\widetilde P}(P_\textbf w)&=\ln\prod_{j=1}^NP_{\textbf w}(Y=y^{(j)}|x^{(j)})\\&=\sum_{j=1}^N\ln P_{\textbf w}(Y=y^{(j)}|x^{(j)})\\&=\sum_{j=1}^N\ln \frac{\exp\sum_{k=1}^Kw_kf_k(y^{(j)},x^{(j)})}{Z_{\textbf w}(x^{(j)})}\\&=\sum_{j=1}^N\biggl(\sum_{k=1}^Kw_kf_k(y^{(j)},x^{(j)})-\ln Z_{\textbf w}(x^{(j)})\biggr)\end{aligned}$$

      或者可以这样写：

$$\begin{aligned}L(\textbf w)=L_{\widetilde P}(P_\textbf w)&=\ln\prod_{x,y}P_{\textbf w}(Y=y|x)^{\widetilde P(x,y)}\\&=\sum_{x,y}\widetilde P(x,y)\ln P_{\textbf w}(Y=y|x)\\&=\sum_{x,y}\widetilde P(x,y)\ln \frac{\exp\sum_{k=1}^Kw_kf_k(y,x)}{Z_{\textbf w}(x)}\\&=\sum_{x,y}\widetilde P(x,y)\sum_{k=1}^Kw_kf_k(y,x)-\sum_{x,y}\widetilde P(x,y)\ln Z_{\textbf w}(x)\\&=\sum_{x,y}\widetilde P(x,y)\sum_{k=1}^Kw_kf_k(y,x)-\sum_{x}\widetilde P(x)\ln Z_{\textbf w}(x)\end{aligned}$$

最后一个等号是因为 $\sum_yP(Y=y|x)=1$ 。顺便求个导：

$$\begin{aligned}\frac{\partial L(\textbf w)}{\partial w_i}&=\sum_{x,y}\widetilde P(x,y)f_i(x,y)-\sum_{x,y}\widetilde P(x)P_{\textbf w}(Y=y|x)f_i(x,y)\\&=\mathbb E_{\widetilde P(X,Y)}[f_i]-\sum_{x,y}\widetilde P(x)P_{\textbf w}(Y=y|x)f_i(x,y)\end{aligned}$$

      似然函数中的 $\ln Z_{\textbf w}(x)$ 项是一个指数函数的和的对数的形式。关于这一项在编程过程中需要注意的地方可以参考[这篇博客](http://www.hankcs.com/ml/computing-log-sum-exp.html)。

参考：

[1] 统计学习方法

[2] [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](http://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers) 

[3] [Conditional Random Fields: An Introduction](http://people.cs.umass.edu/~wallach/technical_reports/wallach04conditional.pdf)

[4] [An Introduction to Conditional Random Fields for Relational Learning](https://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf)

[5] [Introduction to Conditional Random Fields](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)

[6] [Log-Linear Models, MEMMs, and CRFs](http://www.cs.columbia.edu/~mcollins/crf.pdf)  

[7] 基于条件随机场的中文命名实体识别（向晓雯，2006 thesis）

[8] [CRF++](http://taku910.github.io/crfpp/)： [CRF++ 代码分析](http://www.hankcs.com/ml/crf-code-analysis.html) [CRF++ 中文分词](http://x-algo.cn/index.php/2016/02/27/crf-of-chinese-word-segmentation/) [CRF++ 词性标注](http://x-algo.cn/index.php/2016/02/28/crf-tagging/)

[9] [数值优化：理解 L-BFGS 算法](http://www.hankcs.com/ml/l-bfgs.html)  [牛顿法与拟牛顿法学习笔记（五）L-BFGS 算法](http://blog.csdn.net/itplus/article/details/21897715)

[10] [漫步条件随机场系列文章](http://www.cnblogs.com/baiboy/p/crf1.html)