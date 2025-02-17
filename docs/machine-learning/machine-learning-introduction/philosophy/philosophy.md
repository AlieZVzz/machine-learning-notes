# 机器学习的哲学思想

* [返回上层目录](../machine-learning-introduction.md)
* [机器学习的哲学思想](#机器学习的哲学思想)
* [没有免费的午餐定理—教条主义的危害](#没有免费的午餐定理—教条主义的危害)
* [奥卡姆剃刀定理—少即是多](#奥卡姆剃刀定理—少即是多)
* [三个臭皮匠的智慧—初看集成学习](#三个臭皮匠的智慧—初看集成学习)
* [民主自由与专制—再看集成学习](#民主自由与专制—再看集成学习)
* [频率学派和贝叶斯学派—不可知论](#频率学派和贝叶斯学派—不可知论)
* [后记：妥协、矛盾与独立思考](#后记：妥协、矛盾与独立思考)


**不仅仅是机器学习，大部分理工科的思想都可以从文史学科的角度去理解。正所谓大道至简，很多道理是共通的**。本文的内容是从哲学角度看待机器学习。文章的结构如下：

1. 天下没有免费的午餐—教条主义的危害
2. 奥卡姆剃刀定理—少即是多
3. 三个臭皮匠的智慧—初看集成学习
4. 民主自由与专制—再看集成学习
5. 频率学派和贝叶斯学派—不可知论
6. 后记：妥协、矛盾与独立思考

# 没有免费的午餐定理—教条主义的危害

（No Free Lunch Theorem / NFL定理）

NFL定理由Wolpert在1996年提出，其应用领域原本为经济学。我们耳熟能详的“天下没有免费的午餐”最早是说，十九世纪初很多欧美酒吧和旅店为了提升销售额向顾客提供免费的三明治，而客人贪图免费的三明治却意外的买了很多杯酒，酒吧从而获利更多了，以此来教育大家不要贪小便宜吃大亏。和那句家喻户晓的"天下没有免费的午餐"有所不同， NFL讲的是**优化模型的评估问题**。

在机器学习领域，NFL的意义在于告诉机器学习从业者："假设所有数据的分布可能性相等，当我们用任一分类做法来预测未观测到的新数据时，对于误分的预期是相同的。" 简而言之，NFL的定律指明，**如果我们对要解决的问题一无所知且并假设其分布完全随机且平等，那么任何算法的预期性能都是相似的**。

**这个定理对于“盲目的算法崇拜”有毁灭性的打击**。例如，现在很多人沉迷“深度学习”不可自拔，那是不是深度学习就比其他任何算法都要好？在任何时候表现都更好呢？未必，我们必须要加深对于问题的理解，不能盲目的说某一个算法可以包打天下。然而，从另一个角度说，我们对于要解决的问题往往不是一无所知，因此大部分情况下我们的确知道什么算法可以得到较好的结果。举例，我们如果知道用于预测的特征之间有强烈的相关性，那么我们可以推测Naive Bayes（简单贝叶斯分类器）不会给出特别好的结果，因为其假设就是特征之间的独立性。

**在某个领域、特定假设下表现卓越的算法不一定在另一个领域也能是“最强者”。正因如此，我们才需要研究和发明更多的机器学习算法来处理不同的假设和数据**。George Box早在上世纪八十年代就说过一句很经典的话："**All models are wrong, but some are useful**（所有的模型的都是错的，但其中一部分是有用的）。" 这也可以理解为是NFL的另一种表述。周志华老师在《机器学习》一书中也简明扼要的总结：“NFL定理最重要的寓意，是让我们清楚的认识到，**脱离具体问题，空泛的谈‘什么学习算法更好’毫无意义**。”  

# 奥卡姆剃刀定理—少即是多

奥卡姆剃刀定理(Occam's Razor - Ockham定理）

奥卡姆剃刀是由十二世纪的英国教士及哲学家奥卡姆提出的：“ **如无必要，勿增实体**”。用通俗的语言来说，如果两个模型A和B对数据的解释能力完全相同，那么选择较为简单的那个模型。**在统计学和数学领域，我们偏好优先选择最简单的那个假设，如果与其他假设相比，其对于观察的描述度一致**。

奥卡姆剃刀定理对于机器学习的意义在于**它给出了一种模型选择的方法**，对待**过拟合**问题有一定的指导意义。就像我在其他文章中提到的过的，如果简单的线性回归和复杂的深度学习在某个问题上的表现相似(如相同的误分率)，那么我们应该选择较为简单的线性回归。

Murphy在MLAPP中用Tenenbaum的强抽样假设（strong sampling assumption）来类比奥卡姆剃刀原理。首先他给出了下面的公式，$Pr(D|h)$代表了我们重置抽样（sampling with replacement）$N$次后得到集合$D$时，某种假设$h$为真的概率。
$$
Pr(D|h)=\left[ \frac{1}{size(h)} \right]^N=\left[ \frac{1}{|h|} \right]^N
$$
举例，我们有0~99共100个数字，每次从其中随机抽出一个数字并放回，重复抽取$N$次。若随机抽了并放回了5次，我们抽出了$\{ 2, 4, 8, 16, 32 \}$，于是我们想要推断抽取到底遵循什么规则。我们有两种假设：

- $h_1$：我们是从$\{ 2, 4, 6, 8, ... ,98 \}$中抽取的，即从偶数中抽取 : 
- $h_2$：我们是从$\{ 2^n \}$中抽取的

根据上文给出的公式进行计算，我们发现$Pr(D|h_2)$远大于$Pr(D|h_1)$，即我们相信$D = \{ 2, 4, 8, 16, 32 \}$从$h_2：\{ 2^n \}$中产生的可能更大，但是$h_1：\{ 2, 4, 6, 8, ... ,98 \}$似乎也能解释我们的结果。这个时候我们就应选择概率更高的那个。  

从奥卡姆剃刀角度思考的话，$h_2：\{ 2^n \}$在0~99中只有5个满足要求的元素，而$h_1：\{ 2, 4, 6, 8, ... ,98 \}$却有50个满足要求的元素。那么**$h_2$更加简单**，更加符合尝试，选择它:)  

提供这个例子的原因是为了提供一个量化方法来评估假设，其与奥卡姆剃刀有相同的哲学内涵。有兴趣的读者应该会发现奥卡姆剃刀的思想与贝叶斯推断是一致的，更细致的讨论可以看[刘未鹏 | Mind Hacks](http://mindhacks.cn/2008/09/21/the-magical-bayesian-method/)关于贝叶斯推断的介绍。 

**但读者应该注意，奥卡姆剃刀定理只是一种对于模型选择的指导方向，不同的选择方向如集成学习(Ensemble Learning)就给出了近似相反的选择标准。现实世界是非常复杂的，切勿滥用**。  

# 三个臭皮匠的智慧—初看集成学习

集成学习（Ensemble Learning）的哲学思想是“众人拾柴火焰高”，和其他机器学习模型不同，集成学习将多个较弱的机器学习（臭皮匠）模型合并起来来一起决策（诸葛亮）。比较常见的方法有多数投票法（majority vote），即少数服从多数。如果我们有10个"子分类器"通过一个人的疾病史来推断他能活多大，其中8个说他会活过200岁，其中2个说他在200岁前会死，那么我们相信他可以活过200岁。

集成学习的思想无处不在，比较著名的有随机森林等。从某种意义上说，**神经网络也是一种集成学习，每个单独的神经元都可以看做某种意义上的学习器**。

相信敏锐的读者已经发现，集成学习似乎和前面提到的奥卡姆剃刀定理相违背。明明一个分类模型就够麻烦了，现在为什么要做更多？这其实说到了一个很重要观点，就是**奥卡姆剃刀定理并非不可辩驳的真理，而只是一种选择方法。从事科学研究，切勿相信有普遍真理**。人大的周孝正教授曾说："**若一件事情不能证实，也不能证伪，就要存疑**。" 恰巧，**奥卡姆定理就是这样一种不能证实也不能证伪的定理**。

而集成学习的精髓在于假设“子分类器”的**错误相互独立**，随着集成中子分类器的数目上升，集成学习后的"母分类器"的误差将会以指数级别下降，直至为0。然而，这样的假设是过分乐观的，因为我们无法保证"子分类器"的错误是相互独立的。以最简单的Bagging为例，如果为了使k个子分类器的错误互相独立，那么我们将训练数据N分为k份。显然，随着k值上升，每个分类器用于训练的数据量都会降低，每个子训练器的准确性也随之下降。即使我们允许训练数据间有重采样，也还是无法避免子分类器数量和准确性之间的矛盾。周志华老师曾这样说："**个体学习的准确性和多样性本身就存在冲突，一般的，准确性很高后，想要增加多样性，就得要牺牲准确性**。事实上，**如何产生并结合好而不同个体学习器，恰是集合学习的研究核心**。"  

# 民主自由与专制—再看集成学习

细分集成学习的话，也有两种截然相反的设计思路：

- **思路1**：每个子学习器都是**弱**分类器，在融合后达成为一个强力的主体。代表算法：随机森林
- 每个子学习器都是**强**分类器，融合过程中可能：
  - **思路2**（强中取强）：选择最强的那一个。代表算法：dynamic classifier selection
  - **思路3**（公平选择）：一视同仁的对待每个子学习器，融合出一个更强的主体。代表算法：stacking

不难看出，思路1和3虽然期望的子学习器不同（弱vs强），但都比较公平做到了一视同仁，和民主很像。而思路2不同，更加强调自由（或是专制），选出最强的那一个，让它说了算。

**让一堆子学习器集体做决定的缺陷在于低效，容易被平庸的子学习器拖了后腿。而信赖最强者的缺点在于缺乏稳定性，上限可以很高，下限也可以很低**。

试问集成学习到底该选择哪条路？没有答案，但实用主义告诉我们哪个好用用哪个，要结合具体情况进行分析。  

# 频率学派和贝叶斯学派—不可知论

很多统计学习领域的小伙伴们都知道从统计学角度出发，对于概率有两种不同的认知。对于不熟悉的读者来说，无论是机器学习还是统计学习都是一种寻找一种**映射**，或者更广义的说，进行**参数估计**。以线性回归为例，我们得到结果仅仅是一组权重。

如果我们的目标是参数估计，那么有一个无法回避的问题...**参数到底存不存在**？换句话说，茫茫宇宙中是否到处都是不确定性（Uncertainty），而因此并不存在真实的参数，而一切都是处于运动当中的。

**频率学派（Frequentism）相信参数是客观存在的，虽然未知，但不会改变**。因此频率学派的方法一直都是试图估计“哪个值最接近真实值”，相对应的我们使用最大似然估计（Maximum Likelihood Estimation），置信区间（Confidence Level），和p-value。因此这一切都是体现我们对于真实值估算的自信和可靠度。

**而贝叶斯学派（Bayesian）相信参数不是固定的，我们需要发生过的事情来推测参数**，这也是为什么总和先验（Prior）及后验（Posterior）过不去，才有了最大后验（Maximum a Posteriori）即MAP。贝叶斯学派最大的优势在于承认未知（Uncertainty）的存在，因此感觉更符合我们的常识“不可知论”。从此处说，前文提到的周孝正教授大概是贝叶斯学派的（周教授是社会学家而不是统计学家）。  

据我不权威观察，不少统计学出身的人倾向于频率学派而**机器学习出身的人更倾向于贝叶斯学派**。比如著名的机器学习书籍PRML就是一本贝叶斯学习，而Murphy在MLAPP中曾毫无保留的花了一个小节指明频率学派的牵强之处。  

就像豆腐脑是甜的还是咸的，这样的问题还是留给读者们去思考。需要注意的是，两种思想都在现实中都有广泛的应用，切勿以偏概全。更多的思考可以看这篇知乎讨论：[贝叶斯学派与频率学派有何不同？ - 知乎](https://www.zhihu.com/question/20587681)

从哲学角度来看，频率学派和贝叶斯学派间的区别更像是在讨论“**形而上学**”以及“**不可知论**”。和我们高中课本中提到的的“二分法”思想似乎也有一定的联系。 

# 后记：妥协、矛盾与独立思考

在接触机器学习的早期阶段，时间往往都花在了研究算法上。随着学习的深入，相信大家会慢慢发现其实算法思想的精髓是**无处不在的妥协**。本文只能涉及到几种“矛盾”和“妥协”，更多的留给大家慢慢发掘和思考:）比如，本文未敢涉及到“统计学习”和“机器学习”之间的区别，也是一种妥协：模型可解释性与有效性的妥协。无处不在的妥协还包含“模型精度”和“模型效率”的妥协，“欠拟合”和“过拟合”的平衡等。

大部分科学，比如数学还是物理，走到一定程度，都是妥协，都有妥协带来的美感。这给我们的指导是：当我们听到不同的想法的时候，反驳之前先想一想，不要急着捍卫自己的观点。而相反的两种观点，在实际情况下却往往都有不俗的效果，这看似矛盾但却是一种和谐。

**因此，当面对纷杂的信息，各种似是而非的解释与结论时。最重要的不是急着发表观点，而是静下来慢下来，不要放弃思考。只有独立的思考，才能最终帮助我们摆脱重重迷雾，达到所追寻的真理。**

# 参考资料

- [机器学习包含哪些学习思想？](https://www.zhihu.com/question/267135168/answer/329318812)

“机器学习的哲学思想”一节来自这篇知乎回答。