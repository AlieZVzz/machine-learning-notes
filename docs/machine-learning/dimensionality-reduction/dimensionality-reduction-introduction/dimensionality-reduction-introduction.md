# 降维概述

* [返回上层目录](../dimensionality-reduction.md)



机器学习领域中所谓的降维就是指采用某种映射方法，将原高维空间中的数据点映射到低维度的空间中。降维的本质是学习一个映射函数$f : x\rightarrow y$，其中$x$是原始数据点的表达，目前最多使用向量表达形式。 $y$是数据点映射后的低维向量表达，通常$y$的维度小于$x$的维度（当然提高维度也是可以的）。$f$可能是显式的或隐式的、线性的或非线性的。

目前大部分降维算法处理向量表达的数据，也有一些降维算法处理高阶张量表达的数据。之所以使用降维后的数据表示是因为在原始的高维空间中，包含有冗余信息以及噪音信息，在实际应用例如图像识别中造成了误差，降低了准确率；而通过降维,我们希望减少冗余信息所造成的误差,提高识别（或其他应用）的精度。又或者希望通过降维算法来寻找数据内部的本质结构特征。

在很多算法中，降维算法成为了数据预处理的一部分，如PCA。事实上，有一些算法如果没有降维预处理，其实是很难得到很好的效果的。



# 参考文献

* [四大机器学习降维算法：PCA、LDA、LLE、Laplacian Eigenmaps](http://dataunion.org/13451.html)

“降维概述”一节参考这篇博客。
