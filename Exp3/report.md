# Report

## 实验设计

本实验为实现不同的集成学习算法，并对比它们与不同的基分类器结合时的效果。

实验任务为基于评论的评分预测任务，在本实验中该任务基于回归任务来实现。

本实验在模块 `models.py` 中实现了两个类： `Bagging` 和 `Adaboost` ，分别对应两个集成学习算法。每个类有对应的初始化方法 `__init__` 以及拟合与预测方法： `fit` 与 `predict`

在主程序 `main.py` 中会首先读取数据并将其按照 9:1 的比例划分为训练集和测试集。模型使用全部的训练集和测试集进行训练测试。

读取完数据集，程序会根据参数选取相应的基回归器： `svm.LinearSVR` 和 `tree.DecisionTreeRegressor` （使用 `scikit-learn` 中的工具包），然后根据参数选取相应的集成学习算法类。

之后变调用类中的 `fit` 与 `predict` 方法对训练集进行拟合，然后在测试集上输出预测结果并计算指标。

## 实验结果

### Baseline
使用以下命令运行实验
```bash
python main.py --regressor svm --ensemble baseline
python main.py --regressor tree --ensemble baseline
```
得到的结果如下：

|基回归器|MAE|MSE|RMSE|
|---|---|---|---|
|LinearSVR|0.8181|1.4875|1.2196|
|DecisionTreeRegressor|0.7500|1.0916|1.0448|

### Bagging集成

参数选择为 $n=5, ratio=0.8$ 
使用以下命令运行实验
```bash
python main.py --n 5 --ratio 0.8 --regressor svm --ensemble bagging
python main.py --n 5 --ratio 0.8 --regressor tree --ensemble bagging
```

|基回归器|MAE|MSE|RMSE|
|---|---|---|---|
|LinearSVR|0.8112|1.4114|1.1880|
|DecisionTreeRegressor|0.7323|0.9765|0.9882|

### Adaboost集成

参数选择为 $n=5$
使用以下命令运行实验
```bash
python main.py --n 5 --regressor svm --ensemble adaboost
python main.py --n 5 --regressor tree --ensemble adaboost
```

|基回归器|MAE|MSE|RMSE|
|---|---|---|---|
|LinearSVR|0.8037|1.4728|1.2136|
|DecisionTreeRegressor|0.7044|1.0577|1.0284|


## 实验分析

## 实验讨论
