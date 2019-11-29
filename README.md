# DataMining
Data Mining Homework: Spam Classification

数据挖掘实验：垃圾邮件分类-朴素贝叶斯模型与SVM模型的实验与评估

#### 实验环境

Python 3.7.3

PyCharm 2019.1.3 (Community Edition)

#### 实验数据

**数据来源：** http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

**数据描述：**

SMS Spam Collection是一组公开的带有标签的SMS邮件数据，从网上的免费研究资源中收集得到，共包含5574条SMS邮件数据，其中747条为垃圾邮件，4827条为非垃圾邮件。

**数据属性：**

SMS Spam Collection仅有一个文件，文件中每行为一条邮件数据，每条数据包含两列信息：第一列为邮件标签，spam表示垃圾邮件，ham表示非垃圾邮件；第二列为邮件内容。此外，邮件没有按照时间进行排序。

**数据示例：**

```
ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
ham	Ok lar... Joking wif u oni...
spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
ham	U dun say so early hor... U c already then say...
ham	Nah I don't think he goes to usf, he lives around here though
spam	FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv
ham	Even my brother is not like to speak with me. They treat me like aids patent.
ham	As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune
```

#### 文件结构

[/bayes_classify](/bayes_classify)  文件夹下为朴素贝叶斯模型：

- [pre_process_method.py](/bayes_classify/pre_process_method.py) 数据预处理，生成词库 [vocabulary_list.txt](/bayes_classify/vocabulary_list.txt)
- [bayes_method.py](/bayes_classify/bayes_method.py) 朴素贝叶斯算法
- [simple_evaluate.py](/bayes_classify/simple_evaluate.py) sklearn库中的朴素贝叶斯与自实现的朴素贝叶斯在相同训练集与测试集下的评估，生成的ROC曲线为 [bayes-simple.png](/bayes_classify/bayes-simple.png)
- [bootstrap_evaluate.py](/bayes_classify/bootstrap_evaluate.py) 通过bootstrap方法对训练集进行划分，并对朴素贝叶斯模型进行评估，生成的ROC曲线为 [bayes-bootstrap.png](/bayes_classify/bayes-bootstrap.png)
- [k_fold_evaluate.py](/bayes_classify/k_fold_evaluate.py) 通过k-fold方法对朴素贝叶斯模型的评估

[/svm_classify](/svm_classify) 文件夹下为SVM模型：

- [pre_process_method.py](/svm_classify/pre_process_method.py) 数据预处理
- [simple_evaluate.py](/svm_classify/simple_evaluate.py) 不同词频阈值下SVM模型的评估，生成的ROC曲线为 [svm-simple-roc.png](/svm_classify/svm-simple-roc.png)，accuracy、precision、recall、f1值的比较为  [svm-simple-evaluate.png](/svm_classify/svm-simple-evaluate.png)
- [bootstrap_evaluate.py](/svm_classify/bootstrap_evaluate.py) 通过bootstrap方法对训练集进行划分，并对SVM模型进行评估，生成的ROC曲线为 [svm-bootstrap.png](/svm_classify/svm-bootstrap.png)
- [k_fold_evaluate.py](/svm_classify/k_fold_evaluate.py) 通过k-fold方法对SVM模型的评估

[/data](/data) 文件夹下保存数据集：

- [train_data.txt](/data/train_data.txt) 用于simple_evaluate的训练集

- [test_data.txt](/data/test_data.txt) 用于simple_evaluate的测试集
- [data.txt](/data/data.txt) 用于bootstrap评估与k-fold评估的数据集

#### 实验结果
[数据挖掘实验报告.pdf](/数据挖掘实验报告.pdf)
