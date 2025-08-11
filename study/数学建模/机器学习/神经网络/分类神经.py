from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 测试集，画图对预测值和实际值进行比较
def test_validate(x_test, y_test, y_predict, classifier):
    x = range(len(y_test))
    plt.plot(x, y_test, "ro", markersize=5, zorder=3, label=u"true_v")
    plt.plot(x, y_predict, "go", markersize=8, zorder=2, label=u"predict_v,$R$=%.3f" % classifier.score(x_test, y_test))
    plt.legend(loc="upper left")
    plt.xlabel("number")
    plt.ylabel("true?")
    plt.show()

# 神经网络数字分类
def multi_class_nn():
    digits = datasets.load_digits()
    x = digits['data']
    y = digits['target']

    # 对数据的训练集进行标准化
    ss = StandardScaler()
    x_regular = ss.fit_transform(x)
    # 划分训练集与测试集
    x_train, x_test, y_train, y_test = train_test_split(x_regular, y, test_size=0.1)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1)
    clf.fit(x_train, y_train)
    # 模型效果获取
    r = clf.score(x_train, y_train)
    print("R值(准确率):", r)
    # 预测
    y_predict = clf.predict(x_test)  # 预测
    print(y_predict)
    print(y_test)
    # 绘制测试集结果验证
    test_validate(x_test=x_test, y_test=y_test, y_predict=y_predict, classifier=clf)

multi_class_nn()
