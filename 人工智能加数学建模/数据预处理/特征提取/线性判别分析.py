# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions (suppress= True)

def read_iris():
    from sklearn.datasets import load_iris
    from sklearn import preprocessing
    data_set = load_iris()
    data_x = data_set.data
    label = data_set.target + 1
    #preprocessing.step(data_x, axis=0, with_mean=True, with_std=True, copy=False)
    return data_x,label
 # 特征均值,计算每类的均值，返回一个向量

def class_mean(data,label,clusters):
    mean_vectors = []
    for cl in range(1,clusters+1):
        mean_vectors.append(np.mean(data[label==cl,],axis=0))
    #print mean_vectors
    return mean_vectors

# 计算类内散度
def within_class_SW(data,label,clusters):
    m = data.shape[1]
    S_W = np.zeros((m,m))
    mean_vectors = class_mean(data,label,clusters)
    for cl ,mv in zip(range(1,clusters+1),mean_vectors):
        class_sc_mat = np.zeros((m,m))
        # 对每个样本数据进行矩阵乘法
        for row  in data[label == cl]:
            row ,mv =row.reshape(4,1),mv.reshape(4,1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W +=class_sc_mat
    #print S_W
    return S_W

def between_class_SB(data,label,clusters):
    m = data.shape[1]
    all_mean =np.mean(data,axis = 0)
    S_B = np.zeros((m,m))
    mean_vectors = class_mean(data,label,clusters)
    for cl ,mean_vec in enumerate(mean_vectors):
        n = data[label==cl+1,:].shape[0]
        mean_vec = mean_vec.reshape(4,1) # make column vector
        all_mean = all_mean.reshape(4,1)# make column vector
        S_B += n * (mean_vec - all_mean).dot((mean_vec - all_mean).T)
    #print S_B
    return S_B

def lda():
    data,label=read_iris()
    clusters = 3
    S_W = within_class_SW(data,label,clusters)
    S_B = between_class_SB(data,label,clusters)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    #print S_W
    #print S_B
    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(4,1)
        #print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))   #输出特征向量
        #print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))  #输出特征向量的值
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
    #print('Matrix W:\n', W.real)     #输出特征矩阵
    print('降维后的数据\n',data.dot(W))
    return W

def plot_lda():
    data,labels = read_iris()
    W = lda()
    Y = data.dot(W)
    #print Y
    ax= plt.subplot(111)
    for label,marker,color in zip(range(1,4),('^','s','o'),('blue','red','green')):
        plt.scatter(x=Y[:,0][labels == label],
            y=Y[:,1][labels == label],
            marker = marker,
            color = color,
            alpha = 0.5,
            )
    plt.xlabel('LDA1')
    plt.ylabel('LDA2')
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')
    plt.show()


def sklearnLDA():
    from sklearn import datasets
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names

    lda = LDA(n_components=2, solver='eigen',shrinkage='auto')
    ''' solver：LDA算法使用什么优化器，默认为'svd'，也可选择'lsqr'或'eigen'。
        shrinkage：LDA算法的缩减系数，用于对协方差矩阵进行调整，一般为float类型，并可输入  auto  等特定字符串。
        priors：类别的先验概率，如果使用类平衡的先验概率，请将其设置为None。
        tol：模型收敛的阈值，默认为1e-4。
        covariance_estimator：LDA算法中用于估计协方差矩阵的方法，默认为'ledoit_wolf'。'''
    X_r2 = lda.fit(X, y).transform(X)
    print('sklearn_lda')
    print(X_r2)
    return X_r2

if __name__ =="__main__":
    #lda()
    sklearnLDA()
    plot_lda()