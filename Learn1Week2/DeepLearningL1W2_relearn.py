import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.misc
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# 读取数据集，其中train_set_x_orig为训练集里面的图像数据，train_set_y is label
# test_set_y同理

index = 4
"""use plt.ishow function to show the [index] picture in train_set"""
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print(train_set_y[0,index])# why [:,index]?    answer:维度为[1,209]，则想要打印[1,index]的数据
#                                              就要打印[0,index],试了一下，可以，而且直接输出的是数字
#                                              而非维度，这 样就不用squeeze，但是为什么不这样做，应该是
#                                              为了增强可扩展性吧
# print("【使用np.squeeze：" + str(np.squeeze(train_set_y[:,index])) + "，不使用np.squeeze： " + str(train_set_y[:,index]) + "】")
# print("y=" + str(train_set_y[:, index]) + ", it's a " + classes[train_set_y[0, index]].decode("utf-8") + "' picture")
# print("y=" + str(train_set_y[:, index]) + ", it's a " + classes[np.squeeze(train_set_y[:, index])].decode("utf-8")+"'picture'")
m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]
# print("训练集的数量：{}".format(m_train))
# print("测试集的数量：{}".format(m_test))
# print("训练图像的高度：{}".format(num_px))
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# print(train_set_x_flatten.shape)
# print(test_set_x_flatten.shape)
# 对数据集进行居中和标准化
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# print(str(sigmoid(np.array([0,2]))))

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = np.zeros((1, m))
    A = sigmoid(np.dot(w.T, X) + b)
    # print(A.shape)
    cost = -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m
    # 反向传播导数
    dw = (np.dot(X, (A - Y).T)) / m
    db = np.sum(A - Y) / m
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {
        "dw": dw,
        "db": db
    }
    return grads, cost

# w, b=initialize_with_zeros(train_set_x_flatten.shape[0])
# grads, cost = propagate(w, b, train_set_x_flatten, train_set_y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))

# dim=train_set_x_flatten.shape[0]
# w, b = initialize_with_zeros(train_set_x_flatten.shape[0])


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i:%f" % (i, cost))
    params = {
        "w": w,
        "b": b
    }
    grads = {
        "dw": dw,
        "db": db
    }
    return params, grads, costs


# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print(costs)

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(m):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    assert (Y_prediction.shape == (1, m))
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    dim = X_train.shape[0]
    w, b = initialize_with_zeros(dim)
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    Y_prediction_train = predict(params["w"], params["b"], X_train)
    Y_prediction_test = predict(params["w"], params["b"], X_test)
    w = params["w"]
    b = params["b"]
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d
d= model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
# index=5
#
# plt.imshow(test_set_x_orig[index])
# plt.show()
# print("y={},你的预测是：{}".format(test_set_y[0,index],d["Y_prediction_test"][0,index]))
# print(d["Y_prediction_test"])
# print(test_set_y)
# costs = np.squeeze(d['costs'])
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds)')
# plt.title("Learning rate =" + str(d["learning_rate"]))
# plt.show()
# learning_rate=[0.01,0.001,0.0001]
# models={}
# for i in learning_rate:
#     print("learning rate is :"+str(i))
#     models[str(i)]=model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=i, print_cost=True)
#     print("\n-------------------------------\n")
# for i in learning_rate:
#     plt.plot(np.squeeze(models[str(i)]["costs"]),label=str(models[str(i)]["learning_rate"]))
#
# plt.ylabel('cost')
# plt.xlabel('iteration(per hundreds)')
# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()
