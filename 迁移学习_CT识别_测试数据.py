import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from tensorflow.python.platform import gfile

#迁移学习前的预处理
image_dir = r'D:\Users\datasets\CT Medical Image\image_jpg'
tensor_dir = r'D:\Users\datasets\CT Medical Image\image_tensor'
label_dir = r'D:\Users\datasets\CT Medical Image\overview.csv'
model_dir = r'D:/Users/datasets/CT Medical Image/model/transfer_learn'
BATCH = 50
#将数据分为测试集和训练集两类
def classify_train_and_test(image_dir,train_size):
    train_id_list = []
    test_id_list = []
    image_basenames = os.listdir(image_dir)
    i = 0
    for image_basename in image_basenames:
        if i<train_size:
            train_id_list.append(image_basename.split('.')[0])
        else:
            test_id_list.append(image_basename.split('.')[0])
        i = i+1
    return train_id_list,test_id_list

#不区分训练集和测试集
def no_class(image_dir):
    id_list = []
    image_basenames = os.listdir(image_dir)
    for image_basename in image_basenames:
        id_list.append(image_basename.split('.')[0])
    return id_list

#获取BSTCH个训练样本
def get_random_train_bottlenecks(train_id_list,labels,BATCH=50):
    train_bottlenecks = []
    train_labels = []
    for _ in range(BATCH):
        image_index = random.randrange(len(train_id_list))
        bottleneck_path = os.path.join(tensor_dir,train_id_list[image_index]+'.txt')
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        train_bottlenecks.append(bottleneck_values)
        ground_truth = np.zeros(2, dtype=np.float32)
        train_label = labels[int(train_id_list[image_index])]
        ground_truth[train_label] = 1.0
        train_labels.append(ground_truth)
    return train_bottlenecks,train_labels

#获取测试集数据
def get_test_bottlenecks(test_id_list,labels):
    test_bottlenecks = []
    test_labels = []
    for image_index in range(len(test_id_list)):
        bottleneck_path = os.path.join(tensor_dir,test_id_list[image_index]+'.txt')
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        test_bottlenecks.append(bottleneck_values)
        ground_truth = np.zeros(2, dtype=np.float32)
        test_label = labels[int(test_id_list[image_index])]
        ground_truth[test_label] = 1.0
        test_labels.append(ground_truth)
    return test_bottlenecks,test_labels

#保存模型数据
def save(sess,path):
    saver = tf.train.Saver()
    saver.save(sess, path, write_meta_graph=False)

#加载标签
overview_df = pd.read_csv(label_dir)
overview_df.columns = ['idx']+list(overview_df.columns[1:])
overview_df['Contrast'] = overview_df['Contrast'].map(lambda x:1 if x else 0)#1：Contrast,0:No Contrast
labels = dict(zip(overview_df['idx'],overview_df['Contrast']))

#将数据分类
# train_id_list,test_id_list = classify_train_and_test(image_dir,400)
id_list = no_class(image_dir)

#定义神经网络
bottleneck_input = tf.placeholder(tf.float32, [None, 2048], name='BottleneckInputPlaceholder')
label_input = tf.placeholder(tf.float32, [None, 2], name='GroundTruthInput')
lr = tf.Variable(0.001, dtype=tf.float32)

with tf.name_scope('final_training_ops'):
    weights = tf.Variable(tf.truncated_normal([2048, 2], stddev=0.001))
    biases = tf.Variable(tf.zeros([2]))
    logits = tf.matmul(bottleneck_input, weights) + biases
    final_tensor = tf.nn.softmax(logits)

#计算损失函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label_input)
#计算平均损失，reduce_mean(?,[0/1]),0为计算每一列的均值，1为计算每一行的均值
cross_entropy_mean = tf.reduce_mean(cross_entropy)
#通过梯度下降算法进行优化，第一个参数为学习率，第二个参数为当前总损失
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_mean)#正确率很高近似100%

with tf.name_scope('evaluation'):
    # tf.equal(A,B)对比A,B是否相等，tf.argmax即返回最大值的下标，1位比较行，0位比较列
    correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(label_input, 1))
    # 求均值
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    #初始化所有变量
    init = tf.initialize_all_variables()
    sess.run(init)

    saver = tf.train.Saver()
    saver.restore(sess,model_dir)

    # test_bottlenecks, test_labels = get_test_bottlenecks(test_id_list, labels)
    test_bottlenecks, test_labels = get_test_bottlenecks(id_list, labels)
    test_accuracy = sess.run(evaluation_step,feed_dict={bottleneck_input: test_bottlenecks, label_input: test_labels})
    print('最终的正确率为%.1f%%'%(test_accuracy*100))

"""
最终的正确率为99.4%
"""