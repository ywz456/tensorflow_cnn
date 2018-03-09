import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tensorflow.python.platform import gfile

#迁移学习前的预处理
image_dir = r'D:\Users\datasets\CT Medical Image\image_jpg'
MODEL_DIR = r'D:\Users\datasets\CT Medical Image\inception_dec_2015'
image_tensor = r'D:\Users\datasets\CT Medical Image\image_tensor'
MODEL_FILE = r'tensorflow_inception_graph.pb'

#获取图片地址字典，字典的键为图片ID，值为图片地址
def get_image_path_dict(image_dir):
    image_path_dict = {}
    image_path_lists = os.listdir(image_dir)
    for image_path_list in image_path_lists:
        image_path = os.path.join(image_dir,image_path_list)
        image_basename = os.path.basename(image_path).split('.')[0]
        image_path_dict[image_basename] = image_path
    return image_path_dict

#获取图片的特征向量字典，键为图片ID，值为图片的特征向量
def get_image_tensor_dict(sess,image_path_dict,bottleneck_tensor,jpeg_data_tensor):
    image_tensor_path_dict = {}
    for image_id in list(image_path_dict.keys()):
        image_data = gfile.FastGFile(image_path_dict[image_id], 'rb').read()
        bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        image_tensor_path = os.path.join(image_tensor, image_id + '.txt')
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(image_tensor_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
        image_tensor_path_dict[image_id] = image_tensor_path
    return image_tensor_path_dict

with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:  # 阅读器上下文,读取文件
    graph_def = tf.GraphDef()  # 生成图
    graph_def.ParseFromString(f.read())  # 图加载模型

# bottleneck_tensor,jpeg_data_tensor类型为：Tensor("import/pool_3/_reshape:0", shape=(1, 2048), dtype=float32) Tensor("import/DecodeJpeg/contents:0", shape=(), dtype=string)
bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=['pool_3/_reshape:0','DecodeJpeg/contents:0'])

with tf.Session() as sess:
    image_path_dict = get_image_path_dict(image_dir)
    image_tensor_dict = get_image_tensor_dict(sess,image_path_dict,bottleneck_tensor,jpeg_data_tensor)
