import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

overview_df = pd.read_csv(r'D:\Users\datasets\CT Medical Image\overview.csv')
overview_df.columns = ['idx']+list(overview_df.columns[1:])
overview_df['Contrast'] = overview_df['Contrast'].map(lambda x:1 if x else 0)#1：Contrast,0:No Contrast

with np.load(r'D:\Users\datasets\CT Medical Image\full_archive.npz') as im_data:
    full_image_dict = dict(zip(im_data['idx'],im_data['image']))

Error_List = []
for x in list(full_image_dict.keys()):
    try:
        x_path = os.path.join(r'D:\Users\datasets\CT Medical Image\image_jpg',str(x)+'.jpg')
        print(x_path)
        plt.imsave(x_path,full_image_dict[x])
    except:
        Error_List.append(x)

print(Error_List)