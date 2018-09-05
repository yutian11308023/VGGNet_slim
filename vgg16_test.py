#coding:utf-8


#读待测图，给出可视化结果

import numpy as np
import tensorflow as tf
import vgg16
import utils
from Nclasses import labels
import matplotlib.pyplot as plt


img_path = raw_input('Please input the correct path of image:')
img = utils.load_image(img_path) #读图
batch = img.reshape((1, 224, 224, 3))

fig=plt.figure(u"Top-5 预测结果")
with tf.Session() as sess:
    images = tf.placeholder("float32", [1, 224, 224, 3])
    vgg = vgg16.Vgg16()
    vgg.build_model(images)
    
    prob = sess.run(vgg.prob, feed_dict={images:batch}) #预测结果
    top5 = np.argsort(prob[0])[-1:-6:-1] #最高5个概率的索引号
    values = []
    bar_label = []
    for n, i in enumerate(top5):

        values.append(prob[0][i])
        bar_label.append(labels[i])
        
    ax = fig.add_subplot(111)
    ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
    ax.set_ylabel(u'probit')
    ax.set_title(u'Top-5')
    for a,b in zip(range(len(values)), values):
	ax.text(a, b+0.0005, '%.2f%%' % (b*100), ha='center', va = 'bottom', fontsize=7)
    plt.show()

    
