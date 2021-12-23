from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import matplotlib.image as mpimg
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from skimage import feature as skft
from sklearn.svm import SVC

def lbp(imagename):
    """
    lbp
    :param imagename: 一张人脸部分的照片的名字（string）
    :return: 这张照片的特征向量
    """
    image = cv2.imread(imagename)
    # image:灰度照片
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imagepart=[]
    block=5
    # 读取输入图片的高度与宽度,也即是各有多少像素点数#
    height, width = image.shape[0], image.shape[1]
    column = width // block
    # 计算切割后，每个部分应该有多少行多少列,也即是各有多少像素点数
    row = height//block
    hist = np.array([])
    for i in range(block*block):
        lbp1 = local_binary_pattern(image[row*(i//block):row*((i//block)+1), column*(i % block):column*((i % block)+1)], 8, 1,  'default')
        hist1, _ = np.histogram(lbp1, normed=True, bins=256, range=(0, 256))
        # 特征向量get,是一个array含256*9=2304元素
        hist = np.concatenate((hist, hist1))
    return hist

def train_and_test_and_score(train_label,train_histogram,test_label,test_histogram):
    """
    训练并测试
    :param train_label:
    :param train_histogram:
    :param test_label: 含3600个1或-1（对应3600张图片的标签）
    :param test_histogram: 含有3600个histogram（对应3600张图片的特征向量）
    :return: score
    """
    svc = SVC(kernel='linear', degree=2, gamma=1, coef0=0)
    # 训练
    svc.fit(train_histogram,train_label)
    # 测试，predict含有400个1或-1
    predict_result=svc.predict(test_histogram)
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(predict_result)):
        if test_label[i] == 1:
            if predict_result[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if predict_result[i] == 1:
                FP += 1
            else:
                TN += 1
    print('F1:', 2*TP/(2*TP+FP+FN))
    print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN', FN)


if __name__ == '__main__':
    for time in range(10):
        train_hist = lbp("train_and_test/0/file0010.jpg")
        train_label = np.array([1])

        for abc in range(time):
            face= open('train_and_test/'+str(abc)+'/faces.txt', 'r')
            while True:
                theline = face.readline()
                if len(theline) == 0 or len(theline) == 1:
                    break
                # 照片名字
                content = theline[:-1].split(' ')[0]
                label = int(theline[:-1].split(' ')[1])
                num = int(content[4:8])//10
                train_label = np.vstack((label, train_label))
                train_hist = np.vstack((lbp("train_and_test/"+str(abc)+"/"+content), train_hist))
            face.close()

        for abc in range(time+1, 10):
            face= open('train_and_test/'+str(abc)+'/faces.txt', 'r')
            while True:
                theline = face.readline()
                if len(theline) == 0 or len(theline) == 1:
                    break
                # 照片名字
                content = theline[:-1].split(' ')[0]
                label = int(theline[:-1].split(' ')[1])
                num = int(content[4:8])//10
                train_label = np.vstack((label, train_label))
                train_hist = np.vstack((lbp("train_and_test/"+str(abc)+"/"+content), train_hist))
            face.close()
        face = open('train_and_test/'+str(time)+'/faces.txt', 'r')
        while True:

            theline = face.readline()
            if len(theline) == 0 or len(theline) == 1:
                break
            content = theline[:-1].split(' ')[0]  # 照片名字
            label = int(theline[:-1].split(' ')[1])
            num = int(content[4:8]) // 10
            if num == 0 or num == 1:
                test_hist = lbp("train_and_test/" + str(time) + "/" + content)
                test_label = np.array([1])
            else:
                test_label = np.vstack((label, test_label))
                test_hist = np.vstack((lbp("train_and_test/"+str(time)+"/" + content), test_hist))
        print(time)
        train_and_test_and_score(train_label, train_hist, test_label, test_hist)
        face.close()
