"""
 1.在该py文件根目录下有：
                    ①files_of_faces文件夹：
                                          A.存储提供的所有的照片
                                          B.所有照片的名字（利用cmd指令实现
                    ②the_frame_of_faces文件夹：
                                          存储能框出脸部且框出脸部后的照片
 2.若人脸无法检测出来，则不显示
 3.显示人脸辨别率
"""

import cv2, os
from PIL import Image, ImageDraw
import time


def detectFaces(image_name):
    """
    利用opencv自带分类器haarcascade_frontalface_default.xml来检测人脸
    :param image_name: 图片名字
    :return: 检测到的人脸
    """
    img = cv2.imread(image_name)
    # 调出opencv中人脸分类器
    face_cascade = cv2.CascadeClassifier("trained_files/haarcascade_frontalface_default.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x+width, y+height))
    return result


def drawFaces(image_name):
    """
    用红线划出人脸部分
    :param image_name: 图片名字
    :return: 画出人脸并保存
    """
    global count
    faces = detectFaces(image_name)
    if True:
        img = Image.open(image_name)
        draw_instance = ImageDraw.Draw(img)
        for (x1, y1, x2, y2) in faces:
            draw_instance.rectangle((x1, y1, x2, y2), outline=255)
            img.save('the_frame_of_faces/'+image_name.split('/')[1])
            count += 1


if __name__ == '__main__':
    """
        以下是主程序
        变量说明：
                count   识别的人脸数   int
                sum     总人脸数       int
                faces   文件          file
                theline 文件某一行     string
                content 照片地址       string

    """
    count = 0
    sum = 0
    a = ''
    faces = open('files_of_faces/faces.txt', 'r')
    # 若不存在the_frame_of_faces文件，则创建
    if not os.path.exists('the_frame_of_faces'):
        os.mkdir('the_frame_of_faces')
    time_start = time.time()
    while True:
        theline = faces.readline()
        if len(theline) == 0 or len(theline) == 1:
            break
        sum += 1
        content = theline[:-1]
        drawFaces('files_of_faces/'+content)
    time_end = time.time()
    faces.close()
    print('Total picture number: '+str(sum))
    print('Recognized picture number: '+str(count))
    print('Accuracy: '+str(count/sum))
    print('Time cost:', time_end - time_start, 's')
