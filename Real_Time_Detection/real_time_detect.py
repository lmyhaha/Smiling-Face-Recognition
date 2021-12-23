import cv2
import numpy as np


def detect(img):
    """
    检测人脸和笑脸
    :param img: 待检测图片
    :return: 画出人脸和笑脸区域
    """
    # 先进行灰度转换
    if img.ndim >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸：1.利用opencv自带的分类器haarcascade_frontalface_default.xml
    #         2.利用我们自己建立的分类器face_detect.xml(在网上找代码，利用cmd实现）
    facePath = "database/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(facePath)

    # 检测笑脸:1.利用opencv自带的分类器haarcascade_smile.xml
    #         2.利用我们建立的分类器smile_detect.xml
    smilePath = "database/haarcascade_smile.xml"
    smileCascade = cv2.CascadeClassifier(smilePath)

    #检测人脸
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 原图像， 内容，左上角的坐标，字体，大小，颜色，厚度
    cv2.rectangle(img, (225, 150), (375, 300), (255, 0, 0), 2)
    cv2.putText(img, 'Best Detecting Area', (190, 320), 4, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, 'Author : Mao&Yang&Zhao', (20, 20), 4, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    # 画出每一个人脸，提取出人脸所在区域
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # 对人脸进行笑脸检测
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.16,
            minNeighbors=35,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # 框出上扬的嘴角并对笑脸打上Smile标签
        for (x2, y2, w2, h2) in smile:
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
            cv2.putText(img, 'Smiling', (x, y - 7), 3, 1.2, (0, 255, 0), 1, cv2.LINE_AA)
    return img


cap = cv2.VideoCapture(0)

# 检测摄像头抓捕照片的宽度以及高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)

# 视频编码方式
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

# 以下是主程序
while 1:
    ret, frame = cap.read()
    if ret:
        frame = detect(frame)
        out.write(frame)
        cv2.imshow('Real-Time Smile Detect', frame)
        if cv2.waitKey(1) & 0xFF <= 126:
            break
cap.release()
cv2.destroyAllWindows()

