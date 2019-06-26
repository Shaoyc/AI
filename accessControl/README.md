## 门禁系统

#### 开发背景

近几年,作为生物特征识别技术中非常重要的一项,人脸识别技术具有独特的发展优势,逐渐成为人工智能以及模式识别的一项焦点。人脸识别技术具有方便、直接以及友好的优势,因此在很多领域得到了广泛的应用。在家庭安防或者公寓安防中人脸识别的门禁系统有良好的表现。安全人员相对固定的情况下，基于人脸识别的门禁系统能够很好的进行区分。

本项目基础功能：基于人脸识别的门禁系统。

本项目可拓展功能：

（1）【打卡系统】可以实现基于人脸识别的打卡系统，能够有效的避免代替打卡等一系列都打卡产生的问题，能够极大的提高打卡签到速度。

（2）【自动驾驶】通过驾驶者面部识别进行身份安全确认。

#### 开发环境

Windows10 X64

python3.6

keras 2.1.6

Opencv 3.4.1

#### 思路

（1）通过对视频中的人脸进行识别

（2）通过活体检测等方式防止照片攻击

#### 预测效果

预测结果：能够准确识别出安全人员并且能够防止照片攻击等方式的恶意攻击。

#### 人脸识别代码详解

```
# -*- coding:utf-8 -*-
import cv2
def discern(img):
    #灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #调用级联分类器
    cap = cv2.CascadeClassifier(
        "C:\Python36\Lib\site-packages\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml"
    )
    #调用多尺度检测函数
    faceRects = cap.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
     #循环
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)  # 框出人脸
    #展示检测结果
    cv2.imshow("Image", img)
# 获取摄像头0表示第一个摄像头
cap = cv2.VideoCapture(0)
while (1):  # 逐帧显示
    ret, img = cap.read()
    # cv2.imshow("Image", img)
    discern(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 释放窗口资源

```
