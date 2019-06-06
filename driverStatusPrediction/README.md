# 基于人脸识别的驾驶员状态预警系统详解

## 开发背景

   交通事故统计结果显示，人为因素是交通事故的主要因素。引发交通事故及造成损失的驾驶员主要违规行为包括疏忽大意、超速行驶、措施不当、违规超车、不按规定让行这5个因素。其中驾驶员在行车过程中注意力分散、疲劳过度、睡眠不足、酒后驾车、身体健康状况欠佳等潜在的心理、生理性原因，造成反应迟缓而酿成交通事故。交通事故调查统计结果表明，如果驾驶员反应操作速度能够块0.5s，可以避免近60%的交通事故。
   
   本项目基于人脸识别的驾驶员情绪状态分析。通过判断出来的驾驶员的情绪状态给予驾驶员相应的预警信息。提示驾驶员是否处于疲劳驾驶和情绪异常。
   
   本项目基础功能：基于人脸识别对驾驶员提供预警服务。
   
   本项目可拓展功能：
   
   （1）【打车软件】基于驾驶员的情绪状态进行任务调度，当驾驶员处于平静以下时，进行短距离驾驶作业分配或建议其进行充分休息，避免因为长时间进行驾驶任务导致驾驶员疲劳驾驶和由情绪异常导致的驾驶风险，提高确保驾驶员行车的安全性。当驾驶员处于平静以上时，推荐长距离驾驶作业分配或者连续作业分配，以提高订单完成率和降低驾驶风险。
   
   （2）【自动驾驶】根据驾驶员的状态，建议其在不同自动驾驶级别中进行切换。避免由于驾驶员状态和自动驾驶级别之间的不同步产生安全问题。

## 开发环境

Windows10 X64

python3.6

keras 2.1.6

Opencv 3.4.1

## 思路

（1）通过OpenCV进行人脸识别

（2）使用keras进行表情识别

## 预测效果
  
  预测结果：生气、厌恶、恐惧、开心、难过、惊喜、平静等。

## 代码详解
```
#coding=utf-8
#表情识别

import cv2
from keras.models import load_model
import numpy as np
import chineseText
import datetime

startTime = datetime.datetime.now()
emotion_classifier = load_model(
    'model/simple_CNN.530-0.65.hdf5')
endTime = datetime.datetime.now()
print(endTime - startTime)

emotion_labels = {
    0: '生气',
    1: '厌恶',
    2: '恐惧',
    3: '开心',
    4: '难过',
    5: '惊喜',
    6: '平静'
}

img = cv2.imread("img/emotion.png")
face_classifier = cv2.CascadeClassifier(
    "C:\Python36\Lib\site-packages\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml"
)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
color = (255, 0, 0)

for (x, y, w, h) in faces:
    gray_face = gray[(y):(y + h), (x):(x + w)]
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face / 255.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion = emotion_labels[emotion_label_arg]
    cv2.rectangle(img, (x + 10, y + 10), (x + h - 10, y + w - 10),
                  (255, 255, 255), 2)
    img = chineseText.cv2ImgAddText(img, emotion, x + h * 0.3, y, color, 20)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

```
