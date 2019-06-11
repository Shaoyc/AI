## 基于人脸识别的驾驶员状态预警系统

#### 开发背景

   交通事故统计结果显示，人为因素是交通事故的主要因素。引发交通事故及造成损失的驾驶员主要违规行为包括疏忽大意、超速行驶、措施不当、违规超车、不按规定让行这5个因素。其中驾驶员在行车过程中注意力分散、疲劳过度、睡眠不足、酒后驾车、身体健康状况欠佳等潜在的心理、生理性原因，造成反应迟缓而酿成交通事故。交通事故调查统计结果表明，如果驾驶员反应操作速度能够块0.5s，可以避免近60%的交通事故。
   
   本项目基于人脸识别的驾驶员情绪状态分析。通过判断出来的驾驶员的情绪状态给予驾驶员相应的预警信息。提示驾驶员是否处于疲劳驾驶和情绪异常。
   
   本项目基础功能：基于人脸识别对驾驶员提供预警服务。
   
   本项目可拓展功能：
   
   （1）【打车软件】基于驾驶员的情绪状态进行任务调度，当驾驶员处于平静以下时，进行短距离驾驶作业分配或建议其进行充分休息，避免因为长时间进行驾驶任务导致驾驶员疲劳驾驶和由情绪异常导致的驾驶风险，提高确保驾驶员行车的安全性。当驾驶员处于平静以上时，推荐长距离驾驶作业分配或者连续作业分配，以提高订单完成率和降低驾驶风险。
   
   （2）【自动驾驶】根据驾驶员的状态，建议其在不同自动驾驶级别中进行切换。避免由于驾驶员状态和自动驾驶级别之间的不同步产生安全问题。

#### 开发环境

Windows10 X64

python3.6

keras 2.1.6

Opencv 3.4.1

#### 思路

（1）通过OpenCV进行人脸识别

（2）使用keras进行表情识别

#### 预测效果
  
  预测结果：生气、厌恶、恐惧、开心、难过、惊喜、平静等。

#### 代码详解
```
#coding=utf-8
#状态预测

#导入相关资源
import cv2
from keras.models import load_model
import numpy as np
import chineseText
import datetime

#记录项目开始时间
startTime = datetime.datetime.now()
#导入模型文件
emotion_classifier = load_model(
    'model/simple_CNN.530-0.65.hdf5')
#记录项目结束时间
endTime = datetime.datetime.now()
#时间信息
print(endTime - startTime)

#建立状态标签
emotion_labels = {
    0: '生气',
    1: '厌恶',
    2: '恐惧',
    3: '开心',
    4: '难过',
    5: '惊喜',
    6: '平静'
}

#调用cv函数读入图片
img = cv2.imread("img/emotion.png")
#建立cv中的级联分类器
face_classifier = cv2.CascadeClassifier(
    "C:\Python36\Lib\site-packages\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml"
)
#调用cv中的cvtColor函数进行图片色彩空间转换（灰度化）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#调用模型中的detectMultiScale函数进行对尺度检测
faces = face_classifier.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
#face_classifier.detectMultiScale参数说明
#gray：转换的灰图
#scaleFactor：图像缩放比例，可理解为相机的X倍镜
#minNeighbors：对特征检测点周边多少有效点同时检测，这样可避免因选取的特征检测点太小而导致遗漏
#minSize：特征检测点的最小尺寸
color = (255, 0, 0)

#遍历像素
for (x, y, w, h) in faces:
    gray_face = gray[(y):(y + h), (x):(x + w)]
    #设置图片大小
    gray_face = cv2.resize(gray_face, (48, 48))
    #图像归一化处理
    gray_face = gray_face / 255.0
    #np.expand_dims:用于扩展数组的形状
    #np.expand_dims(a, axis=0)表示在0位置添加数据
    gray_face = np.expand_dims(gray_face, 0)
    #np.expand_dims(a, axis=1)表示在1位置添加数据
    gray_face = np.expand_dims(gray_face, -1)
    #np.argmax() 返回最大值索引号（模型输出结果）
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    #获取情绪标签
    emotion = emotion_labels[emotion_label_arg]
    #在图片上画出矩形
    cv2.rectangle(img, (x + 10, y + 10), (x + h - 10, y + w - 10),
                  (255, 255, 255), 2)
    #调用cv函数在图片上添加文字
    img = chineseText.cv2ImgAddText(img, emotion, x + h * 0.3, y, color, 20)
    
#cv2.imShow()函数可以在窗口中显示图像。
cv2.imshow("Image", img)
#waitKey()函数的功能是不断刷新图像，频率时间为delay，单位为ms。
cv2.waitKey(0)
cv2.destroyAllWindows()

```
