## 危险物品检测

#### 研究背景及意义

随着计算机领域和人工智能技术的不断发展,计算机对图像处理方法不断完善，推动图像识别技术向前发展，基于图像识别的方法应用到不同的领域。尤其是在公共场所环境下，实现对危险物品及时和准确的目标识别具有很重要的理论价值和实际意义。

图像识别技术是计算机对获取到的图像进行处理和识别，以识别目标物体的技术。图像识别过程包括计算机对图像的预处理、图像的分割、重要特征提取和图像特征匹配等。在公共场所使用基于深度学习的图像识别技术，对摄像头获取的图片进行处理。通过深度学习模型对危险物品的种类和可信度进行及时的判断，能够达到快速准确的识别出公共场所的危险物品和潜在的危险物品，从而避免安全隐患的发生。

#### 算法流程

S1从公共场所摄像头循环获取图像信息；

S2将图像输入到神经网络当中；

S3获取输出结果可信度最高的5件物体；

S4判断图像中是否存在危险品；

S5若不存在危险品则 goto S1；

S6若存在危险品则 goto S7；

S7判断危险品的可信度是否高于设置阈值；

S8若危险品可信度高于阈值 goto S9；

S9进行警报并通知相关人员进行处理；

S10若危险品可信度不高于阈值 goto S11；

S11获取该摄像头的下一张图像信息进行再次判断；

S12重复上述过程。

#### 运行流程

危险物品检测系统是在Ubuntu环境下运行，使用Python编写的基于Tensorflow 后端的高层神经网络API--Keras，利用 Keras 所提供的5个卷积神经网络，分别为 VGG16、VGG19、ResNet50、Inception V3、Xception，这些神经网络在 ImageNet 数据集中已经进行了预训练，该数据集训练大约 120 万个训练图像，另有 50,000 个图像用于验证，100,000 个图像用于测试。这1000个图像类别代表我们在日常生活中遇到的对象类，例如狗，猫，各种家用物品，车辆类型等等。然后，使用 Keras 创建了一个Python识别程序，可以从磁盘中加载这些预先训练好的网络架构并对输入的图像进行分类。最后看到输入图像的识别效果，以下是 Ubuntu 上对危险物品检测系统的终端说明：

#### Python 3.6安装

Ubuntu 默认安装了 Python2.7 和 3.5，在此我们选用 python3.6.0 作为主要编程环境,打开终端，输入命令

    sudo apt-get update
    
    sudo apt-get install python3.6 
    
按照终端提示命令输入 y 确认安装。

    python –version 
    
此时可以看到显示 python 版本为 3.6.0 即安装成功。

安装 pip：输入命令

    sudo apt-get install python3-pip
    
更新 pip：输入命令

    sudo pip3 install --upgrade pip 

#### TensorFlow安装

安装 python 对应版本的 pip 和依赖包：

    sudo apt-get install python3-pip python3-dev
    
更新 pip 版本：

    sudo pip3 install --upgrade pip
    
安装 Tensorflow(CPU 版本):

    sudo pip3 install tensorflow

#### Keras 框架的安装
 
系统软件更新

    sudo apt-get update
    
    sudo apt-get upgrade
    
安装 python 基础开发包

    sudo apt install -y python-dev python-pip python-nose gcc g++ git gfortranvim
    
安装运算加速库

    sudo apt install -y libopenblas-dev liblapack-dev libatlas-base-dev
    
#### Keras 开发包安装：

    sudo pip install -U --pre pip setuptools wheel
    
    sudo pip install -U --pre numpy scipy matplotlib scikit-learn scikit-image 
    
    sudo pip install -U --pre tensorflow
    
    sudo pip install -U --pre keras
    
安装完成后在终端输入

    python3
    
    import tensorflow 
    
    import keras

#### OpenCV 视觉库的安装

终端下 pip 方式安装 OpenCV 

    pip install cv2
    
    pip apt-get install cv2
    
    pip install opencv-python
    

#### Ubuntu 终端运行

到项目文件夹中打开终端，输入命令

    python deeplearning.py --image images/1_test.jpeg --model xception

即可开始对输入的摄像头图片进行识别，其中 images/1_text.jpeg表示的是输入的识别摄像头图片的路径，xception 为本次训练需要的模型，当上一步中环境配置完毕后，再输入命令,若系统中没有该模型，命令输入完毕后系统会自动下载该模型，下载完毕后继续完成训练，因此第一次识别时速度较慢，下载完模型之后的识别速度正常。

#### 分析

从模型输出结果的可信度方面来说，对于安保工作中一般的危险物品：当危险物品可信度大于30%的时候，应该认为可能存在危险物品。当危险物品可信度大于50%时，就应该引起注意。进行必要的安全检查。当危险物品可信度大于90%时，就认为该图像中存在危险物品，应该及时处理危险物品，尽可能降低危险物品可能带来的严重影响。从模型输出结果的物品类型方面来说，对于不同危险物品的种类不同可信度需要采取不同处理方法。当危险物品输出的种类属于高度危险物品时，较低可信度时就需要迅速核实处理，当危险品输出的种类属于低度危险物品时，可以进行正常处理。

通过对公共场所摄像头采集到的图像进行分析，得到图片中危险物品出现的类型和物品的可信度。通过两个性能指标（平均正确率和速度）进行测试实验。试验结果表明，本文所提出的检测方法能够在保证较高精度的基础上，初步达到实时检测。本文提出的基于深度学习的危险物品检测系统可以辅助安检人员检测一些较为常见的危险物品，进而提高安检效率和节约人力物力。

基于深度学习的危险物品检测系统对于改变外形的危险物品不能够有良好的识别，检测在某些方面存在局限性。仅可以作为辅助安保方式使用。
