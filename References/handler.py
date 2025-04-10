import torch
import torchvision.transforms as tf
import numpy as np
import base64
import io
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from ultralytics import YOLO  # 导入YOLO类
from ultralytics.models.yolo.detect import DetectionPredictor  # 导入YOLOv11检测类
from ultralytics.data import build_dataloader, build_yolo_dataset  # 导入数据集构建函数
from ultralytics.utils import yaml_load


class YOLOv11Handler(BaseHandler):
    """
    YOLOv11 Custom Model Handler for TorchServe
    """

    def __init__(self):
        super(YOLOv11Handler, self).__init__()
        self.model = None
        self.device = None
        self.initialized = False
        
        # 定义类别索引到中文名称的映射
        self.index_to_name = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog"
        }


    def initialize(self, context):
        """
        初始化模型及相关参数
        """
        # 读取模型的配置和权重路径
        properties = context.system_properties

        # 确保模型文件路径正确
        model_dir = properties.get("model_dir")
        model_path = f"/data/AI_train/ultralytics/ultralytics/yolo11n.pt"
        data_path = f"/data/AI_train/ultralytics/ultralytics/cfg/datasets/coco8.yaml"  # 更新数据集文件路径

        # 设置设备为GPU或CPU
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        )

        try:
            # 加载YOLOv11模型并指定数据集文件路径
            self.model = YOLO(model=model_path, task="detect")
            if self.model is None:
                raise ValueError("Model not loaded correctly")

            # 设置数据集路径
            self.model.overrides['data'] = data_path

            # 读取数据集配置
            self.data = yaml_load(data_path)

            # 将模型移动到指定设备并设为评估模式
            self.model.to(self.device).predict()
            self.initialized = True
            print(f"Model loaded successfully from {model_path} with data from {data_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}")

    def preprocess(self, data):
        """ 
        预处理输入数据
        """
        image_list = []
        for row in data:
            image = row.get("data") or row.get("body")

            if isinstance(image, str):
                # 处理 base64 编码的图像
                image = base64.b64decode(image)
                image = Image.open(io.BytesIO(image))

            elif isinstance(image, (bytearray, bytes)):
                # 处理 bytes 类型图像
                image = Image.open(io.BytesIO(image))

            # 应用预处理步骤 (YOLO模型默认输入为640x640的图像)
            transform = tf.Compose([
                tf.Resize((640, 640)),  # 调整大小
                tf.ToTensor()  # 转换为张量
            ])

            image = transform(image).unsqueeze(0)  # 添加 batch 维度
            image_list.append(image)

        images_tensor = torch.cat(image_list).to(self.device)  # 将图像批量合并并传输到设备
        return images_tensor

    def inference(self, data):
        """
        使用模型进行推理
        """
        with torch.no_grad():
            # YOLOv11 推理
            predictions = self.model(data)
        return predictions

    def postprocess(self, inference_output):
        """
        后处理推理结果
        """
        results = []
        for output in inference_output:
            detections = []
            for det in output.boxes:
                # 获取检测框坐标、信心度和类别信息
                box = det.xyxy.cpu().numpy().tolist()  # [x1, y1, x2, y2]
                conf = det.conf.cpu().numpy().tolist()  # 信心度
                cls = det.cls.cpu().numpy().tolist()  # 类别标签

                # 检查 cls 是否为列表，并获取类别索引
                if isinstance(cls, list):
                    cls = cls[0] if cls else -1  # 防止为空的情况

                # 根据类别索引获取英文名称
                class_name = self.index_to_name.get(int(cls), "未知类别")  # 获取对应英文名称，找不到时返回"未知类别"

                detections.append({
                    "box": box,
                    "confidence": conf,
                    "class": class_name  # 使用英文类别名称
                })
            results.append(detections)
        return results

    def handle(self, data, context):
        """
        处理请求数据
        """
        # 预处理
        images = self.preprocess(data)

        # 推理
        outputs = self.inference(images)

        # 后处理
        result = self.postprocess(outputs)

        return result
