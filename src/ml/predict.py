"""
统一预测接口
整合MobileNetV2和SVM+HOG两种模型的预测功能
"""

import torch
import numpy as np
import cv2
import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models import MobileNetV2, load_mobilenet_model
from ml.traditional_ml import SVMHOGClassifier


class ImagePredictor:
    """
    图像预测器
    支持MobileNetV2和SVM+HOG两种模型
    """
    
    def __init__(self):
        """初始化预测器"""
        self.mobilenet_model = None
        self.svm_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mobilenet_loaded = False
        self.svm_loaded = False
        
    def load_models(self, mobilenet_path='./models/mnist_mobilenet.pth', 
                   svm_path='./models/svm_hog.pkl'):
        """
        加载所有模型
        
        Args:
            mobilenet_path: MobileNetV2模型路径
            svm_path: SVM模型路径
        """
        # 加载MobileNetV2模型
        try:
            self.mobilenet_model = load_mobilenet_model(mobilenet_path)
            self.mobilenet_model.to(self.device)
            self.mobilenet_loaded = True
            print("MobileNetV2模型加载成功")
        except Exception as e:
            print(f"加载MobileNetV2模型失败: {e}")
            self.mobilenet_loaded = False
            
        # 加载SVM模型
        try:
            self.svm_model = SVMHOGClassifier(svm_path)
            self.svm_loaded = True
            print("SVM+HOG模型加载成功")
        except Exception as e:
            print(f"加载SVM模型失败: {e}")
            self.svm_loaded = False
        
        return self.mobilenet_loaded or self.svm_loaded
    
    def preprocess_for_mobilenet(self, image):
        """
        为MobileNetV2预处理图像
        
        Args:
            image: 输入图像 (numpy数组)
            
        Returns:
            torch.Tensor: 预处理后的张量
        """
        # 确保是单通道灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 调整大小为224x224
        # image_resized = cv2.resize(image, (224, 224))
        image_resized = image.copy()
        # 归一化
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - 0.1307) / 0.3081
        
        # 转换为tensor并添加batch和channel维度
        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict_with_mobilenet(self, image):
        """
        使用MobileNetV2进行预测
        
        Args:
            image: 输入图像
            
        Returns:
            tuple: (预测类别, 置信度)
        """
        if not self.mobilenet_loaded or self.mobilenet_model is None:
            raise ValueError("MobileNetV2模型未加载")
        
        # 预处理
        input_tensor = self.preprocess_for_mobilenet(image)
        
        # 预测
        with torch.no_grad():
            self.mobilenet_model.eval()
            outputs = self.mobilenet_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
        return prediction.item(), confidence.item()
    
    def predict_with_svm(self, image):
        """
        使用SVM+HOG进行预测
        
        Args:
            image: 输入图像
            
        Returns:
            tuple: (预测类别, 置信度)
        """
        if not self.svm_loaded or self.svm_model is None:
            raise ValueError("SVM模型未加载")
        
        # 确保是单通道灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 调整大小为28x28（MNIST原始尺寸）
        image_resized = cv2.resize(image, (28, 28))
        
        return self.svm_model.predict(image_resized)
    
    def predict(self, image, model_type='mobilenet'):
        """
        统一预测接口
        
        Args:
            image: 输入图像
            model_type: 模型类型 ('mobilenet' 或 'svm')
            
        Returns:
            tuple: (预测类别, 置信度)
        """
        if model_type.lower() == 'mobilenet':
            return self.predict_with_mobilenet(image)
        elif model_type.lower() == 'svm':
            return self.predict_with_svm(image)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def predict_ensemble(self, image):
        """
        集成预测（结合两个模型的结果）
        
        Args:
            image: 输入图像
            
        Returns:
            dict: 预测结果详情
        """
        results = {}
        
        # MobileNetV2预测
        if self.mobilenet_loaded:
            try:
                mobilenet_pred, mobilenet_conf = self.predict_with_mobilenet(image)
                results['mobilenet'] = {
                    'prediction': mobilenet_pred,
                    'confidence': mobilenet_conf
                }
            except Exception as e:
                print(f"MobileNetV2预测失败: {e}")
                results['mobilenet'] = {'error': str(e)}
        
        # SVM预测
        if self.svm_loaded:
            try:
                svm_pred, svm_conf = self.predict_with_svm(image)
                results['svm'] = {
                    'prediction': svm_pred,
                    'confidence': svm_conf
                }
            except Exception as e:
                print(f"SVM预测失败: {e}")
                results['svm'] = {'error': str(e)}
        
        # 如果两个模型都成功，计算集成结果
        if 'mobilenet' in results and 'svm' in results and \
           'prediction' in results['mobilenet'] and 'prediction' in results['svm']:
            
            # 简单的加权平均（基于置信度）
            mobilenet_weight = results['mobilenet']['confidence']
            svm_weight = results['svm']['confidence']
            total_weight = mobilenet_weight + svm_weight
            
            if results['mobilenet']['prediction'] == results['svm']['prediction']:
                # 两个模型预测一致
                final_prediction = results['mobilenet']['prediction']
                final_confidence = max(mobilenet_weight, svm_weight)
            else:
                # 两个模型预测不一致，选择置信度更高的
                if mobilenet_weight > svm_weight:
                    final_prediction = results['mobilenet']['prediction']
                    final_confidence = mobilenet_weight
                else:
                    final_prediction = results['svm']['prediction']
                    final_confidence = svm_weight
            
            results['ensemble'] = {
                'prediction': final_prediction,
                'confidence': final_confidence,
                'agreement': results['mobilenet']['prediction'] == results['svm']['prediction']
            }
        
        return results
    
    def get_model_status(self):
        """
        获取模型加载状态
        
        Returns:
            dict: 模型状态信息
        """
        return {
            'mobilenet_loaded': self.mobilenet_loaded,
            'svm_loaded': self.svm_loaded,
            'device': str(self.device),
            'available_models': []
        }
    
    def batch_predict(self, images, model_type='mobilenet'):
        """
        批量预测
        
        Args:
            images: 图像列表
            model_type: 模型类型
            
        Returns:
            list: 预测结果列表
        """
        results = []
        for image in images:
            try:
                pred, conf = self.predict(image, model_type)
                results.append({'prediction': pred, 'confidence': conf, 'error': None})
            except Exception as e:
                results.append({'prediction': None, 'confidence': None, 'error': str(e)})
        
        return results


def test_predictor():
    """测试预测器功能"""
    print("测试图像预测器...")
    
    # 创建预测器
    predictor = ImagePredictor()
    
    # 检查模型状态
    status = predictor.get_model_status()
    print("模型状态:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # 尝试加载模型
    models_loaded = predictor.load_models()
    if not models_loaded:
        print("警告: 没有成功加载任何模型")
        return
    
    # 创建测试图像
    test_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    
    # 测试预测
    try:
        if status['mobilenet_loaded']:
            pred, conf = predictor.predict(test_image, 'mobilenet')
            print(f"MobileNetV2预测: {pred}, 置信度: {conf:.4f}")
        
        if status['svm_loaded']:
            pred, conf = predictor.predict(test_image, 'svm')
            print(f"SVM预测: {pred}, 置信度: {conf:.4f}")
        
        # 测试集成预测
        ensemble_results = predictor.predict_ensemble(test_image)
        print("集成预测结果:")
        for model, result in ensemble_results.items():
            print(f"  {model}: {result}")
            
    except Exception as e:
        print(f"预测测试失败: {e}")


if __name__ == "__main__":
    test_predictor()