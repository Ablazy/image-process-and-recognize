"""
传统机器学习模型实现
SVM + HOG特征提取用于手写数字识别
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import pickle
import os
import sys
import time
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset import MNISTManager


class SVMHOGClassifier:
    """
    SVM + HOG特征分类器
    """
    
    def __init__(self, model_path=None):
        """
        初始化SVM+HOG分类器
        
        Args:
            model_path: 预训练模型路径
        """
        self.model = None
        self.model_path = model_path
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',
            'visualize': False
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model()
    
    def extract_hog_features(self, images):
        """
        提取HOG特征
        
        Args:
            images: 图像列表，每个图像为numpy数组
            
        Returns:
            np.array: HOG特征矩阵
        """
        features = []
        for img in images:
            # 确保图像是2D的
            if len(img.shape) > 2:
                img = img.squeeze()
            
            # 提取HOG特征
            fd = hog(
                img,
                **self.hog_params
            )
            features.append(fd)
        
        return np.array(features)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练SVM分类器
        
        Args:
            X_train: 训练图像
            y_train: 训练标签
            X_val: 验证图像（可选）
            y_val: 验证标签（可选）
            
        Returns:
            dict: 训练结果
        """
        print("提取HOG特征...")
        start_time = time.time()
        
        # 提取HOG特征
        X_train_features = self.extract_hog_features(X_train)
        feature_time = time.time() - start_time
        
        print(f"HOG特征提取完成，用时: {feature_time:.2f}秒")
        print(f"特征维度: {X_train_features.shape}")
        
        # 训练SVM
        print("训练SVM分类器...")
        start_time = time.time()
        
        self.model = SVC(
            kernel='rbf', 
            C=10, 
            gamma='scale', 
            probability=True,
            random_state=42
        )
        
        self.model.fit(X_train_features, y_train)
        train_time = time.time() - start_time
        
        print(f"SVM训练完成，用时: {train_time:.2f}秒")
        
        # 计算训练准确率
        y_train_pred = self.model.predict(X_train_features)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        results = {
            'train_accuracy': train_accuracy,
            'feature_extraction_time': feature_time,
            'training_time': train_time,
            'feature_dim': X_train_features.shape[1]
        }
        
        # 如果有验证集，计算验证准确率
        if X_val is not None and y_val is not None:
            val_results = self.evaluate(X_val, y_val, dataset_type='验证')
            results.update(val_results)
        
        print(f"训练准确率: {train_accuracy:.4f}")
        
        return results
    
    def predict(self, image):
        """
        预测单张图像
        
        Args:
            image: 输入图像
            
        Returns:
            tuple: (预测类别, 置信度)
        """
        if self.model is None:
            raise ValueError("模型未训练或加载")
        
        # 提取HOG特征
        features = self.extract_hog_features([image])
        
        # 预测
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        
        return prediction, confidence
    
    def predict_batch(self, images):
        """
        批量预测
        
        Args:
            images: 图像列表
            
        Returns:
            tuple: (预测类别列表, 置信度列表)
        """
        if self.model is None:
            raise ValueError("模型未训练或加载")
        
        # 提取HOG特征
        features = self.extract_hog_features(images)
        
        # 预测
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences
    
    def evaluate(self, X_test, y_test, dataset_type='测试'):
        """
        评估模型性能
        
        Args:
            X_test: 测试图像
            y_test: 测试标签
            dataset_type: 数据集类型名称
            
        Returns:
            dict: 评估结果
        """
        print(f"评估{dataset_type}集性能...")
        
        # 提取特征
        X_test_features = self.extract_hog_features(X_test)
        
        # 预测
        y_pred = self.model.predict(X_test_features)
        y_pred_proba = self.model.predict_proba(X_test_features)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        
        # 详细分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            f'{dataset_type.lower()}_accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        print(f"{dataset_type}准确率: {accuracy:.4f}")
        
        return results
    
    def save_model(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型和参数
        model_data = {
            'model': self.model,
            'hog_params': self.hog_params
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {path}")
    
    def load_model(self):
        """
        加载模型
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.hog_params = model_data.get('hog_params', self.hog_params)
        
        print(f"模型已从 {self.model_path} 加载")


def prepare_mnist_data(mnist_manager, sample_size=5000, test_size=0.2):
    """
    准备MNIST数据用于SVM训练
    
    Args:
        mnist_manager: MNIST数据集管理器
        sample_size: 采样数量
        test_size: 测试集比例
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"准备MNIST数据，采样数量: {sample_size}")
    
    # 从训练集中采样
    total_size = mnist_manager.get_dataset_size('train')
    indices = np.random.choice(total_size, min(sample_size, total_size), replace=False)
    
    images = []
    labels = []
    
    for idx in indices:
        image, label = mnist_manager.get_image_by_index(idx, 'train')
        images.append(image)
        labels.append(label)
    
    # 转换为numpy数组
    X = np.array(images)
    y = np.array(labels)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"类别分布: {np.bincount(y_train)}")
    
    return X_train, X_test, y_train, y_test


def main():
    """主训练函数"""
    print("开始SVM+HOG模型训练...")
    
    # 加载数据
    mnist_manager = MNISTManager()
    
    # 准备数据
    X_train, X_test, y_train, y_test = prepare_mnist_data(
        mnist_manager, sample_size=5000, test_size=0.2
    )
    
    # 创建并训练分类器
    classifier = SVMHOGClassifier()
    results = classifier.train(X_train, y_train, X_test, y_test)
    
    # 评估模型
    test_results = classifier.evaluate(X_test, y_test, '测试')
    results.update(test_results)
    
    # 打印详细结果
    print("\n训练结果:")
    for key, value in results.items():
        if key not in ['classification_report', 'confusion_matrix']:
            print(f"  {key}: {value}")
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"./models/svm_hog_{timestamp}.pkl"
    classifier.save_model(model_path)
    
    # 也保存一个默认路径的模型
    default_path = "./models/svm_hog.pkl"
    classifier.save_model(default_path)
    
    return classifier, results


if __name__ == "__main__":
    classifier, results = main()