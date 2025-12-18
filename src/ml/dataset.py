"""
MNIST数据集管理模块
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import random


class MNISTManager:
    """MNIST数据集管理器"""
    
    def __init__(self, data_dir='./data'):
        """
        初始化MNIST数据集管理器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = None
        self.test_dataset = None
        self.load_dataset()
    
    def load_dataset(self):
        """下载并加载MNIST数据集"""
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        try:
            # 加载训练集
            self.dataset = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transform
            )
            
            # 加载测试集
            self.test_dataset = datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.transform
            )
            
            print(f"MNIST数据集已加载")
            print(f"训练集大小: {len(self.dataset)}")
            print(f"测试集大小: {len(self.test_dataset)}")
            
        except Exception as e:
            print(f"加载MNIST数据集时出错: {e}")
            raise
    
    def get_image_by_index(self, index, dataset_type='train'):
        """
        通过索引获取单张图片和标签
        
        Args:
            index: 图像索引
            dataset_type: 数据集类型 ('train' 或 'test')
            
        Returns:
            tuple: (image_array, label) 或 (None, None) 如果索引无效
        """
        dataset = self.dataset if dataset_type == 'train' else self.test_dataset
        
        if dataset is None or index < 0 or index >= len(dataset):
            return None, None
        
        image, label = dataset[index]
        
        # 转换为numpy数组并反归一化
        image_np = image.squeeze().numpy()
        image_np = (image_np * 0.3081 + 0.1307) * 255
        image_np = image_np.astype(np.uint8)
        
        return image_np, label
    
    def get_random_image(self, dataset_type='train'):
        """
        获取随机图片
        
        Args:
            dataset_type: 数据集类型 ('train' 或 'test')
            
        Returns:
            tuple: (image_array, label)
        """
        dataset = self.dataset if dataset_type == 'train' else self.test_dataset
        index = random.randint(0, len(dataset) - 1)
        return self.get_image_by_index(index, dataset_type)
    
    def get_dataset_size(self, dataset_type='train'):
        """
        获取数据集大小
        
        Args:
            dataset_type: 数据集类型 ('train' 或 'test')
            
        Returns:
            int: 数据集大小
        """
        dataset = self.dataset if dataset_type == 'train' else self.test_dataset
        return len(dataset) if dataset else 0
    
    def get_dataloader(self, batch_size=64, shuffle=True, dataset_type='train'):
        """
        获取数据加载器
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据
            dataset_type: 数据集类型 ('train' 或 'test')
            
        Returns:
            DataLoader: 数据加载器
        """
        dataset = self.dataset if dataset_type == 'train' else self.test_dataset
        
        if dataset is None:
            raise ValueError(f"数据集 {dataset_type} 未加载")
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0  # Windows上设置为0避免多进程问题
        )
    
    def get_class_names(self):
        """
        获取类别名称列表
        
        Returns:
            list: 类别名称列表
        """
        return [str(i) for i in range(10)]
    
    def get_statistics(self):
        """
        获取数据集统计信息
        
        Returns:
            dict: 统计信息
        """
        train_size = self.get_dataset_size('train')
        test_size = self.get_dataset_size('test')
        
        return {
            'train_size': train_size,
            'test_size': test_size,
            'total_size': train_size + test_size,
            'num_classes': 10,
            'image_size': (28, 28),
            'data_dir': self.data_dir
        }


# 测试代码
if __name__ == "__main__":
    # 创建数据集管理器
    mnist_manager = MNISTManager()
    
    # 获取统计信息
    stats = mnist_manager.get_statistics()
    print("数据集统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试获取图像
    image, label = mnist_manager.get_image_by_index(0)
    if image is not None:
        print(f"\n获取到第一张图片，标签: {label}")
        print(f"图片形状: {image.shape}")
        print(f"图片数据类型: {image.dtype}")
        print(f"像素值范围: [{image.min()}, {image.max()}]")
    
    # 测试数据加载器
    train_loader = mnist_manager.get_dataloader(batch_size=32, shuffle=True)
    print(f"\n训练数据加载器创建成功，批次数量: {len(train_loader)}")