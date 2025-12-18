"""
MobileNetV2模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset import MNISTManager
from ml.models import MobileNetV2


def train_model(model, train_loader, test_loader, device, epochs=10, learning_rate=0.001):
    """
    训练模型
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 训练设备
        epochs: 训练轮数
        learning_rate: 学习率
        
    Returns:
        dict: 训练历史记录
    """
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_time': []
    }
    
    print(f"开始训练，使用设备: {device}")
    print(f"训练轮数: {epochs}, 学习率: {learning_rate}")
    print("-" * 60)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # 打印进度
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%')
        
        # 更新学习率
        scheduler.step()
        
        # 计算训练指标
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()
        
        # 计算测试指标
        test_loss_avg = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total
        
        # 记录历史
        epoch_time = time.time() - epoch_start_time
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss_avg)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        
        # 打印epoch结果
        print(f'Epoch [{epoch+1}/{epochs}] 完成:')
        print(f'  训练 - 损失: {train_loss_avg:.4f}, 准确率: {train_acc:.2f}%')
        print(f'  测试 - 损失: {test_loss_avg:.4f}, 准确率: {test_acc:.2f}%')
        print(f'  用时: {epoch_time:.2f}秒')
        print("-" * 60)
    
    return history


def save_model(model, save_path, history=None):
    """
    保存模型和训练历史
    
    Args:
        model: 要保存的模型
        save_path: 保存路径
        history: 训练历史记录
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存模型状态字典
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")
    
    # 保存训练历史
    if history:
        history_path = save_path.replace('.pth', '_history.pth')
        torch.save(history, history_path)
        print(f"训练历史已保存到: {history_path}")


def main():
    """主训练函数"""
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据准备
    print("准备数据...")
    mnist_manager = MNISTManager()
    
    # 创建数据加载器
    train_loader = mnist_manager.get_dataloader(batch_size=64, shuffle=True, dataset_type='train')
    test_loader = mnist_manager.get_dataloader(batch_size=64, shuffle=False, dataset_type='test')
    
    print(f"训练数据批次数: {len(train_loader)}")
    print(f"测试数据批次数: {len(test_loader)}")
    
    # 创建模型
    print("创建模型...")
    model = MobileNetV2(num_classes=10).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {total_params:,}")
    
    # 训练参数
    epochs = 10
    learning_rate = 0.001
    
    # 开始训练
    start_time = time.time()
    history = train_model(model, train_loader, test_loader, device, epochs, learning_rate)
    total_time = time.time() - start_time
    
    print(f"训练完成！总用时: {total_time/60:.2f}分钟")
    print(f"最终测试准确率: {history['test_acc'][-1]:.2f}%")
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"./models/mnist_mobilenet_{timestamp}.pth"
    save_model(model, model_path, history)
    
    # 也保存一个默认路径的模型
    default_path = "./models/mnist_mobilenet.pth"
    save_model(model, default_path, history)
    
    return model, history


if __name__ == "__main__":
    model, history = main()