"""
深度学习模型定义
MobileNetV2网络结构实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class InvertedResidual(nn.Module):
    """
    倒残差块 (Inverted Residual Block)
    MobileNetV2的核心组件
    """
    
    def __init__(self, inp, oup, stride, expand_ratio):
        """
        初始化倒残差块
        
        Args:
            inp: 输入通道数
            oup: 输出通道数
            stride: 步长
            expand_ratio: 扩展比例
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        layers = []
        if expand_ratio != 1:
            # 扩展层 (1x1卷积)
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        # 深度卷积 (3x3卷积，groups=hidden_dim)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 投影层 (1x1卷积)
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2网络实现
    适配MNIST数据集（单通道输入，10类输出）
    """
    
    def __init__(self, num_classes=10, width_mult=1.0, input_size=224):
        """
        初始化MobileNetV2
        
        Args:
            num_classes: 分类数量
            width_mult: 宽度倍数
            input_size: 输入图像尺寸
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        
        # MobileNetV2的配置
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # 构建第一层，适应单通道输入（MNIST是灰度图）
        self.features = [nn.Conv2d(1, input_channel, 3, 2, 1, bias=False)]
        self.features.extend([
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        ])
        
        # 构建倒残差块
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
        
        # 构建最后几层
        self.features.extend([
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ])
        
        self.features = nn.Sequential(*self.features)
        
        # 自适应平均池化，处理不同输入尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Linear(last_channel, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 1, height, width)
            
        Returns:
            torch.Tensor: 分类结果 (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def load_mobilenet_model(model_path='./models/mnist_mobilenet.pth', num_classes=10):
    """
    加载预训练的MobileNetV2模型
    
    Args:
        model_path: 模型文件路径
        num_classes: 分类数量
        
    Returns:
        MobileNetV2: 加载的模型
    """
    model = MobileNetV2(num_classes=num_classes)
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"模型已从 {model_path} 加载")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("使用未训练的模型")
    else:
        print(f"模型文件 {model_path} 不存在，使用未训练的模型")
    
    model.eval()
    return model


def test_model():
    """测试模型功能"""
    # 创建模型
    model = MobileNetV2(num_classes=10)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试前向传播
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 224, 224)
    
    with torch.no_grad():
        output = model(input_tensor)
        print(f"输入形状: {input_tensor.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
    
    # 测试不同输入尺寸
    test_sizes = [(28, 28), (56, 56), (112, 112), (224, 224)]
    for h, w in test_sizes:
        test_input = torch.randn(1, 1, h, w)
        with torch.no_grad():
            model.eval()  # 设置为评估模式
            test_output = model(test_input)
            print(f"输入尺寸 {h}x{w} -> 输出形状: {test_output.shape}")


if __name__ == "__main__":
    test_model()