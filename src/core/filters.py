"""
滤波模块
实现各种图像滤波功能
"""

import cv2
import numpy as np


def apply_gaussian_blur(image, kernel_size, sigma_x=0, sigma_y=None):
    """
    应用高斯滤波
    
    Args:
        image: 输入图像 (numpy数组)
        kernel_size: 核大小（必须为奇数）
        sigma_x: x方向标准差
        sigma_y: y方向标准差（如果为None，则等于sigma_x）
        
    Returns:
        numpy.ndarray: 滤波后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if kernel_size <= 0:
        raise ValueError("核大小必须为正数")
    
    # 确保核大小为奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    if sigma_y is None:
        sigma_y = sigma_x
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x, sigma_y)


def apply_mean_blur(image, kernel_size):
    """
    应用均值滤波
    
    Args:
        image: 输入图像 (numpy数组)
        kernel_size: 核大小
        
    Returns:
        numpy.ndarray: 滤波后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if kernel_size <= 0:
        raise ValueError("核大小必须为正数")
    
    return cv2.blur(image, (kernel_size, kernel_size))


def apply_median_blur(image, kernel_size):
    """
    应用中值滤波
    
    Args:
        image: 输入图像 (numpy数组)
        kernel_size: 核大小（必须为奇数）
        
    Returns:
        numpy.ndarray: 滤波后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if kernel_size <= 0:
        raise ValueError("核大小必须为正数")
    
    # 确保核大小为奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.medianBlur(image, kernel_size)


def apply_bilateral_filter(image, d, sigma_color, sigma_space):
    """
    应用双边滤波（保边滤波）
    
    Args:
        image: 输入图像 (numpy数组)
        d: 滤波时每个像素邻域的直径
        sigma_color: 颜色空间标准差
        sigma_space: 坐标空间标准差
        
    Returns:
        numpy.ndarray: 滤波后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if d <= 0:
        raise ValueError("邻域直径必须为正数")
    
    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("标准差必须为正数")
    
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_box_filter(image, kernel_size):
    """
    应用方框滤波
    
    Args:
        image: 输入图像 (numpy数组)
        kernel_size: 核大小
        
    Returns:
        numpy.ndarray: 滤波后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if kernel_size <= 0:
        raise ValueError("核大小必须为正数")
    
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)


def apply_sharpen_filter(image, strength=1.0):
    """
    应用锐化滤波
    
    Args:
        image: 输入图像 (numpy数组)
        strength: 锐化强度
        
    Returns:
        numpy.ndarray: 锐化后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if strength < 0:
        raise ValueError("锐化强度不能为负数")
    
    # 锐化核
    kernel = np.array([[-1, -1, -1],
                      [-1, 8 + strength, -1],
                      [-1, -1, -1]]) / (8 + strength)
    
    return cv2.filter2D(image, -1, kernel)


def apply_emboss_filter(image):
    """
    应用浮雕滤波
    
    Args:
        image: 输入图像 (numpy数组)
        
    Returns:
        numpy.ndarray: 浮雕效果图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    # 浮雕核
    kernel = np.array([[-2, -1, 0],
                      [-1, 1, 1],
                      [0, 1, 2]])
    
    embossed = cv2.filter2D(image, -1, kernel)
    
    # 归一化到0-255范围
    embossed = cv2.normalize(embossed, None, 0, 255, cv2.NORM_MINMAX)
    
    return embossed.astype(np.uint8)


def apply_edge_enhance_filter(image):
    """
    应用边缘增强滤波
    
    Args:
        image: 输入图像 (numpy数组)
        
    Returns:
        numpy.ndarray: 边缘增强后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    # 边缘检测核
    kernel = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
    
    return cv2.filter2D(image, -1, kernel)


def apply_motion_blur(image, size, angle):
    """
    应用运动模糊
    
    Args:
        image: 输入图像 (numpy数组)
        size: 模糊核大小
        angle: 运动角度（度）
        
    Returns:
        numpy.ndarray: 运动模糊后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if size <= 0:
        raise ValueError("模糊核大小必须为正数")
    
    # 创建运动模糊核
    kernel = np.zeros((size, size))
    
    # 计算运动方向
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # 在核中画线
    center = size // 2
    for i in range(size):
        x = int(center + (i - center) * cos_angle)
        y = int(center + (i - center) * sin_angle)
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
    
    # 归一化核
    kernel = kernel / np.sum(kernel)
    
    return cv2.filter2D(image, -1, kernel)


def get_filter_info():
    """
    获取可用滤波器信息
    
    Returns:
        dict: 滤波器信息
    """
    filters_info = {
        'gaussian': {
            'name': '高斯滤波',
            'description': '使用高斯函数进行平滑，有效去除高斯噪声',
            'parameters': ['kernel_size', 'sigma_x', 'sigma_y']
        },
        'mean': {
            'name': '均值滤波',
            'description': '使用均值进行平滑，简单快速的滤波方法',
            'parameters': ['kernel_size']
        },
        'median': {
            'name': '中值滤波',
            'description': '使用中值进行平滑，有效去除椒盐噪声',
            'parameters': ['kernel_size']
        },
        'bilateral': {
            'name': '双边滤波',
            'description': '保边滤波，在平滑的同时保持边缘清晰',
            'parameters': ['d', 'sigma_color', 'sigma_space']
        },
        'sharpen': {
            'name': '锐化滤波',
            'description': '增强图像边缘和细节',
            'parameters': ['strength']
        },
        'emboss': {
            'name': '浮雕滤波',
            'description': '创建浮雕效果',
            'parameters': []
        },
        'edge_enhance': {
            'name': '边缘增强',
            'description': '增强图像边缘',
            'parameters': []
        },
        'motion_blur': {
            'name': '运动模糊',
            'description': '模拟运动造成的模糊效果',
            'parameters': ['size', 'angle']
        }
    }
    
    return filters_info


def test_filters():
    """测试滤波功能"""
    print("测试滤波功能...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    print(f"原始图像形状: {test_image.shape}")
    
    # 测试各种滤波器
    try:
        # 高斯滤波
        gaussian = apply_gaussian_blur(test_image, 5)
        print(f"高斯滤波后图像形状: {gaussian.shape}")
    except Exception as e:
        print(f"高斯滤波测试失败: {e}")
    
    try:
        # 均值滤波
        mean = apply_mean_blur(test_image, 5)
        print(f"均值滤波后图像形状: {mean.shape}")
    except Exception as e:
        print(f"均值滤波测试失败: {e}")
    
    try:
        # 中值滤波
        median = apply_median_blur(test_image, 5)
        print(f"中值滤波后图像形状: {median.shape}")
    except Exception as e:
        print(f"中值滤波测试失败: {e}")
    
    try:
        # 双边滤波
        bilateral = apply_bilateral_filter(test_image, 9, 75, 75)
        print(f"双边滤波后图像形状: {bilateral.shape}")
    except Exception as e:
        print(f"双边滤波测试失败: {e}")
    
    try:
        # 锐化滤波
        sharpen = apply_sharpen_filter(test_image, 1.0)
        print(f"锐化滤波后图像形状: {sharpen.shape}")
    except Exception as e:
        print(f"锐化滤波测试失败: {e}")
    
    try:
        # 浮雕滤波
        emboss = apply_emboss_filter(test_image)
        print(f"浮雕滤波后图像形状: {emboss.shape}")
    except Exception as e:
        print(f"浮雕滤波测试失败: {e}")
    
    try:
        # 运动模糊
        motion = apply_motion_blur(test_image, 15, 45)
        print(f"运动模糊后图像形状: {motion.shape}")
    except Exception as e:
        print(f"运动模糊测试失败: {e}")
    
    # 显示滤波器信息
    print("\n可用滤波器:")
    filters_info = get_filter_info()
    for key, info in filters_info.items():
        print(f"  {key}: {info['name']} - {info['description']}")
    
    print("滤波功能测试完成")


if __name__ == "__main__":
    test_filters()