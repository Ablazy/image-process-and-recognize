"""
形态学操作模块
实现膨胀、腐蚀、开运算、闭运算等形态学操作
"""

import cv2
import numpy as np


def apply_morphology(image, op_type, kernel_size, iterations=1, kernel_shape=cv2.MORPH_RECT):
    """
    应用形态学操作
    
    Args:
        image: 输入图像 (numpy数组)，应为二值图像
        op_type: 操作类型 ('dilate', 'erode', 'open', 'close', 'gradient', 'tophat', 'blackhat')
        kernel_size: 核大小
        iterations: 迭代次数
        kernel_shape: 核形状 (cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS)
        
    Returns:
        numpy.ndarray: 形态学操作后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if kernel_size <= 0:
        raise ValueError("核大小必须为正数")
    
    if iterations <= 0:
        raise ValueError("迭代次数必须为正数")
    
    # 确保核大小为奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 创建核
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    
    # 根据操作类型应用相应的形态学操作
    op_type = op_type.lower()
    
    if op_type == 'dilate':
        result = cv2.dilate(image, kernel, iterations=iterations)
    elif op_type == 'erode':
        result = cv2.erode(image, kernel, iterations=iterations)
    elif op_type == 'open':
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op_type == 'close':
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif op_type == 'gradient':
        result = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    elif op_type == 'tophat':
        result = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    elif op_type == 'blackhat':
        result = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
    else:
        raise ValueError(f"不支持的操作类型: {op_type}")
    
    return result


def dilate_image(image, kernel_size, iterations=1, kernel_shape=cv2.MORPH_RECT):
    """
    膨胀操作
    
    Args:
        image: 输入图像
        kernel_size: 核大小
        iterations: 迭代次数
        kernel_shape: 核形状
        
    Returns:
        numpy.ndarray: 膨胀后的图像
    """
    return apply_morphology(image, 'dilate', kernel_size, iterations, kernel_shape)


def erode_image(image, kernel_size, iterations=1, kernel_shape=cv2.MORPH_RECT):
    """
    腐蚀操作
    
    Args:
        image: 输入图像
        kernel_size: 核大小
        iterations: 迭代次数
        kernel_shape: 核形状
        
    Returns:
        numpy.ndarray: 腐蚀后的图像
    """
    return apply_morphology(image, 'erode', kernel_size, iterations, kernel_shape)


def open_image(image, kernel_size, iterations=1, kernel_shape=cv2.MORPH_RECT):
    """
    开运算（先腐蚀后膨胀）
    
    Args:
        image: 输入图像
        kernel_size: 核大小
        iterations: 迭代次数
        kernel_shape: 核形状
        
    Returns:
        numpy.ndarray: 开运算后的图像
    """
    return apply_morphology(image, 'open', kernel_size, iterations, kernel_shape)


def close_image(image, kernel_size, iterations=1, kernel_shape=cv2.MORPH_RECT):
    """
    闭运算（先膨胀后腐蚀）
    
    Args:
        image: 输入图像
        kernel_size: 核大小
        iterations: 迭代次数
        kernel_shape: 核形状
        
    Returns:
        numpy.ndarray: 闭运算后的图像
    """
    return apply_morphology(image, 'close', kernel_size, iterations, kernel_shape)


def gradient_image(image, kernel_size, iterations=1, kernel_shape=cv2.MORPH_RECT):
    """
    形态学梯度（膨胀图减腐蚀图）
    
    Args:
        image: 输入图像
        kernel_size: 核大小
        iterations: 迭代次数
        kernel_shape: 核形状
        
    Returns:
        numpy.ndarray: 形态学梯度图像
    """
    return apply_morphology(image, 'gradient', kernel_size, iterations, kernel_shape)


def tophat_image(image, kernel_size, iterations=1, kernel_shape=cv2.MORPH_RECT):
    """
    顶帽变换（原图减去开运算结果）
    
    Args:
        image: 输入图像
        kernel_size: 核大小
        iterations: 迭代次数
        kernel_shape: 核形状
        
    Returns:
        numpy.ndarray: 顶帽变换后的图像
    """
    return apply_morphology(image, 'tophat', kernel_size, iterations, kernel_shape)


def blackhat_image(image, kernel_size, iterations=1, kernel_shape=cv2.MORPH_RECT):
    """
    黑帽变换（闭运算减去原图）
    
    Args:
        image: 输入图像
        kernel_size: 核大小
        iterations: 迭代次数
        kernel_shape: 核形状
        
    Returns:
        numpy.ndarray: 黑帽变换后的图像
    """
    return apply_morphology(image, 'blackhat', kernel_size, iterations, kernel_shape)


def adaptive_threshold(image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                   threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
    """
    自适应阈值处理
    
    Args:
        image: 输入灰度图像
        max_value: 阈值化后的最大值
        adaptive_method: 自适应方法
        threshold_type: 阈值类型
        block_size: 邻域块大小
        C: 常数
        
    Returns:
        numpy.ndarray: 阈值化后的二值图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if len(image.shape) != 2:
        raise ValueError("输入图像必须是灰度图像")
    
    if block_size % 2 == 0:
        block_size += 1
    
    return cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, C)


def otsu_threshold(image, max_value=255):
    """
    大津法阈值处理
    
    Args:
        image: 输入灰度图像
        max_value: 阈值化后的最大值
        
    Returns:
        tuple: (二值图像, 阈值)
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if len(image.shape) != 2:
        raise ValueError("输入图像必须是灰度图像")
    
    _, binary_image, threshold = cv2.threshold(image, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_image, threshold


def remove_small_objects(binary_image, min_size=50):
    """
    移除小对象
    
    Args:
        binary_image: 输入二值图像
        min_size: 最小对象大小（像素数）
        
    Returns:
        numpy.ndarray: 移除小对象后的二值图像
    """
    if binary_image is None:
        raise ValueError("输入图像不能为None")
    
    # 找到连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)
    
    # 创建输出图像
    output = np.zeros_like(binary_image)
    
    # 保留大于最小尺寸的对象
    for i in range(1, num_labels):  # 跳过背景标签0
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255
    
    return output.astype(np.uint8)


def fill_holes(binary_image):
    """
    填充孔洞
    
    Args:
        binary_image: 输入二值图像
        
    Returns:
        numpy.ndarray: 填充孔洞后的二值图像
    """
    if binary_image is None:
        raise ValueError("输入图像不能为None")
    
    # 使用形态学操作填充孔洞
    kernel = np.ones((3, 3), np.uint8)
    
    # 先膨胀再腐蚀，填充小孔洞
    filled = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    return filled


def get_morphology_info():
    """
    获取形态学操作信息
    
    Returns:
        dict: 形态学操作信息
    """
    morphology_info = {
        'dilate': {
            'name': '膨胀',
            'description': '扩大白色区域，连接断开的区域',
            'parameters': ['kernel_size', 'iterations', 'kernel_shape']
        },
        'erode': {
            'name': '腐蚀',
            'description': '缩小白色区域，分离连接的区域',
            'parameters': ['kernel_size', 'iterations', 'kernel_shape']
        },
        'open': {
            'name': '开运算',
            'description': '先腐蚀后膨胀，去除小噪声',
            'parameters': ['kernel_size', 'iterations', 'kernel_shape']
        },
        'close': {
            'name': '闭运算',
            'description': '先膨胀后腐蚀，填充小孔洞',
            'parameters': ['kernel_size', 'iterations', 'kernel_shape']
        },
        'gradient': {
            'name': '形态学梯度',
            'description': '膨胀图减腐蚀图，提取边缘',
            'parameters': ['kernel_size', 'iterations', 'kernel_shape']
        },
        'tophat': {
            'name': '顶帽变换',
            'description': '原图减去开运算结果，提取亮细节',
            'parameters': ['kernel_size', 'iterations', 'kernel_shape']
        },
        'blackhat': {
            'name': '黑帽变换',
            'description': '闭运算减去原图，提取暗细节',
            'parameters': ['kernel_size', 'iterations', 'kernel_shape']
        }
    }
    
    return morphology_info


def test_morphology():
    """测试形态学操作功能"""
    print("测试形态学操作功能...")
    
    # 创建测试图像（带噪声的圆形）
    test_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(test_image, (50, 50), 30, 255, -1)
    
    # 添加一些噪声
    noise = np.random.randint(0, 2, (100, 100)) * 255
    test_image = np.maximum(test_image, noise.astype(np.uint8))
    
    print(f"原始图像形状: {test_image.shape}")
    
    # 测试各种形态学操作
    try:
        # 膨胀
        dilated = dilate_image(test_image, 5)
        print(f"膨胀后图像形状: {dilated.shape}")
    except Exception as e:
        print(f"膨胀测试失败: {e}")
    
    try:
        # 腐蚀
        eroded = erode_image(test_image, 5)
        print(f"腐蚀后图像形状: {eroded.shape}")
    except Exception as e:
        print(f"腐蚀测试失败: {e}")
    
    try:
        # 开运算
        opened = open_image(test_image, 5)
        print(f"开运算后图像形状: {opened.shape}")
    except Exception as e:
        print(f"开运算测试失败: {e}")
    
    try:
        # 闭运算
        closed = close_image(test_image, 5)
        print(f"闭运算后图像形状: {closed.shape}")
    except Exception as e:
        print(f"闭运算测试失败: {e}")
    
    try:
        # 形态学梯度
        gradient = gradient_image(test_image, 5)
        print(f"形态学梯度后图像形状: {gradient.shape}")
    except Exception as e:
        print(f"形态学梯度测试失败: {e}")
    
    try:
        # 自适应阈值
        adaptive = adaptive_threshold(test_image, block_size=11)
        print(f"自适应阈值后图像形状: {adaptive.shape}")
    except Exception as e:
        print(f"自适应阈值测试失败: {e}")
    
    try:
        # 大津法阈值
        binary, threshold = otsu_threshold(test_image)
        print(f"大津法阈值后图像形状: {binary.shape}, 阈值: {threshold}")
    except Exception as e:
        print(f"大津法阈值测试失败: {e}")
    
    # 显示形态学操作信息
    print("\n可用形态学操作:")
    morphology_info = get_morphology_info()
    for key, info in morphology_info.items():
        print(f"  {key}: {info['name']} - {info['description']}")
    
    print("形态学操作功能测试完成")


if __name__ == "__main__":
    test_morphology()