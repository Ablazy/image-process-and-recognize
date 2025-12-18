"""
基础图像处理模块
实现旋转、缩放、裁剪等基本操作
"""

import cv2
import numpy as np


def rotate_image(image, angle):
    """
    旋转图像
    
    Args:
        image: 输入图像 (numpy数组)
        angle: 旋转角度（度），正值为顺时针，负值为逆时针
        
    Returns:
        numpy.ndarray: 旋转后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后的新边界
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # 调整旋转矩阵以考虑平移
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # 执行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    return rotated_image


def scale_image(image, scale_factor, interpolation=cv2.INTER_LINEAR):
    """
    缩放图像
    
    Args:
        image: 输入图像 (numpy数组)
        scale_factor: 缩放因子，>1为放大，<1为缩小
        interpolation: 插值方法
        
    Returns:
        numpy.ndarray: 缩放后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if scale_factor <= 0:
        raise ValueError("缩放因子必须为正数")
    
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # 选择合适的插值方法
    if scale_factor < 1:
        interpolation = cv2.INTER_AREA  # 缩小时使用INTER_AREA避免锯齿
    
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    return scaled_image


def crop_image(image, x, y, w, h):
    """
    裁剪图像
    
    Args:
        image: 输入图像 (numpy数组)
        x, y: 裁剪区域左上角坐标
        w, h: 裁剪区域宽度和高度
        
    Returns:
        numpy.ndarray: 裁剪后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    height, width = image.shape[:2]
    
    # 边界检查和修正
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    
    # 执行裁剪
    cropped_image = image[y:y+h, x:x+w]
    
    return cropped_image


def flip_image(image, flip_code):
    """
    翻转图像
    
    Args:
        image: 输入图像 (numpy数组)
        flip_code: 翻转代码
                  0: 垂直翻转
                  1: 水平翻转
                  -1: 同时水平和垂直翻转
                  
    Returns:
        numpy.ndarray: 翻转后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if flip_code not in [0, 1, -1]:
        raise ValueError("翻转代码必须是 0, 1, 或 -1")
    
    return cv2.flip(image, flip_code)


def resize_image(image, width, height, interpolation=cv2.INTER_LINEAR):
    """
    调整图像尺寸
    
    Args:
        image: 输入图像 (numpy数组)
        width: 目标宽度
        height: 目标高度
        interpolation: 插值方法
        
    Returns:
        numpy.ndarray: 调整尺寸后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if width <= 0 or height <= 0:
        raise ValueError("宽度和高度必须为正数")
    
    resized_image = cv2.resize(image, (width, height), interpolation=interpolation)
    
    return resized_image


def pad_image(image, top, bottom, left, right, border_type=cv2.BORDER_CONSTANT, value=0):
    """
    为图像添加边框
    
    Args:
        image: 输入图像 (numpy数组)
        top, bottom, left, right: 各个方向的边框大小
        border_type: 边框类型
        value: 边框值（对于BORDER_CONSTANT）
        
    Returns:
        numpy.ndarray: 添加边框后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if any(size < 0 for size in [top, bottom, left, right]):
        raise ValueError("边框大小不能为负数")
    
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=value)
    
    return padded_image


def translate_image(image, dx, dy):
    """
    平移图像
    
    Args:
        image: 输入图像 (numpy数组)
        dx: x方向平移距离（正值为向右）
        dy: y方向平移距离（正值为向下）
        
    Returns:
        numpy.ndarray: 平移后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    height, width = image.shape[:2]
    
    # 创建平移矩阵
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # 执行平移
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
    
    return translated_image


def get_image_info(image):
    """
    获取图像基本信息
    
    Args:
        image: 输入图像 (numpy数组)
        
    Returns:
        dict: 图像信息
    """
    if image is None:
        return {"error": "输入图像不能为None"}
    
    info = {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "size": image.size,
        "channels": 1 if len(image.shape) == 2 else image.shape[2],
        "height": image.shape[0],
        "width": image.shape[1]
    }
    
    # 添加像素值范围信息
    info["min_value"] = image.min()
    info["max_value"] = image.max()
    info["mean_value"] = image.mean()
    info["std_value"] = image.std()
    
    return info


def validate_image(image):
    """
    验证图像是否有效
    
    Args:
        image: 输入图像 (numpy数组)
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if image is None:
        return False, "图像为None"
    
    if not isinstance(image, np.ndarray):
        return False, "图像不是numpy数组"
    
    if image.size == 0:
        return False, "图像为空"
    
    if len(image.shape) not in [2, 3]:
        return False, "图像维度不正确，应为2D或3D"
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False, "图像通道数不正确，应为1、3或4"
    
    return True, ""


def test_basic_ops():
    """测试基础图像处理功能"""
    print("测试基础图像处理功能...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    print(f"原始图像形状: {test_image.shape}")
    
    # 测试旋转
    try:
        rotated = rotate_image(test_image, 45)
        print(f"旋转45度后图像形状: {rotated.shape}")
    except Exception as e:
        print(f"旋转测试失败: {e}")
    
    # 测试缩放
    try:
        scaled = scale_image(test_image, 1.5)
        print(f"缩放1.5倍后图像形状: {scaled.shape}")
    except Exception as e:
        print(f"缩放测试失败: {e}")
    
    # 测试裁剪
    try:
        cropped = crop_image(test_image, 10, 10, 50, 50)
        print(f"裁剪后图像形状: {cropped.shape}")
    except Exception as e:
        print(f"裁剪测试失败: {e}")
    
    # 测试翻转
    try:
        flipped_h = flip_image(test_image, 1)
        flipped_v = flip_image(test_image, 0)
        print(f"水平翻转后图像形状: {flipped_h.shape}")
        print(f"垂直翻转后图像形状: {flipped_v.shape}")
    except Exception as e:
        print(f"翻转测试失败: {e}")
    
    # 测试图像信息
    try:
        info = get_image_info(test_image)
        print("图像信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"图像信息获取失败: {e}")
    
    # 测试图像验证
    try:
        is_valid, error_msg = validate_image(test_image)
        print(f"图像验证结果: {'有效' if is_valid else '无效'}")
        if error_msg:
            print(f"错误信息: {error_msg}")
    except Exception as e:
        print(f"图像验证失败: {e}")
    
    print("基础图像处理功能测试完成")


if __name__ == "__main__":
    test_basic_ops()