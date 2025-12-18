"""
特征提取模块
实现边缘检测、直方图计算等特征提取功能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def detect_edges(image, low_threshold, high_threshold, aperture_size=3, l2_gradient=False):
    """
    Canny边缘检测
    
    Args:
        image: 输入图像 (numpy数组)
        low_threshold: 低阈值
        high_threshold: 高阈值
        aperture_size: Sobel算子孔径大小
        l2_gradient: 是否使用L2梯度幅度
        
    Returns:
        numpy.ndarray: 边缘图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if low_threshold >= high_threshold:
        raise ValueError("低阈值必须小于高阈值")
    
    # 确保是单通道图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return cv2.Canny(image, low_threshold, high_threshold, apertureSize=aperture_size, L2gradient=l2_gradient)


def sobel_edge_detection(image, ksize=3, scale=1, delta=0):
    """
    Sobel边缘检测
    
    Args:
        image: 输入图像
        ksize: Sobel核大小
        scale: 缩放因子
        delta: 添加到结果的增量
        
    Returns:
        tuple: (x方向梯度, y方向梯度, 梯度幅值)
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    # 确保是单通道图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算x和y方向梯度
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)
    
    # 计算梯度幅值
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 转换为uint8
    grad_x = np.uint8(np.absolute(grad_x))
    grad_y = np.uint8(np.absolute(grad_y))
    magnitude = np.uint8(magnitude)
    
    return grad_x, grad_y, magnitude


def laplacian_edge_detection(image, ksize=3, scale=1, delta=0):
    """
    拉普拉斯边缘检测
    
    Args:
        image: 输入图像
        ksize: 拉普拉斯核大小
        scale: 缩放因子
        delta: 添加到结果的增量
        
    Returns:
        numpy.ndarray: 拉普拉斯边缘图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    # 确保是单通道图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)
    
    # 转换为uint8
    laplacian = np.uint8(np.absolute(laplacian))
    
    return laplacian


def calc_histogram(image, bins=256, mask=None):
    """
    计算灰度直方图
    
    Args:
        image: 输入图像
        bins: 直方图箱数
        mask: 掩码
        
    Returns:
        tuple: (直方图数据, 箱边界)
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    # 确保是单通道图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    hist, bins = np.histogram(image.flatten(), bins, [0, 256])
    
    return hist, bins


def calc_color_histogram(image, bins=256):
    """
    计算彩色直方图
    
    Args:
        image: 输入彩色图像
        bins: 直方图箱数
        
    Returns:
        dict: 各通道的直方图数据
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if len(image.shape) != 3:
        raise ValueError("输入图像必须是彩色图像")
    
    # 分离BGR通道
    b, g, r = cv2.split(image)
    
    # 计算各通道直方图
    hist_b, _ = np.histogram(b.flatten(), bins, [0, 256])
    hist_g, _ = np.histogram(g.flatten(), bins, [0, 256])
    hist_r, _ = np.histogram(r.flatten(), bins, [0, 256])
    
    return {
        'blue': hist_b,
        'green': hist_g,
        'red': hist_r,
        'bins': np.arange(bins)
    }


def calc_histogram_equalization(image):
    """
    直方图均衡化
    
    Args:
        image: 输入图像
        
    Returns:
        numpy.ndarray: 均衡化后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    if len(image.shape) == 3:
        # 彩色图像，转换到YUV空间，只对Y通道均衡化
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # 灰度图像
        equalized = cv2.equalizeHist(image)
    
    return equalized


def calc_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    对比度限制自适应直方图均衡化
    
    Args:
        image: 输入图像
        clip_limit: 对比度限制
        tile_grid_size: 网格大小
        
    Returns:
        numpy.ndarray: CLAHE处理后的图像
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(image.shape) == 3:
        # 彩色图像，转换到LAB空间，只对L通道处理
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # 灰度图像
        clahe_img = clahe.apply(image)
    
    return clahe_img


def hough_lines_detection(image, rho=1, theta=np.pi/180, threshold=100, 
                      min_line_length=50, max_line_gap=10):
    """
    霍夫直线检测
    
    Args:
        image: 输入边缘图像
        rho: 距离分辨率
        theta: 角度分辨率
        threshold: 阈值
        min_line_length: 最小线长
        max_line_gap: 最大线段间隙
        
    Returns:
        numpy.ndarray: 检测到的直线
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    lines = cv2.HoughLinesP(image, rho, theta, threshold, 
                          minLineLength=min_line_length, 
                          maxLineGap=max_line_gap)
    
    return lines


def hough_circles_detection(image, method=cv2.HOUGH_GRADIENT, dp=1, min_dist=100,
                         param1=50, param2=30, min_radius=0, max_radius=0):
    """
    霍夫圆检测
    
    Args:
        image: 输入图像
        method: 检测方法
        dp: 累加器分辨率
        min_dist: 检测到的圆的最小距离
        param1: Canny边缘检测高阈值
        param2: 累加器阈值
        min_radius: 最小半径
        max_radius: 最大半径
        
    Returns:
        numpy.ndarray: 检测到的圆
    """
    if image is None:
        raise ValueError("输入图像不能为None")
    
    # 确保是单通道图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(image, method, dp, min_dist,
                             param1=param1, param2=param2,
                             minRadius=min_radius, maxRadius=max_radius)
    
    return circles


def create_histogram_plot(hist_data, title="Histogram", xlabel="Value", ylabel="Frequency"):
    """
    创建直方图图像
    
    Args:
        hist_data: 直方图数据
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        
    Returns:
        numpy.ndarray: 直方图图像
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(hist_data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    # 将matplotlib图形转换为numpy数组
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # 获取RGB图像
    buf = canvas.buffer_rgba()
    hist_image = np.asarray(buf)
    hist_image = cv2.cvtColor(hist_image, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    
    return hist_image


def create_color_histogram_plot(color_hist_data):
    """
    创建彩色直方图图像
    
    Args:
        color_hist_data: 彩色直方图数据
        
    Returns:
        numpy.ndarray: 彩色直方图图像
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bins = color_hist_data['bins']
    ax.plot(bins, color_hist_data['blue'], 'b-', label='Blue', alpha=0.7)
    ax.plot(bins, color_hist_data['green'], 'g-', label='Green', alpha=0.7)
    ax.plot(bins, color_hist_data['red'], 'r-', label='Red', alpha=0.7)
    
    ax.set_title("Color Histogram")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 将matplotlib图形转换为numpy数组
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # 获取RGB图像
    buf = canvas.buffer_rgba()
    hist_image = np.asarray(buf)
    hist_image = cv2.cvtColor(hist_image, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    
    return hist_image


def get_features_info():
    """
    获取特征提取功能信息
    
    Returns:
        dict: 特征提取功能信息
    """
    features_info = {
        'canny': {
            'name': 'Canny边缘检测',
            'description': '使用Canny算子进行边缘检测',
            'parameters': ['low_threshold', 'high_threshold', 'aperture_size']
        },
        'sobel': {
            'name': 'Sobel边缘检测',
            'description': '使用Sobel算子计算梯度',
            'parameters': ['ksize', 'scale', 'delta']
        },
        'laplacian': {
            'name': '拉普拉斯边缘检测',
            'description': '使用拉普拉斯算子检测边缘',
            'parameters': ['ksize', 'scale', 'delta']
        },
        'histogram': {
            'name': '直方图计算',
            'description': '计算图像的灰度或彩色直方图',
            'parameters': ['bins']
        },
        'histogram_equalization': {
            'name': '直方图均衡化',
            'description': '增强图像对比度',
            'parameters': []
        },
        'clahe': {
            'name': '自适应直方图均衡化',
            'description': '对比度限制自适应直方图均衡化',
            'parameters': ['clip_limit', 'tile_grid_size']
        },
        'hough_lines': {
            'name': '霍夫直线检测',
            'description': '检测图像中的直线',
            'parameters': ['rho', 'theta', 'threshold', 'min_line_length', 'max_line_gap']
        },
        'hough_circles': {
            'name': '霍夫圆检测',
            'description': '检测图像中的圆形',
            'parameters': ['dp', 'min_dist', 'param1', 'param2', 'min_radius', 'max_radius']
        }
    }
    
    return features_info


def test_features():
    """测试特征提取功能"""
    print("测试特征提取功能...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    print(f"原始图像形状: {test_image.shape}")
    
    # 测试边缘检测
    try:
        # Canny边缘检测
        edges = detect_edges(test_image, 50, 150)
        print(f"Canny边缘检测后图像形状: {edges.shape}")
    except Exception as e:
        print(f"Canny边缘检测测试失败: {e}")
    
    try:
        # Sobel边缘检测
        grad_x, grad_y, magnitude = sobel_edge_detection(test_image)
        print(f"Sobel边缘检测后图像形状: {grad_x.shape}")
    except Exception as e:
        print(f"Sobel边缘检测测试失败: {e}")
    
    try:
        # 拉普拉斯边缘检测
        laplacian = laplacian_edge_detection(test_image)
        print(f"拉普拉斯边缘检测后图像形状: {laplacian.shape}")
    except Exception as e:
        print(f"拉普拉斯边缘检测测试失败: {e}")
    
    # 测试直方图
    try:
        hist, bins = calc_histogram(test_image)
        print(f"直方图数据形状: {hist.shape}")
    except Exception as e:
        print(f"直方图计算测试失败: {e}")
    
    try:
        # 直方图均衡化
        equalized = calc_histogram_equalization(test_image)
        print(f"直方图均衡化后图像形状: {equalized.shape}")
    except Exception as e:
        print(f"直方图均衡化测试失败: {e}")
    
    try:
        # CLAHE
        clahe_img = calc_clahe(test_image)
        print(f"CLAHE处理后图像形状: {clahe_img.shape}")
    except Exception as e:
        print(f"CLAHE测试失败: {e}")
    
    # 显示特征提取功能信息
    print("\n可用特征提取功能:")
    features_info = get_features_info()
    for key, info in features_info.items():
        print(f"  {key}: {info['name']} - {info['description']}")
    
    print("特征提取功能测试完成")


if __name__ == "__main__":
    test_features()