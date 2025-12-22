"""
图像工具函数
提供图像格式转换、显示等工具函数
"""

import cv2
import numpy as np

# 尝试导入PyQt6，如果失败则提供替代方案
try:
    from PyQt6.QtGui import QPixmap, QImage
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("警告: PyQt6不可用，某些功能将受限")


def numpy_to_qpixmap(image):
    """
    将numpy数组转换为QPixmap
    
    Args:
        image: numpy数组图像
        
    Returns:
        QPixmap: PyQt6可显示的图像
    """
    if not PYQT_AVAILABLE:
        raise ImportError("PyQt6不可用，无法创建QPixmap")
    
    if image is None:
        return QPixmap()
    
    height, width = image.shape[:2]
    
    # 确保图像是连续的
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)
    
    if len(image.shape) == 3:
        # 彩色图像 (BGR to RGB)
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * width
            # 将memoryview转换为bytes
            image_data = image_rgb.data.tobytes() if hasattr(image_rgb.data, 'tobytes') else image_rgb.data
            q_image = QImage(image_data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        elif image.shape[2] == 4:
            # RGBA图像
            image_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            bytes_per_line = 4 * width
            # 将memoryview转换为bytes
            image_data = image_rgba.data.tobytes() if hasattr(image_rgba.data, 'tobytes') else image_rgba.data
            q_image = QImage(image_data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        else:
            # 其他通道数，取前3个通道
            image_rgb = image[:, :, :3]
            if image_rgb.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * width
            # 将memoryview转换为bytes
            image_data = image_rgb.data.tobytes() if hasattr(image_rgb.data, 'tobytes') else image_rgb.data
            q_image = QImage(image_data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    else:
        # 灰度图像
        bytes_per_line = width
        # 将memoryview转换为bytes
        image_data = image.data.tobytes() if hasattr(image.data, 'tobytes') else image.data
        q_image = QImage(image_data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
    
    return QPixmap.fromImage(q_image)


def qpixmap_to_numpy(pixmap):
    """
    将QPixmap转换为numpy数组
    
    Args:
        pixmap: QPixmap对象
        
    Returns:
        numpy.ndarray: OpenCV可处理的图像数组
    """
    if not PYQT_AVAILABLE:
        raise ImportError("PyQt6不可用，无法处理QPixmap")
    
    if pixmap.isNull():
        return None
    
    # 转换为QImage
    qimage = pixmap.toImage()
    
    # 转换为numpy数组
    width = qimage.width()
    height = qimage.height()
    
    # 根据格式转换
    format = qimage.format()
    
    if format == QImage.Format.Format_Grayscale8:
        # 灰度图像
        ptr = qimage.bits()
        ptr.setsize(height * width)
        arr = np.array(ptr).reshape(height, width)
        return arr.copy()
    
    elif format == QImage.Format.Format_RGB888:
        # RGB图像
        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        arr = np.array(ptr).reshape(height, width, 3)
        # RGB to BGR for OpenCV
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    elif format == QImage.Format.Format_RGBA8888:
        # RGBA图像
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape(height, width, 4)
        # RGBA to BGRA for OpenCV
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    
    else:
        # 其他格式，转换为RGB888
        qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        arr = np.array(ptr).reshape(height, width, 3)
        # RGB to BGR for OpenCV
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def scale_image_to_fit(image, max_width, max_height, keep_aspect_ratio=True):
    """
    缩放图像以适应指定大小
    
    Args:
        image: 输入图像
        max_width: 最大宽度
        max_height: 最大高度
        keep_aspect_ratio: 是否保持宽高比
        
    Returns:
        numpy.ndarray: 缩放后的图像
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    if not keep_aspect_ratio:
        return cv2.resize(image, (max_width, max_height), interpolation=cv2.INTER_AREA)
    
    # 计算缩放比例
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    
    if scale >= 1:
        return image  # 不需要缩放
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def ensure_grayscale(image):
    """
    确保图像是灰度图
    
    Args:
        image: 输入图像
        
    Returns:
        numpy.ndarray: 灰度图像
    """
    if image is None:
        return None
    
    if len(image.shape) == 2:
        return image.copy()
    
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image


def ensure_color(image):
    """
    确保图像是彩色图
    
    Args:
        image: 输入图像
        
    Returns:
        numpy.ndarray: 彩色图像
    """
    if image is None:
        return None
    
    if len(image.shape) == 3:
        return image.copy()
    
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    return image


def normalize_image(image, min_val=0, max_val=255):
    """
    归一化图像到指定范围
    
    Args:
        image: 输入图像
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        numpy.ndarray: 归一化后的图像
    """
    if image is None:
        return None
    
    # 转换为float
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # 归一化到0-1
    image_norm = (image - image.min()) / (image.max() - image.min())
    
    # 缩放到目标范围
    image_norm = image_norm * (max_val - min_val) + min_val
    
    return image_norm.astype(np.uint8)


def add_text_to_image(image, text, position=(10, 30), 
                   font_scale=1.0, color=(255, 255, 255), thickness=2):
    """
    在图像上添加文字
    
    Args:
        image: 输入图像
        text: 要添加的文字
        position: 文字位置 (x, y)
        font_scale: 字体大小
        color: 文字颜色 (B, G, R)
        thickness: 文字粗细
        
    Returns:
        numpy.ndarray: 添加文字后的图像
    """
    if image is None:
        return None
    
    result = image.copy()
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness)
    
    return result


def create_grid_image(images, grid_size=(2, 2), cell_size=(200, 200)):
    """
    创建图像网格
    
    Args:
        images: 图像列表
        grid_size: 网格大小 (rows, cols)
        cell_size: 每个单元格的大小
        
    Returns:
        numpy.ndarray: 网格图像
    """
    if not images:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)
    
    rows, cols = grid_size
    cell_h, cell_w = cell_size
    
    # 创建网格图像
    grid_image = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    # 填充网格
    for i, img in enumerate(images[:rows * cols]):
        row = i // cols
        col = i % cols
        
        # 调整图像大小
        if img is not None:
            img_resized = cv2.resize(img, (cell_w, cell_h))
            
            # 确保是彩色图像
            if len(img_resized.shape) == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            
            # 放置到网格中
            y_start = row * cell_h
            y_end = y_start + cell_h
            x_start = col * cell_w
            x_end = x_start + cell_w
            
            grid_image[y_start:y_end, x_start:x_end] = img_resized
    
    return grid_image


def compare_images(original, processed, titles=None):
    """
    并排比较原图和处理后的图像
    
    Args:
        original: 原始图像
        processed: 处理后的图像
        titles: 图像标题列表
        
    Returns:
        numpy.ndarray: 比较图像
    """
    if original is None or processed is None:
        return np.zeros((200, 400, 3), dtype=np.uint8)
    
    # 确保两个图像大小相同
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]
    
    max_h = max(h1, h2)
    max_w = max(w1, w2)
    
    # 调整图像大小
    orig_resized = cv2.resize(original, (max_w, max_h))
    proc_resized = cv2.resize(processed, (max_w, max_h))
    
    # 确保都是彩色图像
    if len(orig_resized.shape) == 2:
        orig_resized = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2BGR)
    if len(proc_resized.shape) == 2:
        proc_resized = cv2.cvtColor(proc_resized, cv2.COLOR_GRAY2BGR)
    
    # 创建比较图像
    comparison = np.hstack([orig_resized, proc_resized])
    
    # 添加标题
    if titles:
        cv2.putText(comparison, titles[0], (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, titles[1], (max_w + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return comparison


def test_image_utils():
    """测试图像工具函数"""
    print("测试图像工具函数...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    print(f"原始图像形状: {test_image.shape}")
    
    # 测试numpy到QPixmap转换
    try:
        pixmap = numpy_to_qpixmap(test_image)
        print(f"QPixmap转换成功，尺寸: {pixmap.size()}")
    except Exception as e:
        print(f"QPixmap转换失败: {e}")
    
    # 测试QPixmap到numpy转换
    try:
        back_to_numpy = qpixmap_to_numpy(pixmap)
        if back_to_numpy is not None:
            print(f"QPixmap到numpy转换成功，形状: {back_to_numpy.shape}")
        else:
            print("QPixmap到numpy转换失败")
    except Exception as e:
        print(f"QPixmap到numpy转换失败: {e}")
    
    # 测试缩放函数
    try:
        scaled = scale_image_to_fit(test_image, 50, 50)
        print(f"缩放后图像形状: {scaled.shape}")
    except Exception as e:
        print(f"缩放测试失败: {e}")
    
    # 测试灰度转换
    try:
        gray = ensure_grayscale(test_image)
        print(f"灰度转换后形状: {gray.shape}")
    except Exception as e:
        print(f"灰度转换失败: {e}")
    
    # 测试彩色转换
    try:
        color = ensure_color(gray)
        print(f"彩色转换后形状: {color.shape}")
    except Exception as e:
        print(f"彩色转换失败: {e}")
    
    # 测试归一化
    try:
        normalized = normalize_image(test_image)
        print(f"归一化后形状: {normalized.shape}")
        print(f"归一化后范围: [{normalized.min()}, {normalized.max()}]")
    except Exception as e:
        print(f"归一化失败: {e}")
    
    print("图像工具函数测试完成")


if __name__ == "__main__":
    test_image_utils()