"""
主窗口UI实现
图像处理与识别软件的主界面
"""

import sys
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QLabel, QPushButton, QTabWidget, QSlider, 
    QSpinBox, QComboBox, QTextEdit, QGroupBox,
    QGridLayout, QApplication, QSplitter, QMessageBox,
    QFileDialog, QProgressBar, QCheckBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon

# 添加相对导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import numpy_to_qpixmap, qpixmap_to_numpy


class ImageProcessingThread(QThread):
    """图像处理线程"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, image, processing_func, *args, **kwargs):
        super().__init__()
        self.image = image
        self.processing_func = processing_func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.processing_func(self.image, *self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理与识别软件 v1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # 初始化数据
        self.current_image = None
        self.original_image = None
        self.current_index = 0
        self.dataset_size = 0
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px;
                border: 1px solid #cccccc;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
        """)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # 创建左侧控制栏
        self.left_control_panel = self.create_left_control_panel()
        splitter.addWidget(self.left_control_panel)
        splitter.setSizes([400, 1000])  # 左侧400px，右侧1000px
        
        # 创建右侧区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 创建中央图像显示区
        self.image_display_area = self.create_image_display_area()
        right_layout.addWidget(self.image_display_area, 3)
        
        # 创建底部状态栏
        self.status_bar = self.create_status_bar()
        right_layout.addWidget(self.status_bar, 1)
        
        splitter.addWidget(right_widget)
        
        # 初始化定时器用于状态更新
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # 每秒更新一次
        
        # 应用按钮模式的全局变量
        self.apply_mode = True
        
        # 初始化日志
        self.log_message("应用程序启动")
        self.log_message("请加载图像或数据集开始使用")
    
    def create_left_control_panel(self):
        """创建左侧控制栏"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 数据集导航组
        nav_group = QGroupBox("数据集导航")
        nav_layout = QHBoxLayout(nav_group)
        
        self.prev_btn = QPushButton("上一张")
        self.next_btn = QPushButton("下一张")
        self.index_spinbox = QSpinBox()
        self.index_spinbox.setRange(0, 9999)
        self.index_spinbox.setValue(0)
        self.jump_btn = QPushButton("跳转")
        self.load_dataset_btn = QPushButton("加载数据集")
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addWidget(QLabel("图号:"))
        nav_layout.addWidget(self.index_spinbox)
        nav_layout.addWidget(self.jump_btn)
        nav_layout.addWidget(self.load_dataset_btn)
        
        layout.addWidget(nav_group)
        
        # 文件操作组
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        
        self.load_image_btn = QPushButton("加载图像")
        self.save_image_btn = QPushButton("保存图像")
        self.reset_btn = QPushButton("恢复原图")
        
        file_layout.addWidget(self.load_image_btn)
        file_layout.addWidget(self.save_image_btn)
        file_layout.addWidget(self.reset_btn)
        
        layout.addWidget(file_group)
        
        # 功能选项卡
        self.tab_widget = QTabWidget()
        
        # 基础操作选项卡
        self.basic_tab = self.create_basic_tab()
        self.tab_widget.addTab(self.basic_tab, "基础")
        
        # 滤波选项卡
        self.filter_tab = self.create_filter_tab()
        self.tab_widget.addTab(self.filter_tab, "滤波")
        
        # 形态学选项卡
        self.morphology_tab = self.create_morphology_tab()
        self.tab_widget.addTab(self.morphology_tab, "形态学")
        
        # 特征选项卡
        self.feature_tab = self.create_feature_tab()
        self.tab_widget.addTab(self.feature_tab, "特征")
        
        # 识别选项卡
        self.recognition_tab = self.create_recognition_tab()
        self.tab_widget.addTab(self.recognition_tab, "识别")
        
        layout.addWidget(self.tab_widget)
        
        # 应用模式选择
        mode_group = QGroupBox("处理模式")
        mode_layout = QVBoxLayout(mode_group)
        
        self.apply_mode_checkbox = QCheckBox("应用按钮模式")
        self.apply_mode_checkbox.setChecked(True)
        self.apply_mode_checkbox.setToolTip("启用后需要点击应用按钮才显示处理结果")
        
        mode_layout.addWidget(self.apply_mode_checkbox)
        layout.addWidget(mode_group)
        
        return panel
    
    def create_basic_tab(self):
        """创建基础操作选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 旋转控制
        rotate_group = QGroupBox("旋转")
        rotate_layout = QVBoxLayout(rotate_group)
        
        self.rotate_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotate_slider.setRange(-180, 180)
        self.rotate_slider.setValue(0)
        self.rotate_label = QLabel("角度: 0°")
        
        rotate_layout.addWidget(self.rotate_label)
        rotate_layout.addWidget(self.rotate_slider)
        layout.addWidget(rotate_group)
        
        # 缩放控制
        scale_group = QGroupBox("缩放")
        scale_layout = QVBoxLayout(scale_group)
        
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(0.1, 5.0)
        self.scale_spinbox.setValue(1.0)
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setSuffix("倍")
        
        scale_layout.addWidget(QLabel("缩放比例:"))
        scale_layout.addWidget(self.scale_spinbox)
        layout.addWidget(scale_group)
        
        # 裁剪控制
        crop_group = QGroupBox("裁剪")
        crop_layout = QGridLayout(crop_group)
        
        self.crop_x_spinbox = QSpinBox()
        self.crop_y_spinbox = QSpinBox()
        self.crop_w_spinbox = QSpinBox()
        self.crop_h_spinbox = QSpinBox()
        
        crop_layout.addWidget(QLabel("X:"), 0, 0)
        crop_layout.addWidget(self.crop_x_spinbox, 0, 1)
        crop_layout.addWidget(QLabel("Y:"), 0, 2)
        crop_layout.addWidget(self.crop_y_spinbox, 0, 3)
        crop_layout.addWidget(QLabel("宽:"), 1, 0)
        crop_layout.addWidget(self.crop_w_spinbox, 1, 1)
        crop_layout.addWidget(QLabel("高:"), 1, 2)
        crop_layout.addWidget(self.crop_h_spinbox, 1, 3)
        
        self.crop_btn = QPushButton("应用裁剪")
        crop_layout.addWidget(self.crop_btn, 2, 0, 1, 4)
        
        layout.addWidget(crop_group)
        
        # 翻转控制
        flip_group = QGroupBox("翻转")
        flip_layout = QHBoxLayout(flip_group)
        
        self.flip_h_btn = QPushButton("水平翻转")
        self.flip_v_btn = QPushButton("垂直翻转")
        
        flip_layout.addWidget(self.flip_h_btn)
        flip_layout.addWidget(self.flip_v_btn)
        layout.addWidget(flip_group)
        
        # 应用按钮
        self.apply_basic_btn = QPushButton("应用基础操作")
        self.apply_basic_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                font-size: 14px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.apply_basic_btn)
        
        return tab
    
    def create_filter_tab(self):
        """创建滤波选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 滤波类型选择
        filter_type_group = QGroupBox("滤波类型")
        filter_type_layout = QVBoxLayout(filter_type_group)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "高斯滤波", "均值滤波", "中值滤波", 
            "双边滤波", "锐化滤波", "浮雕滤波"
        ])
        
        filter_type_layout.addWidget(self.filter_combo)
        layout.addWidget(filter_type_group)
        
        # 核大小控制
        kernel_group = QGroupBox("核大小")
        kernel_layout = QVBoxLayout(kernel_group)
        
        self.kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.kernel_slider.setRange(1, 31)
        self.kernel_slider.setValue(5)
        self.kernel_slider.setSingleStep(2)  # 只允许奇数
        self.kernel_label = QLabel("核大小: 5")
        
        kernel_layout.addWidget(self.kernel_label)
        kernel_layout.addWidget(self.kernel_slider)
        layout.addWidget(kernel_group)
        
        # 高级参数
        params_group = QGroupBox("高级参数")
        params_layout = QGridLayout(params_group)
        
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setRange(0.1, 10.0)
        self.sigma_spinbox.setValue(1.0)
        self.sigma_spinbox.setSingleStep(0.1)
        
        params_layout.addWidget(QLabel("Sigma:"), 0, 0)
        params_layout.addWidget(self.sigma_spinbox, 0, 1)
        
        layout.addWidget(params_group)
        
        # 应用按钮
        self.apply_filter_btn = QPushButton("应用滤波")
        self.apply_filter_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                font-size: 14px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.apply_filter_btn)
        
        return tab
    
    def create_morphology_tab(self):
        """创建形态学选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 形态学操作类型
        morph_type_group = QGroupBox("操作类型")
        morph_type_layout = QVBoxLayout(morph_type_group)
        
        self.morph_combo = QComboBox()
        self.morph_combo.addItems([
            "膨胀", "腐蚀", "开运算", "闭运算", 
            "形态学梯度", "顶帽变换", "黑帽变换"
        ])
        
        morph_type_layout.addWidget(self.morph_combo)
        layout.addWidget(morph_type_group)
        
        # 核大小和迭代次数
        params_group = QGroupBox("参数")
        params_layout = QGridLayout(params_group)
        
        self.morph_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.morph_kernel_slider.setRange(1, 31)
        self.morph_kernel_slider.setValue(5)
        self.morph_kernel_slider.setSingleStep(2)
        self.morph_kernel_label = QLabel("核大小: 5")
        
        self.morph_iter_spinbox = QSpinBox()
        self.morph_iter_spinbox.setRange(1, 10)
        self.morph_iter_spinbox.setValue(1)
        
        params_layout.addWidget(self.morph_kernel_label, 0, 0)
        params_layout.addWidget(self.morph_kernel_slider, 0, 1)
        params_layout.addWidget(QLabel("迭代次数:"), 1, 0)
        params_layout.addWidget(self.morph_iter_spinbox, 1, 1)
        
        layout.addWidget(params_group)
        
        # 应用按钮
        self.apply_morph_btn = QPushButton("应用形态学操作")
        self.apply_morph_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                font-size: 14px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.apply_morph_btn)
        
        return tab
    
    def create_feature_tab(self):
        """创建特征提取选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 边缘检测
        edge_group = QGroupBox("边缘检测")
        edge_layout = QGridLayout(edge_group)
        
        self.low_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.low_threshold_slider.setRange(0, 255)
        self.low_threshold_slider.setValue(50)
        self.low_threshold_label = QLabel("低阈值: 50")
        
        self.high_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.high_threshold_slider.setRange(0, 255)
        self.high_threshold_slider.setValue(150)
        self.high_threshold_label = QLabel("高阈值: 150")
        
        edge_layout.addWidget(self.low_threshold_label, 0, 0)
        edge_layout.addWidget(self.low_threshold_slider, 0, 1)
        edge_layout.addWidget(self.high_threshold_label, 1, 0)
        edge_layout.addWidget(self.high_threshold_slider, 1, 1)
        
        self.edge_btn = QPushButton("应用边缘检测")
        edge_layout.addWidget(self.edge_btn, 2, 0, 1, 2)
        
        layout.addWidget(edge_group)
        
        # 直方图
        hist_group = QGroupBox("直方图")
        hist_layout = QVBoxLayout(hist_group)
        
        self.hist_btn = QPushButton("显示直方图")
        self.hist_equalize_btn = QPushButton("直方图均衡化")
        
        hist_layout.addWidget(self.hist_btn)
        hist_layout.addWidget(self.hist_equalize_btn)
        
        layout.addWidget(hist_group)
        
        # 阈值化
        threshold_group = QGroupBox("阈值化")
        threshold_layout = QVBoxLayout(threshold_group)
        
        self.otsu_btn = QPushButton("大津法阈值")
        self.adaptive_btn = QPushButton("自适应阈值")
        
        threshold_layout.addWidget(self.otsu_btn)
        threshold_layout.addWidget(self.adaptive_btn)
        
        layout.addWidget(threshold_group)
        
        return tab
    
    def create_recognition_tab(self):
        """创建识别选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["MobileNetV2", "SVM+HOG", "集成预测"])
        
        model_layout.addWidget(self.model_combo)
        layout.addWidget(model_group)
        
        # 模型状态
        status_group = QGroupBox("模型状态")
        status_layout = QVBoxLayout(status_group)
        
        self.model_status_label = QLabel("模型未加载")
        self.model_status_label.setStyleSheet("color: red; font-weight: bold;")
        
        self.load_models_btn = QPushButton("加载模型")
        
        status_layout.addWidget(self.model_status_label)
        status_layout.addWidget(self.load_models_btn)
        layout.addWidget(status_group)
        
        # 识别按钮
        self.recognize_btn = QPushButton("开始识别")
        self.recognize_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                font-size: 16px;
                padding: 12px;
            }
        """)
        layout.addWidget(self.recognize_btn)
        
        # 结果显示
        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout(result_group)
        
        self.result_label = QLabel("未识别")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2196F3;")
        self.confidence_label = QLabel("置信度: -")
        self.confidence_label.setStyleSheet("font-size: 14px; color: #666666;")
        
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.confidence_label)
        
        layout.addWidget(result_group)
        
        return tab
    
    def create_image_display_area(self):
        """创建图像显示区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 图像显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                background-color: white;
                min-height: 400px;
            }
        """)
        self.image_label.setText("请加载图像")
        self.image_label.setFont(QFont("Arial", 16))
        
        layout.addWidget(self.image_label)
        
        return widget
    
    def create_status_bar(self):
        """创建状态栏"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 日志文本区域
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #cccccc;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        
        layout.addWidget(QLabel("操作日志:"))
        layout.addWidget(self.log_text)
        
        return widget
    
    def log_message(self, message):
        """添加日志消息"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def display_image(self, image):
        """显示图像"""
        if image is None:
            self.image_label.setText("无图像")
            return
        
        # 转换为QPixmap
        pixmap = numpy_to_qpixmap(image)
        
        if pixmap.isNull():
            self.image_label.setText("图像格式错误")
            return
        
        # 缩放图像以适应显示区域
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def update_status(self):
        """更新状态信息"""
        # 这里可以添加状态更新逻辑
        pass
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        # 重新调整图像显示
        if self.current_image is not None:
            self.display_image(self.current_image)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        reply = QMessageBox.question(
            self, '确认退出',
            '确定要退出应用程序吗？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.log_message("应用程序退出")
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())