#!/usr/bin/env python3
"""
图像处理与识别软件主程序
Image Processing & Recognition GUI - Main Entry Point
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from ui.main_window import MainWindow

def main():
    """主函数"""
    # 创建QApplication实例
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("图像处理与识别软件")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("ImageProcessing Lab")
    
    # 设置高DPI支持 (修复属性名)
    try:
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        # 如果属性不存在，跳过高DPI设置
        print("警告: 高DPI属性不可用，跳过高DPI设置")
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序事件循环
    sys.exit(app.exec())

if __name__ == "__main__":
    main()