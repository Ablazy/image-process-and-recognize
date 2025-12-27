// 设置文档
#set page(
  paper: "a4",
  margin: (left: 2.5cm, right: 2.5cm, top: 3cm, bottom: 2.5cm),
  footer: context align(center, text(size: 10pt, "-" + str(counter(page).display()) + "-"))
)

// 设置字体
#set text(font: ("Source Han Serif SC", "Times New Roman"), size: 12pt, hyphenate: true)

// 设置段落首行缩进为两个字符，1.5倍行距
#set par(first-line-indent:(amount: 2em, all: true), leading: 1.5em)

// 设置figure的分隔符号
#set figure.caption(separator: "  ")
// 自定义caption的显示格式
#show figure.caption: it => {
  text(
    font: ("Source Han Serif SC", "Times New Roman"),
    size: 10.5pt,
    weight: "regular",
    it
  )
}
// 设置图片
#show figure.where(kind: image):set figure(supplement: [图])
#show figure.where(kind: image):set figure.caption(position: bottom)

// 设置表格的caption
#show figure.where(kind: table):set figure(supplement: [表])
#show figure.where(kind: table):set figure.caption(position: top)

// 定义子图标签样式
// 创建子图计数器
#let subfig-counter = counter("subfig")

#let subfig-label = (label) => {
  subfig-counter.step()
  align(center, text(
    size: 10.5pt, 
    weight: "regular", 
    "(" + context subfig-counter.display("a") + ") " + label
  ))
}

// 每个figure会重置计数器
#show figure: it => {
  subfig-counter.update(0)
  it
}

// 设置代码块样式
#show raw.where(lang: "python"): it => {
  set par(first-line-indent: 0pt, leading: 0.5em)
  set text(font: ("Source Han Serif SC", "Times New Roman") )
  block(
    fill: rgb("f5f5f5"),
    inset: 10pt,
    radius: 5pt,
    breakable: false, 
    
    it
  )
}

// 创建状态变量来跟踪编号
#let h1-num = state("h1-num", 0)
#let h2-num = state("h2-num", 0)
#let h3-num = state("h3-num", 0)

// 使用 show 规则自定义 heading 显示
#show heading.where(level: 1): it => {
  let num = h1-num.get() + 1
  h1-num.update(num)
  h2-num.update(0)
  h3-num.update(0)
  set par(first-line-indent: 0pt)
  align(left, text(weight: "bold", size: 14pt, str(num) + " " + it.body))
}

#show heading.where(level: 2): it => {
  let parent = h1-num.get()
  let num = h2-num.get() + 1
  h2-num.update(num)
  h3-num.update(0)
  set par(first-line-indent: 0pt)
  align(left,text(weight: "bold", size: 12pt, str(parent) + "." + str(num) + " " + it.body))
}

#show heading.where(level: 3): it => {
  let num = h3-num.get() + 1
  h3-num.update(num)
  set par(first-line-indent: 0pt)
  align(left,text(weight: "bold", size: 12pt, "(" + str(num) + ") " + it.body))
}

// 标题部分
#align(center, text(size: 16pt, weight: "bold", [基于PyQt6的图像处理与手写数字识别软件设计]))

#align(right, text(size: 16pt, [大作业题目编号]))

#align(center, text(size: 10.5pt, [姓名，学号]))
#align(center, text(size: 10.5pt, [班级，个人email]))

// 摘要部分（标题左对齐，无编号）
#par(first-line-indent: 0em)[#align(left, text(size: 14pt, weight: "bold", [摘要]))]

本设计旨在开发一个基于PyQt6的图像处理与识别软件，解决图像识别和图像处理的实际应用需求。设计采用模块化架构，将用户界面层、业务逻辑层、数据处理层和机器学习层进行解耦，实现了图像旋转、缩放、裁剪、滤波、形态学操作、边缘检测等多种图像处理功能，以及基于深度学习模型和传统机器学习模型的双模态手写数字识别方案。系统通过PyQt6构建友好的图形用户界面，集成OpenCV进行图像处理，使用PyTorch和Scikit-learn分别实现深度学习和传统机器学习算法。设计结果实现了完整的图像处理流程和准确的手写数字识别功能，提供了相关的GUI界面。

#par(first-line-indent: 0em)[#text(weight: "bold")[关键字：] 图像处理，手写数字识别，PyQt6，MobileNetV2，SVM，HOG]

// 正文部分
#heading(level: 1, [引言])

随着计算机视觉和人工智能技术的快速发展，图像处理与模式识别在各个领域得到了广泛应用。手写数字识别作为模式识别的经典问题，在邮政编码识别、银行支票处理、表单数字化等场景中具有重要应用价值#cite(<lecun1998mnist>)。传统的手写数字识别方法主要依赖于人工提取的特征和机器学习算法，如支持向量机（SVM）#cite(<cortes1995svm>)、k近邻（KNN）等，这些方法在特定条件下取得了较好的效果，但对复杂背景和书写变化的适应性有限。

近年来，深度学习技术在图像识别领域取得了突破性进展，卷积神经网络（CNN）能够自动学习图像的层次化特征表示，显著提高了识别准确率。MobileNetV2作为一种轻量级的深度卷积神经网络，通过深度可分离卷积和倒残差结构，在保持较高识别精度的同时大幅减少了模型参数量和计算复杂度，非常适合在资源受限的设备上部署#cite(<sandler2018mobilenetv2>)。然而，深度学习模型通常需要大量标注数据和计算资源进行训练，而传统机器学习方法在小样本场景下仍具有优势。

在实际应用中，用户往往需要对手写数字图像进行预处理操作，如图像旋转、缩放、滤波、边缘检测等，以提高识别准确率或满足特定应用需求。因此，开发一个集图像处理与手写数字识别于一体的软件系统具有重要的实用价值。目前，虽然存在一些图像处理和数字识别的工具，但大多数工具功能单一，缺乏良好的用户界面和灵活的操作方式，难以满足用户的多样化需求。

本设计的目的是开发具有GUI的图像处理与手写数字识别软件，整合多种图像处理算法和两种不同的识别模型，为用户提供便捷、高效的图像处理和数字识别功能。设计采用模块化架构，将用户界面、业务逻辑、数据处理和机器学习模型进行解耦，提高了系统的可维护性和可扩展性。本设计的创新点在于：一是实现了深度学习模型和传统机器学习模型的集成预测，能够根据置信度自动选择最佳识别结果；二是设计了友好的图形用户界面，支持实时预览和参数调整；三是提供了丰富的图像处理功能，涵盖了基础操作、滤波、形态学操作和特征提取等多个方面。

本设计报告将首先介绍图像处理和手写数字识别的相关背景知识和现有方法，然后详细阐述系统的设计原理、架构设计和实现过程，最后通过仿真验证展示系统的功能和性能。

#heading(level: 1, [系统架构设计])

本设计采用分层架构设计，将系统划分为用户界面层、业务逻辑层、数据处理层和机器学习层四个层次，各层之间通过明确的接口进行通信。

用户界面层负责与用户进行交互，接收用户的操作指令并将处理结果展示给用户。该层采用PyQt6框架构建，包含主窗口、左侧控制栏、中央图像显示区和底部状态栏等组件。左侧控制栏集成了数据集导航、基础操作、滤波、形态学、特征提取和识别等功能选项卡，用户可以通过选项卡切换不同的功能模块。中央图像显示区使用QLabel或QGraphicsView组件显示当前处理的图像，支持图像的缩放和拖拽操作。底部状态栏用于显示操作日志和系统状态信息，为用户提供及时的反馈。PyQt6的相关资料参考#cite(<pyqt6book>)。

业务逻辑层负责协调各个功能模块的执行，处理用户的操作请求并调用相应的处理模块。该层包含图像处理管理器、识别管理器和数据集管理器三个核心组件。图像处理管理器负责管理各种图像处理操作，包括基础操作、滤波、形态学操作和特征提取等；识别管理器负责管理不同的识别模型，包括MobileNetV2深度学习模型和SVM结合HOG#cite(<dalal2005hog>)的传统机器学习模型；数据集管理器负责MNIST数据集的加载、缓存和管理，支持通过索引快速访问指定的图像。

数据处理层封装了各种图像处理算法的具体实现，基于OpenCV库提供高效的图像处理功能#cite(<opencvbook>)。该层包含基础操作模块、滤波模块、形态学模块和特征提取模块。基础操作模块实现了图像旋转、缩放、裁剪和翻转等基本几何变换；滤波模块实现了高斯滤波、均值滤波、中值滤波、双边滤波、锐化滤波、浮雕滤波和运动模糊等多种滤波算法；形态学模块实现了膨胀、腐蚀、开运算、闭运算、形态学梯度、顶帽变换和黑帽变换等形态学操作；特征提取模块实现了Canny边缘检测#cite(<canny1986edge>)、Sobel边缘检测、拉普拉斯边缘检测、灰度直方图、彩色直方图、直方图均衡化、自适应直方图均衡化等特征提取功能。

机器学习层实现了手写数字识别的核心算法，包含深度学习模型和传统机器学习模型两个子层。深度学习模型基于MobileNetV2网络结构，针对MNIST数据集的特点进行了适配，将输入层调整为单通道，输出层设置为10个类别。MobileNetV2采用深度可分离卷积和倒残差块结构，在保证识别精度的同时大幅减少了模型参数量。传统机器学习模型采用SVM分类器结合HOG特征提取的方法，首先使用HOG算法提取图像的方向梯度直方图特征，然后使用SVM进行分类，该方法在小样本场景下具有较好的性能。

数据层负责存储和管理系统所需的数据，包括MNIST数据集#cite(<lecun1998mnist>)、训练好的模型文件和临时图像缓存。MNIST数据集包含60000张训练图像和10000张测试图像，每张图像为28×28像素的灰度图像。模型文件存储目录用于保存训练好的MobileNetV2模型（.pth文件）和SVM+HOG模型（.pkl文件）。临时图像缓存用于存储处理过程中的中间结果，提高系统的响应速度。

#heading(level: 1, [核心功能实现])

#heading(level: 2, [图像处理功能实现])

图像处理功能是本系统的重要组成部分，涵盖了基础操作、滤波、形态学操作和特征提取四个方面。基础操作模块实现了图像的旋转、缩放、裁剪和翻转功能。图像旋转通过OpenCV的getRotationMatrix2D函数生成旋转矩阵，然后使用warpAffine函数进行仿射变换，支持-180°到+180°范围内的任意角度旋转。图像缩放通过resize函数实现，支持0.1倍到5.0倍的缩放比例，采用双线性插值算法保证缩放质量。图像裁剪通过NumPy数组切片实现，用户可以通过指定裁剪区域的左上角坐标和宽高来提取感兴趣区域。图像翻转通过flip函数实现，支持水平翻转和垂直翻转两种方式，具体实现如下。

```python
def rotate_image(image, angle):
    """旋转图像"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 执行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    return rotated_image
```
滤波模块实现了多种滤波算法，用于图像去噪和增强。高斯滤波通过高斯核对图像进行卷积，能够有效去除高斯噪声同时保持图像边缘信息，核大小可调。均值滤波通过均值核对图像进行卷积，适用于去除随机噪声，但会导致图像模糊。中值滤波通过取邻域像素的中值来替代中心像素，对椒盐噪声具有良好的去除效果。双边滤波结合了空间邻近度和像素值相似度，能够在去噪的同时保持边缘信息。锐化滤波通过拉普拉斯算子增强图像的高频成分，使图像边缘更加清晰。浮雕滤波通过卷积核产生浮雕效果，常用于艺术处理。运动模糊通过特定方向的卷积核模拟运动模糊效果。不同滤波算法的效果对比如@filter-comparison 所示。

#figure(
    grid(
    columns: 5,
    gutter: 0em,
    [#image("images/filter_original_noisy.png", width: 80%)
    #subfig-label([加噪原图])],
    [#image("images/filter_gaussian.png", width: 80%)
    #subfig-label([高斯滤波])],
    [#image("images/filter_mean.png", width: 80%)
    #subfig-label([均值滤波])],
    [#image("images/filter_median.png", width: 80%)
    #subfig-label([中值滤波])],
    [#image("images/filter_bilateral.png", width: 80%)
    #subfig-label([双边滤波])],
  ),
  caption: [不同滤波效果对比图],
  // placement: auto,
)<filter-comparison>

```python
def apply_gaussian_blur(image, kernel_size, sigma_x=0, sigma_y=None):
    """应用高斯滤波"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    if sigma_y is None:
        sigma_y = sigma_x
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x, sigma_y)
```

形态学操作模块实现了基于膨胀和腐蚀的各种形态学运算。膨胀操作通过结构元素对图像进行最大值滤波，能够填充图像中的小孔洞和连接断裂的区域。腐蚀操作通过结构元素对图像进行最小值滤波，能够去除图像中的小噪声和分离连接的区域。开运算是先腐蚀后膨胀的组合操作，能够去除小噪声同时保持目标区域大小不变。闭运算是先膨胀后腐蚀的组合操作，能够填充小孔洞同时保持目标区域大小不变。形态学梯度是膨胀图像与腐蚀图像的差值，能够突出图像的边缘信息。顶帽变换是原图像与开运算结果的差值，能够提取比结构元素小的亮区域。黑帽变换是闭运算结果与原图像的差值，能够提取比结构元素小的暗区域。各种形态学操作的效果对比如@morphology-comparison 所示。

#figure(
    grid(
    columns: 6,
    gutter: 0em,
    [#image("images/morphology_original.png", width: 80%)
    #subfig-label([原图])],
    [#image("images/morphology_dilate.png", width: 80%)
    #subfig-label([膨胀])],
    [#image("images/morphology_erode.png", width: 80%)
    #subfig-label([腐蚀])],
    [#image("images/morphology_opening.png", width: 80%)
    #subfig-label([开运算])],
    [#image("images/morphology_closing.png", width: 80%)
    #subfig-label([闭运算])],
    [#image("images/morphology_gradient.png", width: 80%)
    #subfig-label([梯度])],
  ),
  caption: [形态学操作效果对比图],
)<morphology-comparison>

特征提取模块实现了多种特征提取算法，用于图像分析和识别。Canny边缘检测通过多阶段算法检测图像中的边缘，包括高斯滤波、梯度计算、非极大值抑制和双阈值检测四个步骤，能够产生细而连续的边缘。Sobel边缘检测通过Sobel算子计算图像的梯度，能够检测水平和垂直方向的边缘。拉普拉斯边缘检测通过拉普拉斯算子计算图像的二阶导数，能够检测各个方向的边缘。灰度直方图统计图像中各灰度级的像素数量，反映图像的灰度分布特征。彩色直方图统计RGB三个通道的像素分布，反映图像的颜色特征。直方图均衡化通过累积分布函数变换图像的灰度级，增强图像的对比度。自适应直方图均衡化将图像分成小块分别进行均衡化，能够在增强对比度的同时避免噪声放大。不同边缘检测算法的效果对比如@edge-comparison 所示。

#figure(
    grid(
    columns: 4,
    gutter: 0em,
    [#image("images/edge_original.png", width: 80%)
    #subfig-label([原图])],
    [#image("images/edge_canny.png", width: 80%)
    #subfig-label([Canny])],
    [#image("images/edge_sobel.png", width: 80%)
    #subfig-label([Sobel])],
    [#image("images/edge_laplacian.png", width: 80%)
    #subfig-label([Laplacian])],
    ),
  caption: [不同边缘检测算法效果对比图],
)<edge-comparison>

```python
def apply_canny_edge_detection(image, threshold1, threshold2):
    """应用Canny边缘检测"""
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 高斯滤波去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny边缘检测
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    return edges
```

#heading(level: 2, [手写数字识别功能实现])

手写数字识别功能是本系统的核心功能，实现了基于MobileNetV2深度学习模型和SVM+HOG传统机器学习模型的双模态识别方案。MobileNetV2模型采用深度可分离卷积和倒残差块结构，大大减少了模型参数量和计算复杂度。倒残差块由1×1升维卷积、3×3深度可分离卷积和1×1降维卷积组成，中间层使用ReLU6激活函数（即最大取值为6的ReLU函数），输入输出层使用线性激活函数，通过残差连接缓解梯度消失问题。MobileNetV2网络包含多个倒残差块，通过逐步增加通道数和减少空间分辨率提取图像的层次化特征。针对MNIST数据集的特点，本设计将MobileNetV2的输入层调整为单通道，第一层卷积核数量设置为32，输出层全连接层设置为10个神经元，对应数字0到9的类别。MobileNetV2网络结构如@mobilenetv2-architecture 所示，核心的倒残差块实现如下。

#figure(
  image("images/mobilenetv2_architecture.png", width: 90%),
  caption: [MobileNetV2网络结构图],
  placement: auto,
)<mobilenetv2-architecture>

```python
class InvertedResidual(nn.Module):
    """倒残差块"""
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        # 1x1升维卷积
        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # 3x3深度可分离卷积
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                               groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # 1x1降维卷积
        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)
```

SVM+HOG模型采用传统机器学习方法，首先使用HOG算法提取图像特征，然后使用SVM分类器进行识别。HOG特征提取将图像划分为小块，计算每个小块内像素的梯度方向直方图，然后将小块的直方图组合成特征向量。HOG特征对光照变化和几何形变具有较强的鲁棒性，适合描述手写数字的形状特征。SVM分类器采用径向基函数（RBF）作为核函数，通过超参数优化获得最佳分类性能。SVM+HOG模型在小样本场景下具有较好的性能，训练速度快，适合快速部署。HOG的具体实现如下。

```python
def extract_hog_features(image):
    """提取HOG特征"""
    # 计算梯度
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    
    # 计算梯度幅值和方向
    magnitude, angle = cv2.cartToPolar(gx, gy)
    
    # 计算方向直方图
    bins = 9
    hist = np.zeros(bins)
    bin_size = 180 / bins
    
    for i in range(angle.shape[0]):
        for j in range(angle.shape[1]):
            bin_idx = int(angle[i, j] / bin_size) % bins
            hist[bin_idx] += magnitude[i, j]
    
    return hist
```

集成预测模块结合MobileNetV2模型和SVM+HOG模型的预测结果，通过加权平均的方式提高识别准确率。集成预测首先计算两个模型的预测置信度，然后根据置信度对预测结果进行加权，选择置信度最高的预测结果作为最终输出。集成预测能够充分利用两个模型的优势，在复杂场景下提高识别的鲁棒性，具体代码如下。

```python
def predict_ensemble(self, image):
    """集成预测"""
    results = {}
    
    # MobileNetV2预测
    mobilenet_pred, mobilenet_conf = self.predict_with_mobilenet(image)
    results['mobilenet'] = {'prediction': mobilenet_pred,'confidence': mobilenet_conf}
    
    # SVM预测
    svm_pred, svm_conf = self.predict_with_svm(image)
    results['svm'] = {'prediction': svm_pred,'confidence': svm_conf}
    
    # 根据置信度选择最佳结果
    if mobilenet_conf > svm_conf:
        final_prediction = mobilenet_pred
        final_confidence = mobilenet_conf
    else:
        final_prediction = svm_pred
        final_confidence = svm_conf
    
    return final_prediction, final_confidence
```

数据集管理模块负责MNIST数据集的加载、缓存和管理。MNIST数据集包含60000张训练图像和10000张测试图像，每张图像为28×28像素的灰度图像，标签为0到9的整数。数据集管理模块使用PyTorch的torchvision.datasets.MNIST接口自动下载数据集，并通过DataLoader实现批量加载和预处理。为了提高系统响应速度，数据集管理模块实现了图像缓存机制，将频繁访问的图像存储在内存中，减少重复加载的开销。MNIST数据集的样本示例如@mnist-samples 所示。

#figure(
    grid(
    columns: 10,
    gutter: 0em,
    [#image("images/mnist_digit_0.png", width: 80%)
    #subfig-label([0])],
    [#image("images/mnist_digit_1.png", width: 80%)
    #subfig-label([1])],
    [#image("images/mnist_digit_2.png", width: 80%)
    #subfig-label([2])],
    [#image("images/mnist_digit_3.png", width: 80%)
    #subfig-label([3])],
    [#image("images/mnist_digit_4.png", width: 80%)
    #subfig-label([4])],
    [#image("images/mnist_digit_5.png", width: 80%)
    #subfig-label([5])],
    [#image("images/mnist_digit_6.png", width: 80%)
    #subfig-label([6])],
    [#image("images/mnist_digit_7.png", width: 80%)
    #subfig-label([7])],
    [#image("images/mnist_digit_8.png", width: 80%)
    #subfig-label([8])],
    [#image("images/mnist_digit_9.png", width: 80%)
    #subfig-label([9])],
    ),
  caption: [MNIST数据集样本示例],
)<mnist-samples>

预测接口模块提供了统一的预测接口，支持单张图像预测和批量预测。预测接口接收OpenCV格式的图像，自动进行预处理包括尺寸调整、归一化等操作，然后调用相应的模型进行预测，返回预测类别和置信度。预测接口支持三种模式：MobileNetV2模式、SVM+HOG模式和集成预测模式，用户可以根据需求选择合适的模式。

```python
class ImagePredictor:
    """统一预测接口"""
    
    def predict(self, image, model_type='mobilenet'):
        """统一预测接口"""
        if model_type.lower() == 'mobilenet':
            return self.predict_with_mobilenet(image)
        elif model_type.lower() == 'svm':
            return self.predict_with_svm(image)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def predict_ensemble(self, image):
        """集成预测（结合两个模型的结果）"""
        ...
```

#heading(level: 2, [用户界面实现])

用户界面采用PyQt6框架构建，提供了友好的图形用户界面。主窗口采用QMainWindow类实现，包含菜单栏、工具栏、中央部件和状态栏。中央部件使用QHBoxLayout水平布局，左侧为控制栏，右侧为图像显示区。控制栏使用QTabWidget实现选项卡切换，每个选项卡对应一个功能模块。主窗口界面如@main-window 所示。

#figure(
  image("images/main_window.png", width: 90%),
  caption: [主窗口界面截图],
)<main-window>

```python
class MainWindow(QMainWindow):
    """主窗口类"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理与手写数字识别")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 水平布局
        layout = QHBoxLayout(central_widget)
        
        # 左侧控制栏
        self.control_panel = QTabWidget()
        self.control_panel.setFixedWidth(300)
        layout.addWidget(self.control_panel)
        
        # 右侧图像显示区
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_display)
```

数据集导航选项卡包含"加载数据集"按钮、"上一张"按钮、"下一张"按钮和图号跳转输入框。用户点击"加载数据集"按钮后，系统自动下载MNIST数据集并显示第一张图像。用户可以通过"上一张"和"下一张"按钮浏览数据集中的图像，也可以在图号跳转输入框中输入图号直接跳转到指定图像。数据集导航选项卡界面如@dataset-tab 所示。

#figure(
  image("images/dataset_tab.png", width: 60%),
  caption: [数据集导航选项卡界面],
)<dataset-tab>

基础操作选项卡包含旋转滑块、缩放比例输入框、裁剪按钮和恢复原图按钮。旋转滑块支持-180°到+180°的范围调节，用户拖动滑块时可以实时预览旋转效果。缩放比例输入框支持0.1倍到5.0倍的缩放，用户输入缩放比例后点击应用按钮执行缩放操作。裁剪按钮支持鼠标框选裁剪区域，用户在图像上拖拽鼠标选择裁剪区域后点击确认按钮完成裁剪。恢复原图按钮将图像恢复到加载时的原始状态。基础操作选项卡界面如@basic-ops-tab 所示。

#figure(
  image("images/basic_ops_tab.png", width: 60%),
  caption: [基础操作选项卡界面],
)<basic-ops-tab>

滤波选项卡包含滤波类型下拉菜单、核大小滑块和应用按钮。滤波类型下拉菜单提供高斯滤波、均值滤波、中值滤波、双边滤波、锐化滤波、浮雕滤波和运动模糊等选项。核大小滑块支持3到31的奇数范围调节，用户选择滤波类型和核大小后点击应用按钮执行滤波操作。滤波选项卡界面如@filter-tab 所示。

#figure(
  image("images/filter_tab.png", width: 60%),
  caption: [滤波选项卡界面],
)<filter-tab>

形态学选项卡包含操作类型下拉菜单、核大小滑块、迭代次数输入框和应用按钮。操作类型下拉菜单提供膨胀、腐蚀、开运算、闭运算、形态学梯度、顶帽变换和黑帽变换等选项。核大小滑块支持3到31的奇数范围调节，迭代次数输入框支持1到10的整数范围，用户选择操作类型、核大小和迭代次数后点击应用按钮执行形态学操作。形态学选项卡界面如@morphology-tab 所示。

#figure(
  image("images/morphology_tab.png", width: 60%),
  caption: [形态学选项卡界面],
)<morphology-tab>

特征提取选项卡包含边缘检测类型下拉菜单、阈值滑块、显示直方图按钮和应用按钮。边缘检测类型下拉菜单提供Canny边缘检测、Sobel边缘检测和拉普拉斯边缘检测等选项。阈值滑块用于设置边缘检测的阈值参数，用户选择边缘检测类型和阈值后点击应用按钮执行边缘检测。显示直方图按钮弹出直方图显示窗口，使用Matplotlib绘制图像的灰度直方图或彩色直方图。特征提取选项卡界面如@features-tab 所示。

#figure(
  image("images/features_tab.png", width: 60%),
  caption: [特征提取选项卡界面],
)<features-tab>

识别选项卡包含模型类型下拉菜单、加载模型按钮、开始识别按钮和结果显示标签。模型类型下拉菜单提供MobileNetV2、SVM+HOG和集成预测等选项。用户点击加载模型按钮后，系统从models目录加载对应的模型文件。用户点击开始识别按钮后，系统对当前显示的图像进行识别，结果显示标签显示预测的数字类别和置信度。识别选项卡界面如@recognition-tab 所示，手写数字识别结果示例如@recognition-result 所示。

#figure(
  image("images/recognition_tab.png", width: 60%),
  caption: [识别选项卡界面],
)<recognition-tab>

#figure(
  image("images/recognition_result.png", width: 70%),
  caption: [手写数字识别结果示例],
)<recognition-result>

#heading(level:1, [参考文献])

#bibliography("ref.bib", title: none)

#heading(level:1, [源代码])