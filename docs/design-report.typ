// 设置文档
#set page(
  paper: "a4",
  margin: (left: 1.25in, right: 1.25in, top: 1in, bottom: 1in),
  footer: context align(center, text(size: 9pt, "- " + str(counter(page).display()) + " -"))
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
  set text(font: ("Source Han Serif SC", "Times New Roman"), size: 10.5pt)
  block(
    // fill: rgb("f5f5f5"),
    inset: 0pt,
    radius: 5pt,
    it
  )
}
#show raw.where(lang: "python"): set block(breakable: true)

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

#align(right, text(size: 16pt, [3. 图像处理与识别软件]))

#align(center, text(size: 10.5pt, [张峻，25171214010]))
#align(center, text(size: 10.5pt, [班级，ablazy.zj\@protonmail.com]))

// 摘要部分（标题左对齐，无编号）
#par(first-line-indent: 0em)[#align(left, text(size: 14pt, weight: "bold", [摘要]))]

本设计旨在开发一个基于PyQt6的图像处理与识别软件，解决图像识别和图像处理的实际应用需求。设计采用模块化架构，将用户界面层、业务逻辑层、数据处理层和机器学习层进行解耦，实现了图像旋转、缩放、裁剪、滤波、形态学操作、边缘检测等多种图像处理功能，以及基于深度学习模型和传统机器学习模型的双模态手写数字识别方案。系统通过PyQt6构建友好的图形用户界面，集成OpenCV进行图像处理，使用PyTorch和Scikit-learn分别实现深度学习和传统机器学习算法。设计结果实现了完整的图像处理流程和准确的手写数字识别功能，提供了相关的GUI界面。
完整代码可以在#text(blue)[#link("https://github.com/Ablazy/image-process-and-recognize")[https://github.com/Ablazy/image-process-and-recognize]]中查看。

#v(0.5em)
#par(first-line-indent: 0em)[#text(weight: "bold")[关键字：] 图像处理，手写数字识别，PyQt6，MobileNetV2，SVM，HOG]
\


// 正文部分
#heading(level: 1, [引言])

随着计算机视觉和人工智能技术的快速发展，图像处理与模式识别在各个领域得到了广泛应用。手写数字识别作为模式识别的经典问题，在邮政编码识别、银行支票处理、表单数字化等场景中具有重要应用价值#cite(<lecun1998mnist>)。传统的手写数字识别方法主要依赖于人工提取的特征和机器学习算法，如支持向量机（SVM）#cite(<cortes1995svm>)、k近邻（KNN）等，这些方法在特定条件下取得了较好的效果，但对复杂背景和书写变化的适应性有限。

近年来，深度学习技术在图像识别领域取得了突破性进展，卷积神经网络（CNN）能够自动学习图像的层次化特征表示，显著提高了识别准确率。MobileNetV2作为一种轻量级的深度卷积神经网络，通过深度可分离卷积和倒残差结构，在保持较高识别精度的同时大幅减少了模型参数量和计算复杂度，非常适合在资源受限的设备上部署#cite(<sandler2018mobilenetv2>)。然而，深度学习模型通常需要大量标注数据和计算资源进行训练，而传统机器学习方法在小样本场景下仍具有优势。

在实际应用中，用户往往需要对手写数字图像进行预处理操作，如图像旋转、缩放、滤波、边缘检测等，以提高识别准确率或满足特定应用需求。因此，开发一个集图像处理与手写数字识别于一体的软件系统具有重要的实用价值。目前，虽然存在一些图像处理和数字识别的工具，但大多数工具功能单一，缺乏良好的用户界面和灵活的操作方式，难以满足用户的多样化需求。

本设计的目的是开发具有GUI的图像处理与手写数字识别软件，整合多种图像处理算法和两种不同的识别模型，为用户提供便捷、高效的图像处理和数字识别功能。设计采用模块化架构，将用户界面、业务逻辑、数据处理和机器学习模型进行解耦，提高了系统的可维护性和可扩展性。本设计的创新点在于：一是实现了深度学习模型和传统机器学习模型的集成预测，能够根据置信度自动选择最佳识别结果；二是设计了友好的图形用户界面，支持实时预览和参数调整；三是提供了丰富的图像处理功能，涵盖了基础操作、滤波、形态学操作和特征提取等多个方面。

本设计报告将首先介绍图像处理和手写数字识别的相关背景知识和现有方法，然后详细阐述系统的设计原理、架构设计和实现过程，最后通过仿真验证展示系统的功能和性能。

#v(0.5em)
#heading(level: 1, [系统架构设计])

本设计采用分层架构设计，将系统划分为用户界面层、业务逻辑层、数据处理层和机器学习层四个层次，各层之间通过明确的接口进行通信。

用户界面层负责与用户进行交互，接收用户的操作指令并将处理结果展示给用户。该层采用PyQt6框架构建，包含主窗口、左侧控制栏、中央图像显示区和底部状态栏等组件。左侧控制栏集成了数据集导航、基础操作、滤波、形态学、特征提取和识别等功能选项卡，用户可以通过选项卡切换不同的功能模块。中央图像显示区使用QLabel或QGraphicsView组件显示当前处理的图像，支持图像的缩放和拖拽操作。底部状态栏用于显示操作日志和系统状态信息，为用户提供及时的反馈。PyQt6的相关资料参考#cite(<pyqt6book>)。

业务逻辑层负责协调各个功能模块的执行，处理用户的操作请求并调用相应的处理模块。该层包含图像处理管理器、识别管理器和数据集管理器三个核心组件。图像处理管理器负责管理各种图像处理操作，包括基础操作、滤波、形态学操作和特征提取等；识别管理器负责管理不同的识别模型，包括MobileNetV2深度学习模型和SVM结合HOG#cite(<dalal2005hog>)的传统机器学习模型；数据集管理器负责MNIST数据集的加载、缓存和管理，支持通过索引快速访问指定的图像。

数据处理层封装了各种图像处理算法的具体实现，基于OpenCV库提供高效的图像处理功能#cite(<opencvbook>)。该层包含基础操作模块、滤波模块、形态学模块和特征提取模块。基础操作模块实现了图像旋转、缩放、裁剪和翻转等基本几何变换；滤波模块实现了高斯滤波、均值滤波、中值滤波、双边滤波、锐化滤波、浮雕滤波和运动模糊等多种滤波算法；形态学模块实现了膨胀、腐蚀、开运算、闭运算、形态学梯度、顶帽变换和黑帽变换等形态学操作；特征提取模块实现了Canny边缘检测#cite(<canny1986edge>)、Sobel边缘检测、拉普拉斯边缘检测、灰度直方图、彩色直方图、直方图均衡化、自适应直方图均衡化等特征提取功能。

机器学习层实现了手写数字识别的核心算法，包含深度学习模型和传统机器学习模型两个子层。深度学习模型基于MobileNetV2网络结构，针对MNIST数据集的特点进行了适配，将输入层调整为单通道，输出层设置为10个类别。MobileNetV2采用深度可分离卷积和倒残差块结构，在保证识别精度的同时大幅减少了模型参数量。传统机器学习模型采用SVM分类器结合HOG特征提取的方法，首先使用HOG算法提取图像的方向梯度直方图特征，然后使用SVM进行分类，该方法在小样本场景下具有较好的性能。

数据层负责存储和管理系统所需的数据，包括MNIST数据集#cite(<lecun1998mnist>)、训练好的模型文件和临时图像缓存。MNIST数据集包含60000张训练图像和10000张测试图像，每张图像为28×28像素的灰度图像。模型文件存储目录用于保存训练好的MobileNetV2模型（.pth文件）和SVM+HOG模型（.pkl文件）。临时图像缓存用于存储处理过程中的中间结果，提高系统的响应速度。

#v(0.5em)
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

#v(0.5em)
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

#v(0.5em)
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

#show raw.where(lang: "python"): set block(breakable: true)

#text(size: 10.5pt)[本设计的部分核心代码在下面列出，完整的源代码可以在：#text(blue)[#link("https://github.com/Ablazy/image-process-and-recognize")[https://github.com/Ablazy/image-process-and-recognize]]中查看。]

#text(size: 10.5pt, weight: "bold")[模块basic_ops.py]
#text(size: 10.5pt)[（功能：基础图像处理，实现图像旋转、缩放、裁剪、翻转等基本几何变换）]

```python
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
```

#text(size: 10.5pt, weight: "bold")[模块filters.py]
#text(size: 10.5pt)[（功能：图像滤波，实现高斯滤波、均值滤波、中值滤波、双边滤波等多种滤波算法）]

```python
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
```

#text(size: 10.5pt, weight: "bold")[模块morphology.py]
#text(size: 10.5pt)[（功能：形态学操作，实现膨胀、腐蚀、开运算、闭运算等形态学变换）]

```python
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
```

#text(size: 10.5pt, weight: "bold")[模块features.py]
#text(size: 10.5pt)[（功能：特征提取，实现Canny边缘检测、Sobel边缘检测、拉普拉斯边缘检测等特征提取功能）]

```python
"""
特征提取模块
实现边缘检测、直方图计算等特征提取功能
"""

import cv2
import numpy as np


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
```

#text(size: 10.5pt, weight: "bold")[模块models.py]
#text(size: 10.5pt)[（功能：深度学习模型定义，实现MobileNetV2网络结构）]

```python
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
```

#text(size: 10.5pt, weight: "bold")[模块traditional_ml.py]
#text(size: 10.5pt)[（功能：传统机器学习模型，实现SVM+HOG特征提取用于手写数字识别）]

```python
"""
传统机器学习模型实现
SVM + HOG特征提取用于手写数字识别
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import pickle
import os
import sys
import time
from datetime import datetime


class SVMHOGClassifier:
    """
    SVM + HOG特征分类器
    """
    
    def __init__(self, model_path=None):
        """
        初始化SVM+HOG分类器
        
        Args:
            model_path: 预训练模型路径
        """
        self.model = None
        self.model_path = model_path
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',
            'visualize': False
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model()
    
    def extract_hog_features(self, images):
        """
        提取HOG特征
        
        Args:
            images: 图像列表，每个图像为numpy数组
            
        Returns:
            np.array: HOG特征矩阵
        """
        features = []
        for img in images:
            # 确保图像是2D的
            if len(img.shape) > 2:
                img = img.squeeze()
            
            # 提取HOG特征
            fd = hog(
                img,
                **self.hog_params
            )
            features.append(fd)
        
        return np.array(features)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练SVM分类器
        
        Args:
            X_train: 训练图像
            y_train: 训练标签
            X_val: 验证图像（可选）
            y_val: 验证标签（可选）
            
        Returns:
            dict: 训练结果
        """
        print("提取HOG特征...")
        start_time = time.time()
        
        # 提取HOG特征
        X_train_features = self.extract_hog_features(X_train)
        feature_time = time.time() - start_time
        
        print(f"HOG特征提取完成，用时: {feature_time:.2f}秒")
        print(f"特征维度: {X_train_features.shape}")
        
        # 训练SVM
        print("训练SVM分类器...")
        start_time = time.time()
        
        self.model = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        self.model.fit(X_train_features, y_train)
        train_time = time.time() - start_time
        
        print(f"SVM训练完成，用时: {train_time:.2f}秒")
        
        # 计算训练准确率
        y_train_pred = self.model.predict(X_train_features)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        results = {
            'train_accuracy': train_accuracy,
            'feature_extraction_time': feature_time,
            'training_time': train_time,
            'feature_dim': X_train_features.shape[1]
        }
        
        # 如果有验证集，计算验证准确率
        if X_val is not None and y_val is not None:
            val_results = self.evaluate(X_val, y_val, dataset_type='验证')
            results.update(val_results)
        
        print(f"训练准确率: {train_accuracy:.4f}")
        
        return results
    
    def predict(self, image):
        """
        预测单张图像
        
        Args:
            image: 输入图像
            
        Returns:
            tuple: (预测类别, 置信度)
        """
        if self.model is None:
            raise ValueError("模型未训练或加载")
        
        # 提取HOG特征
        features = self.extract_hog_features([image])
        
        # 预测
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        
        return prediction, confidence
    
    def evaluate(self, X_test, y_test, dataset_type='测试'):
        """
        评估模型性能
        
        Args:
            X_test: 测试图像
            y_test: 测试标签
            dataset_type: 数据集类型名称
            
        Returns:
            dict: 评估结果
        """
        print(f"评估{dataset_type}集性能...")
        
        # 提取特征
        X_test_features = self.extract_hog_features(X_test)
        
        # 预测
        y_pred = self.model.predict(X_test_features)
        y_pred_proba = self.model.predict_proba(X_test_features)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        
        # 详细分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            f'{dataset_type.lower()}_accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        print(f"{dataset_type}准确率: {accuracy:.4f}")
        
        return results
    
    def save_model(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型和参数
        model_data = {
            'model': self.model,
            'hog_params': self.hog_params
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {path}")
    
    def load_model(self):
        """
        加载模型
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.hog_params = model_data.get('hog_params', self.hog_params)
        
        print(f"模型已从 {self.model_path} 加载")
```

#text(size: 10.5pt, weight: "bold")[模块predict.py]
#text(size: 10.5pt)[（功能：统一预测接口，整合MobileNetV2和SVM+HOG两种模型的预测功能）]

```python
"""
统一预测接口
整合MobileNetV2和SVM+HOG两种模型的预测功能
"""

import torch
import numpy as np
import cv2
import os
import sys

from ml.models import MobileNetV2, load_mobilenet_model
from ml.traditional_ml import SVMHOGClassifier


class ImagePredictor:
    """
    图像预测器
    支持MobileNetV2和SVM+HOG两种模型
    """
    
    def __init__(self):
        """初始化预测器"""
        self.mobilenet_model = None
        self.svm_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mobilenet_loaded = False
        self.svm_loaded = False
        
    def load_models(self, mobilenet_path='./models/mnist_mobilenet.pth',
                   svm_path='./models/svm_hog.pkl'):
        """
        加载所有模型
        
        Args:
            mobilenet_path: MobileNetV2模型路径
            svm_path: SVM模型路径
        """
        # 加载MobileNetV2模型
        try:
            self.mobilenet_model = load_mobilenet_model(mobilenet_path)
            self.mobilenet_model.to(self.device)
            self.mobilenet_loaded = True
            print("MobileNetV2模型加载成功")
        except Exception as e:
            print(f"加载MobileNetV2模型失败: {e}")
            self.mobilenet_loaded = False
            
        # 加载SVM模型
        try:
            self.svm_model = SVMHOGClassifier(svm_path)
            self.svm_loaded = True
            print("SVM+HOG模型加载成功")
        except Exception as e:
            print(f"加载SVM模型失败: {e}")
            self.svm_loaded = False
        
        return self.mobilenet_loaded or self.svm_loaded
    
    def preprocess_for_mobilenet(self, image):
        """
        为MobileNetV2预处理图像
        
        Args:
            image: 输入图像 (numpy数组)
            
        Returns:
            torch.Tensor: 预处理后的张量
        """
        # 确保是单通道灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 调整大小为224x224
        image_resized = image.copy()
        # 归一化
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - 0.1307) / 0.3081
        
        # 转换为tensor并添加batch和channel维度
        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict_with_mobilenet(self, image):
        """
        使用MobileNetV2进行预测
        
        Args:
            image: 输入图像
            
        Returns:
            tuple: (预测类别, 置信度)
        """
        if not self.mobilenet_loaded or self.mobilenet_model is None:
            raise ValueError("MobileNetV2模型未加载")
        
        # 预处理
        input_tensor = self.preprocess_for_mobilenet(image)
        
        # 预测
        with torch.no_grad():
            self.mobilenet_model.eval()
            outputs = self.mobilenet_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
        return prediction.item(), confidence.item()
    
    def predict_with_svm(self, image):
        """
        使用SVM+HOG进行预测
        
        Args:
            image: 输入图像
            
        Returns:
            tuple: (预测类别, 置信度)
        """
        if not self.svm_loaded or self.svm_model is None:
            raise ValueError("SVM模型未加载")
        
        # 确保是单通道灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 调整大小为28x28（MNIST原始尺寸）
        image_resized = cv2.resize(image, (28, 28))
        
        return self.svm_model.predict(image_resized)
    
    def predict(self, image, model_type='mobilenet'):
        """
        统一预测接口
        
        Args:
            image: 输入图像
            model_type: 模型类型 ('mobilenet' 或 'svm')
            
        Returns:
            tuple: (预测类别, 置信度)
        """
        if model_type.lower() == 'mobilenet':
            return self.predict_with_mobilenet(image)
        elif model_type.lower() == 'svm':
            return self.predict_with_svm(image)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def predict_ensemble(self, image):
        """
        集成预测（结合两个模型的结果）
        
        Args:
            image: 输入图像
            
        Returns:
            dict: 预测结果详情
        """
        results = {}
        
        # MobileNetV2预测
        if self.mobilenet_loaded:
            try:
                mobilenet_pred, mobilenet_conf = self.predict_with_mobilenet(image)
                results['mobilenet'] = {
                    'prediction': mobilenet_pred,
                    'confidence': mobilenet_conf
                }
            except Exception as e:
                print(f"MobileNetV2预测失败: {e}")
                results['mobilenet'] = {'error': str(e)}
        
        # SVM预测
        if self.svm_loaded:
            try:
                svm_pred, svm_conf = self.predict_with_svm(image)
                results['svm'] = {
                    'prediction': svm_pred,
                    'confidence': svm_conf
                }
            except Exception as e:
                print(f"SVM预测失败: {e}")
                results['svm'] = {'error': str(e)}
        
        # 如果两个模型都成功，计算集成结果
        if 'mobilenet' in results and 'svm' in results and \
           'prediction' in results['mobilenet'] and 'prediction' in results['svm']:
            
            # 简单的加权平均（基于置信度）
            mobilenet_weight = results['mobilenet']['confidence']
            svm_weight = results['svm']['confidence']
            total_weight = mobilenet_weight + svm_weight
            
            if results['mobilenet']['prediction'] == results['svm']['prediction']:
                # 两个模型预测一致
                final_prediction = results['mobilenet']['prediction']
                final_confidence = max(mobilenet_weight, svm_weight)
            else:
                # 两个模型预测不一致，选择置信度更高的
                if mobilenet_weight > svm_weight:
                    final_prediction = results['mobilenet']['prediction']
                    final_confidence = mobilenet_weight
                else:
                    final_prediction = results['svm']['prediction']
                    final_confidence = svm_weight
            
            results['ensemble'] = {
                'prediction': final_prediction,
                'confidence': final_confidence,
                'agreement': results['mobilenet']['prediction'] == results['svm']['prediction']
            }
        
        return results
    
    def get_model_status(self):
        """
        获取模型加载状态
        
        Returns:
            dict: 模型状态信息
        """
        return {
            'mobilenet_loaded': self.mobilenet_loaded,
            'svm_loaded': self.svm_loaded,
            'device': str(self.device),
            'available_models': []
        }
    
    def batch_predict(self, images, model_type='mobilenet'):
        """
        批量预测
        
        Args:
            images: 图像列表
            model_type: 模型类型
            
        Returns:
            list: 预测结果列表
        """
        results = []
        for image in images:
            try:
                pred, conf = self.predict(image, model_type)
                results.append({'prediction': pred, 'confidence': conf, 'error': None})
            except Exception as e:
                results.append({'prediction': None, 'confidence': None, 'error': str(e)})
        
        return results
```

#text(size: 10.5pt, weight: "bold")[模块main_window.py]
#text(size: 10.5pt)[（功能：主窗口UI实现，构建图形用户界面）]

```python
"""
主窗口UI实现
图像处理与识别软件的主界面
"""

import sys
import os
import cv2
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QTabWidget, QSlider,
    QSpinBox, QComboBox, QTextEdit, QGroupBox,
    QGridLayout, QApplication, QSplitter, QMessageBox,
    QFileDialog, QProgressBar, QCheckBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon

from utils.image_utils import numpy_to_qpixmap, qpixmap_to_numpy
from ml.dataset import MNISTManager
from ml.predict import ImagePredictor
from core.basic_ops import rotate_image, scale_image, crop_image, flip_image
from core.filters import apply_gaussian_blur, apply_mean_blur, apply_median_blur, apply_bilateral_filter, apply_sharpen_filter, apply_emboss_filter
from core.morphology import apply_morphology
from core.features import detect_edges, sobel_edge_detection, laplacian_edge_detection, calc_histogram, calc_histogram_equalization, calc_clahe, create_histogram_plot, create_color_histogram_plot


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
        
        # 添加标志位防止重复缩放
        self._is_resizing = False
        
        # 初始化数据集管理器和预测器
        self.mnist_manager = None
        self.predictor = None
        self.dataset_type = 'train'  # 默认使用训练集
        
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
        
        # 连接所有信号和槽
        self.connect_signals()
    
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
        self.index_spinbox.setMinimumWidth(80)
        self.index_spinbox.setMinimumHeight(25)
        self.index_spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
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
        self.scale_spinbox.setMinimumWidth(100)
        self.scale_spinbox.setMinimumHeight(25)
        self.scale_spinbox.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        
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
        
        # 设置裁剪SpinBox的最小尺寸
        for spinbox in [self.crop_x_spinbox, self.crop_y_spinbox, self.crop_w_spinbox, self.crop_h_spinbox]:
            spinbox.setMinimumWidth(60)
            spinbox.setMinimumHeight(25)
            spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        
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
        self.kernel_group = QGroupBox("核大小")
        kernel_layout = QVBoxLayout(self.kernel_group)
        
        self.kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.kernel_slider.setRange(1, 31)
        self.kernel_slider.setValue(5)
        self.kernel_slider.setSingleStep(2)  # 只允许奇数
        self.kernel_label = QLabel("核大小: 5")
        
        kernel_layout.addWidget(self.kernel_label)
        kernel_layout.addWidget(self.kernel_slider)
        layout.addWidget(self.kernel_group)
        
        # 高级参数
        self.params_group = QGroupBox("高级参数")
        params_layout = QGridLayout(self.params_group)
        
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setRange(0.1, 10.0)
        self.sigma_spinbox.setValue(1.0)
        self.sigma_spinbox.setSingleStep(0.1)
        self.sigma_spinbox.setMinimumWidth(80)
        self.sigma_spinbox.setMinimumHeight(25)
        self.sigma_spinbox.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        
        params_layout.addWidget(QLabel("Sigma:"), 0, 0)
        params_layout.addWidget(self.sigma_spinbox, 0, 1)
        
        layout.addWidget(self.params_group)
        
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
        
        # 初始化参数显示状态
        self.update_filter_params_visibility()
        
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
        self.morph_params_group = QGroupBox("参数")
        params_layout = QGridLayout(self.morph_params_group)
        
        self.morph_kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.morph_kernel_slider.setRange(1, 31)
        self.morph_kernel_slider.setValue(5)
        self.morph_kernel_slider.setSingleStep(2)
        self.morph_kernel_label = QLabel("核大小: 5")
        
        self.morph_iter_spinbox = QSpinBox()
        self.morph_iter_spinbox.setRange(1, 10)
        self.morph_iter_spinbox.setValue(1)
        self.morph_iter_spinbox.setMinimumWidth(60)
        self.morph_iter_spinbox.setMinimumHeight(25)
        self.morph_iter_spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        
        params_layout.addWidget(self.morph_kernel_label, 0, 0)
        params_layout.addWidget(self.morph_kernel_slider, 0, 1)
        params_layout.addWidget(QLabel("迭代次数:"), 1, 0)
        params_layout.addWidget(self.morph_iter_spinbox, 1, 1)
        
        layout.addWidget(self.morph_params_group)
        
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
        
        # 初始化参数显示状态
        self.update_morph_params_visibility()
        
        return tab
    
    def create_feature_tab(self):
        """创建特征提取选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 边缘检测
        edge_group = QGroupBox("边缘检测")
        edge_layout = QVBoxLayout(edge_group)
        
        # 边缘检测方法选择
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("检测方法:"))
        self.edge_method_combo = QComboBox()
        self.edge_method_combo.addItems(["Canny边缘检测", "Sobel边缘检测", "拉普拉斯边缘检测"])
        method_layout.addWidget(self.edge_method_combo)
        edge_layout.addLayout(method_layout)
        
        # Canny边缘检测参数
        self.canny_params_group = QGroupBox("Canny参数")
        canny_layout = QGridLayout(self.canny_params_group)
        
        self.low_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.low_threshold_slider.setRange(0, 255)
        self.low_threshold_slider.setValue(50)
        self.low_threshold_label = QLabel("低阈值: 50")
        
        self.high_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.high_threshold_slider.setRange(0, 255)
        self.high_threshold_slider.setValue(150)
        self.high_threshold_label = QLabel("高阈值: 150")
        
        canny_layout.addWidget(self.low_threshold_label, 0, 0)
        canny_layout.addWidget(self.low_threshold_slider, 0, 1)
        canny_layout.addWidget(self.high_threshold_label, 1, 0)
        canny_layout.addWidget(self.high_threshold_slider, 1, 1)
        
        edge_layout.addWidget(self.canny_params_group)
        
        # Sobel边缘检测参数
        self.sobel_params_group = QGroupBox("Sobel参数")
        sobel_layout = QGridLayout(self.sobel_params_group)
        
        sobel_layout.addWidget(QLabel("核大小:"), 0, 0)
        self.sobel_ksize_combo = QComboBox()
        self.sobel_ksize_combo.addItems(["1", "3", "5", "7"])
        self.sobel_ksize_combo.setCurrentIndex(1)  # 默认选择3
        sobel_layout.addWidget(self.sobel_ksize_combo, 0, 1)
        
        edge_layout.addWidget(self.sobel_params_group)
        
        # 拉普拉斯边缘检测参数
        self.laplacian_params_group = QGroupBox("拉普拉斯参数")
        laplacian_layout = QGridLayout(self.laplacian_params_group)
        
        laplacian_layout.addWidget(QLabel("核大小:"), 0, 0)
        self.laplacian_ksize_combo = QComboBox()
        self.laplacian_ksize_combo.addItems(["1", "3", "5", "7"])
        self.laplacian_ksize_combo.setCurrentIndex(1)  # 默认选择3
        laplacian_layout.addWidget(self.laplacian_ksize_combo, 0, 1)
        
        edge_layout.addWidget(self.laplacian_params_group)
        
        self.edge_btn = QPushButton("应用边缘检测")
        edge_layout.addWidget(self.edge_btn)
        
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
        
        # 初始化参数显示状态
        self.update_edge_params_visibility()
        
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
                color: #666666;
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
                background-color: #f8f8f8;
                border: 1px solid #cccccc;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                color: #333333;
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
        
        # 缩放图像以适应显示区域，但保持原始尺寸比例
        # 获取图像显示区域的大小
        label_size = self.image_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            # 计算缩放比例，确保图像能完整显示在标签内
            scale_w = label_size.width() / pixmap.width()
            scale_h = label_size.height() / pixmap.height()
            scale = min(scale_w, scale_h)  # 使用较小的缩放比例，确保图像完全适应显示区域
            
            # 应用缩放
            scaled_pixmap = pixmap.scaled(
                label_size.width(),
                label_size.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        else:
            # 如果显示区域大小无效，使用原始图像
            scaled_pixmap = pixmap
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def update_status(self):
        """更新状态信息"""
        # 这里可以添加状态更新逻辑
        pass
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        
        # 防止重复调用
        if hasattr(self, '_is_resizing') and self._is_resizing:
            return
        
        # 只有当窗口大小确实改变时才重新显示图像
        if hasattr(self, '_last_size') and self._last_size == event.size():
            return
        
        self._last_size = event.size()
        self._is_resizing = True
        
        # 从原始图像重新显示，而不是对已缩放的pixmap再次缩放
        if self.current_image is not None:
            self.display_image(self.current_image)
        
        # 使用定时器重置标志位，确保所有resize事件处理完毕
        QTimer.singleShot(100, lambda: setattr(self, '_is_resizing', False))
    
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
    
    def connect_signals(self):
        """连接所有信号和槽"""
        # 数据集导航按钮
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.jump_btn.clicked.connect(self.jump_to_image)
        self.load_dataset_btn.clicked.connect(self.load_dataset)
        
        # 文件操作按钮
        self.load_image_btn.clicked.connect(self.load_image)
        self.save_image_btn.clicked.connect(self.save_image)
        self.reset_btn.clicked.connect(self.reset_image)
        
        # 基础操作选项卡
        self.rotate_slider.valueChanged.connect(self.update_rotate_label)
        self.apply_basic_btn.clicked.connect(self.apply_basic_operations)
        self.flip_h_btn.clicked.connect(self.flip_horizontal)
        self.flip_v_btn.clicked.connect(self.flip_vertical)
        self.crop_btn.clicked.connect(self.apply_crop)
        
        # 滤波选项卡
        self.kernel_slider.valueChanged.connect(self.update_kernel_label)
        self.filter_combo.currentTextChanged.connect(self.update_filter_params_visibility)
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        
        # 形态学选项卡
        self.morph_kernel_slider.valueChanged.connect(self.update_morph_kernel_label)
        self.morph_combo.currentTextChanged.connect(self.update_morph_params_visibility)
        self.apply_morph_btn.clicked.connect(self.apply_morphology_operation)
        
        # 特征提取选项卡
        self.edge_method_combo.currentTextChanged.connect(self.update_edge_params_visibility)
        self.low_threshold_slider.valueChanged.connect(self.update_low_threshold_label)
        self.high_threshold_slider.valueChanged.connect(self.update_high_threshold_label)
        self.edge_btn.clicked.connect(self.apply_edge_detection)
        self.hist_btn.clicked.connect(self.show_histogram)
        self.hist_equalize_btn.clicked.connect(self.apply_histogram_equalization)
        self.otsu_btn.clicked.connect(self.apply_otsu_threshold)
        self.adaptive_btn.clicked.connect(self.apply_adaptive_threshold)
        
        # 识别选项卡
        self.load_models_btn.clicked.connect(self.load_models)
        self.recognize_btn.clicked.connect(self.recognize_image)
        
        # 应用模式切换
        self.apply_mode_checkbox.toggled.connect(self.toggle_apply_mode)
    
    # 数据集和图像加载功能
    def load_dataset(self):
        """加载MNIST数据集"""
        try:
            self.log_message("正在加载MNIST数据集...")
            self.mnist_manager = MNISTManager()
            self.dataset_size = self.mnist_manager.get_dataset_size(self.dataset_type)
            self.current_index = 0
            
            # 更新UI
            self.index_spinbox.setRange(0, self.dataset_size - 1)
            self.index_spinbox.setValue(0)
            
            # 加载第一张图像
            self.load_current_image()
            
            stats = self.mnist_manager.get_statistics()
            self.log_message(f"数据集加载成功 - 训练集: {stats['train_size']}, 测试集: {stats['test_size']}")
            
        except Exception as e:
            self.log_message(f"加载数据集失败: {str(e)}")
            self.show_error_message("加载数据集失败", str(e))
    
    def load_current_image(self):
        """加载当前索引的图像"""
        if self.mnist_manager is None:
            return
        
        try:
            image, label = self.mnist_manager.get_image_by_index(self.current_index, self.dataset_type)
            if image is not None:
                self.current_image = image
                self.original_image = image.copy()
                self.display_image(image)
                self.log_message(f"加载图像 #{self.current_index}, 标签: {label}")
            else:
                self.log_message(f"无法加载图像 #{self.current_index}")
        except Exception as e:
            self.log_message(f"加载图像失败: {str(e)}")
    
    def prev_image(self):
        """显示上一张图像"""
        if self.current_index > 0:
            self.current_index -= 1
            self.index_spinbox.setValue(self.current_index)
            self.load_current_image()
    
    def next_image(self):
        """显示下一张图像"""
        if self.current_index < self.dataset_size - 1:
            self.current_index += 1
            self.index_spinbox.setValue(self.current_index)
            self.load_current_image()
    
    def jump_to_image(self):
        """跳转到指定图像"""
        index = self.index_spinbox.value()
        if 0 <= index < self.dataset_size:
            self.current_index = index
            self.load_current_image()
        else:
            self.log_message(f"无效的图像索引: {index}")
    
    def load_image(self):
        """加载本地图像文件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图像文件", "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff);;所有文件 (*)"
            )
            
            if file_path:
                image = cv2.imread(file_path)
                if image is not None:
                    # 转换为灰度图（如果是MNIST风格的单通道图像）
                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    self.current_image = image
                    self.original_image = image.copy()
                    self.display_image(image)
                    self.log_message(f"成功加载图像: {file_path}")
                else:
                    self.log_message("无法读取图像文件")
                    self.show_error_message("加载失败", "无法读取图像文件")
        except Exception as e:
            self.log_message(f"加载图像失败: {str(e)}")
            self.show_error_message("加载失败", str(e))
    
    def save_image(self):
        """保存当前图像"""
        if self.current_image is None:
            self.log_message("没有可保存的图像")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "",
                "PNG文件 (*.png);;JPEG文件 (*.jpg);;所有文件 (*)"
            )
            
            if file_path:
                success = cv2.imwrite(file_path, self.current_image)
                if success:
                    self.log_message(f"图像已保存到: {file_path}")
                else:
                    self.log_message("保存图像失败")
                    self.show_error_message("保存失败", "无法保存图像到指定位置")
        except Exception as e:
            self.log_message(f"保存图像失败: {str(e)}")
            self.show_error_message("保存失败", str(e))
    
    def reset_image(self):
        """恢复原始图像"""
        if self.original_image is not None:
            # 确保original_image是numpy数组
            if hasattr(self.original_image, 'shape'):
                self.current_image = self.original_image.copy()
                self.display_image(self.current_image)
                self.log_message("已恢复原始图像")
            else:
                self.log_message("原始图像数据格式错误")
                self.show_error_message("操作失败", "原始图像数据格式错误")
        else:
            self.log_message("没有原始图像可恢复")
    
    # UI更新函数
    def update_rotate_label(self, value):
        """更新旋转角度标签"""
        self.rotate_label.setText(f"角度: {value}°")
        if not self.apply_mode and self.current_image is not None:
            self.apply_rotation_preview(value)
    
    def update_kernel_label(self, value):
        """更新核大小标签"""
        self.kernel_label.setText(f"核大小: {value}")
    
    def update_filter_params_visibility(self):
        """根据滤波类型更新参数可见性"""
        filter_type = self.filter_combo.currentText()
        
        # 根据滤波类型决定哪些参数可见
        if filter_type in ["高斯滤波", "均值滤波", "中值滤波"]:
            # 这些滤波需要核大小
            self.kernel_group.setVisible(True)
            # 只有高斯滤波需要sigma参数
            self.params_group.setVisible(filter_type == "高斯滤波")
        elif filter_type == "双边滤波":
            # 双边滤波不需要核大小滑块，但需要sigma参数
            self.kernel_group.setVisible(False)
            self.params_group.setVisible(True)
            # 修改sigma标签为更具体的描述
            self.params_group.setTitle("双边滤波参数")
        elif filter_type in ["锐化滤波", "浮雕滤波"]:
            # 这些滤波不需要核大小和sigma参数
            self.kernel_group.setVisible(False)
            self.params_group.setVisible(False)
    
    def update_morph_kernel_label(self, value):
        """更新形态学核大小标签"""
        self.morph_kernel_label.setText(f"核大小: {value}")
    
    def update_morph_params_visibility(self):
        """根据形态学操作类型更新参数可见性"""
        morph_type = self.morph_combo.currentText()
        
        # 所有形态学操作都需要核大小参数，所以保持显示
        self.morph_params_group.setVisible(True)
    
    def update_low_threshold_label(self, value):
        """更新低阈值标签"""
        self.low_threshold_label.setText(f"低阈值: {value}")
    
    def update_high_threshold_label(self, value):
        """更新高阈值标签"""
        self.high_threshold_label.setText(f"高阈值: {value}")
    
    def update_edge_params_visibility(self):
        """根据边缘检测方法更新参数可见性"""
        method = self.edge_method_combo.currentText()
        
        if method == "Canny边缘检测":
            self.canny_params_group.setVisible(True)
            self.sobel_params_group.setVisible(False)
            self.laplacian_params_group.setVisible(False)
        elif method == "Sobel边缘检测":
            self.canny_params_group.setVisible(False)
            self.sobel_params_group.setVisible(True)
            self.laplacian_params_group.setVisible(False)
        elif method == "拉普拉斯边缘检测":
            self.canny_params_group.setVisible(False)
            self.sobel_params_group.setVisible(False)
            self.laplacian_params_group.setVisible(True)
    
    def toggle_apply_mode(self, checked):
        """切换应用模式"""
        self.apply_mode = checked
        mode_text = "应用按钮模式" if checked else "实时预览模式"
        self.log_message(f"已切换到{mode_text}")
    
    # 错误处理和消息显示
    def show_error_message(self, title, message):
        """显示错误消息"""
        QMessageBox.critical(self, title, message)
    
    def show_info_message(self, title, message):
        """显示信息消息"""
        QMessageBox.information(self, title, message)
    
    def show_warning_message(self, title, message):
        """显示警告消息"""
        QMessageBox.warning(self, title, message)
    
    # 基础操作功能
    def apply_rotation_preview(self, angle):
        """应用旋转预览（实时模式）"""
        if self.current_image is None:
            return
        
        try:
            rotated = rotate_image(self.current_image, angle)
            self.display_image(rotated)
        except Exception as e:
            self.log_message(f"旋转预览失败: {str(e)}")
    
    def apply_basic_operations(self):
        """应用基础操作"""
        if self.current_image is None:
            self.log_message("没有可处理的图像")
            return
        
        try:
            result = self.original_image.copy()
            
            # 应用旋转
            angle = self.rotate_slider.value()
            if angle != 0:
                result = rotate_image(result, angle)
                self.log_message(f"应用旋转: {angle}°")
            
            # 应用缩放
            scale = self.scale_spinbox.value()
            if scale != 1.0:
                result = scale_image(result, scale)
                self.log_message(f"应用缩放: {scale}倍")
            
            self.current_image = result
            self.display_image(self.current_image)
            self.log_message("基础操作应用完成")
            
        except Exception as e:
            self.log_message(f"应用基础操作失败: {str(e)}")
            self.show_error_message("操作失败", str(e))
    
    def flip_horizontal(self):
        """水平翻转"""
        if self.current_image is None:
            self.log_message("没有可处理的图像")
            return
        
        try:
            self.current_image = flip_image(self.current_image, 1)
            self.display_image(self.current_image)
            self.log_message("已应用水平翻转")
        except Exception as e:
            self.log_message(f"水平翻转失败: {str(e)}")
            self.show_error_message("操作失败", str(e))
    
    def flip_vertical(self):
        """垂直翻转"""
        if self.current_image is None:
            self.log_message("没有可处理的图像")
            return
        
        try:
            self.current_image = flip_image(self.current_image, 0)
            self.display_image(self.current_image)
            self.log_message("已应用垂直翻转")
        except Exception as e:
            self.log_message(f"垂直翻转失败: {str(e)}")
            self.show_error_message("操作失败", str(e))
    
    def apply_crop(self):
        """应用裁剪"""
        if self.current_image is None:
            self.log_message("没有可处理的图像")
            return
        
        try:
            x = self.crop_x_spinbox.value()
            y = self.crop_y_spinbox.value()
            w = self.crop_w_spinbox.value()
            h = self.crop_h_spinbox.value()
            
            # 如果裁剪参数为0，使用图像尺寸的一半作为默认值
            if w == 0 or h == 0:
                height, width = self.current_image.shape[:2]
                x = width // 4
                y = height // 4
                w = width // 2
                h = height // 2
                self.crop_x_spinbox.setValue(x)
                self.crop_y_spinbox.setValue(y)
                self.crop_w_spinbox.setValue(w)
                self.crop_h_spinbox.setValue(h)
            
            self.current_image = crop_image(self.current_image, x, y, w, h)
            self.display_image(self.current_image)
            self.log_message(f"已应用裁剪: ({x},{y}) 大小: {w}x{h}")
        except Exception as e:
            self.log_message(f"裁剪失败: {str(e)}")
            self.show_error_message("操作失败", str(e))
    
    # 滤波功能
    def apply_filter(self):
        """应用滤波"""
        if self.current_image is None:
            self.log_message("没有可处理的图像")
            return
        
        try:
            filter_type = self.filter_combo.currentText()
            
            if filter_type == "高斯滤波":
                kernel_size = self.kernel_slider.value()
                sigma = self.sigma_spinbox.value()
                result = apply_gaussian_blur(self.current_image, kernel_size, sigma)
            elif filter_type == "均值滤波":
                kernel_size = self.kernel_slider.value()
                result = apply_mean_blur(self.current_image, kernel_size)
            elif filter_type == "中值滤波":
                kernel_size = self.kernel_slider.value()
                result = apply_median_blur(self.current_image, kernel_size)
            elif filter_type == "双边滤波":
                sigma = self.sigma_spinbox.value()
                result = apply_bilateral_filter(self.current_image, 9, sigma * 10, sigma * 10)
            elif filter_type == "锐化滤波":
                sigma = self.sigma_spinbox.value()
                result = apply_sharpen_filter(self.current_image, sigma)
            elif filter_type == "浮雕滤波":
                result = apply_emboss_filter(self.current_image)
            else:
                self.log_message(f"未知的滤波类型: {filter_type}")
                return
            
            self.current_image = result
            self.display_image(self.current_image)
            self.log_message(f"已应用{filter_type}")
            
        except Exception as e:
            self.log_message(f"应用滤波失败: {str(e)}")
            self.show_error_message("操作失败", str(e))
    
    # 形态学操作功能
    def apply_morphology_operation(self):
        """应用形态学操作"""
        if self.current_image is None:
            self.log_message("没有可处理的图像")
            return
        
        try:
            morph_type = self.morph_combo.currentText()
            kernel_size = self.morph_kernel_slider.value()
            iterations = self.morph_iter_spinbox.value()
            
            # 将中文操作类型转换为英文
            morph_type_map = {
                "膨胀": "dilate",
                "腐蚀": "erode",
                "开运算": "open",
                "闭运算": "close",
                "形态学梯度": "gradient",
                "顶帽变换": "tophat",
                "黑帽变换": "blackhat"
            }
            
            op_type = morph_type_map.get(morph_type, "dilate")
            
            # 确保图像是二值图像
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            # 应用阈值处理
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            
            result = apply_morphology(binary, op_type, kernel_size, iterations)
            
            self.current_image = result
            self.display_image(self.current_image)
            self.log_message(f"已应用{morph_type}")
            
        except Exception as e:
            self.log_message(f"应用形态学操作失败: {str(e)}")
            self.show_error_message("操作失败", str(e))
    
    # 特征提取功能
    def apply_edge_detection(self):
        """应用边缘检测"""
        if self.current_image is None:
            self.log_message("没有可处理的图像")
            return
        
        try:
            method = self.edge_method_combo.currentText()
            
            if method == "Canny边缘检测":
                low_threshold = self.low_threshold_slider.value()
                high_threshold = self.high_threshold_slider.value()
                result = detect_edges(self.current_image, low_threshold, high_threshold)
                self.log_message(f"已应用Canny边缘检测: 低阈值={low_threshold}, 高阈值={high_threshold}")
                
            elif method == "Sobel边缘检测":
                ksize = int(self.sobel_ksize_combo.currentText())
                grad_x, grad_y, magnitude = sobel_edge_detection(self.current_image, ksize=ksize)
                # 使用梯度幅值作为结果
                result = magnitude
                self.log_message(f"已应用Sobel边缘检测: 核大小={ksize}")
                
            elif method == "拉普拉斯边缘检测":
                ksize = int(self.laplacian_ksize_combo.currentText())
                result = laplacian_edge_detection(self.current_image, ksize=ksize)
                self.log_message(f"已应用拉普拉斯边缘检测: 核大小={ksize}")
            else:
                self.log_message(f"未知的边缘检测方法: {method}")
                return
            
            self.current_image = result
            self.display_image(self.current_image)
            
        except Exception as e:
            self.log_message(f"边缘检测失败: {str(e)}")
            self.show_error_message("操作失败", str(e))
    
    def show_histogram(self):
        """显示直方图"""
        if self.current_image is None:
            self.log_message("没有可处理的图像")
            return
        
        try:
            # 计算直方图
            if len(self.current_image.shape) == 3:
                # 彩色图像
                hist_data = calc_color_histogram(self.current_image)
                hist_image = create_color_histogram_plot(hist_data)
            else:
                # 灰度图像
                hist, bins = calc_histogram(self.current_image)
                hist_image = create_histogram_plot(hist, "灰度直方图", "像素值", "频率")
            
            # 确保hist_image是numpy数组
            if hist_image is not None and hasattr(hist_image, 'shape'):
                # 直接显示直方图，但不替换current_image
                self.display_image(hist_image)
                self.log_message("已显示直方图")
            else:
                self.log_message("直方图生成失败：返回的不是有效的图像数据")
                self.show_error_message("操作失败", "直方图生成失败：返回的不是有效的图像数据")
            
        except Exception as e:
            self.log_message(f"显示直方图失败: {str(e)}")
            self.show_error_message("操作失败", str(e))
    
    def apply_histogram_equalization(self):
        """应用直方图均衡化"""
        if self.current_image is None:
            self.log_message("没有可处理的图像")
            return
        
        try:
            result = calc_histogram_equalization(self.current_image)
            
            self.current_image = result
            self.display_image(self.current_image)
            self.log_message("已应用直方图均衡化")
            
        except Exception as e:
            self.log_message(f"直方图均衡化失败: {str(e)}")
            self.show_error_message("操作失败", str(e))
    
    def apply_otsu_threshold(self):
        """应用大津法阈值"""
        if self.current_image is None:
            self.log_message("没有可处理的图像")
            return
        
        try:
            # 确保是灰度图像
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.current_image = binary
            self.display_image(self.current_image)
            self.log_message(f"已应用大津法阈值，阈值: {threshold}")
            
        except Exception as e:
            self.log_message(f"大津法阈值失败: {str(e)}")
            self.show_error_message("操作失败", str(e))
    
    def apply_adaptive_threshold(self):
        """应用自适应阈值"""
        if self.current_image is None:
            self.log_message("没有可处理的图像")
            return
        
        try:
            # 确保是灰度图像
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            
            self.current_image = binary
            self.display_image(self.current_image)
            self.log_message("已应用自适应阈值")
            
        except Exception as e:
            self.log_message(f"自适应阈值失败: {str(e)}")
            self.show_error_message("操作失败", str(e))
    
    # 识别功能
    def load_models(self):
        """加载识别模型"""
        try:
            self.log_message("正在加载模型...")
            
            if self.predictor is None:
                self.predictor = ImagePredictor()
            
            success = self.predictor.load_models()
            
            if success:
                status = self.predictor.get_model_status()
                model_status = "模型已加载: "
                if status['mobilenet_loaded']:
                    model_status += "MobileNetV2 "
                if status['svm_loaded']:
                    model_status += "SVM+HOG "
                
                self.model_status_label.setText(model_status)
                self.model_status_label.setStyleSheet("color: green; font-weight: bold;")
                self.log_message(model_status)
            else:
                self.model_status_label.setText("模型加载失败")
                self.model_status_label.setStyleSheet("color: red; font-weight: bold;")
                self.log_message("模型加载失败，请检查模型文件是否存在")
                
        except Exception as e:
            self.log_message(f"加载模型失败: {str(e)}")
            self.show_error_message("加载失败", str(e))
    
    def recognize_image(self):
        """识别当前图像"""
        if self.current_image is None:
            self.log_message("没有可识别的图像")
            return
        
        if self.predictor is None:
            self.log_message("请先加载模型")
            self.show_warning_message("提示", "请先点击'加载模型'按钮")
            return
        
        try:
            model_type = self.model_combo.currentText()
            
            if model_type == "MobileNetV2":
                pred, conf = self.predictor.predict(self.current_image, 'mobilenet')
                self.result_label.setText(f"预测结果: {pred}")
                self.confidence_label.setText(f"置信度: {conf:.4f}")
                self.log_message(f"MobileNetV2识别结果: {pred}, 置信度: {conf:.4f}")
                
            elif model_type == "SVM+HOG":
                pred, conf = self.predictor.predict(self.current_image, 'svm')
                self.result_label.setText(f"预测结果: {pred}")
                self.confidence_label.setText(f"置信度: {conf:.4f}")
                self.log_message(f"SVM+HOG识别结果: {pred}, 置信度: {conf:.4f}")
                
            elif model_type == "集成预测":
                results = self.predictor.predict_ensemble(self.current_image)
                if 'ensemble' in results:
                    pred = results['ensemble']['prediction']
                    conf = results['ensemble']['confidence']
                    agreement = results['ensemble']['agreement']
                    self.result_label.setText(f"预测结果: {pred}")
                    self.confidence_label.setText(f"置信度: {conf:.4f}")
                    self.log_message(f"集成识别结果: {pred}, 置信度: {conf:.4f}, 模型一致: {agreement}")
                else:
                    self.result_label.setText("识别失败")
                    self.confidence_label.setText("置信度: -")
                    self.log_message("集成识别失败")
            else:
                self.log_message(f"未知的模型类型: {model_type}")
                
        except Exception as e:
            self.log_message(f"识别失败: {str(e)}")
            self.show_error_message("识别失败", str(e))
```
