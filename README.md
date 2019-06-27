# A Font Selection Algorithm For Assisting
    基于机器学习的字体辅助选择系统
## main.py
    这是调用其它模块的主程序
## harris.py
    这是harris角点检测模块，检测字体特征尖锐程度
## feature.py
    goodFeaturesToTrack函数（Shi Tomasi）角点检测
## Fourier.py
    傅里叶算子模块，基于边缘检测后转换为Fourier算子表达边缘，便于训练
## model文件夹
    SVM训练得到的预测模型
    ![字体包](https://github.com/Rndlab/A-Font-Selection-Algorithm-For-Assisting/blob/master/Bss/8_4.jpg)
## 运行示例
  前端部分工程文件丢失了，只剩下截图
  ![示例](https://github.com/Rndlab/A-Font-Selection-Algorithm-For-Assisting/blob/master/example.png)
