本项目为Xjtu-Graphics课程实验项目
实验文档https://dandelion-docs.readthedocs.io/zh-cn

本实验我实现的部分如下：
1. 基础部分：
    - 变化矩阵：平移旋转缩放
    - 透视投影矩阵：MVP矩阵
    - 欧拉角到四元数的转换
2. 渲染部分：
    - 光栅化渲染管线：实现了软光栅渲染器
    - Whitted-Style Ray-Tracing
3. 物理模拟部分：
    - 前向欧拉法，隐式欧拉法，半隐式欧拉法，四阶龙格-库塔法 (4-th Runge-Kutta Integration)的实现
    - 朴素的碰撞检测实现。