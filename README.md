
## KITTI Objcet Classification
### 文件功能
1. Calibration extract_object　从数据集中提取目标点云
2. data_visualize 点云提取效果可视化
3. goup_pointcloud VexelNet中的VFELayer层
4. classify 　特征提取后的分类网络
5. model 组合group_pointcloud和classify两部分形成整体的网络
6. dataloader 对点云数据进行预处理，分块，映射加入到待处理队列中
7. train 加载数据进行训练