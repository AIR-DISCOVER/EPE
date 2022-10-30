```
conda activate epe
```

## G-Buffer 格式

尝试用 `./epe/dataset/generate_fake_gbuffers.py` 生成假的 G-Buffer，以此推断 G-Buffer 的格式。

一张图片的 G-Buffer 存在一个 `npz` 文件里，其中 key 为 data，value 的 shape 为 (720, 1080, 32)，数据类型为 float16.


我需要做什么？
1. **搞清楚 G-Buffer 的格式**，都有哪些类型的数据，然后在 Carla 中获取。[O] 5.5

撰写了 `./Carla/generate_dataset_file.py`，自动生成每行为 (img_path, robust_label_path, gbuffer_path, gt_label_path) 的 csv 文件。

+ img_path 为图像原地址
+ robust_label_path 本来应该是使用 Robust Segmentation Network 的语义分割结果，但是这里直接指向了 Ground Truth
+ gbuffer_path 为 G-Buffer 的原地址，格式为 npz
  + npz 中的 img 项为 RGB 彩色图像
  + npz 中的 gbuffers 项 目前只有 depth, #channels = 1
  + npz 中的 shader 项为 12 个 mask
+ gt_label_path 为 Ground Truth 的原地址

2. 搞 Fake Data 和 Real Data（Cityscapes），然后写成对应配置文件。[O] 5.6

3. 找 Matching，建立索引。
+ 先用 `./matching/feature_based/collect_crops.py` 生成 Crops [O] 5.7
+ 再用 `./matching/feature_based/find_knn.py` 生成 Matching [O] 5.7
+ 再用 `./matching/filter.py` 进行过滤 [O]
+ 再用 `./matching/compute_weights.py` 计算权重 [O]

4. 撰写 config 中的 yaml 文件，训练模型 [...]
5. 跑模型 [...]