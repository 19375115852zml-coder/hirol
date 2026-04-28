# `hirol_lerobot_v3_dataset.py` 数据流流程图

对应源码：

* [diffusion_policy/dataset/hirol_lerobot_v3_dataset.py](/mnt/code/dp_hirol-main/diffusion_policy/dataset/hirol_lerobot_v3_dataset.py)
* [data_converter/converter_lerobot_v3.py](/mnt/code/dp_hirol-main/data_converter/converter_lerobot_v3.py)
* [diffusion_policy/common/lerobot_v3_io.py](/mnt/code/dp_hirol-main/diffusion_policy/common/lerobot_v3_io.py)

这份文档讲的是**当前训练主线**的数据流：

```text
HiROL 原生数据 -> converter_lerobot_v3.py -> 标准 LeRobot v3 目录
-> HirolLeRobotV3Dataset -> DataLoader -> policy 训练
```

原有 `zarr -> HirolDataset` 训练流没有被删除，但已经不是这份文档的重点。

---

## 1. 总流程图

```mermaid
flowchart TD
    A[HiROL 原生数据目录 episode_*/data.json + 图像] --> B[converter_lerobot_v3.py]
    B --> C[HiROLEpisodeReader 逐 episode 读取]
    C --> D[抽取图像 状态 动作 timestamp task]
    D --> E[CustomLeRobotV3Writer]
    E --> F[标准 LeRobot v3 目录]

    F --> G[HirolLeRobotV3Dataset 初始化]
    G --> H[CustomLeRobotV3Dataset 打开 parquet mp4 meta]
    H --> I[加载 lowdim/action/timestamp]
    I --> J[构建 episode ranges]
    J --> K[create_indices 生成窗口索引]
    K --> L[__getitem__ 取一个序列]
    L --> M[读图像窗口 + lowdim 窗口 + action 窗口]
    M --> N[转换为 torch.Tensor]
    N --> O[DataLoader batch]
    O --> P[Diffusion Policy 训练]
```

---

## 2. 当前训练流要点

当前链路不是直接从原生数据训练，而是分成两段：

1. 离线转换  
   把原生 `episode_*/data.json` 和图像文件，整理成标准 LeRobot v3 目录。

2. 训练读取  
   `HirolLeRobotV3Dataset` 读取 LeRobot v3 数据，并继续输出当前 DP 训练需要的：

```python
{
  "obs": {
    "ee_cam_color": ...,
    "third_person_cam_color": ...,
    "side_cam_color": ...,
    "state": ...,
  },
  "action": ...
}
```

也就是说，**存储格式变了，但训练批次契约没变**。

---

## 3. 原生数据到 LeRobot v3 的转换流程

```mermaid
flowchart TD
    A[converter_lerobot_v3.py] --> B[list_episode_dirs]
    B --> C[遍历每个 episode_*]
    C --> D[HiROLEpisodeReader 打开 data.json]
    D --> E[识别 camera_keys 和 primary_stream]
    E --> F[逐 step 调用 get_step]
    F --> G[取图像 图像时间戳 图像有效位]
    G --> H[取 ee_pose joint_position gripper_width]
    H --> I[取 action 和主 timestamp]
    I --> J[组装 frame dict]
    J --> K[CustomLeRobotV3Writer.add_frame]
    K --> L[episode 结束后 save_episode]
    L --> M[全部结束后 finalize]
    M --> N[写出 parquet mp4 info stats tasks episodes]
```

---

## 4. 输出的 LeRobot v3 目录结构

```text
dataset_root/
  data/
    chunk-000/
      file-000.parquet
  meta/
    info.json
    stats.json
    tasks.parquet
    episodes/
      chunk-000/
        file-000.parquet
  videos/
    observation.images.ee_cam_color/
      chunk-000/
        file-000.mp4
    observation.images.third_person_cam_color/
      chunk-000/
        file-000.mp4
    observation.images.side_cam_color/
      chunk-000/
        file-000.mp4
```

其中：

* `data/...parquet`：逐帧低维数据表
* `videos/...mp4`：每路相机的视频
* `meta/info.json`：feature schema、路径模板、fps、robot_type
* `meta/tasks.parquet`：任务表
* `meta/episodes/...parquet`：episode 边界和长度

---

## 5. 写入的关键 feature

当前 converter 写入的训练相关字段主要有：

* `observation.images.ee_cam_color`
* `observation.images.third_person_cam_color`
* `observation.images.side_cam_color`
* `observation.state`
* `observation.state.ee_pose`
* `observation.state.joint_position`
* `observation.state.gripper_width`
* `action`
* `action.joint_position`
* `action.gripper_width`
* `timestamp`
* `episode_index`
* `frame_index`
* `index`
* `task_index`
* `next.done`

另外每路图像还会保存：

* `observation.images.<cam>.timestamp`
* `observation.images.<cam>.is_valid`

---

## 6. 训练侧初始化流程图

```mermaid
flowchart TD
    A[HirolLeRobotV3Dataset.__init__] --> B[读取 shape_meta]
    B --> C[分离 rgb_keys 和 lowdim_keys]
    C --> D[读取 image_feature_map]
    D --> E[读取 lowdim_feature_groups]
    E --> F[读取 action_feature_fields]
    F --> G[打开 CustomLeRobotV3Dataset]
    G --> H[加载 timestamp 列]
    H --> I[构建 episode_index / episode_ranges]
    I --> J[估计每个 episode 的 step_sec]
    J --> K[加载 lowdim_data]
    K --> L[加载 action_data]
    L --> M[检查 shape_meta 对齐]
    M --> N[get_val_mask / downsample_mask]
    N --> O[create_indices]
    O --> P[初始化完成]
```

这里和旧版 `HirolDataset` 的最大区别是：

* 不再先构建 `ReplayBuffer`
* 直接从 LeRobot v3 的 parquet / mp4 / meta 中取数
* 但窗口采样仍然保留了 DP 风格的 `create_indices` 语义

---

## 7. `idx` 与 `timestamp` 两种窗口策略

训练 config 可以控制：

* `window_sampling_strategy: idx`
* `window_sampling_strategy: timestamp`

它们共享同一个 batch 输出格式，只是窗口构造不同。

### `idx` 模式

```mermaid
flowchart TD
    A[sample idx] --> B[create_indices 取 buffer_start/end]
    B --> C[拼出 sequence_indices]
    C --> D[前 n_obs_steps 作为 obs_indices]
    D --> E[整个 sequence 作为 action window]
```

含义是：

* 按数组位置采样
* 最接近原 zarr 训练语义
* 默认推荐先用这个模式

### `timestamp` 模式

```mermaid
flowchart TD
    A[sample idx] --> B[先生成 idx 窗口]
    B --> C[取 anchor timestamp]
    C --> D[根据 step_sec 构造目标时间序列]
    D --> E[在当前 episode 内做 nearest search]
    E --> F[得到重定时后的 sequence_indices]
```

含义是：

* 仍然在 episode 内采样
* 但窗口会按时间近邻重对齐
* 适合后续实验时对比时序语义

---

## 8. `__getitem__` 取样流程图

```mermaid
flowchart TD
    A[DataLoader 调用 dataset idx] --> B[_sample_indices_to_sequence]
    B --> C{window_sampling_strategy}
    C -- idx --> D[直接用 sequence_indices]
    C -- timestamp --> E[_retime_sequence_indices]
    D --> F[obs_indices = 前 n_obs_steps]
    E --> F

    F --> G[处理 RGB keys]
    G --> H[逐帧从 mp4 读图像]
    H --> I[_coerce_image resize HWC->CHW /255]
    I --> J[stack 成 T C H W]

    F --> K[处理 lowdim keys]
    K --> L[从 lowdim_data 直接切片]

    D --> M[处理 action]
    E --> M
    M --> N[从 action_data 切片]
    N --> O{n_latency_steps > 0}
    O -- 是 --> P[裁掉前几步]
    O -- 否 --> Q[保持不变]
    P --> R[dict_apply -> torch]
    Q --> R
    J --> R
    L --> R
    R --> S[返回 obs/action]
```

---

## 9. 图像处理流程

```mermaid
flowchart TD
    A[从 mp4 解码得到 RGB 图像] --> B[_coerce_image]
    B --> C{是否已经是目标分辨率}
    C -- 否 --> D[cv2.resize]
    C -- 是 --> E[跳过 resize]
    D --> F[统一到 HWC]
    E --> F
    F --> G[HWC -> CHW]
    G --> H[astype float32]
    H --> I[如果像素大于1则 /255]
    I --> J[assert 输出 shape]
```

最终图像张量形状为：

```text
T_obs x C x H x W
```

---

## 10. 训练视角的数据形状变化

```mermaid
flowchart LR
    A[视频帧 H W C] --> B[_coerce_image]
    B --> C[单帧 C H W]
    C --> D[多帧 stack]
    D --> E[RGB 序列 T_obs C H W]

    F[lowdim 全局表 N D] --> G[按 obs_indices 切片]
    G --> H[lowdim 序列 T_obs D]

    I[action 全局表 N A] --> J[按 sequence_indices 切片]
    J --> K[考虑 latency 裁剪]
    K --> L[action 序列 T_action A]
```

默认这条任务里的关键形状是：

* `ee_cam_color`: `T_obs x 3 x 480 x 640`
* `third_person_cam_color`: `T_obs x 3 x 480 x 640`
* `side_cam_color`: `T_obs x 3 x 480 x 640`
* `state`: `T_obs x 15`
* `action`: `T_action x 8`

---

## 11. 最关键的对象关系图

```mermaid
flowchart TD
    A[HiROL 原生 episode 数据] --> B[converter_lerobot_v3.py]
    B --> C[LeRobot v3 目录]
    C --> D[CustomLeRobotV3Dataset]
    D --> E[HirolLeRobotV3Dataset]
    E --> F[__getitem__]
    F --> G[torch Tensor]
    G --> H[DataLoader batch]
    H --> I[模型训练]
```

你可以把它理解成：

* 原生数据：采集结果
* converter：离线标准化
* LeRobot v3 目录：训练前的数据资产
* `CustomLeRobotV3Dataset`：底层 reader
* `HirolLeRobotV3Dataset`：面向 DP 的 adapter
* `DataLoader`：批量打包
* 模型训练：真正消费数据

---

## 12. 一眼看懂版总结

如果只记当前主线，可以记这一条：

```text
native hirol -> converter_lerobot_v3.py -> LeRobot v3
-> CustomLeRobotV3Dataset -> HirolLeRobotV3Dataset
-> __getitem__ -> torch.Tensor -> DataLoader -> model
```

和旧版相比，最大的变化是：

```text
旧版: zarr -> ReplayBuffer -> SequenceSampler -> __getitem__
新版: LeRobot v3 -> 直接 reader + adapter -> __getitem__
```

但训练最终拿到的 `obs/action` 契约没有变。

---

## 13. 下一步建议

如果你还想继续顺着数据流看，最适合接着看的文件是：

1. [data_converter/hirol_reader.py](/mnt/code/dp_hirol-main/data_converter/hirol_reader.py)
2. [data_converter/converter_lerobot_v3.py](/mnt/code/dp_hirol-main/data_converter/converter_lerobot_v3.py)
3. [diffusion_policy/common/lerobot_v3_io.py](/mnt/code/dp_hirol-main/diffusion_policy/common/lerobot_v3_io.py)
4. [diffusion_policy/dataset/hirol_lerobot_v3_dataset.py](/mnt/code/dp_hirol-main/diffusion_policy/dataset/hirol_lerobot_v3_dataset.py)
