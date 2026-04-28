# LeRobot v3 + ResNet + U-Net 数据与策略链路说明

本文说明当前 `lerobotv3-resnet-Unet` 链路中，数据如何从原生 HIROL episode 转成 LeRobot v3，再被 dataset/loader 送进 workspace，最后进入 `DiffusionUnetImagePolicy` 做训练加噪、推理去噪并输出 action。

当前文档对应的主配置是：

```text
diffusion_policy/config/train_lerobot_v3/train_hirol_fr3_pnp_cam_state_to_ee_unet_h16o2a8.yaml
```

对应 task 配置是：

```text
diffusion_policy/config/task_lerobot_v3/hirol_fr3_pnp_cam_state_to_ee_unet.yaml
```

## 0. 当前链路总览

```text
原生 HIROL 数据
  episode_*/data.json + 图像文件
        |
        v
data_converter/converter_lerobot_v3.py.convert_dataset
        |
        v
data_converter/hirol_reader.py.HiROLEpisodeReader.get_lerobot_frame
        |
        v
diffusion_policy/common/lerobot_v3_io.py.CustomLeRobotV3Writer.add_frame/save_episode/finalize
        |
        v
LeRobot v3 数据集
  meta/info.json
  meta/episodes/chunk-000/file-000.parquet
  meta/tasks.parquet
  data/chunk-000/file-000.parquet
  videos/<video_key>/chunk-000/file-000.mp4
        |
        v
diffusion_policy/dataset/hirol_lerobot_v3_dataset.py.HirolLeRobotV3Dataset.__getitem__
        |
        v
torch.utils.data.DataLoader
        |
        v
diffusion_policy/workspace/train_diffusion_unet_image_workspace.py.TrainDiffusionUnetImageWorkspace.run
        |
        v
diffusion_policy/policy/diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.compute_loss
        |
        v
MultiImageObsEncoder(ResNet18) + ConditionalUnet1D + DDPMScheduler
        |
        v
训练: loss
推理: DiffusionUnetImagePolicy.predict_action -> action
```

## 1. 配置入口

### train.py.main

`train.py.main` 是训练命令的 Hydra 入口。它做三件关键事：

1. `OmegaConf.resolve(cfg)` 解析 `${eval:...}`、`${task.xxx}` 等配置引用。
2. `hydra.utils.get_class(cfg._target_)` 找到 workspace 类。
3. 创建 workspace 并调用 `workspace.run()`。

当前配置中：

```yaml
_target_: diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace
```

所以 `train.py.main` 最终进入：

```text
diffusion_policy/workspace/train_diffusion_unet_image_workspace.py.TrainDiffusionUnetImageWorkspace.run
```

### train_hirol_fr3_pnp_cam_state_to_ee_unet_h16o2a8.yaml

这个文件定义当前训练链路的核心对象：

```yaml
policy:
  _target_: diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy

  noise_scheduler:
    _target_: diffusers.schedulers.scheduling_ddpm.DDPMScheduler
    num_train_timesteps: 100
    prediction_type: epsilon

  obs_encoder:
    _target_: diffusion_policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder
    rgb_model:
      _target_: diffusion_policy.model.vision.model_getter.get_resnet
      name: resnet18
      weights: IMAGENET1K_V1
    crop_shape: [202,202]
    random_crop: True
    use_group_norm: True
    share_rgb_model: True
    imagenet_norm: True

  horizon: 16
  n_obs_steps: 2
  n_action_steps: 8
  num_inference_steps: 100
  obs_as_global_cond: True
  diffusion_step_embed_dim: 128
  down_dims: [128, 256, 512]
  kernel_size: 5
```

含义：

- `DiffusionUnetImagePolicy` 是最终 policy。
- `DDPMScheduler` 负责训练时加噪、推理时一步步去噪。
- `MultiImageObsEncoder` 负责把多相机图像和低维状态编码成一个 observation feature。
- `get_resnet(name=resnet18)` 创建 ResNet18，并把最后的 `fc` 替换成 `Identity`，输出 512 维视觉特征。
- `ConditionalUnet1D` 是真正预测噪声的 1D U-Net。

### hirol_fr3_pnp_cam_state_to_ee_unet.yaml

这个 task 配置定义数据字段和 shape：

```yaml
shape_meta:
  obs:
    ee_cam_color:
      shape: [3, 224, 224]
      type: rgb
    third_person_cam_color:
      shape: [3, 224, 224]
      type: rgb
    side_cam_color:
      shape: [3, 224, 224]
      type: rgb
    state_ee:
      shape: [15]
      type: low_dim
  action:
    shape: [8]
```

当前链路的数据选择是：

- 观测图像：`ee_cam_color`、`third_person_cam_color`、`side_cam_color`
- 低维观测：`state_ee`，15 维，内容是 `ee_pose(7) + joint_position(7) + gripper_width(1)`
- action：8 维，内容是 `action.ee_pose(7) + action.gripper_width(1)`

对应配置：

```yaml
image_feature_map:
  ee_cam_color: observation.images.ee_cam_color
  third_person_cam_color: observation.images.third_person_cam_color
  side_cam_color: observation.images.side_cam_color

lowdim_feature_groups:
  state_ee:
    - observation.state

action_feature_fields:
  - action.ee_pose
  - action.gripper_width
```

注意：虽然 converter 会写入 `action.joint_position`，但当前 task 的 policy action 只使用 `action.ee_pose + action.gripper_width`，所以 action_dim 是 8。

## 2. 原生 HIROL -> LeRobot v3

### 原生 HIROL 数据结构

原生数据通常是：

```text
<input_root>/
  episode_000/
    data.json
    <image files>
  episode_001/
    data.json
    <image files>
```

每个 `episode_*/data.json` 里包含：

- `info`
- `text`
- `data`

其中 `data` 是 step 列表。每个 step 里可能包含：

- `colors`: 相机图像路径和时间戳
- `joint_states`: 关节状态
- `ee_states`: 末端位姿
- `tools`: 夹爪或工具状态
- `actions`: 动作，包括 joint、ee、tool

### converter_lerobot_v3.py.main

`data_converter/converter_lerobot_v3.py.main` 解析命令行参数，然后调用：

```text
data_converter/converter_lerobot_v3.py.convert_dataset
```

常见参数：

```bash
python data_converter/converter_lerobot_v3.py \
  --input-root data/pnp_30_ep/pick_and_place \
  --output-dir data/pnp_30_ep/pick_and_place_lerobotv3 \
  --missing-policy zeros \
  --robot-type fr3
```

### converter_lerobot_v3.py.convert_dataset

`convert_dataset` 是转换主流程。它做这些事：

1. 调用 `data_converter/hirol_reader.py.HiROLEpisodeReader.list_episode_dirs` 找到所有 `episode_*` 目录。
2. 创建第一个 `HiROLEpisodeReader` 推断图像尺寸和相机 key。
3. 如果没有显式传 `fps`，调用 `converter_lerobot_v3.py._infer_fps` 从时间戳估计 fps。
4. 调用 `converter_lerobot_v3.py._build_feature_spec` 定义 LeRobot v3 的 features。
5. 创建 `diffusion_policy/common/lerobot_v3_io.py.CustomLeRobotV3Writer`。
6. 遍历每个 episode，每个 step 调用 `HiROLEpisodeReader.get_lerobot_frame`。
7. 每一帧调用 `CustomLeRobotV3Writer.add_frame` 写入 buffer、parquet 或视频。
8. 每个 episode 结束调用 `CustomLeRobotV3Writer.save_episode` 记录 episode 边界。
9. 所有 episode 结束调用 `CustomLeRobotV3Writer.finalize` 写出 meta、parquet、视频和 stats。

### converter_lerobot_v3.py._build_feature_spec

`_build_feature_spec` 定义 LeRobot v3 输出数据有哪些列。

当前会写入的重要字段：

```text
timestamp
episode_index
frame_index
index
task_index
next.done

observation.state                         # 15维: ee_pose + joint_position + gripper_width
observation.state.ee_pose                 # 7维
observation.state.joint_position          # 7维
observation.state.gripper_width           # 1维

action                                    # 15维: action_ee + action_joint + action_gripper
action.ee_pose                            # 7维
action.joint_position                     # 7维
action.gripper_width                      # 1维

observation.images.<camera_key>           # 视频或图像
observation.images.<camera_key>.timestamp
observation.images.<camera_key>.is_valid
```

### hirol_reader.py.HiROLEpisodeReader.__init__

`HiROLEpisodeReader.__init__` 读取一个 episode：

1. 打开 `<episode_dir>/data.json`。
2. 读取 `info`、`text`、`data`。
3. 调用 `_infer_cameras` 推断相机 key。
4. 调用 `_infer_stream_keys` 推断 `joint_states`、`ee_states`、`tools`、`actions` 里的 role。
5. 根据 episode 路径和文本信息选择 `primary_stream`。

`primary_stream` 的作用：如果一个 step 有多个 role，比如 left/right/head/single，后续会优先选择当前 episode 的主 stream。

### hirol_reader.py.HiROLEpisodeReader.get_lerobot_frame

`get_lerobot_frame` 是从原生 step 生成 LeRobot frame 的核心函数。

它输入：

```text
index
fallback_timestamp
episode_index
task_index
fill_missing
image_shape
image_color_space
```

它输出一个 dict，供 writer 写入：

```python
{
    "episode_index": np.asarray([episode_index], dtype=np.int64),
    "task_index": np.asarray([task_index], dtype=np.int64),
    "timestamp": np.asarray([primary_timestamp], dtype=np.float32),
    "observation.images.<cam>": image,
    "observation.images.<cam>.timestamp": ...,
    "observation.images.<cam>.is_valid": ...,
    "observation.state": ee_pose + joint_position + gripper_width,
    "observation.state.ee_pose": ee_pose,
    "observation.state.joint_position": joint_position,
    "observation.state.gripper_width": gripper_width,
    "action": action_ee + action_joint + action_gripper,
    "action.ee_pose": action_ee,
    "action.joint_position": action_joint,
    "action.gripper_width": action_gripper,
}
```

关键操作：

- `hirol_reader.py.HiROLEpisodeReader._path_exists` 判断图像是否存在。
- `hirol_reader.py.load_rgb_image` 或 `hirol_reader.py.load_bgr_image` 读取图像。
- `hirol_reader.py.HiROLEpisodeReader.extract_primary_timestamp` 选择主时间戳。
- `hirol_reader.py._as_float_vector` 把 pose/joint 转成固定长度 float32 向量。
- `hirol_reader.py._as_float_scalar_array` 把 gripper 转成 1 维 float32。

### lerobot_v3_io.py.CustomLeRobotV3Writer.add_frame

`CustomLeRobotV3Writer.add_frame` 接收 `get_lerobot_frame` 的输出，并把每个 feature 放进内部 buffer。

对视频字段：

- 如果 feature 是 `video` 且在 `video_keys` 里，就调用 `_get_video_writer` 获取 OpenCV VideoWriter。
- 调用 `_write_video_frame` 把图像写入 mp4。
- 在 parquet 里不直接存图像，而是存：

```text
observation.images.<cam>.__video_file__
observation.images.<cam>.__frame_index__
```

对非视频字段：

- 转成 feature spec 指定的 dtype。
- 放入 `_buffer_columns`。
- 对数值字段调用 `_vector_stats_update` 累计统计量。

### lerobot_v3_io.py.CustomLeRobotV3Writer.save_episode

`save_episode` 在一个 episode 写完后记录边界：

```text
episode_index
start_frame_index
end_frame_index
from_index
to_index
length
task
task_index
chunk_index
file_index
```

这些 episode 边界后面会被 dataset 用来做“不跨 episode”的窗口采样。

### lerobot_v3_io.py.CustomLeRobotV3Writer.finalize

`finalize` 是转换结束的收尾函数：

1. 释放所有 video writer。
2. `_flush_buffer` 把剩余数据写入 `data/chunk-000/file-000.parquet`。
3. 写 `meta/info.json`，里面包含 features、video_keys、fps、路径模板。
4. 写 `meta/episodes/chunk-000/file-000.parquet`，里面包含 episode 边界。
5. 写 `meta/tasks.parquet` 和 `meta/tasks.jsonl`。
6. 写 `meta/stats.json`，里面是数值字段统计。

## 3. LeRobot v3 -> Dataset

### lerobot_v3_io.py.CustomLeRobotV3Dataset.__init__

`CustomLeRobotV3Dataset` 是本仓库自定义的 LeRobot v3 读取器。

初始化时：

1. 读取 `meta/info.json`。
2. 根据 `info["data_path"]` 读取 data parquet。
3. 根据 `info["episodes_path"]` 读取 episodes parquet。
4. 构造 `episode_data_index`：

```python
{
    "from": start_frame_index列表,
    "to": end_frame_index列表,
}
```

5. 构造 `frame_to_episode_index`，把每一帧映射回 episode。

### lerobot_v3_io.py.CustomLeRobotV3Dataset.__getitem__

`CustomLeRobotV3Dataset.__getitem__(idx)` 读取单帧。

对普通 parquet 字段：

- 从 parquet column 读取第 `idx` 行。
- 转成 numpy array。

对视频字段：

- 调用 `CustomLeRobotV3Dataset._read_video_frame`。
- `_read_video_frame` 找到 mp4 文件和 frame index。
- 用 OpenCV `VideoCapture` seek 到对应帧。
- 读出 BGR，再转 RGB。

输出是一个单帧 dict，例如：

```python
{
    "timestamp": ...,
    "observation.images.ee_cam_color": HWC uint8 image,
    "observation.state": shape (15,),
    "action.ee_pose": shape (7,),
    "action.gripper_width": shape (1,),
    ...
}
```

## 4. Dataset -> DataLoader batch

### hirol_lerobot_v3_dataset.py.HirolLeRobotV3Dataset.__init__

`HirolLeRobotV3Dataset` 是真正给训练用的 dataset。它把单帧 LeRobot 数据整理成固定 horizon 的训练样本。

初始化流程：

1. 从 `shape_meta["obs"]` 找出 RGB keys 和 low_dim keys。

```text
rgb_keys = ["ee_cam_color", "side_cam_color", "third_person_cam_color"]  # 内部会排序
lowdim_keys = ["state_ee"]
```

2. 根据 task 配置建立字段映射：

```text
ee_cam_color -> observation.images.ee_cam_color
third_person_cam_color -> observation.images.third_person_cam_color
side_cam_color -> observation.images.side_cam_color
state_ee -> observation.state
action -> action.ee_pose + action.gripper_width
```

3. 创建 `CustomLeRobotV3Dataset(self.dataset_path)`。
4. 调用 `_load_column("timestamp")` 读取所有 timestamp。
5. 调用 `_load_episode_index` 从 `episode_data_index` 构造每帧 episode id。
6. 调用 `_build_episode_ends` 得到每个 episode 的结束 frame index。
7. 调用 `_build_episode_ranges` 得到每个 episode 的 frame 范围。
8. 调用 `_concat_columns` 读取低维状态和 action。
9. 根据 `preload_images` 和 `load_result_add` 决定是否预加载图像或建立磁盘 cache。
10. 调用 `get_val_mask`、`downsample_mask` 拆分 train/val episode。
11. 调用 `diffusion_policy/common/sampler.py.create_indices` 生成训练窗口索引。

当前配置：

```yaml
horizon: 16
n_obs_steps: 2
n_action_steps: 8
pad_before: n_obs_steps - 1 + n_latency_steps
pad_after: n_action_steps - 1
window_sampling_strategy: idx
preload_images: true
load_result_add: ssd
```

含义：

- 每个训练样本的 action 序列长度是 `horizon=16`。
- 每个样本只取前 `n_obs_steps=2` 帧 observation。
- policy 推理时最终只输出 `n_action_steps=8` 个动作。
- `pad_before/pad_after` 允许窗口靠近 episode 起止位置时重复边界帧，避免跨 episode。

### hirol_lerobot_v3_dataset.py.HirolLeRobotV3Dataset.__getitem__

`__getitem__(idx)` 是训练样本生成的核心函数。

流程：

1. 调用 `_sample_indices_to_sequence(idx)` 把 sample index 转成全局 frame index 序列。
2. 如果 `window_sampling_strategy == "timestamp"`，再调用 `_retime_sequence_indices` 按时间戳重采样；当前配置是 `idx`，所以不走这一步。
3. 取 observation index：

```python
obs_indices = sequence_indices[: self.n_obs_steps]
```

4. 对每个 RGB key 读取 `obs_indices` 对应图像：

- 如果已经有 `image_data` cache，就调用 `read_image_result`。
- 否则调用 `_load_frame_feature` 通过 `CustomLeRobotV3Dataset.__getitem__` 读视频帧。
- `_coerce_image` 把 HWC/RGB/uint8 图像转成 CHW/float32/[0,1]，必要时 resize 到 `shape_meta` 的 `[3,224,224]`。

5. 对每个 low_dim key，从 `self.lowdim_data` 取 `obs_indices`。
6. 从 `self.action_data` 取完整 `sequence_indices` 对应动作，长度是 `horizon + n_latency_steps`。
7. 如果有 `n_latency_steps`，裁掉前面的延迟动作；当前是 0。
8. 返回 torch tensor：

```python
{
    "obs": {
        "ee_cam_color": Tensor[n_obs_steps, 3, 224, 224],
        "third_person_cam_color": Tensor[n_obs_steps, 3, 224, 224],
        "side_cam_color": Tensor[n_obs_steps, 3, 224, 224],
        "state_ee": Tensor[n_obs_steps, 15],
    },
    "action": Tensor[horizon, 8],
}
```

经过 `torch.utils.data.DataLoader` 后，一个 batch 变成：

```python
batch = {
    "obs": {
        "ee_cam_color": Tensor[B, 2, 3, 224, 224],
        "third_person_cam_color": Tensor[B, 2, 3, 224, 224],
        "side_cam_color": Tensor[B, 2, 3, 224, 224],
        "state_ee": Tensor[B, 2, 15],
    },
    "action": Tensor[B, 16, 8],
}
```

## 5. Workspace 如何把 batch 送进 policy

### train_diffusion_unet_image_workspace.py.TrainDiffusionUnetImageWorkspace.__init__

初始化 workspace 时：

1. 设置随机种子。
2. `hydra.utils.instantiate(cfg.policy)` 创建 `DiffusionUnetImagePolicy`。
3. 如果 `training.use_ema=True`，深拷贝一个 `ema_model`。
4. 创建 optimizer。

### train_diffusion_unet_image_workspace.py.TrainDiffusionUnetImageWorkspace.run

`run` 是训练主循环。

和数据有关的关键步骤：

1. `hydra.utils.instantiate(cfg.task.dataset, **dataset_kwargs)` 创建 `HirolLeRobotV3Dataset`。
2. `DataLoader(dataset, **train_dataloader_kwargs)` 创建训练 loader。
3. `dataset.get_normalizer()` 创建 normalizer。
4. `self.model.set_normalizer(normalizer)` 把 normalizer 放进 policy。
5. 如果开启 EMA，也调用 `self.ema_model.set_normalizer(normalizer)`。
6. 如果 `training.freeze_encoder=True`，调用 `_freeze_obs_encoder` 冻结 ResNet encoder。
7. 把 `self.model` 移到 `training.device`。
8. 训练循环中，取出 batch，调用：

```python
raw_loss = self.model.compute_loss(batch)
```

也就是进入：

```text
diffusion_policy/policy/diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.compute_loss
```

验证阶段也是调用 `self.model.compute_loss(batch)`。

采样评估阶段调用：

```python
result = policy.predict_action(obs_dict)
```

也就是进入：

```text
diffusion_policy/policy/diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.predict_action
```

## 6. Obs Encoder: ResNet 如何处理多相机和低维状态

### model_getter.py.get_resnet

`diffusion_policy/model/vision/model_getter.py.get_resnet` 创建 torchvision ResNet：

```python
resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
resnet.fc = torch.nn.Identity()
```

所以 ResNet18 输出是 512 维 feature。

### multi_image_obs_encoder.py.MultiImageObsEncoder.__init__

`MultiImageObsEncoder.__init__` 根据 `shape_meta` 建立 encoder：

- 对 `type: rgb` 的 key，加入 `rgb_keys`。
- 对 `type: low_dim` 的 key，加入 `low_dim_keys`。
- 如果 `share_rgb_model=True`，三个相机共用同一个 ResNet，放在 `key_model_map["rgb"]`。
- 如果 `use_group_norm=True`，把 ResNet 里的 `BatchNorm2d` 替换成 `GroupNorm`。
- 对每个 RGB key 建立 transform：Resize/RandomCrop/Normalize。

当前配置：

```text
crop_shape: [202,202]
random_crop: True
imagenet_norm: True
share_rgb_model: True
use_group_norm: True
```

训练时随机 crop；eval/predict 时 center crop。

### multi_image_obs_encoder.py.MultiImageObsEncoder.forward

输入是一个 obs dict，但在 policy 中会先把时间维压平：

```python
this_nobs = {
    key: Tensor[B * n_obs_steps, ...]
}
```

`MultiImageObsEncoder.forward` 对 `share_rgb_model=True` 的处理：

1. 对每个 RGB key 做 crop 和 ImageNet normalize。
2. 把三个相机图像沿 batch 维拼起来：

```text
imgs: Tensor[3 * B * n_obs_steps, 3, 202, 202]
```

3. 一次送入共享 ResNet：

```text
feature: Tensor[3 * B * n_obs_steps, 512]
```

4. reshape 回每个样本的多相机特征并拼接：

```text
rgb feature: Tensor[B * n_obs_steps, 3 * 512]
```

5. 对 low_dim key，直接取 `state_ee`：

```text
state_ee: Tensor[B * n_obs_steps, 15]
```

6. 最后 `torch.cat(features, dim=-1)`：

```text
obs feature: Tensor[B * n_obs_steps, 1551]
```

其中：

```text
1551 = 3 cameras * 512 ResNet feature + 15 lowdim state
```

### multi_image_obs_encoder.py.MultiImageObsEncoder.output_shape

`DiffusionUnetImagePolicy.__init__` 会调用 `obs_encoder.output_shape()` 推断 `obs_feature_dim`。

当前链路大约是：

```text
obs_feature_dim = 1551
```

如果相机数量、ResNet 类型或 low_dim 维度变了，这个值也会变。

## 7. Policy 初始化：创建 U-Net、mask 和 scheduler

### diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.__init__

初始化时读取：

```python
action_dim = shape_meta["action"]["shape"][0]  # 当前是 8
obs_feature_dim = obs_encoder.output_shape()[0]  # 当前约 1551
```

因为当前 `obs_as_global_cond=True`：

```python
input_dim = action_dim                 # 8
global_cond_dim = obs_feature_dim * n_obs_steps  # 1551 * 2 = 3102
```

然后创建：

```text
diffusion_policy/model/diffusion/conditional_unet1d.py.ConditionalUnet1D
```

当前 U-Net 配置：

```text
input_dim = 8
global_cond_dim = 3102
diffusion_step_embed_dim = 128
down_dims = [128, 256, 512]
kernel_size = 5
n_groups = 8
cond_predict_scale = True
```

同时创建：

```text
diffusion_policy/model/diffusion/mask_generator.py.LowdimMaskGenerator
```

因为 `obs_as_global_cond=True`，所以：

```python
obs_dim = 0
action_visible = False
```

这意味着训练和推理中的 trajectory 只包含 action，不把 observation 拼进 trajectory。observation 通过 `global_cond` 注入 U-Net。

## 8. 训练时的噪声处理

训练入口：

```text
diffusion_policy/policy/diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.compute_loss
```

### 8.1 normalizer

第一步归一化：

```python
nobs = self.normalizer.normalize(batch["obs"])
nactions = self.normalizer["action"].normalize(batch["action"])
```

normalizer 来自：

```text
diffusion_policy/dataset/hirol_lerobot_v3_dataset.py.HirolLeRobotV3Dataset.get_normalizer
```

它会：

- 对 action 调用 `SingleFieldLinearNormalizer.create_fit(self.action_data)`。
- 对 lowdim obs 调用 `SingleFieldLinearNormalizer.create_fit(self.lowdim_data[key])`。
- 对 RGB obs 调用 `get_image_range_normalizer()`，把 `[0,1]` 图像线性映射到 `[-1,1]`。

### 8.2 observation 编码成 global_cond

因为当前 `obs_as_global_cond=True`，`compute_loss` 只取前 `n_obs_steps=2` 帧 observation：

```python
this_nobs = dict_apply(
    nobs,
    lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
)
nobs_features = self.obs_encoder(this_nobs)
global_cond = nobs_features.reshape(batch_size, -1)
```

形状变化：

```text
batch["obs"][key]: Tensor[B, 2, ...]
this_nobs[key]:    Tensor[B*2, ...]
nobs_features:     Tensor[B*2, 1551]
global_cond:       Tensor[B, 3102]
```

### 8.3 clean trajectory

当前 trajectory 是归一化后的 action：

```python
trajectory = nactions
cond_data = trajectory
```

形状：

```text
trajectory: Tensor[B, 16, 8]
```

### 8.4 mask_generator

调用：

```python
condition_mask = self.mask_generator(trajectory.shape)
```

因为当前 `obs_as_global_cond=True` 且 `obs_dim=0`、`action_visible=False`，所以 condition mask 基本全 False。

含义：

- 没有 observation 被拼进 action trajectory。
- 没有已知 action 需要强制保持。
- U-Net 需要对整个 action trajectory 学习去噪。

如果 `obs_as_global_cond=False`，trajectory 会变成 `action + obs_feature`，mask 会把前几个 obs step 的 obs_feature 固定住，这就是 inpainting 条件方式。

### 8.5 采样噪声和 timestep

`compute_loss` 随机生成噪声：

```python
noise = torch.randn(trajectory.shape, device=trajectory.device)
```

随机采样每个 batch 元素对应的扩散时间步：

```python
timesteps = torch.randint(
    0,
    self.noise_scheduler.config.num_train_timesteps,
    (bsz,),
    device=trajectory.device,
).long()
```

当前 `num_train_timesteps=100`，所以 timestep 在 `[0, 99]`。

### 8.6 DDPMScheduler.add_noise

调用：

```python
noisy_trajectory = self.noise_scheduler.add_noise(
    trajectory,
    noise,
    timesteps
)
```

这是前向扩散过程，把 clean action 加噪成 noisy action：

```text
clean action trajectory + noise + timestep -> noisy trajectory
```

形状不变：

```text
noisy_trajectory: Tensor[B, 16, 8]
```

### 8.7 ConditionalUnet1D 预测噪声

调用：

```python
pred = self.model(
    noisy_trajectory,
    timesteps,
    local_cond=None,
    global_cond=global_cond,
)
```

也就是进入：

```text
diffusion_policy/model/diffusion/conditional_unet1d.py.ConditionalUnet1D.forward
```

由于 scheduler 配置是：

```yaml
prediction_type: epsilon
```

所以 U-Net 的目标是预测刚才加进去的 `noise`。

### 8.8 loss

`compute_loss` 中：

```python
target = noise
loss = F.mse_loss(pred, target, reduction="none")
loss = loss * loss_mask.type(loss.dtype)
loss = reduce(loss, "b ... -> b (...)", "mean")
loss = loss.mean()
```

训练目标：

```text
让 ConditionalUnet1D(noisy_action, timestep, obs_global_cond) 预测出噪声 epsilon
```

最终返回一个 scalar loss 给 workspace：

```text
TrainDiffusionUnetImageWorkspace.run -> raw_loss.backward() -> optimizer.step()
```

## 9. ConditionalUnet1D 内部做了什么

### conditional_unet1d.py.ConditionalUnet1D.__init__

U-Net 初始化时：

1. 用 `SinusoidalPosEmb(diffusion_step_embed_dim)` 编码 timestep。
2. 经过 MLP 得到 diffusion step feature。
3. 如果有 `global_cond`，就把 timestep feature 和 observation feature 拼起来：

```text
global_feature = timestep_embedding + global_cond
```

当前 cond 维度：

```text
cond_dim = diffusion_step_embed_dim + global_cond_dim
         = 128 + 3102
         = 3230
```

4. 根据 `down_dims=[128,256,512]` 创建 down path、mid modules、up path。
5. 每个 block 是：

```text
conditional_unet1d.py.ConditionalResidualBlock1D
```

其中每个 residual block 内部包含：

```text
conv1d_components.py.Conv1dBlock
```

### conditional_unet1d.py.ConditionalResidualBlock1D.forward

`ConditionalResidualBlock1D` 是带条件调制的残差块。

输入：

```text
x:    Tensor[B, channels, horizon]
cond: Tensor[B, cond_dim]
```

流程：

1. `Conv1dBlock` 做一层 `Conv1d -> GroupNorm -> Mish`。
2. `cond_encoder` 把 cond 映射到 channel 维。
3. 如果 `cond_predict_scale=True`，cond 会生成 scale 和 bias：

```python
out = scale * out + bias
```

4. 再过第二个 `Conv1dBlock`。
5. 加 residual connection。

这就是 observation condition 和 timestep condition 注入 U-Net 的主要位置。

### conditional_unet1d.py.ConditionalUnet1D.forward

输入：

```text
sample:      Tensor[B, horizon, input_dim] = Tensor[B, 16, 8]
timestep:    Tensor[B] or int
global_cond: Tensor[B, 3102]
```

内部第一步：

```python
sample = einops.rearrange(sample, "b h t -> b t h")
```

把形状从：

```text
Tensor[B, 16, 8]
```

变成 1D Conv 需要的：

```text
Tensor[B, 8, 16]
```

然后：

1. `diffusion_step_encoder(timesteps)` 得到 timestep embedding。
2. 拼接 `global_cond`。
3. down path 做多层残差卷积和下采样。
4. mid modules 做 bottleneck 处理。
5. up path 和 skip connection 拼接后上采样。
6. `final_conv` 输出和 input_dim 一样的通道数。
7. rearrange 回：

```text
Tensor[B, 16, 8]
```

输出含义：

- 训练时：预测噪声 `epsilon`。
- 推理时：每一步预测当前 noisy action 里的噪声，用 scheduler 去掉它。

## 10. 推理时从 observation 到 action 输出

推理入口：

```text
diffusion_policy/policy/diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.predict_action
```

输入：

```python
obs_dict = {
    "ee_cam_color": Tensor[B, 2, 3, 224, 224],
    "third_person_cam_color": Tensor[B, 2, 3, 224, 224],
    "side_cam_color": Tensor[B, 2, 3, 224, 224],
    "state_ee": Tensor[B, 2, 15],
}
```

### 10.1 normalize observation

```python
nobs = self.normalizer.normalize(obs_dict)
```

RGB 从 `[0,1]` 到 `[-1,1]`，lowdim 根据训练集统计归一化。

### 10.2 obs_encoder 得到 global_cond

和训练时类似：

```python
this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
nobs_features = self.obs_encoder(this_nobs)
global_cond = nobs_features.reshape(B, -1)
```

当前：

```text
To = n_obs_steps = 2
global_cond: Tensor[B, 3102]
```

### 10.3 构造空 action 条件

因为 `obs_as_global_cond=True`：

```python
cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
```

当前：

```text
T = horizon = 16
Da = action_dim = 8
cond_data: Tensor[B, 16, 8]
cond_mask: Tensor[B, 16, 8] all False
```

### 10.4 conditional_sample 从纯噪声开始去噪

`predict_action` 调用：

```python
nsample = self.conditional_sample(
    cond_data,
    cond_mask,
    local_cond=None,
    global_cond=global_cond,
)
```

进入：

```text
diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.conditional_sample
```

`conditional_sample` 做：

1. 随机初始化 action trajectory：

```python
trajectory = torch.randn(size=condition_data.shape)
```

也就是：

```text
trajectory: Tensor[B, 16, 8]
```

2. 设置推理步数：

```python
scheduler.set_timesteps(self.num_inference_steps)
```

当前 `num_inference_steps=100`。

3. 对每个 timestep 循环：

```python
for t in scheduler.timesteps:
    trajectory[condition_mask] = condition_data[condition_mask]
    model_output = self.model(trajectory, t, global_cond=global_cond)
    trajectory = scheduler.step(model_output, t, trajectory).prev_sample
```

这里的含义：

- `trajectory` 当前是 noisy action。
- `ConditionalUnet1D` 根据当前 noisy action、timestep、observation condition 预测噪声。
- `DDPMScheduler.step` 根据预测噪声从 `x_t` 得到更干净的 `x_{t-1}`。
- 循环结束后得到归一化 action trajectory。

### 10.5 unnormalize 并裁出真正要执行的动作

`predict_action` 里：

```python
naction_pred = nsample[..., :Da]
action_pred = self.normalizer["action"].unnormalize(naction_pred)
```

`action_pred` 是完整 horizon 的动作：

```text
action_pred: Tensor[B, 16, 8]
```

然后裁剪出要执行的 action chunk：

```python
start = To - 1
end = start + self.n_action_steps
action = action_pred[:, start:end]
```

当前：

```text
To = 2
n_action_steps = 8
start = 1
end = 9
action: Tensor[B, 8, 8]
```

返回：

```python
{
    "action": action,           # 真正用于执行的动作片段
    "action_pred": action_pred  # 完整 horizon=16 的预测动作
}
```

## 11. 当前链路关键张量形状

按当前配置：

```text
B = batch size
horizon = 16
n_obs_steps = 2
n_action_steps = 8
action_dim = 8
rgb cameras = 3
ResNet18 feature per camera = 512
lowdim state_ee = 15
obs_feature_dim = 3*512 + 15 = 1551
global_cond_dim = 1551*2 = 3102
```

Dataset 输出单样本：

```text
obs.ee_cam_color:           [2, 3, 224, 224]
obs.third_person_cam_color: [2, 3, 224, 224]
obs.side_cam_color:         [2, 3, 224, 224]
obs.state_ee:               [2, 15]
action:                     [16, 8]
```

DataLoader 输出 batch：

```text
obs.ee_cam_color:           [B, 2, 3, 224, 224]
obs.third_person_cam_color: [B, 2, 3, 224, 224]
obs.side_cam_color:         [B, 2, 3, 224, 224]
obs.state_ee:               [B, 2, 15]
action:                     [B, 16, 8]
```

Policy 内部：

```text
this_nobs image: [B*2, 3, 224, 224]
this_nobs state: [B*2, 15]
nobs_features:  [B*2, 1551]
global_cond:    [B, 3102]
trajectory:     [B, 16, 8]
noise:          [B, 16, 8]
timesteps:      [B]
pred:           [B, 16, 8]
loss:           scalar
```

推理输出：

```text
action_pred: [B, 16, 8]
action:      [B, 8, 8]
```

## 12. 文件.类.函数速查表

### 转换阶段

```text
data_converter/converter_lerobot_v3.py.main
  解析 CLI 参数，调用 convert_dataset。

data_converter/converter_lerobot_v3.py.convert_dataset
  遍历原生 HIROL episode，把每个 step 转成 LeRobot frame，并写出 LeRobot v3 数据集。

data_converter/converter_lerobot_v3.py._build_feature_spec
  定义 LeRobot v3 features，包括 observation、action、images、timestamp、episode metadata。

data_converter/converter_lerobot_v3.py._infer_fps
  从 timestamp 差值估计 dataset fps。

data_converter/hirol_reader.py.HiROLEpisodeReader.__init__
  读取 episode/data.json，推断相机、状态/action stream 和 primary_stream。

data_converter/hirol_reader.py.HiROLEpisodeReader.get_lerobot_frame
  把一个原生 HIROL step 转成 LeRobot v3 frame dict。

diffusion_policy/common/lerobot_v3_io.py.CustomLeRobotV3Writer.add_frame
  把 frame 写入 parquet buffer；视频字段写入 mp4，并在 parquet 记录 video path/frame index。

diffusion_policy/common/lerobot_v3_io.py.CustomLeRobotV3Writer.save_episode
  记录 episode 的起止 frame index。

diffusion_policy/common/lerobot_v3_io.py.CustomLeRobotV3Writer.finalize
  写出 data parquet、video、meta/info.json、episodes parquet、tasks、stats。
```

### 数据读取阶段

```text
diffusion_policy/common/lerobot_v3_io.py.CustomLeRobotV3Dataset.__init__
  读取 LeRobot v3 meta、data parquet、episodes parquet，并建立 episode_data_index。

diffusion_policy/common/lerobot_v3_io.py.CustomLeRobotV3Dataset.__getitem__
  读取单帧 LeRobot 数据；视频字段通过 _read_video_frame 解码。

diffusion_policy/common/lerobot_v3_io.py.CustomLeRobotV3Dataset._read_video_frame
  根据 parquet 中的视频文件路径和帧号，从 mp4 读取 RGB 图像。

diffusion_policy/dataset/hirol_lerobot_v3_dataset.py.HirolLeRobotV3Dataset.__init__
  加载 LeRobot 数据，读取状态/action列，建立 episode-aware 的窗口采样 indices。

diffusion_policy/dataset/hirol_lerobot_v3_dataset.py.HirolLeRobotV3Dataset.__getitem__
  根据窗口 index 取 n_obs_steps 帧观测和 horizon 长度 action，返回训练 batch 单样本。

diffusion_policy/dataset/hirol_lerobot_v3_dataset.py.HirolLeRobotV3Dataset.get_normalizer
  根据训练数据创建 action、lowdim、image normalizer。
```

### Workspace 阶段

```text
train.py.main
  Hydra 入口，实例化 cfg._target_ 指定的 workspace。

diffusion_policy/workspace/train_diffusion_unet_image_workspace.py.TrainDiffusionUnetImageWorkspace.__init__
  实例化 policy、EMA policy 和 optimizer。

diffusion_policy/workspace/train_diffusion_unet_image_workspace.py.TrainDiffusionUnetImageWorkspace.run
  创建 dataset/dataloader，设置 normalizer，执行训练、验证、采样和 checkpoint。

diffusion_policy/workspace/train_diffusion_unet_image_workspace.py.TrainDiffusionUnetImageWorkspace._freeze_obs_encoder
  如果 freeze_encoder=True，冻结 ResNet obs encoder。
```

### Policy 和模型阶段

```text
diffusion_policy/policy/diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.__init__
  创建 obs_encoder、ConditionalUnet1D、noise_scheduler、mask_generator，并确定 action_dim/obs_feature_dim。

diffusion_policy/policy/diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.set_normalizer
  从 dataset normalizer 加载归一化参数。

diffusion_policy/policy/diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.compute_loss
  训练入口：归一化 batch，编码 obs，给 action 加噪，U-Net 预测噪声，计算 MSE loss。

diffusion_policy/policy/diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.predict_action
  推理入口：编码 obs，从随机噪声开始采样，反归一化后输出 action。

diffusion_policy/policy/diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.conditional_sample
  推理去噪循环：scheduler.set_timesteps 后反复调用 U-Net 和 scheduler.step。

diffusion_policy/model/vision/model_getter.py.get_resnet
  创建 ResNet18/34/50，去掉分类头，输出视觉 feature。

diffusion_policy/model/vision/multi_image_obs_encoder.py.MultiImageObsEncoder.__init__
  根据 shape_meta 建立多相机 encoder、crop、normalize 和 lowdim 拼接规则。

diffusion_policy/model/vision/multi_image_obs_encoder.py.MultiImageObsEncoder.forward
  对多相机图像跑 ResNet，把 RGB feature 和 lowdim state 拼成 obs feature。

diffusion_policy/model/vision/multi_image_obs_encoder.py.MultiImageObsEncoder.output_shape
  用 dummy obs 跑一遍 forward，推断 obs feature 维度。

diffusion_policy/model/diffusion/conditional_unet1d.py.ConditionalUnet1D.__init__
  创建 timestep embedding、down path、mid modules、up path 和 final conv。

diffusion_policy/model/diffusion/conditional_unet1d.py.ConditionalUnet1D.forward
  输入 noisy trajectory、timestep、global_cond，输出预测噪声。

diffusion_policy/model/diffusion/conditional_unet1d.py.ConditionalResidualBlock1D.forward
  使用 cond 生成 scale/bias 或 bias，对 Conv1D 特征做条件调制。

diffusion_policy/model/diffusion/conv1d_components.py.Conv1dBlock.forward
  执行 Conv1d -> GroupNorm -> Mish。

diffusion_policy/model/diffusion/conv1d_components.py.Downsample1d.forward
  用 stride=2 的 Conv1d 下采样时间维。

diffusion_policy/model/diffusion/conv1d_components.py.Upsample1d.forward
  用 ConvTranspose1d 上采样时间维。

diffusion_policy/model/diffusion/mask_generator.py.LowdimMaskGenerator.forward
  生成 condition mask；当前 obs_as_global_cond=True 时基本不固定 action trajectory。
```

## 13. 最容易混淆的点

### observation.state 和 state_ee

converter 写出的字段叫：

```text
observation.state
```

task 配置里 policy 使用的 obs key 叫：

```text
state_ee
```

这两者通过 dataset 配置映射起来：

```yaml
lowdim_feature_groups:
  state_ee:
    - observation.state
```

所以 policy 看到的是 `batch["obs"]["state_ee"]`，但数据源来自 LeRobot v3 的 `observation.state`。

### action 是 8 维，不是 converter 写出的完整 15 维

converter 同时写：

```text
action: 15维
action.ee_pose: 7维
action.joint_position: 7维
action.gripper_width: 1维
```

但当前 task 配置选择：

```yaml
action_feature_fields:
  - action.ee_pose
  - action.gripper_width
```

所以训练 action 是：

```text
action_dim = 7 + 1 = 8
```

### obs_as_global_cond=True 时，observation 不在 trajectory 里

当前 policy：

```yaml
obs_as_global_cond: True
```

所以：

```text
trajectory = action only
global_cond = obs_encoder(obs[前2帧])
```

U-Net 每一层通过 condition modulation 使用 observation。

如果改成 `obs_as_global_cond=False`，逻辑会变成：

```text
trajectory = action + obs_feature
condition_mask 固定前 n_obs_steps 的 obs_feature
```

这是另一条 inpainting 条件链路。

### num_train_timesteps 和 num_inference_steps

训练加噪使用：

```yaml
noise_scheduler.num_train_timesteps: 100
```

也就是训练时随机 timestep 范围是 `[0, 99]`。

推理去噪使用：

```yaml
policy.num_inference_steps: 100
```

也就是 `conditional_sample` 里 scheduler 迭代 100 步。

两者可以不同，但当前配置相同。

### policy 输出 action 的裁剪

U-Net 推理生成完整 horizon：

```text
action_pred: [B, 16, 8]
```

真正返回给执行器的是：

```python
start = n_obs_steps - 1 = 1
end = start + n_action_steps = 9
action = action_pred[:, 1:9]
```

所以：

```text
action: [B, 8, 8]
```

这不是全部 16 步，只是从当前观测时间对齐后取出的 8 步 action chunk。

## 14. 读代码建议顺序

如果目标是完全理解代码，建议按这个顺序读：

1. `diffusion_policy/config/train_lerobot_v3/train_hirol_fr3_pnp_cam_state_to_ee_unet_h16o2a8.yaml`
2. `diffusion_policy/config/task_lerobot_v3/hirol_fr3_pnp_cam_state_to_ee_unet.yaml`
3. `data_converter/converter_lerobot_v3.py.convert_dataset`
4. `data_converter/hirol_reader.py.HiROLEpisodeReader.get_lerobot_frame`
5. `diffusion_policy/common/lerobot_v3_io.py.CustomLeRobotV3Writer`
6. `diffusion_policy/common/lerobot_v3_io.py.CustomLeRobotV3Dataset`
7. `diffusion_policy/dataset/hirol_lerobot_v3_dataset.py.HirolLeRobotV3Dataset.__getitem__`
8. `diffusion_policy/workspace/train_diffusion_unet_image_workspace.py.TrainDiffusionUnetImageWorkspace.run`
9. `diffusion_policy/policy/diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.compute_loss`
10. `diffusion_policy/model/vision/multi_image_obs_encoder.py.MultiImageObsEncoder.forward`
11. `diffusion_policy/model/diffusion/conditional_unet1d.py.ConditionalUnet1D.forward`
12. `diffusion_policy/policy/diffusion_unet_image_policy.py.DiffusionUnetImagePolicy.predict_action`

按这个顺序读，能先建立数据字段概念，再理解 batch 形状，最后理解 diffusion 训练和推理。
