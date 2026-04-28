# LeRobot V3 Operation Guide

本文对应当前仓库里的 `lerobot_v3` 数据链路。

当前方案是：

* 输入是 **HiROL 原生数据目录**，不是 zarr
* 输出是 **标准 LeRobot v3 目录结构**
* 训练阶段继续使用仓库内 `lerobot_v3` reader / dataset adapter 接入 `diffusion_policy`
* 原有 zarr 训练流保持不变

## 1. 依赖

新增链路只额外依赖：

* `pyarrow`

现有环境描述已经补到：

* [conda_environment.yaml](/mnt/code/dp_hirol-main/conda_environment.yaml)
* [conda_environment_real.yaml](/mnt/code/dp_hirol-main/conda_environment_real.yaml)
* [conda_environment_macos.yaml](/mnt/code/dp_hirol-main/conda_environment_macos.yaml)
* [uv-common.txt](/mnt/code/dp_hirol-main/requirements/uv-common.txt)

如果你在 Docker 里重建镜像，或在本地重建 conda 环境，就会带上 `pyarrow`。

## 2. 数据转换

把原生 HiROL 数据转换成 LeRobot v3 目录。

这里的输入不是 zarr，而是你的 **原生数据目录**，也就是包含 `episode_*` 子目录的根目录。

```bash
python data_converter/converter_lerobot_v3.py \
  --input-root /home/rei/mnt/code/dataset/1113_left_fr3_insert_pinboard_53ep \
  --output-dir /home/rei/mnt/code/dataset/1113_left_fr3_insert_pinboard_53ep_lerobot_v3
```

常用参数：

* `--fps 10`：手动指定帧率；不写则尝试从时间戳推断
* `--missing-policy zeros|skip|error`：缺图像时的处理方式
* `--no-videos`：不写 MP4，直接把图像塞进 parquet，通常只建议做调试

输出目录结构大致为：

```text
your_dataset_lerobot_v3/
  data/chunk-000/file-000.parquet
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  videos/observation.images.<camera>/chunk-000/file-000.mp4
```

`meta/info.json` 里会写入：

* `codebase_version: v3.0`
* 标准 `data_path` 模板
* 标准 `video_path` 模板
* `robot_type`
* `fps`
* `features`

## 3. 数据验证

第一层验证：用仓库内置脚本检查目录结构、元数据和训练 adapter 契约。

```bash
python data_converter/validate_lerobot_v3.py \
  --dataset-path /home/rei/mnt/code/dataset/1113_left_fr3_insert_pinboard_53ep_lerobot_v3
```

如果还想同时验证训练 adapter 输出契约：

```bash
python data_converter/validate_lerobot_v3.py \
  --dataset-path /home/rei/mnt/code/dataset/1113_left_fr3_insert_pinboard_53ep_lerobot_v3 \
  --task-config diffusion_policy/config/task_zarr/hirol_fr3_abs_jps_ee_state_lerobot_v3.yaml
```

这个脚本会检查：

* 关键元数据文件是否存在
* parquet 是否可读
* 视频文件是否存在
* `info.json` 是否使用 `v3.0` 和标准 chunk/file 路径模板
* 本地 `lerobot_v3` dataset reader 是否能读取首个 sample
* 可选：训练 adapter 是否能产出 `obs/action` batch 契约

第二层验证：如果你要验“官方 `lerobot` 是否能加载”，建议在**独立的 Python 3.10+ 环境**里做，只用于格式验收，不用于本项目训练。

示例：

```bash
python -m venv /tmp/lerobot-v3-check
source /tmp/lerobot-v3-check/bin/activate
pip install -U pip
pip install lerobot
python - <<'PY'
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id=None, root="/home/rei/mnt/code/dataset/1113_left_fr3_insert_pinboard_53ep_lerobot_v3")
print("len =", len(ds))
sample = ds[0]
print("keys =", sorted(sample.keys()))
PY
```

如果官方导入路径将来调整，以安装后的 `lerobot` 版本文档为准；核心目标是不报格式错误并能读取首个 sample。

## 4. 训练命令

使用新增 task 配置训练：

```bash
python train.py \
  --config-dir=diffusion_policy/config/train_lerobot_v3 \
  --config-name=train_hirol_fr3_unet_abs_jp_ee_state \
  dataset_path=/home/rei/mnt/code/dataset/1113_left_fr3_insert_pinboard_53ep_lerobot_v3
```

如果使用插管任务配置：

```bash
python train.py \
  --config-dir=diffusion_policy/config/train_lerobot_v3 \
  --config-name=train_hirol_fr3_3cam_insert_tube_unet \
  dataset_path=/home/rei/mnt/code/dataset/1113_left_fr3_insert_pinboard_53ep_lerobot_v3
```

切换为 timestamp 窗口采样：

```bash
python train.py \
  --config-dir=diffusion_policy/config/train_lerobot_v3 \
  --config-name=train_hirol_fr3_unet_abs_jp_ee_state \
  dataset_path=/home/rei/mnt/code/dataset/1113_left_fr3_insert_pinboard_53ep_lerobot_v3 \
  task.dataset.window_sampling_strategy=timestamp
```

## 5. 训练 Config 编写

新增 dataset adapter 是：

* [hirol_lerobot_v3_dataset.py](/mnt/code/dp_hirol-main/diffusion_policy/dataset/hirol_lerobot_v3_dataset.py)

推荐保留这些核心参数：

* `window_sampling_strategy: idx|timestamp`
* `image_feature_map`
* `lowdim_feature_groups`
* `action_feature_fields`
* `timestamp_key`
* `timestamp_step_sec`
* `timestamp_tolerance_sec`

说明：

* `window_sampling_strategy=idx` 是当前默认值，最接近原 zarr 训练语义
* `window_sampling_strategy=timestamp` 仍使用本地 adapter 的近邻时间重采样逻辑
* 是否使用哪些原子字段参与训练，由 `lowdim_feature_groups` 和 `action_feature_fields` 控制

最小示例：

```yaml
dataset:
  _target_: diffusion_policy.dataset.hirol_lerobot_v3_dataset.HirolLeRobotV3Dataset
  shape_meta: *shape_meta
  dataset_path: ${dataset_path}
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1+${n_latency_steps}'}
  pad_after: ${eval:'${n_action_steps}-1'}
  n_obs_steps: ${dataset_obs_steps}
  n_latency_steps: ${n_latency_steps}
  window_sampling_strategy: idx
  image_feature_map:
    ee_cam_color: observation.images.ee_cam_color
    third_person_cam_color: observation.images.third_person_cam_color
    side_cam_color: observation.images.side_cam_color
  lowdim_feature_groups:
    state:
      - observation.state
  action_feature_fields:
    - action
```

如果你想用更细粒度的原子字段做训练，只改 `lowdim_feature_groups` / `action_feature_fields` 即可，例如：

```yaml
lowdim_feature_groups:
  state:
    - observation.state.ee_pose
    - observation.state.joint_position
    - observation.state.gripper_width
action_feature_fields:
  - action.joint_position
  - action.gripper_width
```

## 6. 现状说明

当前实现的定位是：

* 训练环境继续保持现有 Python 3.9 链路
* 不要求训练环境安装官方 `lerobot` 包
* 输出目录按标准 LeRobot v3 路径和 metadata 组织
* 保持现有 DP 训练批次契约不变

推荐顺序：

1. 先跑 `converter_lerobot_v3.py`
2. 再跑 `validate_lerobot_v3.py`
3. 如需更强格式验收，再用独立 Python 3.10+ 环境安装官方 `lerobot`
4. 最后再切到训练
