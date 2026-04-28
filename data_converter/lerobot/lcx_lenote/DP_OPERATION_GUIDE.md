# DP Operation Guide

这份文档面向当前仓库的 `diffusion policy` 训练流程，重点覆盖：

- 如何启动一次新的训练
- 如何从旧 run 继续训练
- 训练 config 和 task config 应该怎么写
- 如何通过配置限制 CPU RAM 占用

## 1. 训练入口

训练入口是根目录的 [train.py](/mnt/code/dp_hirol-main/train.py)。

默认情况下，`train.py` 会从 [diffusion_policy/config](/mnt/code/dp_hirol-main/diffusion_policy/config) 读取配置，所以新训练通常不需要额外传 `--config-dir`。

常见配置分两层：

- `diffusion_policy/config/train_*.yaml`
  - 定义 workspace、policy、optimizer、dataloader、training、logging、checkpoint、hydra 输出目录等
- `diffusion_policy/config/task/*.yaml`
  - 定义任务名、`shape_meta`、`env_runner`、dataset 类型和 dataset 参数

## 2. 新训练命令

### 2.1 最常见写法

以 Hirol FR3 的 UNet 图像策略为例：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name train_hirol_fr3_unet_abs_jp.yaml \
  dataset_path=data_converter/dataset/1113_left_fr3_insert_pinboard_53ep.zarr \
  training.device=cuda:0
```

说明：

- `CUDA_VISIBLE_DEVICES=0` 控制当前进程只看到 1 张卡
- `training.device=cuda:0` 指的是“可见 GPU 列表中的第 1 张”，不是整机物理编号
- `dataset_path=...` 会传给 task config 里的 `HirolDataset`

### 2.2 指定输出目录

仓库默认把训练输出写到：

```text
data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
```

如果你想自己指定输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name train_hirol_fr3_unet_abs_jp.yaml \
  dataset_path=data_converter/dataset/1113_left_fr3_insert_pinboard_53ep.zarr \
  training.device=cuda:0 \
  hydra.run.dir=data/outputs/manual/debug_run_01
```

### 2.3 常用覆盖项

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name train_hirol_fr3_unet_abs_jp.yaml \
  dataset_path=data_converter/dataset/1113_left_fr3_insert_pinboard_53ep.zarr \
  training.device=cuda:0 \
  dataloader.batch_size=12 \
  val_dataloader.batch_size=8 \
  training.gradient_accumulate_every=4 \
  training.num_epochs=3000
```

## 3. 续训练命令

### 3.1 推荐方式

推荐直接从旧 run 的 `.hydra` 目录启动。这样会复用那次训练保存下来的完整配置。

示例：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-dir data/outputs/2026.03.20/18.10.04_train_hirol_dp_fr3_pick_N_place_unet_abs_jp_hirol_fr3_3cam_insert_tube/.hydra \
  --config-name config \
  training.max_ram_gb=13 \
  training.memory_reserve_gb=3.0
```

当前仓库已经支持这种续训方式自动做两件事：

- 把 `output_dir` 绑定回旧 run 目录
- 从旧 run 下的 `checkpoints/latest.ckpt` 自动恢复

只要旧目录里存在下面这些文件即可：

- `.../.hydra/config.yaml`
- `.../.hydra/hydra.yaml`
- `.../checkpoints/latest.ckpt`

### 3.2 续训时的注意事项

- `--config-dir` 后面的路径必须写成一整段，不能换行拆开
- 续训时 `--config-name` 应写 `config`
- 最稳妥的做法是直接使用旧 run 的 `.hydra` 目录，而不是手工重新拼一遍所有参数
- 如果旧 run 的配置里保存了 `logging.id`，W&B 会继续写回同一个 run

错误示例：

```bash
python train.py \
  --config-dir data/
  outputs/2026.03.20/.../.hydra \
  --config-name config
```

上面这种写法会被 shell 拆成两条命令。

### 3.3 续训时继续覆盖参数

即使是续训，也可以覆盖一部分参数，最常见的是内存和 dataloader：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-dir data/outputs/2026.03.20/18.10.04_train_hirol_dp_fr3_pick_N_place_unet_abs_jp_hirol_fr3_3cam_insert_tube/.hydra \
  --config-name config \
  training.max_ram_gb=13 \
  training.memory_reserve_gb=3.0 \
  dataloader.num_workers=1 \
  val_dataloader.num_workers=4
```

## 4. Config 怎么写

### 4.1 train config 的职责

一个典型训练配置见 [train_hirol_fr3_unet_abs_jp.yaml](/mnt/code/dp_hirol-main/diffusion_policy/config/train_hirol_fr3_unet_abs_jp.yaml)。

建议关注这些字段：

- `defaults`
  - 指向 task config，例如 `task: hirol_fr3_abs_jps`
- `_target_`
  - 指向 workspace 类，例如 `TrainDiffusionUnetImageWorkspace`
- `dataset_path`
  - 数据集路径，占位符通常是 `???`，建议在命令行覆盖
- `policy`
  - 模型结构和超参数
- `dataloader` / `val_dataloader`
  - batch size、worker 数量、pin memory 等
- `optimizer`
  - 学习率、weight decay 等
- `training`
  - device、resume、epoch、EMA、debug、内存预算等
- `logging`
  - W&B project、resume、name、id
- `checkpoint`
  - top-k 保存策略和 latest checkpoint
- `hydra`
  - 输出目录规则

最小 train config 骨架可以参考：

```yaml
defaults:
  - _self_
  - task: hirol_fr3_abs_jps

name: train_hirol_dp_fr3_unet_abs_jp
_target_: diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace

dataset_path: ???
task_name: ${task.name}
shape_meta: ${task.shape_meta}

policy:
  _target_: diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy
  shape_meta: ${shape_meta}

dataloader:
  batch_size: 12
  num_workers: 1

val_dataloader:
  batch_size: 8
  num_workers: 4

training:
  device: cuda:0
  resume: True
  num_epochs: 3000
  gradient_accumulate_every: 4
  max_ram_gb: 13
  memory_reserve_gb: 3.0

logging:
  project: diffusion_policy_debug
  resume: True
  mode: online
  id: null
```

### 4.2 task config 的职责

一个典型 task 配置见 [hirol_fr3_3cam_insert_tube.yaml](/mnt/code/dp_hirol-main/diffusion_policy/config/task/hirol_fr3_3cam_insert_tube.yaml)。

task config 主要负责：

- 定义观测和动作的 `shape_meta`
- 指定 `env_runner`
- 指定 dataset 类型
- 给 dataset 注入 `dataset_path`、`horizon`、`n_obs_steps` 等

最关键的是 `shape_meta` 要与真实数据一致：

- `obs` 下的每个 key 要和 zarr 里的字段对应
- `rgb` / `low_dim` 类型要写对
- `action.shape` 要和实际动作维度一致

task config 最小骨架示例：

```yaml
name: hirol_custom_task

shape_meta:
  obs:
    ee_cam_color:
      shape: [3, 128, 128]
      type: rgb
    state_ee:
      shape: [15]
      type: low_dim
  action:
    shape: [8]

env_runner:
  _target_: diffusion_policy.env_runner.hirol_fr3_runner.HirolFr3Runner
  test_start_seed: 100000

dataset:
  _target_: diffusion_policy.dataset.hirol_dataset.HirolDataset
  shape_meta: ${task.shape_meta}
  dataset_path: ${dataset_path}
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1+${n_latency_steps}'}
  pad_after: ${eval:'${n_action_steps}-1'}
  n_obs_steps: ${dataset_obs_steps}
  n_latency_steps: ${n_latency_steps}
  val_ratio: 0.02
```

## 5. 如何限制运存

这里说的“运存”在当前仓库里主要是 CPU RAM，不是显存。

内存预算入口在 `training` 下：

```yaml
training:
  max_ram_gb: 13
  memory_reserve_gb: 3.0
  enforce_process_ram_limit: False
```

含义：

- `max_ram_gb`
  - 允许训练进程使用的总 RAM 上限估计
- `memory_reserve_gb`
  - 给系统和其他进程预留的 RAM
- `effective budget`
  - 实际可用于 dataset 和 dataloader 的预算，等于 `max_ram_gb - memory_reserve_gb`

例如：

- `training.max_ram_gb=13`
- `training.memory_reserve_gb=3.0`

则有效预算约为 `10 GiB`。

### 5.1 预算生效在哪里

当前代码会把预算应用到两处：

1. `HirolDataset`
2. dataloader

#### `HirolDataset`

[hirol_dataset.py](/mnt/code/dp_hirol-main/diffusion_policy/dataset/hirol_dataset.py) 会估算：

- 整个数据集如果完整拷进内存需要多少 RAM
- 所有图片如果预加载成 `float32` 需要多少 RAM

如果超出预算，会自动关闭这些高 RAM 选项：

- `load_into_memory`
- `preload_images`

推荐的低 RAM 配置：

```yaml
task:
  dataset:
    _target_: diffusion_policy.dataset.hirol_dataset.HirolDataset
    load_into_memory: False
    preload_images: False
    use_parallel_loading: False
```

#### dataloader

[memory_budget.py](/mnt/code/dp_hirol-main/diffusion_policy/common/memory_budget.py) 会在开启 `max_ram_gb` 时自动调整：

- `num_workers`
- `prefetch_factor`
- `pin_memory`
- `persistent_workers`

也就是说，哪怕 config 里原本写了：

```yaml
dataloader:
  num_workers: 8
  pin_memory: True
  persistent_workers: True
```

一旦启用 RAM 预算，运行时也可能被自动降成更保守的值。

### 5.2 最实用的降内存顺序

如果训练容易把主机 RAM 打满，建议按这个顺序调：

1. 先加：

```bash
training.max_ram_gb=13 training.memory_reserve_gb=3.0
```

2. 保证 dataset 不做整库预载：

```yaml
load_into_memory: False
preload_images: False
use_parallel_loading: False
```

3. 降 dataloader 压力：

```bash
dataloader.num_workers=1 val_dataloader.num_workers=4
```

4. 降 batch size：

```bash
dataloader.batch_size=12 val_dataloader.batch_size=8
```

5. 用梯度累积补回等效 batch：

```bash
training.gradient_accumulate_every=4
```

6. 继续不够就降观测规模：

- 减少 `n_obs_steps`
- 降低 `task.image_shape`
- 减少相机路数

### 5.3 `enforce_process_ram_limit` 要不要开

这个字段会尝试对进程施加 `RLIMIT_AS`。

但在当前仓库里，如果是 CUDA 训练，即使你把它设为 `True`，代码也会跳过这个限制，因为它会破坏 `cuDNN/CUDA` 初始化。

所以实际建议是：

- CUDA 训练：主要依赖 dataset 和 dataloader 的预算控制
- CPU 调试：如果需要，可以再尝试 `enforce_process_ram_limit=True`

## 6. 推荐命令模板

### 6.1 新训练模板

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name train_hirol_fr3_unet_abs_jp.yaml \
  dataset_path=data_converter/dataset/1113_left_fr3_insert_pinboard_53ep.zarr \
  training.device=cuda:0 \
  training.max_ram_gb=13 \
  training.memory_reserve_gb=3.0 \
  dataloader.batch_size=12 \
  dataloader.num_workers=1 \
  val_dataloader.batch_size=8 \
  val_dataloader.num_workers=4 \
  training.gradient_accumulate_every=4
```

### 6.2 续训练模板

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-dir data/outputs/2026.03.20/18.10.04_train_hirol_dp_fr3_pick_N_place_unet_abs_jp_hirol_fr3_3cam_insert_tube/.hydra \
  --config-name config \
  training.max_ram_gb=13 \
  training.memory_reserve_gb=3.0 \
  dataloader.num_workers=1 \
  val_dataloader.num_workers=4
```

## 7. 排错速查

### 报错：`Key '_target_' is not in struct`

通常是命令写断了，尤其是 `--config-dir` 被拆成了两行。

### 报错：W&B `output_dir` 冲突

当前仓库已经修过续训逻辑。优先使用旧 run 的 `.hydra` 目录续训，不要手工拼一套新 config 再强行 `resume`。

### 日志里显示 `effective=10.00 GiB`

说明内存预算已生效，算法是：

```text
effective_budget = max_ram_gb - memory_reserve_gb
```

如果你想额外再留 `1 GiB` 安全冗余，最直接的做法是把：

```text
training.memory_reserve_gb: 3.0 -> 4.0
```

例如：

- `training.max_ram_gb=13`
- `training.memory_reserve_gb=4.0`

则有效预算约为 `9 GiB`。

### 续训时没找到 checkpoint

检查旧目录里是否存在：

```text
checkpoints/latest.ckpt
```

如果没有，就不能按“自动续训”方式恢复。

### W&B 提示 `Step xxx < yyy. Dropping entry`

这通常表示：

- 本地 checkpoint 里的 `global_step`
- W&B 云端 run 记录的 step

两边已经不同步。

这种情况不会阻止训练继续，但旧 step 的指标会被 W&B 丢弃，直到本地 step 重新追上云端 step。

如果你更关心训练本身能继续，而不是强行接回同一个 W&B 曲线，推荐续训时新开一个 W&B run：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-dir outputs/2026-03-21/15-41-59/.hydra \
  --config-name config \
  training.max_ram_gb=13 \
  training.memory_reserve_gb=4.0 \
  dataloader.num_workers=1 \
  val_dataloader.num_workers=4 \
  logging.resume=false \
  logging.id=null \
  logging.name=2026.03.21-15.41.59_resume_epoch551_safe
```

## 8. 当前这次中断训练的直接续训命令

本次已确认的中断 run 是：

```text
outputs/2026-03-21/15-41-59
```

已确认存在并可读取：

- `outputs/2026-03-21/15-41-59/checkpoints/latest.ckpt`
- `outputs/2026-03-21/15-41-59/.hydra/config.yaml`

如果你要沿用旧 W&B run，直接执行：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-dir outputs/2026-03-21/15-41-59/.hydra \
  --config-name config \
  training.max_ram_gb=13 \
  training.memory_reserve_gb=4.0 \
  dataloader.num_workers=1 \
  val_dataloader.num_workers=4
```

如果你要避免 W&B step 错位继续刷 warning，推荐执行：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-dir outputs/2026-03-21/15-41-59/.hydra \
  --config-name config \
  training.max_ram_gb=13 \
  training.memory_reserve_gb=4.0 \
  dataloader.num_workers=1 \
  val_dataloader.num_workers=4 \
  logging.resume=false \
  logging.id=null \
  logging.name=2026.03.21-15.41.59_resume_epoch551_safe
```

说明：

- 当前 checkpoint 文件大小约 `1.4G`，zip 结构可正常读取，不像是坏档。
- `latest.ckpt` 对应的训练已跑到 `epoch 551` 附近，`logs.json.txt` 最后记录到 `global_step 332810`。
- 这个 workspace 的 `training.num_epochs=50` 表示“本次续训再跑 50 个 epoch”，不是总 epoch 上限。
