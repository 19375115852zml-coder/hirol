# BCpolicy 操作文档

`BCpolicy` 放在工程根目录，和 `data_converter` 同级，用来保存独立的
Behavior Cloning policy 与训练配置。训练入口、workspace、dataset、task
配置仍复用原 `diffusion_policy` 包。

## 目录结构

```text
/home/hirol/code/diffusion_policy/
├── data_converter/
├── diffusion_policy/
└── BCpolicy/
    ├── BCpolicy/
    │   └── policy/
    │       └── bc_image_policy.py
    ├── config/
    │   ├── train_lerobot_v3/
    │   │   └── train_hirol_fr3_pnp_cam_state_to_ee_bc_h16o2a8.yaml
    │   └── train_zarr/
    │       └── train_hirol_pick_N_place_ee_state_bc.yaml
    └── README.md
```

## 运行前准备

从原 `diffusion_policy` 仓库根目录运行训练脚本：

```bash
cd /home/hirol/code/diffusion_policy
export PYTHONPATH=$PWD/BCpolicy:$PYTHONPATH
```

配置文件已经包含 Hydra search path，默认会继续从下面路径读取
`task_lerobot_v3`、`task_zarr` 等原工程配置：

```bash
/home/hirol/code/diffusion_policy/diffusion_policy/config
```

如果原工程路径变化，运行前覆盖这个环境变量：

```bash
export DIFFUSION_POLICY_CONFIG_DIR=/path/to/diffusion_policy/diffusion_policy/config
```

## LeRobot V3 BC 训练

```bash
python train.py \
  -c BCpolicy/config/train_lerobot_v3/train_hirol_fr3_pnp_cam_state_to_ee_bc_h16o2a8.yaml
```

常用覆盖项：

```bash
python train.py \
  -c BCpolicy/config/train_lerobot_v3/train_hirol_fr3_pnp_cam_state_to_ee_bc_h16o2a8.yaml \
  training.device=cuda:0 \
  dataloader.batch_size=128 \
  training.max_train_steps=1000
```

该配置默认数据路径为：

```text
data/pnp_30_ep/pick_and_place_lerobotv3
```

需要换数据集时直接覆盖：

```bash
python train.py \
  -c BCpolicy/config/train_lerobot_v3/train_hirol_fr3_pnp_cam_state_to_ee_bc_h16o2a8.yaml \
  dataset_path=/path/to/lerobot_dataset
```

## Zarr BC 训练

`train_hirol_pick_N_place_ee_state_bc.yaml` 里的 `dataset_path` 仍是占位值
`???`，运行时必须传入真实 zarr 数据集路径：

```bash
python train.py \
  -c BCpolicy/config/train_zarr/train_hirol_pick_N_place_ee_state_bc.yaml \
  dataset_path=/path/to/dataset.zarr
```

## 输出与恢复

默认输出目录仍在当前运行目录下的 `data/outputs/...`：

```text
data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
```

如果要指定输出目录：

```bash
python train.py \
  -c BCpolicy/config/train_lerobot_v3/train_hirol_fr3_pnp_cam_state_to_ee_bc_h16o2a8.yaml \
  hydra.run.dir=data/outputs/bc_debug
```

配置中 `training.resume: True`。当从已有 `.hydra` 输出目录恢复时，训练脚本会优先查找：

```text
checkpoints/latest.ckpt
```

## 实现说明

- `BCpolicy.policy.bc_image_policy.BCImagePolicy` 直接回归归一化后的 action trajectory。
- 图像 encoder、dataset、normalizer、workspace 仍使用原 `diffusion_policy` 模块。
- 两份外置 YAML 都通过 `hydra.searchpath` 复用原仓库的 task 配置。
- 如果出现 `ModuleNotFoundError: No module named 'BCpolicy'`，检查
  `PYTHONPATH=$PWD/BCpolicy:$PYTHONPATH` 是否已经导出。
- 如果出现 `Could not find task_lerobot_v3` 或 `task_zarr`，检查
  `DIFFUSION_POLICY_CONFIG_DIR` 是否指向原工程的 `diffusion_policy/config`。
