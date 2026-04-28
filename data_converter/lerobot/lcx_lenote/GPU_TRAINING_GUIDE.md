# GPU Training Guide

本指南介绍如何在不同GPU配置下使用diffusion policy进行训练。

## 快速开始

### 1. 单GPU训练

```bash
# 使用默认GPU (通常是cuda:0)
python train.py --config-dir=. --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml

# 指定特定GPU
python train.py --config-dir=. --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml training.device=cuda:1

# 使用CPU (调试模式)
python train.py --config-dir=. --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml training.device=cpu
```

### 2. 多GPU服务器部署

```bash
# 方式1: 使用环境变量限制可见GPU
CUDA_VISIBLE_DEVICES=0,1 python train.py --config-dir=. --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml training.device=cuda:0

# 方式2: 指定特定GPU组合
CUDA_VISIBLE_DEVICES=2,3,4 python train.py --config-dir=. --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml training.device=cuda:0

# 方式3: 在8卡服务器上使用后4张卡
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --config-dir=. --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml training.device=cuda:0
```

## 服务器管理最佳实践

### 1. GPU状态检查

```bash
# 查看GPU使用情况
nvidia-smi

# 实时监控
watch -n 1 nvidia-smi

# 查看可用GPU
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### 2. 并行训练任务

```bash
# 终端1: 使用GPU 0
CUDA_VISIBLE_DEVICES=0 python train.py --config-dir=. --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml training.device=cuda:0 hydra.run.dir='outputs/gpu0_run'

# 终端2: 使用GPU 1
CUDA_VISIBLE_DEVICES=1 python train.py --config-dir=. --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml training.device=cuda:0 hydra.run.dir='outputs/gpu1_run'

# 终端3: 使用GPU 2,3
CUDA_VISIBLE_DEVICES=2,3 python train.py --config-dir=. --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml training.device=cuda:0 hydra.run.dir='outputs/gpu23_run'
```

### 3. 批量提交训练任务

创建训练脚本 `batch_train.sh`:

```bash
#!/bin/bash

# 任务1: GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
    --config-dir=. \diffusion_policy/config/task/hirol_fr3_3cam_insert_tube.yaml
    --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml \
    training.device=cuda:0 \
    hydra.run.dir='outputs/exp1_gpu0' \
    > logs/exp1.log 2>&1 &

# 任务2: GPU 1  
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --config-dir=. \
    --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml \
    training.device=cuda:0 \
    training.num_epochs=5000 \
    hydra.run.dir='outputs/exp2_gpu1' \
    > logs/exp2.log 2>&1 &

echo "All training jobs submitted"
```

## 高级配置

### 1. 内存优化

```bash
# 减少batch size以适应GPU内存
python train.py --config-dir=. --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml \
    training.device=cuda:0 \
    dataloader.batch_size=32 \
    val_dataloader.batch_size=32
```

### 2. 混合精度训练

```bash
# 启用混合精度 (需要在配置中支持)
python train.py --config-dir=. --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml \
    training.device=cuda:0 \
    training.use_amp=true
```

### 3. 断点续训

```bash
# 从检查点继续训练
python train.py --config-dir=. --config-name=train_hirol_fr3_pick_N_place_unet_abs_jp.yaml \
    training.device=cuda:1 \
    training.resume=true \
    checkpoint.load_path='/path/to/checkpoint.ckpt'
```

## 常见问题

### 1. CUDA Out of Memory

```bash
# 解决方案1: 减少batch size
python train.py --config-name=xxx training.device=cuda:0 dataloader.batch_size=16

# 解决方案2: 使用梯度累积
python train.py --config-name=xxx training.device=cuda:0 training.gradient_accumulate_every=2
```

### 2. 指定的GPU不存在

```bash
# 检查可用GPU数量
python -c "import torch; print(torch.cuda.device_count())"

# 使用正确的GPU索引 (0-indexed)
python train.py --config-name=xxx training.device=cuda:0  # 第1块GPU
python train.py --config-name=xxx training.device=cuda:1  # 第2块GPU
```

### 3. 多进程端口冲突

```bash
# 指定不同的端口
MASTER_PORT=29500 python train.py --config-name=xxx
MASTER_PORT=29501 python train.py --config-name=xxx  # 另一个任务
```

## 性能监控

### 1. GPU利用率监控

```bash
# 安装gpustat (可选)
pip install gpustat
gpustat -i 1  # 每秒更新

# 或使用nvidia-smi
nvidia-smi dmon -i 0 -s puct -d 1  # 监控GPU 0
```

### 2. 训练日志

```bash
# 带时间戳的日志
python train.py --config-name=xxx 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
```

---

## 总结

- **单GPU**: 直接使用 `training.device=cuda:X`
- **多GPU服务器**: 使用 `CUDA_VISIBLE_DEVICES` + `training.device=cuda:0`
- **并行任务**: 为每个任务指定不同的GPU和输出目录
- **监控**: 定期检查GPU使用情况和训练进度

如有问题，请检查GPU状态和配置文件设置。