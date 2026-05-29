阶段 1: 明确数据协议

  1. 确认两个 LeRobotV3 数据集字段名：
      - 主数据集：timestamp、images、lowdim、action、episode metadata
      - FT 数据集：timestamp、ft_data 或具体 FT 字段名
  2. 确认两个数据集 episode 数量一致，episode 顺序一致。
  3. 确认 timestamp 是同一时间基准，单位是秒，保留 float64。
  4. 定义 FT 对齐策略：
      - 推荐：对每个 image timestamp t_img，取 (t_img -
        ft_window_sec, t_img]
      - 例如 ft_window_sec=0.05，ft_steps=10
  5. 定义 dataloader 输出 shape：
      - image: [B, To, C, H, W]
      - lowdim: [B, To, D_low]
      - FT: [B, To, K, D_ft]
      - FT mask: [B, To, K]
      - action: [B, horizon, D_action]

  阶段 2: Dataloader 接入 FT 数据集

  1. 给 HirolLeRobotV3Dataset 增加可选 FT 参数：
      - ft_dataset_path
      - ft_feature_fields
      - ft_timestamp_key
      - ft_obs_key
      - ft_window_sec
      - ft_steps
      - ft_time_offset_sec
  2. 初始化时加载第二个 LeRobotV3 FT dataset。
  3. 预加载 FT timestamps 和 FT data 到 numpy。
  4. 为 FT dataset 构建：
      - ft_episode_index
      - ft_episode_ends
      - ft_episode_ranges
  5. 检查 episode 数、timestamp 单调性、shape 是否匹配。
  6. 在 __getitem__ 中：
      - 用主数据集 obs_indices
      - 取每个 obs 的 t_img
      - 在同 episode FT timestamps 里 searchsorted
      - 取 FT window
      - pad/downsample/interpolate 到固定 K
      - 写入 obs_dict["ft_data"] 和 obs_dict["ft_mask"]

  阶段 3: Dataloader 验证

  1. 写一个 debug 脚本，打印单个 sample：
      - image timestamps
      - 每个 obs 对应的 FT timestamp range
      - FT window shape
      - FT mask 有效数量
  2. 检查没有跨 episode 对齐。
  3. 检查没有未来 FT：
      - max(ft_timestamps) <= t_img
  4. 检查 batch collate 后 shape 正确。
  5. 检查训练 yaml 只改配置即可打开/关闭 FT。

  阶段 4: shape_meta / config 设计

  1. 在 shape_meta.obs 增加：

  ft_data:
    type: ft
    shape: [10, 6]

  2. 不要把 FT 当普通 low_dim，否则会被老的 lowdim concat 流程吃掉。
  3. 增加 encoder 配置：
      - image encoder: DINOv3
      - FT encoder: LSTM
      - fusion: cross attention
  4. 保留一个无 FT 的 baseline config，方便 ablation。

  阶段 5: Model 侧 Encoder

  1. 新建 multimodal obs encoder，例如：
      - DinoFtCrossAttentionObsEncoder
  2. 输入：
      - image obs dict
      - ft_data
      - ft_mask
      - optional lowdim
  3. DINOv3 输出 image tokens：
      - 推荐 patch tokens，不只是 pooled feature
  4. FT LSTM 输出 FT tokens：
      - 输入 [B*To, K, D_ft]
      - 输出 [B*To, K, D_lstm]
  5. 投影到统一 embedding dim：
      - image tokens -> [B*To, N_img, D]
      - FT tokens -> [B*To, K, D]
  6. lowdim 可选投影成一个 state token：
      - [B*To, 1, D]

  阶段 6: Cross Attention Fusion

  1. 推荐先用 learned query cross attention：
      - query: [B*To, N_query, D]
      - key/value: concat(image_tokens, ft_tokens, state_token)
  2. 输出 pool 成每个 obs step 一个 feature：
      - [B*To, D_fused]
  3. 返回给 DP policy：
      - policy 内部 reshape 成 [B, To * D_fused]
  4. 支持 ft_mask，避免 padding FT token 被 attention 使用。

  阶段 7: 接入 DP Policy

  1. 让现有 DiffusionUnetImagePolicy 能使用新的 obs encoder。
  2. 保持 DP 训练目标不变：
      - obs condition: image + FT + lowdim fused feature
      - action trajectory: [B, horizon, D_action]
  3. 检查 obs_encoder.output_shape() 正确。
  4. 检查 normalizer 支持：
      - image normalizer
      - lowdim normalizer
      - FT normalizer

  阶段 8: 测试与实验

  1. 单元测试 dataloader 对齐。
  2. 单 batch forward test。
  3. 小 batch 训练 10 step，确认 loss 正常下降且无 shape error。
  4. 对比实验：
      - img + lowdim baseline
      - img + lowdim + FT without cross attention
      - img + lowdim + FT-LSTM + cross attention
  5. 记录关键超参：
      - ft_window_sec
      - ft_steps
      - DINOv3 freeze/unfreeze
      - attention dim/head 数
      - LSTM hidden dim/layers

  建议第一个 commit 只做 dataloader FT 对齐 + debug 验证，不要同时碰
  model。这个阶段稳定后，再进入 FT-LSTM 和 cross attention。

  已完成：

  1. Dataloader 接入 FT 数据集
      - 文件：diffusion_policy/dataset/hirol_lerobot_v3_dataset.py
      - 支持第二个 LeRobotV3 FT dataset。
      - FT 字段默认：

        ft_feature_fields:
          - observation.ft
        ft_timestamp_key: timestamp
      - 按主数据集 obs image timestamp 对齐 FT：

        FT window = (t_img - ft_window_sec, t_img]
      - timestamp 使用 float64，没有转 int。
      - 输出：

        obs["ft_data"]  # [To, K, 6]
        obs["ft_mask"]  # [To, K]
  2. FT shape_meta / task config
      - 新增：diffusion_policy/config/lerobot_v3/task_lerobot_v3/
        pick_n_place_ft/hirol_fr3_pnp_cam_state_ft_to_ee_unet.yaml
      - 加了：

        ft_data:
          shape: [10, 6]
          type: ft
  3. FT train config 初版
      - 新增：diffusion_policy/config/lerobot_v3/train_lerobot_v3/
        pick_n_place_ft/cam_state_ft_ee2ee_dino_lstm_xattn.yaml
      - 包含：

        dataset_path: ???
        ft_dataset_path: ???
        ft_window_sec: 0.05
        ft_steps: 10
      - encoder target 已预留为：

        DinoFtCrossAttentionObsEncoder
  4. 基础验证
      - hirol_lerobot_v3_dataset.py 语法检查通过。
      - git diff --check 通过。
      - YAML parse 没跑，因为当前环境没有 yaml 包。

  还要完成：

  1. 实现 multimodal encoder
      - 新文件建议：

        diffusion_policy/model/vision/
  dino_ft_cross_attention_obs_encoder.py
      - 功能：

        image -> DINOv3 tokens
        ft_data -> LSTM tokens
        lowdim -> state token
        tokens -> cross attention
        return [B*To, D_fused]
  2. 调整 DINOv3 wrapper
      - 当前 DINOv3 wrapper 偏 pooled output。
      - cross attention 最好需要 patch tokens，所以要支持：

        return_patch_tokens=True
  3. 确认 policy 是否无需改动
      - 目标是保持 DiffusionUnetImagePolicy 不改。
      - 新 encoder 继续实现：

        forward(obs_dict) -> [B*To, D]
        output_shape() -> (D,)
  4. 真实数据验证 dataloader
      - 用真实 dataset_path 和 ft_dataset_path 跑一个 sample。
      - 检查：

        obs["ft_data"].shape == [To, 10, 6]
        obs["ft_mask"].shape == [To, 10]
        max(valid_ft_timestamp) <= image_timestamp
      - 确认没有跨 episode 对齐。
  5. 训练前 smoke test
      - 实例化新 config。
      - 跑一个 batch forward。
      - 跑 5-10 step training，确认 loss、shape、显存都正常。

  当前状态一句话：dataloader + config 骨架已完成；还缺核心 multimodal
  encoder 和真实数据/训练验证。