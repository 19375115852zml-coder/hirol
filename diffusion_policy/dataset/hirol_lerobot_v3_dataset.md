HirolLeRobotV3Dataset 的核心作用：把 LeRobot v3 格式的逐帧数据，适配成当前 diffusion_policy 训练需要的 固定时间窗口样本。

  源码位置：diffusion_policy/dataset/hirol_lerobot_v3_dataset.py:95

  主要功能

  1. 读取 LeRobot v3 数据集
     通过 CustomLeRobotV3Dataset(self.dataset_path) 打开数据集：diffusion_policy/dataset/hirol_lerobot_v3_dataset.py:171。
  2. 根据 shape_meta 判断哪些 observation 是图像、哪些是低维状态
     例如配置里 ee_cam_color/third_person_cam_color/side_cam_color 是 rgb，state_ee 是 low_dim：diffusion_policy/config/lerobot_v3/task_lerobot_v3/pick_n_palce/hirol_fr3_pnp_cam_state_to_ee_unet.yaml:5。
  3. 把 LeRobot 字段映射成训练字段
     例如：
      - ee_cam_color -> observation.images.ee_cam_color
      - state_ee -> observation.state
      - action -> action.ee_pose + action.gripper_width

     这部分由 image_feature_map、lowdim_feature_groups、action_feature_fields 控制：diffusion_policy/config/lerobot_v3/task_lerobot_v3/pick_n_palce/hirol_fr3_pnp_cam_state_to_ee_unet.yaml:48。
  4. 构造 episode-aware 的时间窗口索引
     它会读取 timestamp、episode 起止位置，然后用 create_indices(...) 生成训练样本窗口，避免跨 episode 采样：diffusion_policy/dataset/hirol_lerobot_v3_dataset.py:174、diffusion_policy/dataset/
     hirol_lerobot_v3_dataset.py:276。
  5. 提供图像缓存/预加载能力
     图像可以：
      - 预加载到 RAM；
      - 缓存到 SSD；
      - 或在 __getitem__ 时按需读取。

     对应逻辑在 diffusion_policy/dataset/hirol_lerobot_v3_dataset.py:205 到 diffusion_policy/dataset/hirol_lerobot_v3_dataset.py:264。
  6. 输出 PyTorch 训练 batch
     __getitem__ 返回：

  {
      "obs": {
          "ee_cam_color": Tensor[n_obs_steps, 3, H, W],
          "third_person_cam_color": Tensor[n_obs_steps, 3, H, W],
          "side_cam_color": Tensor[n_obs_steps, 3, H, W],
          "state_ee": Tensor[n_obs_steps, state_dim],
      },
      "action": Tensor[horizon, action_dim],
  }

  对应源码：diffusion_policy/dataset/hirol_lerobot_v3_dataset.py:486。

  运行流程

  1. 训练脚本通过 Hydra 创建 dataset
     hydra.utils.instantiate(cfg.task.dataset, **dataset_kwargs) 会实例化这个类：diffusion_policy/workspace/train_diffusion_unet_image_workspace.py:127。
  2. __init__ 初始化
     主要做这些事：
      - 保存 horizon / n_obs_steps / n_latency_steps / pad_before / pad_after；
      - 计算 sequence_length = horizon + n_latency_steps；
      - 解析 RGB key 和 low_dim key；
      - 打开 LeRobot v3 数据；
      - 读取 timestamp、episode 信息；
      - 读取低维状态和 action；
      - 校验 shape 是否和 shape_meta 一致；
      - 根据配置预加载或缓存图像；
      - 划分 train/val episode；
      - 创建训练窗口索引。
  3. DataLoader 调用 __len__
     返回可采样窗口数量，也就是 len(self.indices)：diffusion_policy/dataset/hirol_lerobot_v3_dataset.py:483。
  4. DataLoader 调用 __getitem__(idx)
     流程是：
      - 根据 idx 找到一个时间窗口；
      - 如果窗口靠近 episode 边界，用首帧/末帧 padding；
      - 如果 window_sampling_strategy == "timestamp"，按时间戳重新对齐；
      - 取前 n_obs_steps 作为 observation；
      - 取完整 sequence_length 作为 action 序列；
      - 如果有 n_latency_steps，丢掉前面的 latency action；
      - 把 numpy 转成 torch tensor 返回。
  5. 验证集
     get_validation_dataset() 会浅拷贝当前 dataset，只换成 validation episode 的 indices，并关闭图像增强：diffusion_policy/dataset/hirol_lerobot_v3_dataset.py:457。
  6. Normalizer
     get_normalizer() 会给：
      - action 拟合线性归一化；
      - low_dim observation 拟合线性归一化；
      - RGB 图像使用 image range normalizer：diffusion_policy/dataset/hirol_lerobot_v3_dataset.py:471。

  一句话总结：HirolLeRobotV3Dataset 是一个 LeRobot v3 数据读取器 + diffusion_policy 训练样本构造器，负责把原始逐帧机器人数据变成模型训练时需要的“多帧 observation + 未来 action 序列”。



  • 这两个文件合起来做一件事：把 LeRobot v3 的“逐帧数据表”，变成 diffusion policy 训练需要的“固定长度时间窗口样本”。

  可以按三层理解：

  1. lerobot_v3_io.py：负责用官方 LeRobot API 打开数据集。
  2. hirol_lerobot_v3_dataset.py.__init__：初始化时把低维数据、action、episode 边界读出来，并准备图像缓存。
  3. hirol_lerobot_v3_dataset.py.__getitem__：训练时每次取一个窗口，返回 DP 需要的 obs + action。

  1. LeRobot v3 原始数据长什么样

  LeRobot v3 可以理解成一张按时间排列的大表，每一行是一帧：

  frame 0:
    timestamp
    observation.images.ee_cam_color
    observation.images.third_person_cam_color
    observation.images.side_cam_color
    observation.state
    action.ee_pose
    action.gripper_width

  frame 1:
    ...

  另外 metadata 里还记录每个 episode 从哪一帧开始、到哪一帧结束。

  例如配置里定义：

  obs:
    ee_cam_color: rgb, shape [3,224,224]
    third_person_cam_color: rgb, shape [3,224,224]
    side_cam_color: rgb, shape [3,224,224]
    state_ee: low_dim, shape [15]

  action:
    shape: [8]

  字段映射是：

  image_feature_map:
    ee_cam_color: observation.images.ee_cam_color

  lowdim_feature_groups:
    state_ee:
      - observation.state

  action_feature_fields:
    - action.ee_pose
    - action.gripper_width

  也就是说，DP 里叫 state_ee，但 LeRobot 里真正的列名可能叫 observation.state。

  2. lerobot_v3_io.py：打开 LeRobot 数据

  核心类是 diffusion_policy/common/lerobot_v3_io.py:123。

  它做的事情很薄：

  self.dataset = LeRobotDataset(
      repo_id=...,
      root=self.root,
      video_backend=video_backend,
      download_videos=True,
  )

  也就是调用官方 lerobot.datasets.LeRobotDataset。

  然后它暴露几个接口：

  len(dataset)

  返回总帧数。

  dataset[idx]

  返回第 idx 帧的完整数据，比如图像、状态、动作。

  get_column(name)

  直接从 HuggingFace table 里取某一列，比如 timestamp、observation.state、action.ee_pose。

  它还会从 LeRobot metadata 里读 episode 起止位置：

  episode["dataset_from_index"]
  episode["dataset_to_index"]

  最后整理成：

  episode_data_index = {
      "from": [0, 123, 250, ...],
      "to":   [123, 250, 390, ...],
  }

  这个非常重要，因为训练采样窗口不能跨 episode。

  3. HirolLeRobotV3Dataset.__init__：初始化阶段做什么

  核心在 diffusion_policy/dataset/hirol_lerobot_v3_dataset.py:95。

  初始化时先保存几个关键参数：

  horizon
  n_obs_steps
  n_latency_steps
  pad_before
  pad_after

  其中：

  sequence_length = horizon + n_latency_steps

  sequence_length 是内部采样窗口长度。最后返回 action 时，如果有 latency，会把前面的 latency action 丢掉。

  然后根据 shape_meta 自动分出两类 observation：

  self.rgb_keys = ...
  self.lowdim_keys = ...

  比如：

  rgb_keys:
    ee_cam_color
    third_person_cam_color
    side_cam_color

  lowdim_keys:
    state_ee

  接着打开 LeRobot 数据：

  self.lerobot_dataset = LeRobotV3Dataset(...)
  self.dataset_length = len(self.lerobot_dataset)

  然后读取时间和 episode 信息：

  self.timestamps = self._load_column("timestamp")
  self.episode_index = self._load_episode_index()
  self.episode_ends = self._build_episode_ends(...)

  episode_ends 类似：

  [123, 250, 390]

  表示：

  episode 0: frame 0   到 122
  episode 1: frame 123 到 249
  episode 2: frame 250 到 389

  低维状态和 action 会在初始化时一次性读入 numpy：

  self.lowdim_data["state_ee"] = observation.state
  self.action_data = concat(action.ee_pose, action.gripper_width)

  所以如果：

  action.ee_pose shape = [7]
  action.gripper_width shape = [1]

  拼起来就是：

  action shape = [8]

  代码还会检查 shape 是否和配置一致。不一致就直接报错，避免训练时才发现维度错。

  4. 图像怎么读

  图像有三种方式：

  1. load_result_add 指向磁盘缓存：先把图片解码、resize、转格式，缓存到 SSD。
  2. preload_images=True：初始化时全部读到 RAM。
  3. 都不用：每次 __getitem__ 时按需从 LeRobot 读图。

  图像最终都会经过 _coerce_image 处理：

  HWC / CHW 统一成 CHW
  resize 到 shape_meta 里的尺寸
  转 float32
  如果是 0-255，就除以 255，变成 0-1

  所以最后图像格式是：

  [3, H, W], float32, range 0~1

  比如：

  [3, 224, 224]

  5. 训练样本窗口怎么生成

  初始化最后会做 train/val episode 划分：

  val_mask = get_val_mask(...)
  train_mask = ~val_mask

  然后调用：

  self.indices = create_indices(
      self.episode_ends,
      sequence_length=self.sequence_length,
      pad_before=self.pad_before,
      pad_after=self.pad_after,
      episode_mask=self.train_mask,
  )

  create_indices 来自 diffusion_policy/common/sampler.py:7。

  它不是直接保存每个窗口的所有帧号，而是保存四个数字：

  buffer_start_idx
  buffer_end_idx
  sample_start_idx
  sample_end_idx

  可以理解为：

  从原始数据的哪一段取真实帧；
  如果窗口超出 episode 边界，前面/后面要补多少帧。

  例如一个 episode 开头，模型想要前 2 帧 observation，但第 0 帧前面没有历史帧，就会重复第 0 帧做 padding：

  想要窗口: [-1, 0, 1, 2, ...]
  实际没有 -1
  变成:     [ 0, 0, 1, 2, ...]

  这就是 _sample_indices_to_sequence 做的事情：把 create_indices 的四个数字展开成固定长度的 sequence_indices。

  6. __getitem__：真正给 DataLoader 的数据

  核心在 diffusion_policy/dataset/hirol_lerobot_v3_dataset.py:489。

  流程是：

  sequence_indices = self._sample_indices_to_sequence(idx)

  比如得到：

  [10, 11, 12, 13, ..., 25]

  如果启用了：

  window_sampling_strategy == "timestamp"

  还会根据 timestamp 重新找最接近的帧，适合数据采样间隔不稳定的情况。默认配置一般是 idx，也就是直接按帧号取。

  然后 observation 只取前 n_obs_steps 帧：

  obs_indices = sequence_indices[:self.n_obs_steps]

  比如：

  n_obs_steps = 2

  sequence_indices = [10, 11, 12, 13, ..., 25]
  obs_indices      = [10, 11]

  图像 observation：

  obs_dict[key] = images

  形状是：

  [n_obs_steps, 3, H, W]

  低维 observation：

  obs_dict["state_ee"] = self.lowdim_data["state_ee"][obs_indices]

  形状是：

  [n_obs_steps, 15]

  action 取完整窗口：

  action = self.action_data[sequence_indices]

  如果有 latency：

  action = action[self.n_latency_steps:]

  最后返回：

  {
      "obs": {
          "ee_cam_color": Tensor[n_obs_steps, 3, 224, 224],
          "third_person_cam_color": Tensor[n_obs_steps, 3, 224, 224],
          "side_cam_color": Tensor[n_obs_steps, 3, 224, 224],
          "state_ee": Tensor[n_obs_steps, 15],
      },
      "action": Tensor[horizon, 8],
  }

  DataLoader 再把多个样本叠成 batch：

  {
      "obs": {
          "ee_cam_color": Tensor[B, n_obs_steps, 3, 224, 224],
          "third_person_cam_color": Tensor[B, n_obs_steps, 3, 224, 224],
          "side_cam_color": Tensor[B, n_obs_steps, 3, 224, 224],
          "state_ee": Tensor[B, n_obs_steps, 15],
      },
      "action": Tensor[B, horizon, 8],
  }

  一句话总结

  lerobot_v3_io.py 负责“把 LeRobot v3 数据集打开，并按帧/按列读取”；hirol_lerobot_v3_dataset.py 负责“按 episode 安全地切固定长度时间窗口，把图像、低维状态、动作整理成 diffusion policy 训练需要的 obs + action batch”。