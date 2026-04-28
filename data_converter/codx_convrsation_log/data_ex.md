  joint_states.single 在 data.json#L49：
- position：7 维关节位置，基本就是机械臂 7 个关节角。结合数据集名 left_fr3 和 7 维长度，可以推断是 FR3 左臂的 7 关节。
- velocity：7 维关节速度。
- acceleration：7 维关节加速度，这个 episode 里看到的值全是 0，说明这个字段当前基本没被真正填充。
- torque：7 维关节力矩/effort。
- time_stamp：这组关节状态的采样时间戳。
- 注意：joint_names 是 null，见 data.json#L22，所以文件本身没有写明每一维具体对应 joint1...joint7 的名字。
 
 ee_states.single 在 data.json#L90：
- pose：7 维末端执行器位姿，前3维是x,y,z，后4维是四元数，顺序qw,qx,qy,qz。
- twist：6 维末端速度，通常可理解为线速度 vx,vy,vz 加角速度 wx,wy,wz。
- time_stamp：末端状态时间戳。
- tools.sinzgle 在 data.json#L115：
- position：工具/夹爪位置。这个 episode 里数值大约在 80 到 90 之间。
- 按读取器的实现，它被当作夹爪开口宽度来解释；如果值大于1，就按“毫米转米”处理，见 hirol_reader.py#L73。所以这里  基本可以理解成“观测到的夹爪张开宽度”，原始单位很像 mm。
- time_stamp：夹爪状态时间戳。
  
  actions.single 在 data.json#L121：
- joint.position：7 维动作目标关节位置，也就是控制命令里的关节目标。
- 读取器也确实把 0/1 当作夹爪 close/open 处理，1.0 会映射成默认张开宽度 0.08m，见 hirol_reader.py#L82。
- 每个子动作也各自带 time_stamp，而且不一定和观测时间完全相同。
 
 
- 其他低维相关字段：
- tactiles 是空字典，当前没触觉数据，见 data.json#L112。
- imus 是 null，当前没 IMU 数据，见 data.json#L113。
- audios 是 null，当前没音频低维特征，见 data.json#L114。
- depths 也是 null，当前没有深度图，见 data.json#L48。