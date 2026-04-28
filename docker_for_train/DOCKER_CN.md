# Docker 部署说明

本文分两部分：

* 如何在开发机上重新构建镜像，并打包成 `tar` / `tar.gz`
* 如何在另一台宿主机上解压、加载镜像并启动容器

适用环境：

* Ubuntu 宿主机
* 需要使用 NVIDIA GPU
* 开发机有项目代码目录
* 目标机已拿到项目代码目录和离线镜像包 `dp-hirol-train.tar` 或 `dp-hirol-train.tar.gz`

## 1. 路径

有两个东西，不是同一个路径：

* 项目代码目录：可以放在任意位置，建议类似 `/home/xxx/dp_hirol-main`
* 镜像包：也可以放在任意位置，建议类似 `/home/xxx/packages/dp-hirol-train.tar.gz`

重点只有一条：

* `docker load` 可以在任何目录执行，只要写对镜像包路径
* `docker compose up` 必须在项目里的 `docker/` 目录执行，或者显式指定 `-f docker/compose.yaml`

## 2. 在宿主机安装 Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

重新登录一次终端后执行：

```bash
docker --version
docker compose version
```

## 3. 如果宿主机有 GPU，再安装 NVIDIA Container Toolkit

如果这台机器要跑 GPU 容器，再执行：

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证 GPU 是否可用：

```bash
docker run --rm --gpus all nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 nvidia-smi
```

## 4. 在开发v机上重新构建镜像

如果你修改了依赖、代码或 `docker/Dockerfile`，需要先在开发机构建新镜像。

进入项目的 `docker/` 目录：

```bash
cd /home/xxx/dp_hirol-main/docker
```

构建 `trainer` 服务对应的镜像：

```bash
docker compose build trainer
```

如果怀疑 Docker 缓存导致旧依赖没有刷新，再改用完全重建：

```bash
docker compose build --no-cache trainer
```

构建成功后，`dp-hirol:train` 这个 tag 会指向新镜像。旧镜像不会自动被容器切换，后面还要执行 `docker compose up -d --force-recreate`。

## 5. 在开发机上导出镜像包

把当前镜像导出成 `tar`：

```bash
docker save -o /home/rei/mnt/code/dp_hirol-main/docker/dp-hirol-train.tar dp-hirol:train
```

如果想直接交付压缩包，再压缩成 `tar.gz`：

```bashv
gzip -f /home/rei/mnt/code/dp_hirol-main/docker/dp-hirol-train.tar 
```

说明：

* `docker save -o ...` 如果输出路径已存在，会覆盖原来的 `tar`
* `gzip -f ...` 会覆盖原来的 `tar.gz`
* 也就是说，重复执行这两条命令即可用新版本覆盖旧包

## 6. 更新环境依赖后，如何更新 Docker

这一步对应日v常开发里最常见的场景：

* 你修改了 `conda_environment.yaml`
* 你修改了 `requirements/uv-common.txt`
* 你修改了 `docker/Dockerfile`
* 你希望容器里的 Python 包和镜像一起更新

推荐按下面顺序执行。

先进入项目的 `docker/` 目录：

```bash
cd /home/xxx/dp_hirol-main/docker
```

如果当前有旧容器正在运行，先停掉：

```bash
docker compose down
```

重新构建镜像：

```bash
docker compose build trainer
```

如果怀疑缓存导致旧依赖没有被替换，再改用：

```bash
docker compose build --no-cache trainer
```

用新镜像重建容器：

```bash
docker compose up -d --force-recreate
```

进入容器验证关键依赖版本，例如检查 `wandb`：

```bash
docker compose exec trainer python -c "import wandb; print(wandb.__version__)"
```

如果你还要把新镜像发给别人，继续导出：

```bash
docker save -o /home/xxx/packages/dp-hirol-train.tar dp-hirol:train
gzip -f /home/xxx/packages/dp-hirol-train.tar
```

这套流程的效果是：

* 更新依赖文件
* 重建 `dp-hirol:train` 镜像
* 用新镜像替换旧容器
* 用新的 `tar` 或 `tar.gz` 覆盖旧离线包

## 7. 在目标机上加载离线镜像

把镜像包拷到宿主机后执行。

如果拿到的是 `tar.gz`：

```bash
gunzip /home/xxx/packages/dp-hirol-train.tar.gz
docker load -i /home/xxx/packages/dp-hirol-train.tar
```

如果拿到的是 `tar`：

```bash
docker load -i /home/xxx/packages/dp-hirol-train.tar
docker images | grep dp-hirol
```

## 8. 启动容器

假设项目放在 `/home/xxx/dp_hirol-main`，启动时执行：

```bash
cd /home/xxx/dp_hirol-main/docker
docker compose up -d
```

如果你修改过项目里的依赖文件，例如 `conda_environment.yaml`、`requirements/uv-common.txt` 或 `docker/Dockerfile`，只执行 `docker compose up -d` 不会刷新现有镜像，需要先重建：

```bash
cd /home/xxx/dp_hirol-main/docker
docker compose build --no-cache trainer
docker compose up -d --force-recreate
```

进入容器：

```bash
docker compose exec trainer bash
```

停止容器：

```bash
docker compose down
```

如果要确认容器里实际安装的 `wandb` 版本：

```bash
docker compose exec trainer python -c "import wandb; print(wandb.__version__)"
```

## 9. 目录要求

`docker/compose.yaml` 默认会挂载这些目录，所以项目根目录下需要存在：

* `data/`
* `outputs/`
* `data_converter/dataset/`

如果路径需要调整，直接修改 [compose.yaml](/mnt/code/dp_hirol-main/docker/compose.yaml) 里的 `volumes`。
