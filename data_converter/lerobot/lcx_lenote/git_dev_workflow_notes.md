# Git 在项目开发流程中的常用命令与含义

这份笔记从一个项目的日常开发流程出发，整理 Git 常见命令的用途、含义和推荐使用方式。目标不是覆盖所有 Git 细节，而是保证你在实际开发中能稳定地提交代码、同步远端、使用分支，并在出错时知道如何处理。

## 1. Git 是什么

Git 是版本控制工具，用来记录代码历史、管理分支、协作开发和回退变更。

你可以把它理解成：

- 工作区：你当前正在编辑的文件
- 暂存区：准备提交的改动
- 本地仓库：你本机上的提交历史
- 远端仓库：GitHub、GitLab 等服务器上的仓库

一个最重要的理解是：

- `git add` 是把修改放进“待提交列表”
- `git commit` 是把待提交列表保存成一次历史记录
- `git push` 是把本地提交上传到远端仓库

## 2. 一个项目的典型开发流程

一个较稳妥的开发流程通常是：

1. 先同步主分支最新代码
2. 新建自己的开发分支
3. 修改代码
4. 查看变更内容
5. 将需要的文件加入暂存区
6. 提交本次修改
7. 推送到远端分支
8. 最后合并回主分支

对应常用命令大致如下：

```bash
git switch main
git pull
git switch -c fix-trainer-build

git status
git add .
git commit -m "fix docker trainer build"
git push -u origin fix-trainer-build
```

## 3. 最常用命令及含义

### `git status`

作用：查看当前仓库状态。

你会用它来判断：

- 哪些文件被修改了
- 哪些文件还没加入暂存区
- 哪些文件已经加入暂存区
- 当前在什么分支

命令：

```bash
git status
```

这是最常用、最该先看的命令。

### `git add`

作用：把文件加入暂存区，为提交做准备。

常见用法：

```bash
git add .
git add path/to/file.py
git add path/to/config.yaml
```

含义：

- `git add .`：把当前目录下所有未被忽略的改动加入暂存区
- `git add 某个文件`：只暂存指定文件

注意：

- `git add` 不等于提交
- 如果 `.gitignore` 配置正确，`git add .` 就不会把大文件、数据集、输出目录一起加进去

### `git commit`

作用：把暂存区内容保存成一次本地提交。

命令：

```bash
git commit -m "fix docker compose gpu config"
```

含义：

- `-m` 后面是这次提交的说明
- 一个 commit 就是一条可回溯的历史记录

推荐提交信息写法：

- `fix docker compose gpu config`
- `add lerobot v3 dataset converter`
- `update training yaml for fr3 task`

### `git push`

作用：把本地提交上传到远端仓库。

命令：

```bash
git push
```

第一次推送新分支时通常用：

```bash
git push -u origin fix-trainer-build
```

含义：

- `origin`：默认远端仓库名
- `fix-trainer-build`：要推送的分支名
- `-u`：建立本地分支和远端分支的跟踪关系，后面再 push 就可以直接用 `git push`

### `git pull`

作用：把远端最新代码拉到本地，并尝试合并。

命令：

```bash
git pull
```

常见场景：

- 你开始开发前先同步主分支
- 你准备推送前先同步远端
- 多人协作时避免落后太多

### 多主机开发时如何安全地“用远端替换本地”

如果你在 A 主机上已经更新代码并推到了远端，现在想让 B 主机直接变成“和远端完全一致”，并且：

- 不做 rebase
- 忽略 B 主机上未提交的改动
- 不保留 B 主机上的本地脏工作区

推荐直接使用下面这组命令，而不是直接 `git pull`：

```bash
git fetch origin
git branch --show-current
git reset --hard origin/<当前分支名>
git clean -fd
```

含义：

- `git fetch origin`：只拉取远端最新提交，不改本地工作区
- `git branch --show-current`：确认你当前所在分支名
- `git reset --hard origin/<当前分支名>`：强制把本地已跟踪文件重置成远端分支当前状态
- `git clean -fd`：删除本地未跟踪的文件和目录

这套流程适合“另一台机器只是同步最新代码，不需要保留本地未提交修改”的情况。

例如当前分支是 `main`，可以写成：

```bash
git fetch origin
git reset --hard origin/main
git clean -fd
```

注意：

- `git reset --hard` 会丢弃本地未提交的已跟踪文件改动
- `git clean -fd` 会删除未跟踪文件和目录
- 如果另一台主机上有你还想保留的内容，不要直接执行这组命令

如果你只是想让以后默认 `pull` 不用 rebase，可以设置：

```bash
git config pull.rebase false
```

但在“多主机强制同步远端并忽略本地脏改动”的场景里，`fetch + reset --hard` 仍然比 `pull --no-rebase` 更直接、更可控。

### `git log`

作用：查看提交历史。

命令：

```bash
git log --oneline --graph --decorate -n 20
```

含义：

- `--oneline`：每条提交只显示一行
- `--graph`：用图形方式显示分支关系
- `--decorate`：显示分支名、标签名

### `git diff`

作用：查看代码差异。

常见用法：

```bash
git diff
git diff --staged
```

含义：

- `git diff`：看工作区相对于暂存区的变化
- `git diff --staged`：看已经 add 进去、准备 commit 的内容

这是提交前非常值得检查的一步。

## 4. 分支 branch 的含义与操作

分支可以理解为一条独立开发线。

为什么要用分支：

- 避免直接在 `main` 上乱改
- 不同功能可以分开开发
- 便于多人协作
- 出问题时更容易隔离影响

### 查看分支

```bash
git branch
git branch -a
```

含义：

- `git branch`：看本地分支
- `git branch -a`：看本地和远端所有分支

### 新建分支

```bash
git branch feature-data-converter
```

只创建，不切换。

### 新建并切换分支

```bash
git switch -c feature-data-converter
```

这是更常用的方式。

### 切换分支

```bash
git switch main
git switch feature-data-converter
```

### 删除分支

```bash
git branch -d feature-data-converter
```

如果分支未合并但你确定不要了：

```bash
git branch -D feature-data-converter
```

## 5. 推荐的日常开发节奏

### 开始开发前

```bash
git switch main
git pull
git switch -c fix-my-feature
```

这样可以确保你是从最新主分支切出来的。

### 开发过程中

```bash
git status
git diff
git add .
git diff --staged
git commit -m "describe your change"
```

### 推送到远端

```bash
git push -u origin fix-my-feature
```

后续继续提交后：

```bash
git push
```

## 6. 针对你这种项目更实用的做法

这个项目里有代码、配置、数据、镜像归档、训练输出等多类文件，所以最关键的是区分“该提交的”和“不该提交的”。

推荐原则：

- 提交：`.py`、`.yaml`、`.md`、`Dockerfile`、`compose.yaml`、脚本文件
- 不提交：数据集、模型权重、训练输出、压缩包、镜像包、本地缓存文件

你前面已经修改了 `.gitignore`，目的就是让：

```bash
git add .
```

尽量只加入代码和配置，而不是把大文件也一起带进去。

但即使如此，也建议在每次 commit 前至少看一次：

```bash
git status
git diff --staged
```

这是避免误提交最有效的习惯。

## 7. 常见问题与处理

### 1. `git add .` 之后发现加错了文件

如果还没 commit：

```bash
git reset
```

作用：把暂存区清空，但不删除你的文件修改。

如果只想取消某个文件的暂存：

```bash
git restore --staged path/to/file
```

### 2. 改坏了代码，想放弃本地修改

如果某个文件还没提交：

```bash
git restore path/to/file
```

作用：恢复成最近一次提交的状态。

注意：这会丢失该文件当前未提交的修改。

### 3. commit 信息写错了

如果只是最近一次提交的信息写错，且还没 push：

```bash
git commit --amend -m "new message"
```

### 4. 远端拒绝 push，大文件超限

可能原因：

- 大文件被 add 进来了
- 大文件已经进入某个本地 commit 历史中

处理思路：

- 先修 `.gitignore`
- 再检查当前暂存区和最近提交
- 如果大文件已经进入历史，仅改 `.gitignore` 不够，还要清理历史或重做提交

### 5. 想知道某个文件改了什么

```bash
git diff path/to/file
```

### 6. 想知道当前在哪个分支

```bash
git branch
```

带 `*` 的就是当前分支。

### 7. `rebase`、`merge`、`pull` 在多主机场景下分别会怎样

#### `git pull`

`git pull` 的结果不固定，取决于你本地是否有未提交改动、是否有本地提交、以及 Git 当前配置。

常见情况：

- 如果本地有未提交改动，而且远端更新碰到了同一批文件，`git pull` 很可能直接失败
- 如果本地有未提交改动但不冲突，`git pull` 可能成功，同时把远端更新和本地未提交改动一起留在工作区
- 如果本地有未推送的提交，`git pull` 会按配置走 `merge` 或 `rebase`

所以 `git pull` 不适合“无脑用远端覆盖本地”的需求。

#### `git rebase`

`rebase` 会把“你的本地提交”重新接到远端最新提交后面。

在多主机同步时，它的问题是：

- 如果本地只有未提交改动，常常会先报错，不允许继续
- 如果本地有已提交但未推送的提交，它会尝试保留这些提交并重新播放
- 如果两边改了同一段内容，可能出现冲突，需要手工处理
- 它会改写本地提交历史，不适合你只想“完全替换成本地最新远端代码”的场景

结论：

- `rebase` 适合“我要保留本地提交并整理历史”
- 不适合“我要忽略另一台机器的本地改动并直接同步远端”

#### `git pull --merge`

`git pull --merge` 的本质是：

1. `git fetch`
2. `git merge origin/<当前分支>`

它表示“拉远端后，用 merge 的方式合并”，而不是“用远端强制覆盖本地”。

它的特点是：

- 会尽量保留本地提交
- 可能产生 merge commit
- 有冲突时仍然需要手工解决
- 如果本地有未提交改动，也可能直接失败

因此：

- `git pull --merge` 适合“我要保留本地开发历史，并把远端并进来”
- 不适合“忽略本地未提交改动并直接替换”

## 8. 一套适合个人开发的简洁命令流

如果你是自己维护项目，下面这套流程已经足够稳定：

```bash
git switch main
git pull
git switch -c my-change

git status
git add .
git commit -m "describe the change"
git push -u origin my-change
```

后续继续开发：

```bash
git add .
git commit -m "continue update"
git push
```

## 9. 推荐你记住的最核心理解

只记住下面几句，很多 Git 操作就不容易乱：

- `git status`：先看状态
- `git add`：把改动放进待提交列表
- `git commit`：把待提交列表保存成历史
- `git push`：把本地历史传到远端
- `branch`：独立开发线
- `pull`：同步远端最新内容
- `diff`：看清楚自己到底改了什么

## 10. 最后给你的实际建议

对于这个项目，比较稳的习惯是：

1. 不要直接在 `main` 上长期开发
2. 每次改功能前先切新分支
3. 每次提交前都先看 `git status`
4. 每次 push 前确认没有大文件混进去
5. `.gitignore` 只负责“默认忽略”，不能代替你检查提交内容

如果你后续需要，我还可以再给你补一份“你这个项目专用的 Git 提交流程模板”，比如专门针对 `.py`、`.yaml`、`docker/` 配置修改和大文件防误提交流程。
