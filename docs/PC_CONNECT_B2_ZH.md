# 电脑连接与控制 Unitree B2

本文给出一条从“只读确认通信”到“高层运动控制”的渐进式流程。默认只使用
Unitree 官方 SDK2/ROS 2 接口；仓库内的自研低层控制台属于实验性参考实现，不能
代替物理急停、机械防跌落或现场操作员。

## 1. 适用范围

- 机器人：Unitree B2；B2-W 的服务和话题可能不同，不要直接套用。
- 电脑：优先 Ubuntu 22.04。Unitree ROS 2 官方仓库列出的推荐组合是 Ubuntu
  22.04 + ROS 2 Humble。
- 网络：电脑通过独立有线网卡直接连接 B2 的用户扩展网口。
- 控制层级：首次连接只做状态读取；确认无误后再使用 SDK2 的高层 `SportClient`。
- 不在本文流程中进行：关闭官方运动服务、发布 `rt/lowcmd`、关节级控制。

不同出厂批次和固件可能改变服务可用性。执行运动前应向 Unitree 技术支持确认当前
机身序列号、固件和 SDK2 版本的匹配关系。

## 2. 真机安全准备

首次连接前必须满足：

1. 遥控器在现场人员手中，物理急停/阻尼操作已经验证。
2. B2 四周、上方和脚下没有人员、线缆和障碍物。
3. 首次运动在吊架、防跌落框或等效约束下进行。
4. 只允许一个控制源；不要同时运行 App、导航、第二个 SDK 示例或自研控制器。
5. 先完成只读检查，再运行运动示例。网络连通不等于可以运动。

软件的 `StopMove()` 或零速度命令不是安全等级急停，也不能证明电机已经停止。

## 3. 配置电脑网口

连接网线后，先找出真实的有线网卡名称：

```bash
ip -brief link
ip -brief address
```

下面用 `enp3s0` 举例。不要填写 `lo`、`wlan0`，也不要照抄示例网卡名。

临时配置（重启或断开后可恢复）：

```bash
export B2_IFACE=enp3s0
sudo ip link set "$B2_IFACE" up
sudo ip addr replace 192.168.123.99/24 dev "$B2_IFACE"
ip -brief address show dev "$B2_IFACE"
```

如果该网卡还有公司网、校园网或默认路由，请先断开那些配置，避免 DDS 在错误接口上
发现设备。通常不需要在 B2 专用网卡上设置网关或 DNS。

## 4. 方案 A：使用官方 ROS 2，只读验证

官方仓库：<https://github.com/unitreerobotics/unitree_ros2>

```bash
git clone https://github.com/unitreerobotics/unitree_ros2.git
cd unitree_ros2
```

按官方 README 为本机 ROS 2 版本安装依赖并编译。配置完成后，确保 CycloneDDS 绑定
到实际的 B2 网卡；例如：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name='${B2_IFACE}' priority='default' multicast='default'/></Interfaces></General></Domain></CycloneDDS>"
```

先只查看 ROS 图和状态：

```bash
ros2 topic list
ros2 topic info /lf/sportmodestate -v 2>/dev/null || true
ros2 topic info /sportmodestate -v 2>/dev/null || true
```

以 `ros2 topic list` 的实际结果为准，选择其中一个状态话题读取一次。不要在首次检查时
向任何 `cmd`、`request` 或 `lowcmd` 话题发布消息。

官方仓库提供的只读示例可用于进一步确认：

```bash
cd ~/unitree_ros2/example
colcon build
source install/setup.bash
./install/unitree_ros2_example/bin/read_motion_state
```

能够持续看到位置、速度、IMU 或足端状态，才说明 DDS 发现和状态链路已建立。

## 5. 方案 B：使用官方 Python SDK2

官方仓库：<https://github.com/unitreerobotics/unitree_sdk2_python>

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
git rev-parse HEAD
```

SDK 当前文档要求 Python 3.8+、CycloneDDS 0.10.2、NumPy 和 OpenCV。如果安装时提示
找不到 CycloneDDS，按官方 README 编译 `releases/0.10.x` 并设置
`CYCLONEDDS_HOME`，不要随意混用系统中其他 DDS 版本。

### 高层控制

B2 的官方交互示例是：

```bash
python3 example/b2/high_level/b2_sport_client.py "$B2_IFACE"
```

程序会再次等待操作者按 Enter。第一次运行时：

1. 先输入 `list` 查看动作。
2. 先选择 `StopMove`，确认请求返回值非负。
3. 在机械约束和现场人员就位后，才测试 `BalanceStand`。
4. `Move` 会产生真实位移，不要在无吊架、无急停或空间未清空时选择。
5. 退出或发生异常时，立即使用现场遥控器/物理急停；不要只依赖终端退出。

官方示例明确省略了部分危险接口。需要低层控制时，应先联系 Unitree 技术支持并完成
单独的台架验收。

## 6. 使用本仓库的 B2 Control Center

控制台源码位于仓库的 `src/`、`gateway/`、`scripts/` 和 `tests/`，不再只存在于
压缩包中。它面向仿真、状态诊断以及经过审批的实验性低层控制。

### Python 环境

```bash
git clone --recurse-submodules https://github.com/zc1Q84/b2-control-center.git
cd b2-control-center
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

运行纯 Python 测试：

```bash
PYTHONPATH=src pytest -q
```

### 编译官方 SDK2/MuJoCo 相关组件

先根据实际安装位置设置：

```bash
export UNITREE_SDK2_PREFIX=/absolute/path/to/unitree_sdk2/install
export MUJOCO_PREFIX=/absolute/path/to/mujoco
# 可选：仅在依赖安装到自定义前缀时设置
export B2_DEPS_PREFIX=/absolute/path/to/dependency-prefix
scripts/build_sim_stack.sh
```

### 启动控制台

```bash
scripts/start_b2_control_center.sh --no-open-browser
```

终端会打印一个只监听 `127.0.0.1`、带随机 token 的 URL。启动后保持 `IDLE`，不会
自动连接机器人或发送运动命令。

在页面中选择实际有线网卡后，“连接”阶段只进行 MotionSwitcher/LowState/LowCmd
基线检查。只有只读门禁通过并再次进行现场确认，自研 locomotion 入口才可能开放。
仓库随附硬件配置是 `reference_only`，默认不会批准真实 LowCmd 激活。

## 7. 常见问题

### 看不到 DDS 话题

- 网卡名填错，或 CycloneDDS 绑定到了 Wi-Fi/loopback；
- 电脑地址不是 `192.168.123.0/24` 网段；
- 防火墙、VPN、Docker 网桥或另一个 ROS 2 daemon 干扰组播发现；
- ROS 2/SDK2 与机器人固件所用 DDS 版本不匹配；
- B2 用户网口、线缆或对应服务没有启用。

先检查 `ip -brief address`、`ip route` 和 DDS 绑定接口，不要通过反复发送运动命令来
“测试网络”。

### 能看到状态但控制没有反应

- 当前固件未启用相应高层服务；
- 机器人处于阻尼、故障或其他不接受该命令的模式；
- 同时存在另一个命令源；
- SDK 请求已在本机发送，但机器人没有接受。SDK 返回成功不能单独证明硬件执行。

### 如何停止

优先级应当是：物理急停/现场遥控器 > 官方保护模式 > SDK `StopMove()` > 退出程序。
不要把关闭网页、`Ctrl-C` 或一次零速度消息当作唯一停止链路。

## 8. 官方参考

- Unitree ROS 2：<https://github.com/unitreerobotics/unitree_ros2>
- Unitree SDK2 Python：<https://github.com/unitreerobotics/unitree_sdk2_python>
- B2 官方高层示例：<https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/b2/high_level/b2_sport_client.py>
- Unitree SDK2 C++：<https://github.com/unitreerobotics/unitree_sdk2>

