# senyun_cam 标定与去畸变说明

本文档基于当前 4 个核心脚本整理：

- `ros_intrinsic_calibration.py`
- `ros_fisheye_dist.py`
- `zmq_intrinsic_calibration.py`
- `zmq_fisheye_distortion.py`

目标是围绕同一套 2x2 拼接画面，完成内参标定与去畸变验证。

## 1. 输入画面约定

所有脚本默认输入为 2x2 拼接画面，视图布局为：

- 左上：`left_eye`
- 右上：`right_eye`
- 左下：`right_hand`
- 右下：`left_hand`

若拼接顺序不一致，标定与去畸变结果会错位。

## 2. 依赖与环境

建议 Python 3.9+。

基础依赖：

```bash
pip install numpy opencv-python pyzmq
```

如果 ZMQ 使用 H264 码流，还需要：

```bash
pip install av
```

ROS 流程需要 ROS2 相关依赖：

- `rclpy`
- `sensor_msgs`
- `cv_bridge`
- `camera_calibration`

> 注意：4 个脚本均依赖 OpenCV 窗口交互（按键采样/保存/退出），建议在有 GUI 的环境运行。

## 3. ROS 流程

### 3.1 ROS 双目鱼眼标定桥接

脚本：`ros_intrinsic_calibration.py`

作用：

- 订阅拼接图像话题（默认 `/quad_tile/raw`）
- 切分并裁剪为 4 路图
- 发布为 `/quad_tile/<camera>/image_raw`
- 自动拉起 `camera_calibration` 做左右眼双目标定
- 标定结束后导出 `ros_calibration.yaml`

示例：

```bash
python3 ros_intrinsic_calibration.py \
	--input-topic /quad_tile/raw \
	--crop-width 640 \
	--board-cols 8 \
	--board-rows 11 \
	--square-size 0.035
```

关键参数：

- `--input-topic`：输入拼接图像话题
- `--crop-width`：每路裁剪宽度（高度固定为 480）
- `--board-cols` / `--board-rows`：棋盘格内角点数量
- `--square-size`：棋盘单格尺寸（米）

输出：

- `ros_calibration.yaml`（左右眼 `K/D/R/P`）

### 3.2 ROS 左右眼去畸变与校正预览

脚本：`ros_fisheye_dist.py`

作用：

- 订阅 `/quad_tile/raw`
- 仅提取左右眼并中心裁剪
- 读取 `ros_calibration.yaml` 中左右眼参数
- 执行 `cv2.fisheye.initUndistortRectifyMap` 去畸变+校正
- 显示 Raw/Rectified 四宫格预览
- 按 `s` 保存当前矫正图像对

示例：

```bash
python3 ros_fisheye_dist.py \
	--input-topic /quad_tile/raw \
	--calib-yaml ros_calibration.yaml \
	--crop-width 640 \
	--crop-height 480 \
	--save-dir undistort_output
```

交互按键：

- `s`：保存当前左右眼矫正结果到 `shot_xxxx/`
- `q` 或 `Esc`：退出

输出结构（示例）：

- `undistort_output/shot_0001/left_eye_rectified.png`
- `undistort_output/shot_0001/right_eye_rectified.png`

## 4. ZMQ 流程

### 4.1 ZMQ 四路鱼眼内参标定

脚本：`zmq_intrinsic_calibration.py`

作用：

- 从 ZMQ 订阅拼接图（支持 `raw` 与 `h264`）
- 切分为 4 路后做中心裁剪
- 在预览窗口中检测棋盘角点
- 手动按键采样（只保存当前检测成功的相机）
- 对 4 路分别执行 `cv2.fisheye.calibrate`
- 输出四路内参（仅内参，不含外参）

示例：

```bash
python3 zmq_intrinsic_calibration.py \
	--zmq-endpoint tcp://192.168.1.15:5556 \
	--stream-codec h264 \
	--board-cols 8 \
	--board-rows 11 \
	--square-size 35.0 \
	--crop-width 640 \
	--crop-height 480 \
	--top-extra-rows 0 \
	--min-samples 20 \
	--output-dir calib_output
```

交互按键：

- `s`：采样当前帧（按相机独立记录）
- `c`：开始计算并输出标定结果
- `q` 或 `Esc`：退出

输出：

- `calib_output/calibration_intrinsics.yaml`
- `calib_output/intrinsics/left_eye_intrinsics.npz`
- `calib_output/intrinsics/right_eye_intrinsics.npz`
- `calib_output/intrinsics/right_hand_intrinsics.npz`
- `calib_output/intrinsics/left_hand_intrinsics.npz`
- 同目录下对应的 `*_intrinsics.txt`

### 4.2 ZMQ 四路去畸变预览与抓图

脚本：`zmq_fisheye_distortion.py`

作用：

- 从 ZMQ 订阅拼接图（`raw`/`h264`）
- 加载 `calibration_intrinsics.yaml` 中四路 `*_K/*_D`
- 对每一路执行去畸变映射
- 拼回 2x2 预览并显示 FPS/帧号
- 按 `s` 保存当前去畸变四视图

示例：

```bash
python3 zmq_fisheye_distortion.py \
	--zmq-endpoint tcp://192.168.1.15:5556 \
	--stream-codec h264 \
	--calib-yaml calib_output/calibration_intrinsics.yaml \
	--crop-width 640 \
	--crop-height 480 \
	--top-extra-rows 0 \
	--alpha 0.0 \
	--save-dir undistort_output
```

交互按键：

- `s`：保存当前去畸变结果到 `shot_xxxx/`
- `q` 或 `Esc`：退出

输出结构（示例）：

- `undistort_output/shot_0001/left_eye.png`
- `undistort_output/shot_0001/right_eye.png`
- `undistort_output/shot_0001/right_hand.png`
- `undistort_output/shot_0001/left_hand.png`

## 5. 常见问题

1. 报错 `View size ... is smaller than crop size ...`

- 降低 `--crop-width`/`--crop-height`
- 检查输入拼接分辨率和切分方式是否匹配

2. ZMQ H264 模式下提示等待关键帧

- 这是解码器在等待 SPS/PPS 或关键帧，通常持续接收后会恢复
- 确认推流端持续发送 IDR 帧

3. 标定样本不足（`< min_samples`）

- 增加采样次数（多按 `s`）
- 棋盘覆盖更多姿态/距离/角度
- 检查光照和模糊，提升角点检测成功率

4. ROS 标定结束但未生成 `ros_calibration.yaml`

- 检查 `~/.ros/camera_info` 下是否生成 left/right 对应 yaml
- 确认 `camera_calibration` 进程正常退出

## 6. 建议使用顺序

1. 先跑 `zmq_intrinsic_calibration.py` 或 `ros_intrinsic_calibration.py` 获取标定结果
2. 再跑对应去畸变脚本验证效果（`zmq_fisheye_distortion.py` 或 `ros_fisheye_dist.py`）
3. 通过抓图目录 `undistort_output/shot_xxxx` 做质检与留档