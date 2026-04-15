# senyun_cam 使用说明

本项目用于处理 2x2 拼接相机视频，主要包含：

- 视图裁剪与重拼接（`crop.py`）
- 四路相机标定 + 双目标定（`calibration.py`）
- 左右眼去畸变对比可视化（`distortion.py`）
- ZMQ 鱼眼图客户端接收显示（`zmq_fisheye_client.py`）

---

## 1. 环境准备

建议 Python 3.9+。

安装依赖：

```bash
pip install opencv-python numpy pyzmq
```

如果在服务器/无图形界面环境运行（`cv2.imshow` 无法使用），请在本机图形环境执行，或自行注释脚本中的显示窗口相关代码。

---

## 2. 输入视频约定

项目默认输入是 **2x2 拼接视频**，布局如下：

- 左上：`left_eye`
- 右上：`right_eye`
- 左下：`right_hand`
- 右下：`left_hand`

请确保输入视频与该布局一致，否则标定/去畸变结果会错位。

---

## 3. 推荐流程（从原始视频到标定结果）

### 第一步：裁剪四路视图（可选但推荐）

用于将每路视图做中心裁剪后重新拼接，得到更稳定的标定输入。

```bash
python crop.py \
	--input cameras.mp4 \
	--output cameras_cropped.mp4 \
	--crop-width 640 \
	--crop-height 480
```

参数说明：

- `--input`：输入 2x2 视频（默认 `cameras.mp4`）
- `--output`：输出视频（默认 `cameras_cropped.mp4`）
- `--crop-width/--crop-height`：每个单视图裁剪尺寸

> 说明：当前 `crop.py` 对上排视图切分时包含了额外高度偏移逻辑（`half_h+100`），若输入数据格式变化，需按实际画面微调。

### 第二步：执行四路标定 + 左右眼双目标定

```bash
python calibration.py \
	--video cameras_cropped.mp4 \
	--stitched-width 1280 \
	--stitched-height 960 \
	--board-cols 8 \
	--board-rows 11 \
	--square-size 25.0 \
	--sample-every 5 \
	--min-samples 20 \
	--min-stereo-pairs 20 \
	--output-dir calib_output \
	--visualize
```

核心参数：

- `--board-cols/--board-rows`：棋盘格**内角点**数量（宽/高）
- `--square-size`：格子边长（建议使用真实单位，如 mm）
- `--sample-every`：每 N 帧采样一次
- `--min-samples`：每个相机最少有效帧
- `--min-stereo-pairs`：左右眼双目标定最少有效配对帧
- `--visualize` / `--no-visualize`：是否显示处理窗口（默认开启）

运行后主要输出：

- `calib_output/intrinsics/*_intrinsics.npz`
- `calib_output/intrinsics/*_intrinsics.txt`
- `calib_output/stereo_left_right.npz`
- `calib_output/calibration_all.yaml`

---

## 4. 左右眼去畸变对比

使用 `calibration_all.yaml` 中的左右眼内参，生成原图/去畸变对比视频：

```bash
python distortion.py \
	--video /path/to/stitched.mp4 \
	--stitched-width 1280 \
	--stitched-height 960 \
	--calib-yaml calib_output/calibration_all.yaml \
	--output undistort_compare_lr.mp4
```

输出：

- 对比视频：`undistort_compare_lr.mp4`
- 首帧截图：`undistort_compare_lr_first_frame.jpg`

按 `q` 或 `Esc` 可提前退出。

---

## 5. ZMQ 鱼眼图像客户端

`zmq_fisheye_client.py` 用于连接远端 ZMQ 服务，接收 Base64 编码图像并显示。

运行：

```bash
python zmq_fisheye_client.py
```

默认配置（在脚本顶部修改）：

- `ZMQ_REMOTE_IP = "192.168.1.57"`
- `ZMQ_REMOTE_PORT = 5555`

当前默认使用 `SUB` 模式；如服务端是 `PUSH/PULL`，可改为调用 `run_client("PULL")`。

---

## 6. 常见问题

1) **报错：找不到棋盘角点 / 有效帧不足**

- 检查棋盘格规格是否与 `--board-cols --board-rows` 一致
- 增加视频帧数、降低 `--sample-every`
- 提升画面清晰度、避免运动模糊/反光

2) **报错：Frame size is smaller than requested crop**

- 减小 `crop.py` 的裁剪尺寸
- 或改用更高分辨率输入

3) **窗口无法显示（远程终端）**

- 使用 `--no-visualize` 运行标定
- 去畸变脚本若需无头运行，请注释 `cv2.imshow/cv2.waitKey`

---

## 7. 现有结果文件说明

仓库中已包含部分历史标定结果，例如：

- `calib_output/`
- `intrinsic_0331/calib_output/`
- `intrinsic_0331/calib_output_2560_1984/`

这些文件可直接用于回放、对比或作为后续流程输入。


python3 ros_intrinsic_calibration.py --input-topic /quad_tile/raw --camera right_eye --board-cols 8 --board-rows 11 --square-size 0.035