#!/usr/bin/env python3

import argparse
import re
import threading
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

try:
	import rclpy
	from cv_bridge import CvBridge, CvBridgeError
	from rclpy.node import Node
	from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
	from sensor_msgs.msg import Image
except ImportError as exc:
	raise ImportError(
		"Missing ROS2 dependencies. Please install rclpy, sensor_msgs, and cv_bridge."
	) from exc


def center_crop(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
	h, w = img.shape[:2]
	if w < target_w or h < target_h:
		raise ValueError(f"View size {w}x{h} is smaller than crop size {target_w}x{target_h}.")
	x0 = (w - target_w) // 2
	y0 = (h - target_h) // 2
	return img[y0 : y0 + target_h, x0 : x0 + target_w]


def split_left_right_eyes(frame: np.ndarray, crop_width: int, crop_height: int) -> Dict[str, np.ndarray]:
	h, w = frame.shape[:2]
	if w % 2 != 0 or h % 2 != 0:
		raise ValueError(f"Stitched frame size must be even, got {w}x{h}")

	half_w = w // 2
	half_h = h // 2
	left_eye = frame[0:half_h, 0:half_w]
	right_eye = frame[0:half_h, half_w:w]

	return {
		"left_eye": center_crop(left_eye, crop_width, crop_height),
		"right_eye": center_crop(right_eye, crop_width, crop_height),
	}


def _parse_matrix_data(block_text: str, key: str, rows: int, cols: int) -> np.ndarray:
	pattern = (
		rf"{re.escape(key)}\s*:\s*!!opencv-matrix\s*\n"
		rf"\s*rows\s*:\s*{rows}\s*\n"
		rf"\s*cols\s*:\s*{cols}\s*\n"
		rf"\s*dt\s*:\s*\w+\s*\n"
		rf"\s*data\s*:\s*\[(.*?)\]"
	)
	match = re.search(pattern, block_text, re.DOTALL)
	if not match:
		raise ValueError(f"Missing or invalid OpenCV matrix block: {key}")
	values = [float(v) for v in re.split(r"[\s,]+", match.group(1).strip()) if v]
	if len(values) != rows * cols:
		raise ValueError(
			f"Matrix {key} has {len(values)} values, expected {rows * cols}"
		)
	return np.asarray(values, dtype=np.float64).reshape(rows, cols)


def _extract_camera_block(yaml_text: str, block_name: str) -> str:
	# Supports both "[name]:" and "[name]" styles.
	pattern = rf"\[{re.escape(block_name)}\]\s*:?(.*?)(?=\n\[[^\n]+\]\s*:?|\Z)"
	match = re.search(pattern, yaml_text, re.DOTALL)
	if not match:
		raise ValueError(f"Cannot find camera block: [{block_name}]")
	return match.group(1)


def load_ros_stereo_yaml(yaml_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
	text = yaml_path.read_text(encoding="utf-8")

	left_block = _extract_camera_block(text, "narrow_stereo/left")
	right_block = _extract_camera_block(text, "narrow_stereo/right")

	def parse_block(block_text: str) -> Dict[str, np.ndarray]:
		k = _parse_matrix_data(block_text, "camera_matrix", 3, 3)
		d = _parse_matrix_data(block_text, "distortion_coefficients", 1, 4).reshape(4, 1)
		r = _parse_matrix_data(block_text, "rectification", 3, 3)
		p = _parse_matrix_data(block_text, "projection_matrix", 3, 4)
		return {"K": k, "D": d, "R": r, "P": p}

	return {
		"left_eye": parse_block(left_block),
		"right_eye": parse_block(right_block),
	}


def print_k_vs_p(stereo_params: Dict[str, Dict[str, np.ndarray]]) -> None:
	print("\n=== Camera Matrix (K) vs Projection Matrix (P) Comparison ===")
	for cam in ["left_eye", "right_eye"]:
		k = stereo_params[cam]["K"]
		p3 = stereo_params[cam]["P"][:, :3]
		delta = p3 - k

		print(f"\n[{cam}]")
		print("K =")
		print(np.array2string(k, precision=6, suppress_small=False))
		print("P[:,:3] =")
		print(np.array2string(p3, precision=6, suppress_small=False))
		print("P[:,:3] - K =")
		print(np.array2string(delta, precision=6, suppress_small=False))
		print(
			"fx/fy/cx/cy delta = "
			f"({delta[0,0]:+.6f}, {delta[1,1]:+.6f}, {delta[0,2]:+.6f}, {delta[1,2]:+.6f})"
		)


def build_fisheye_maps(
	stereo_params: Dict[str, Dict[str, np.ndarray]],
	size: Tuple[int, int],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
	maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
	for cam in ["left_eye", "right_eye"]:
		k = stereo_params[cam]["K"]
		d = stereo_params[cam]["D"]
		r = stereo_params[cam]["R"]
		p = stereo_params[cam]["P"][:, :3]

		map1, map2 = cv2.fisheye.initUndistortRectifyMap(
			k,
			d,
			r,
			p,
			size,
			cv2.CV_16SC2,
		)
		maps[cam] = (map1, map2)
	return maps


def draw_label(img: np.ndarray, text: str) -> np.ndarray:
	out = img.copy()
	cv2.putText(
		out,
		text,
		(12, 30),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.9,
		(0, 255, 0),
		2,
		cv2.LINE_AA,
	)
	return out


def make_preview(
	left_raw: np.ndarray,
	right_raw: np.ndarray,
	left_rect: np.ndarray,
	right_rect: np.ndarray,
) -> np.ndarray:
	top = cv2.hconcat([
		draw_label(left_raw, "Left Raw"),
		draw_label(right_raw, "Right Raw"),
	])
	bottom = cv2.hconcat([
		draw_label(left_rect, "Left Rectified"),
		draw_label(right_rect, "Right Rectified"),
	])
	preview = cv2.vconcat([top, bottom])

	# Draw epipolar guide lines to visually check row alignment.
	h, w = preview.shape[:2]
	for y in range(40, h, 80):
		cv2.line(preview, (0, y), (w - 1, y), (255, 180, 0), 1, cv2.LINE_AA)
	return preview


def save_rectified_pair(save_root: Path, shot_idx: int, left_img: np.ndarray, right_img: np.ndarray) -> Path:
	shot_dir = save_root / f"shot_{shot_idx:04d}"
	shot_dir.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(shot_dir / "left_eye_rectified.png"), left_img)
	cv2.imwrite(str(shot_dir / "right_eye_rectified.png"), right_img)
	return shot_dir


class QuadTileSubscriber(Node):
	def __init__(self, topic: str):
		super().__init__("quad_tile_raw_subscriber")
		self.bridge = CvBridge()
		self._lock = threading.Lock()
		self._latest_frame = None

		qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
		self.create_subscription(Image, topic, self._on_image, qos)
		self.get_logger().info(f"Subscribed to {topic}")

	def _on_image(self, msg: Image) -> None:
		try:
			frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
		except CvBridgeError as exc:
			self.get_logger().warning(f"cv_bridge conversion failed: {exc}")
			return

		with self._lock:
			self._latest_frame = frame

	def pop_latest_frame(self) -> np.ndarray:
		with self._lock:
			frame = self._latest_frame
			self._latest_frame = None
		return frame


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Subscribe /quad_tile/raw, crop left/right eye, visualize fisheye rectification, "
			"compare K vs P from ros_calibration.yaml, and press s to save rectified stereo pair."
		)
	)
	parser.add_argument("--input-topic", default="/quad_tile/raw", help="ROS2 image topic")
	parser.add_argument("--calib-yaml", default="ros_calibration.yaml", help="Stereo calibration YAML")
	parser.add_argument("--crop-width", type=int, default=640, help="Center crop width for each eye")
	parser.add_argument("--crop-height", type=int, default=480, help="Center crop height for each eye")
	parser.add_argument("--save-dir", default="undistort_output", help="Output directory")
	parser.add_argument("--window", default="ros-fisheye-rectify", help="OpenCV preview window name")
	return parser


def main() -> int:
	args = build_parser().parse_args()

	yaml_path = Path(args.calib_yaml)
	if not yaml_path.exists():
		raise FileNotFoundError(f"Calibration file not found: {yaml_path}")

	save_root = Path(args.save_dir)
	save_root.mkdir(parents=True, exist_ok=True)

	stereo_params = load_ros_stereo_yaml(yaml_path)
	print_k_vs_p(stereo_params)

	map_size = (args.crop_width, args.crop_height)
	undistort_maps = build_fisheye_maps(stereo_params, map_size)

	rclpy.init()
	node = QuadTileSubscriber(args.input_topic)
	cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

	shot_idx = 1
	last_preview = None
	last_frame_t = time.perf_counter()

	print("\nRunning... Press 's' to save rectified left/right images, 'q' or ESC to quit.")

	try:
		while rclpy.ok():
			rclpy.spin_once(node, timeout_sec=0.01)

			frame = node.pop_latest_frame()
			if frame is not None:
				try:
					eyes = split_left_right_eyes(frame, args.crop_width, args.crop_height)
				except ValueError as exc:
					node.get_logger().warning(str(exc))
					continue

				left_raw = eyes["left_eye"]
				right_raw = eyes["right_eye"]

				lmap1, lmap2 = undistort_maps["left_eye"]
				rmap1, rmap2 = undistort_maps["right_eye"]
				left_rect = cv2.remap(left_raw, lmap1, lmap2, interpolation=cv2.INTER_LINEAR)
				right_rect = cv2.remap(right_raw, rmap1, rmap2, interpolation=cv2.INTER_LINEAR)

				last_preview = make_preview(left_raw, right_raw, left_rect, right_rect)
				cv2.imshow(args.window, last_preview)
				last_frame_t = time.perf_counter()
			else:
				# Keep the last frame visible when topic is temporarily silent.
				if last_preview is not None:
					cv2.imshow(args.window, last_preview)

				# Show a heartbeat in terminal every few seconds if no frame arrives.
				if time.perf_counter() - last_frame_t > 5.0:
					print("Waiting for frames from topic...", flush=True)
					last_frame_t = time.perf_counter()

			key = cv2.waitKey(1) & 0xFF
			if key in (27, ord("q")):
				break
			if key == ord("s") and last_preview is not None and frame is not None:
				out_dir = save_rectified_pair(save_root, shot_idx, left_rect, right_rect)
				print(f"Saved shot {shot_idx} to {out_dir}")
				shot_idx += 1
	finally:
		cv2.destroyAllWindows()
		node.destroy_node()
		rclpy.shutdown()

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
