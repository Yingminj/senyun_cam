#!/usr/bin/env python3

import argparse
import struct
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


ZMQ_MAGIC = 0x514D4D47  # "GMMQ"
ZMQ_H264_MAGIC = 0x48323634  # "H264"
VIEW_ORDER = ["left_eye", "right_eye", "right_hand", "left_hand"]


class ZmqRawSubscriber:
	def __init__(self, endpoint: str):
		import zmq

		self._zmq = zmq
		self._ctx = zmq.Context.instance()
		self._sock = self._ctx.socket(zmq.SUB)
		self._sock.setsockopt(zmq.SUBSCRIBE, b"")
		self._sock.setsockopt(zmq.RCVHWM, 1)
		self._sock.connect(endpoint)

	@staticmethod
	def _parse_header(header: bytes) -> Optional[Tuple[int, int, int, int, int, int, int]]:
		if len(header) >= 40:
			return struct.unpack("<IIIII4xQQ", header[:40])
		if len(header) >= 36:
			return struct.unpack("<IIIIIQQ", header[:36])
		return None

	def recv_latest_image(self) -> Optional[Tuple[np.ndarray, int, int]]:
		try:
			parts = self._sock.recv_multipart()
		except Exception:
			return None

		while True:
			try:
				parts = self._sock.recv_multipart(flags=self._zmq.NOBLOCK)
			except self._zmq.Again:
				break

		if len(parts) != 2:
			return None

		header, payload = parts
		parsed = self._parse_header(header)
		if parsed is None:
			return None

		magic, width, height, channels, step, timestamp_ns, frame_index = parsed
		if magic != ZMQ_MAGIC:
			return None

		expected = int(step) * int(height)
		if len(payload) < expected:
			return None

		if channels not in (1, 3):
			return None

		buf = np.frombuffer(payload, dtype=np.uint8, count=expected)
		image = buf.reshape((height, step))[:, : width * channels]
		image = image.reshape((height, width, channels))

		if channels == 3:
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		else:
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

		return image, int(timestamp_ns), int(frame_index)


class ZmqH264Subscriber:
	def __init__(self, endpoint: str):
		import zmq
		import av

		self._zmq = zmq
		self._ctx = zmq.Context.instance()
		self._sock = self._ctx.socket(zmq.SUB)
		self._sock.setsockopt(zmq.SUBSCRIBE, b"")
		self._sock.setsockopt(zmq.RCVHWM, 1)
		self._sock.connect(endpoint)

		self._av = av
		self._decoder = av.CodecContext.create("h264", "r")
		self._warned_decode_error = False

	@staticmethod
	def _parse_header(header: bytes) -> Optional[Tuple[int, int, int, int, int, int, int, int]]:
		if len(header) < 40:
			return None
		return struct.unpack("<IIIIQQII", header[:40])

	def recv_latest_image(self) -> Optional[Tuple[np.ndarray, int, int]]:
		try:
			parts = self._sock.recv_multipart()
		except Exception:
			return None

		while True:
			try:
				parts = self._sock.recv_multipart(flags=self._zmq.NOBLOCK)
			except self._zmq.Again:
				break

		if len(parts) != 2:
			return None

		header, payload = parts
		parsed = self._parse_header(header)
		if parsed is None:
			return None

		magic, _width, _height, _flags, timestamp_ns, frame_index, payload_bytes, _reserved = parsed
		if magic != ZMQ_H264_MAGIC:
			return None

		if payload_bytes > len(payload):
			return None

		packet_bytes = payload[:payload_bytes]

		try:
			parsed_packets = self._decoder.parse(packet_bytes)
			frames = []
			for pkt in parsed_packets:
				frames.extend(self._decoder.decode(pkt))
		except self._av.error.InvalidDataError:
			if not self._warned_decode_error:
				print("[Distort] H264 decoder waiting for keyframe/SPS-PPS...")
				self._warned_decode_error = True
			return None
		except Exception:
			return None

		if not frames:
			return None

		self._warned_decode_error = False
		frame_bgr = frames[-1].to_ndarray(format="bgr24")
		return frame_bgr, int(timestamp_ns), int(frame_index)


def center_crop(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
	h, w = img.shape[:2]
	if w < target_w or h < target_h:
		raise ValueError(f"View size {w}x{h} is smaller than crop size {target_w}x{target_h}.")

	x0 = (w - target_w) // 2
	y0 = (h - target_h) // 2
	return img[y0 : y0 + target_h, x0 : x0 + target_w]


def split_and_crop_like_crop_py(
	frame: np.ndarray,
	crop_width: int,
	crop_height: int,
	top_extra_rows: int,
) -> Dict[str, np.ndarray]:
	h, w = frame.shape[:2]
	if w % 2 != 0 or h % 2 != 0:
		raise ValueError(f"Stitched frame size must be even, got {w}x{h}")

	half_w = w // 2
	half_h = h // 2
	top_h = min(h, half_h + top_extra_rows)

	left_eye = frame[0:top_h, 0:half_w]
	right_eye = frame[0:top_h, half_w:w]
	right_hand = frame[half_h:h, 0:half_w]
	left_hand = frame[half_h:h, half_w:w]

	return {
		"left_eye": center_crop(left_eye, crop_width, crop_height),
		"right_eye": center_crop(right_eye, crop_width, crop_height),
		"right_hand": center_crop(right_hand, crop_width, crop_height),
		"left_hand": center_crop(left_hand, crop_width, crop_height),
	}


def load_intrinsics_from_yaml(yaml_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
	fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
	if not fs.isOpened():
		raise RuntimeError(f"Failed to open calibration yaml: {yaml_path}")

	intrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
	try:
		for name in VIEW_ORDER:
			k = fs.getNode(f"{name}_K").mat()
			d = fs.getNode(f"{name}_D").mat()
			if k is None or d is None:
				raise ValueError(f"Missing node in yaml: {name}_K or {name}_D")
			intrinsics[name] = (k, d)
	finally:
		fs.release()

	return intrinsics


def build_undistort_maps(
	intrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]],
	view_shape: Tuple[int, int],
	alpha: float,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, np.ndarray]]:
	view_h, view_w = view_shape
	maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
	new_intrinsics: Dict[str, np.ndarray] = {}

	for name in VIEW_ORDER:
		k, d = intrinsics[name]
		new_k, _ = cv2.getOptimalNewCameraMatrix(k, d, (view_w, view_h), alpha)
		map1, map2 = cv2.initUndistortRectifyMap(k, d, None, new_k, (view_w, view_h), cv2.CV_16SC2)
		maps[name] = (map1, map2)
		new_intrinsics[name] = new_k

	return maps, new_intrinsics


def print_undistorted_intrinsics(new_intrinsics: Dict[str, np.ndarray]) -> None:
	print("[Distort] Undistorted camera intrinsics (new_K):")
	for name in VIEW_ORDER:
		print(f"[Distort] {name}_K =")
		print(np.array2string(new_intrinsics[name], precision=8, suppress_small=False))


def stitch_views(views: Dict[str, np.ndarray]) -> np.ndarray:
	top = cv2.hconcat([views["left_eye"], views["right_eye"]])
	bottom = cv2.hconcat([views["right_hand"], views["left_hand"]])
	return cv2.vconcat([top, bottom])


def save_views_once(save_root: Path, shot_idx: int, views: Dict[str, np.ndarray]) -> Path:
	shot_dir = save_root / f"shot_{shot_idx:04d}"
	shot_dir.mkdir(parents=True, exist_ok=True)

	# Keep exact requested names per view.
	cv2.imwrite(str(shot_dir / "left_eye.png"), views["left_eye"])
	cv2.imwrite(str(shot_dir / "right_eye.png"), views["right_eye"])
	cv2.imwrite(str(shot_dir / "left_hand.png"), views["left_hand"])
	cv2.imwrite(str(shot_dir / "right_hand.png"), views["right_hand"])
	return shot_dir


def main() -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Subscribe stitched frames from ZMQ, crop each view like crop.py, undistort each camera view, "
			"and press 's' to save one shot (4 images)."
		)
	)
	parser.add_argument("--zmq-endpoint", default="tcp://192.168.1.15:5556", help="ZMQ endpoint")
	parser.add_argument(
		"--stream-codec",
		choices=["raw", "h264"],
		default="h264",
		help="ZMQ stream codec",
	)
	parser.add_argument(
		"--calib-yaml",
		default="calib_output/calibration_intrinsics.yaml",
		help="Calibration YAML containing *_K and *_D for 4 cameras",
	)
	parser.add_argument("--crop-width", type=int, default=640, help="Crop width per view")
	parser.add_argument("--crop-height", type=int, default=480, help="Crop height per view")
	parser.add_argument(
		"--top-extra-rows",
		type=int,
		default=0,
		help="Extra rows for top row split before center crop, same as crop.py",
	)
	parser.add_argument(
		"--alpha",
		type=float,
		default=0.0,
		help="Undistort alpha in getOptimalNewCameraMatrix, 0.0 keeps less black border",
	)
	parser.add_argument("--save-dir", default="undistort_output", help="Output root directory")
	parser.add_argument("--window", default="zmq-fisheye-undistort", help="Preview window")
	args = parser.parse_args()

	calib_yaml = Path(args.calib_yaml)
	if not calib_yaml.exists():
		raise FileNotFoundError(f"Calibration yaml not found: {calib_yaml}")

	save_root = Path(args.save_dir)
	save_root.mkdir(parents=True, exist_ok=True)

	intrinsics = load_intrinsics_from_yaml(calib_yaml)

	if args.stream_codec == "h264":
		subscriber = ZmqH264Subscriber(args.zmq_endpoint)
	else:
		subscriber = ZmqRawSubscriber(args.zmq_endpoint)

	maps: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
	shot_idx = 1
	fps_counter = 0
	fps = 0.0
	last_t = time.perf_counter()

	print("[Distort] Running. Press 's' to save one shot (4 images), 'q' or ESC to quit.")

	while True:
		recv = subscriber.recv_latest_image()
		if recv is None:
			key = cv2.waitKey(1) & 0xFF
			if key in (27, ord("q")):
				break
			continue

		stitched, timestamp_ns, frame_index = recv
		try:
			cropped_views = split_and_crop_like_crop_py(
				stitched,
				crop_width=args.crop_width,
				crop_height=args.crop_height,
				top_extra_rows=args.top_extra_rows,
			)
		except ValueError as exc:
			print(f"[Distort] Skip frame due to split/crop error: {exc}")
			continue

		if maps is None:
			first = cropped_views["left_eye"]
			maps, new_intrinsics = build_undistort_maps(intrinsics, (first.shape[0], first.shape[1]), args.alpha)
			print_undistorted_intrinsics(new_intrinsics)

		undist_views: Dict[str, np.ndarray] = {}
		for name in VIEW_ORDER:
			map1, map2 = maps[name]
			undist_views[name] = cv2.remap(cropped_views[name], map1, map2, interpolation=cv2.INTER_LINEAR)

		vis = stitch_views(undist_views)

		fps_counter += 1
		now = time.perf_counter()
		if now - last_t >= 1.0:
			fps = fps_counter / (now - last_t)
			fps_counter = 0
			last_t = now

		info = f"frame={frame_index} fps={fps:.2f} ts_ns={timestamp_ns} | s=save q=quit"
		cv2.rectangle(vis, (8, 8), (980, 44), (0, 0, 0), -1)
		cv2.putText(vis, info, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
		cv2.imshow(args.window, vis)

		key = cv2.waitKey(1) & 0xFF
		if key in (27, ord("q")):
			break
		if key == ord("s"):
			shot_dir = save_views_once(save_root, shot_idx, undist_views)
			print(f"[Distort] Saved shot {shot_idx} to: {shot_dir}")
			shot_idx += 1

	cv2.destroyAllWindows()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

