#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from cv_bridge import CvBridge, CvBridgeError
    from sensor_msgs.msg import Image
except ImportError as exc:
    raise ImportError(
        "Missing ROS2 dependencies. Please install rclpy, sensor_msgs, and cv_bridge."
    ) from exc


VIEW_ORDER = ["left_eye", "right_eye"]
CROP_HEIGHT = 480


def center_crop(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w < target_w or h < target_h:
        raise ValueError(f"View size {w}x{h} is smaller than crop size {target_w}x{target_h}.")
    x0 = (w - target_w) // 2
    y0 = (h - target_h) // 2
    return img[y0 : y0 + target_h, x0 : x0 + target_w]


def split_and_crop(frame: np.ndarray, crop_width: int) -> Dict[str, np.ndarray]:
    h, w = frame.shape[:2]
    if w % 2 != 0 or h % 2 != 0:
        raise ValueError(f"Stitched frame size must be even, got {w}x{h}")

    half_w = w // 2
    half_h = h // 2
    left_eye = frame[0:half_h, 0:half_w]
    right_eye = frame[0:half_h, half_w:w]
    right_hand = frame[half_h:h, 0:half_w]
    left_hand = frame[half_h:h, half_w:w]

    return {
        "left_eye": center_crop(left_eye, crop_width, CROP_HEIGHT),
        "right_eye": center_crop(right_eye, crop_width, CROP_HEIGHT),
        "right_hand": center_crop(right_hand, crop_width, CROP_HEIGHT),
        "left_hand": center_crop(left_hand, crop_width, CROP_HEIGHT),
    }


def _extract_int(text: str, key: str) -> int:
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(\d+)\s*$", text, re.MULTILINE)
    if not m:
        raise ValueError(f"Missing key: {key}")
    return int(m.group(1))


def _extract_str(text: str, key: str, default: str = "") -> str:
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    if not m:
        return default
    return m.group(1).strip().strip('"').strip("'")


def _extract_block_data(text: str, key: str) -> np.ndarray:
    pattern = (
        rf"^\s*{re.escape(key)}\s*:\s*\n"
        rf"(?:^\s+.*\n)*?"
        rf"^\s*data\s*:\s*\[(.*?)\]"
    )
    m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not m:
        raise ValueError(f"Missing matrix block/data: {key}")
    values = [v.strip() for v in m.group(1).replace("\n", " ").split(",") if v.strip()]
    return np.asarray([float(v) for v in values], dtype=np.float64)


def load_camera_info_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "image_width": _extract_int(text, "image_width"),
        "image_height": _extract_int(text, "image_height"),
        "camera_name": _extract_str(text, "camera_name", path.stem),
        "distortion_model": _extract_str(text, "distortion_model", "fisheye"),
        "K": _extract_block_data(text, "camera_matrix").reshape(3, 3),
        "D": _extract_block_data(text, "distortion_coefficients").reshape(-1),
        "R": _extract_block_data(text, "rectification_matrix").reshape(3, 3),
    }


def compute_new_projection(k: np.ndarray, d: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    d4 = d[:4].reshape(4, 1)
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        k,
        d4,
        (width, height),
        np.eye(3, dtype=np.float64),
        balance=0.0,
        new_size=(width, height),
        fov_scale=1.0,
    )
    p = np.zeros((3, 4), dtype=np.float64)
    p[:, :3] = new_k
    p[2, 2] = 1.0
    return p


def as_list(arr: np.ndarray) -> list:
    return [float(v) for v in arr.reshape(-1)]


def write_ros_calibration_yaml(left_info: Dict[str, Any], right_info: Dict[str, Any], out_path: Path) -> None:
    left_p = compute_new_projection(left_info["K"], left_info["D"], (left_info["image_width"], left_info["image_height"]))
    right_p = compute_new_projection(
        right_info["K"],
        right_info["D"],
        (right_info["image_width"], right_info["image_height"]),
    )

    def block(name: str, info: Dict[str, Any], p: np.ndarray) -> str:
        return (
            f"{name}:\n"
            f"  calibration_type: stereo_fisheye\n"
            f"  image:\n"
            f"    width: {info['image_width']}\n"
            f"    height: {info['image_height']}\n"
            f"  distortion_model: {info['distortion_model']}\n"
            f"  distortion_coefficients:\n"
            f"    D: {as_list(info['D'])}\n"
            f"  camera_matrix:\n"
            f"    K: {as_list(info['K'])}\n"
            f"  rectification_matrix:\n"
            f"    R: {as_list(info['R'])}\n"
            f"  projection_matrix:\n"
            f"    P: {as_list(p)}\n"
        )

    text = block("left_eye", left_info, left_p) + "\n" + block("right_eye", right_info, right_p)
    out_path.write_text(text, encoding="utf-8")


def find_camera_info_file(camera_key: str) -> Optional[Path]:
    cam_info_dir = Path.home() / ".ros" / "camera_info"
    if not cam_info_dir.exists():
        return None

    candidates = sorted(cam_info_dir.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if camera_key in p.stem:
            return p
    return None


class StereoCalibrationBridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("quad_tile_stereo_fisheye_bridge")
        self.bridge = CvBridge()
        self.crop_width = args.crop_width
        self.should_exit = False
        self.calib_proc: Optional[subprocess.Popen] = None

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.output_prefix = "/quad_tile"
        self.pubs: Dict[str, Any] = {}
        for name in VIEW_ORDER:
            topic = f"{self.output_prefix}/{name}/image_raw"
            self.pubs[name] = self.create_publisher(Image, topic, qos)
            self.get_logger().info(f"Publishing {topic}")

        self.sub = self.create_subscription(Image, args.input_topic, self.on_image, qos)
        self.get_logger().info(f"Subscribed to {args.input_topic}")

        self.calib_proc = self.launch_stereo_calibrator(args.board_cols, args.board_rows, args.square_size)
        self.timer = self.create_timer(0.5, self.check_process)

    def on_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            views = split_and_crop(frame, self.crop_width)
        except (CvBridgeError, ValueError) as exc:
            self.get_logger().warning(f"split/crop failed: {exc}")
            return

        for name in VIEW_ORDER:
            out = self.bridge.cv2_to_imgmsg(views[name], encoding="bgr8")
            out.header = msg.header
            out.header.frame_id = name
            self.pubs[name].publish(out)

    def launch_stereo_calibrator(self, board_cols: int, board_rows: int, square_size: float) -> subprocess.Popen:
        cmd = [
            "ros2",
            "run",
            "camera_calibration",
            "cameracalibrator",
            "--size",
            f"{board_cols}x{board_rows}",
            "--square",
            str(square_size),
            "--queue-size",
            "5",
            "--approximate",
            "0.05",
            "--fisheye-fix-skew",
            "--fisheye-recompute-extrinsicsts",
            "--ros-args",
            "-r",
            f"left:={self.output_prefix}/left_eye/image_raw",
            "-r",
            f"right:={self.output_prefix}/right_eye/image_raw",
            "-r",
            f"left_camera:={self.output_prefix}/left_eye",
            "-r",
            f"right_camera:={self.output_prefix}/right_eye",
        ]
        self.get_logger().info("Launching: " + " ".join(cmd))
        return subprocess.Popen(cmd)

    def check_process(self) -> None:
        if self.calib_proc is None:
            return
        ret = self.calib_proc.poll()
        if ret is None:
            return

        self.get_logger().info(f"camera_calibration exited with code {ret}")
        if ret == 0:
            try:
                left_yaml = find_camera_info_file("left_eye")
                right_yaml = find_camera_info_file("right_eye")
                if left_yaml is None or right_yaml is None:
                    raise RuntimeError("Cannot find left/right camera_info yaml in ~/.ros/camera_info")
                left_info = load_camera_info_yaml(left_yaml)
                right_info = load_camera_info_yaml(right_yaml)
                out = Path("ros_calibration.yaml")
                write_ros_calibration_yaml(left_info, right_info, out)
                self.get_logger().info(f"Saved {out.resolve()}")
            except Exception as exc:
                self.get_logger().error(f"Failed to export ros_calibration.yaml: {exc}")

        self.should_exit = True

    def shutdown(self) -> None:
        if self.calib_proc is not None and self.calib_proc.poll() is None:
            self.calib_proc.terminate()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Subscribe /quad_tile/raw, split to left/right images, run ROS camera_calibration stereo fisheye, "
            "and export ros_calibration.yaml."
        )
    )
    parser.add_argument("--crop-width", type=int, default=640, help="Crop width per view")
    parser.add_argument("--input-topic", default="/quad_tile/raw", help="Input stitched topic")
    parser.add_argument("--board-cols", type=int, default=8, help="Chessboard inner corners along width")
    parser.add_argument("--board-rows", type=int, default=11, help="Chessboard inner corners along height")
    parser.add_argument("--square-size", type=float, default=0.035, help="Chessboard square size in meters")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rclpy.init()
    node = StereoCalibrationBridge(args)

    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
