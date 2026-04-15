import argparse
import sys

import cv2
import numpy as np

try:
	import pyrealsense2 as rs
except ImportError as exc:
	raise ImportError(
		"无法导入 pyrealsense2，请先安装 RealSense Python SDK。"
	) from exc

try:
	import apriltag
except ImportError as exc:
	raise ImportError("无法导入 apriltag，请先执行: pip install apriltag") from exc


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="使用 RealSense + apriltag 检测指定 AprilTag 并估计位姿"
	)
	parser.add_argument("--tag-size", type=float, required=True, help="Tag 边长，单位米")
	parser.add_argument("--tag-id", type=int, required=True, help="目标 Tag 的 ID")
	parser.add_argument("--fx", type=float, required=True, help="相机内参 fx")
	parser.add_argument("--fy", type=float, required=True, help="相机内参 fy")
	parser.add_argument("--cx", type=float, required=True, help="相机内参 cx")
	parser.add_argument("--cy", type=float, required=True, help="相机内参 cy")
	parser.add_argument("--width", type=int, default=640, help="彩色流宽度")
	parser.add_argument("--height", type=int, default=480, help="彩色流高度")
	parser.add_argument("--fps", type=int, default=30, help="彩色流帧率")
	parser.add_argument(
		"--families",
		type=str,
		default="tag36h11",
		help="AprilTag family，例如 tag36h11",
	)
	return parser.parse_args()


def build_detector(families: str) -> apriltag.Detector:
	options = apriltag.DetectorOptions(families=families)
	return apriltag.Detector(options)


def draw_axes(
	image: np.ndarray,
	camera_matrix: np.ndarray,
	dist_coeffs: np.ndarray,
	rotation_matrix: np.ndarray,
	translation: np.ndarray,
	axis_length: float,
) -> None:
	rvec, _ = cv2.Rodrigues(rotation_matrix)
	tvec = translation.reshape(3, 1)
	axis_points = np.float32(
		[
			[0.0, 0.0, 0.0],
			[axis_length, 0.0, 0.0],
			[0.0, axis_length, 0.0],
			[0.0, 0.0, axis_length],
		]
	)
	img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
	p0, px, py, pz = [tuple(np.int32(pt.ravel())) for pt in img_points]
	cv2.line(image, p0, px, (0, 0, 255), 2)
	cv2.line(image, p0, py, (0, 255, 0), 2)
	cv2.line(image, p0, pz, (255, 0, 0), 2)
	cv2.putText(image, "X", px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(image, "Y", py, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	cv2.putText(image, "Z", pz, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def draw_detection_box(image: np.ndarray, corners: np.ndarray) -> None:
	corners = corners.astype(np.int32)
	for i in range(4):
		p1 = tuple(corners[i])
		p2 = tuple(corners[(i + 1) % 4])
		cv2.line(image, p1, p2, (0, 255, 255), 2)


def main() -> int:
	args = parse_args()
	detector = build_detector(args.families)
	camera_params = (args.fx, args.fy, args.cx, args.cy)
	camera_matrix = np.array(
		[[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
		dtype=np.float64,
	)
	dist_coeffs = np.zeros((4, 1), dtype=np.float64)

	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

	try:
		pipeline.start(config)
	except RuntimeError as exc:
		print(f"启动 RealSense 失败: {exc}")
		return 1

	print("按 q 或 Esc 退出")

	try:
		while True:
			frames = pipeline.wait_for_frames()
			color_frame = frames.get_color_frame()
			if not color_frame:
				continue

			image = np.asanyarray(color_frame.get_data())
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			detections = detector.detect(gray)

			has_target = False
			for det in detections:
				if det.tag_id != args.tag_id:
					continue

				has_target = True
				draw_detection_box(image, det.corners)

				pose, _, _ = detector.detection_pose(det, camera_params, args.tag_size)
				rotation_matrix = pose[:3, :3]
				translation = pose[:3, 3]

				draw_axes(
					image,
					camera_matrix,
					dist_coeffs,
					rotation_matrix,
					translation,
					axis_length=args.tag_size * 0.5,
				)

				center = tuple(np.int32(det.center))
				text_lines = [
					f"ID: {det.tag_id}",
					f"t = [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}] m",
				]
				for idx, line in enumerate(text_lines):
					cv2.putText(
						image,
						line,
						(center[0] + 10, center[1] + 20 + idx * 22),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.6,
						(0, 255, 0),
						2,
					)

			if not has_target:
				cv2.putText(
					image,
					f"Tag ID {args.tag_id} not found",
					(20, 40),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.8,
					(0, 0, 255),
					2,
				)

			cv2.imshow("AprilTag Pose (RealSense)", image)
			key = cv2.waitKey(1) & 0xFF
			if key in (27, ord("q")):
				break

	finally:
		pipeline.stop()
		cv2.destroyAllWindows()

	return 0


if __name__ == "__main__":
	sys.exit(main())
