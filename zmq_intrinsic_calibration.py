#!/usr/bin/env python3

import argparse
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


ZMQ_MAGIC = 0x514D4D47  # "GMMQ"
ZMQ_H264_MAGIC = 0x48323634  # "H264"


@dataclass
class CameraCalibrationResult:
    name: str
    rms: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    image_size: Tuple[int, int]


class ZmqFrameSubscriber:
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
        # Sender C++ struct may be 40 bytes (alignment padding) or 36 bytes.
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

        # Keep newest frame only to reduce lag while calibrating.
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

        # Publisher sends rgb8 by default.
        if channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image, int(timestamp_ns), int(frame_index)


class ZmqH264FrameSubscriber:
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
                print("[ZMQ-Calib] H264 decoder waiting for keyframe/SPS-PPS, dropping invalid packet(s)...")
                self._warned_decode_error = True
            return None
        except Exception:
            return None

        if not frames:
            return None

        self._warned_decode_error = False
        frame_bgr = frames[-1].to_ndarray(format="bgr24")
        return frame_bgr, int(timestamp_ns), int(frame_index)


def build_object_points(board_cols: int, board_rows: int, square_size: float) -> np.ndarray:
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    grid = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    return objp


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

    # Keep the same split rule used in crop.py.
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


def detect_corners(gray: np.ndarray, board_cols: int, board_rows: int) -> Tuple[bool, np.ndarray]:
    pattern = (board_cols, board_rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        return False, corners

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def draw_status_text(img: np.ndarray, text: str, color: Tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (8, 8), (700, 44), (0, 0, 0), -1)
    cv2.putText(out, text, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def build_views_visualization(
    views: Dict[str, np.ndarray],
    found_now: Dict[str, Tuple[bool, np.ndarray]],
    board_cols: int,
    board_rows: int,
) -> np.ndarray:
    order = [
        ("left_eye", "TL left_eye"),
        ("right_eye", "TR right_eye"),
        ("right_hand", "BL right_hand"),
        ("left_hand", "BR left_hand"),
    ]
    vis_views: Dict[str, np.ndarray] = {}
    for key, title in order:
        panel = views[key].copy()
        found, corners = found_now[key]
        if found:
            cv2.drawChessboardCorners(panel, (board_cols, board_rows), corners, True)
            panel = draw_status_text(panel, f"{title} | corners: OK", (0, 220, 0))
        else:
            panel = draw_status_text(panel, f"{title} | corners: MISS", (0, 0, 255))
        vis_views[key] = panel

    top = cv2.hconcat([vis_views["left_eye"], vis_views["right_eye"]])
    bottom = cv2.hconcat([vis_views["right_hand"], vis_views["left_hand"]])
    return cv2.vconcat([top, bottom])


def calibrate_single_camera(
    name: str,
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    image_size: Tuple[int, int],
) -> CameraCalibrationResult:
    if len(object_points) != len(image_points):
        raise ValueError(f"{name}: object/image points size mismatch.")
    if not object_points:
        raise ValueError(f"{name}: no valid samples for calibration.")

    # cv2.fisheye.calibrate requires per-view points with shape (N, 1, 3)/(N, 1, 2) in float64.
    obj_points_fisheye = [pts.reshape(-1, 1, 3).astype(np.float64) for pts in object_points]
    img_points_fisheye = [pts.reshape(-1, 1, 2).astype(np.float64) for pts in image_points]

    k = np.eye(3, dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    rms, k, dist, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_points_fisheye,
        img_points_fisheye,
        image_size,
        k,
        dist,
        None,
        None,
        flags,
        criteria,
    )

    return CameraCalibrationResult(
        name=name,
        rms=rms,
        camera_matrix=k,
        dist_coeffs=dist,
        rvecs=rvecs,
        tvecs=tvecs,
        image_size=image_size,
    )


def save_intrinsic_text(result: CameraCalibrationResult, file_path: Path) -> None:
    d = result.dist_coeffs.reshape(-1)
    d1 = d[0] if d.size > 0 else 0.0
    d2 = d[1] if d.size > 1 else 0.0
    d3 = d[2] if d.size > 2 else 0.0
    d4 = d[3] if d.size > 3 else 0.0

    text = [
        f"NAME:{result.name}",
        "MODEL:FISHEYE",
        f"IMAGE_WIDTH:{result.image_size[0]}",
        f"IMAGE_HEIGHT:{result.image_size[1]}",
        f"FX:{result.camera_matrix[0, 0]:.10f}",
        f"FY:{result.camera_matrix[1, 1]:.10f}",
        f"CX:{result.camera_matrix[0, 2]:.10f}",
        f"CY:{result.camera_matrix[1, 2]:.10f}",
        f"D1:{d1:.10f}",
        f"D2:{d2:.10f}",
        f"D3:{d3:.10f}",
        f"D4:{d4:.10f}",
        f"RMS:{result.rms:.6f}",
    ]
    file_path.write_text("\n".join(text) + "\n", encoding="utf-8")


def run_intrinsic_calibration(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir = output_dir / "intrinsics"
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    if args.stream_codec == "h264":
        subscriber = ZmqH264FrameSubscriber(args.zmq_endpoint)
    else:
        subscriber = ZmqFrameSubscriber(args.zmq_endpoint)
    objp = build_object_points(args.board_cols, args.board_rows, args.square_size)

    per_cam_obj: Dict[str, List[np.ndarray]] = {
        "left_eye": [],
        "right_eye": [],
        "right_hand": [],
        "left_hand": [],
    }
    per_cam_img: Dict[str, List[np.ndarray]] = {
        "left_eye": [],
        "right_eye": [],
        "right_hand": [],
        "left_hand": [],
    }

    frame_idx = 0
    sampled_frames = 0
    capture_events = 0
    image_size: Optional[Tuple[int, int]] = None

    last_t = time.perf_counter()
    fps_counter = 0
    fps = 0.0

    if not args.visualize:
        raise ValueError("Manual capture requires visualization. Please use --visualize.")

    print(f"[ZMQ-Calib] Connected (codec={args.stream_codec}).")
    print("[ZMQ-Calib] Controls: s=capture current detections, c=compute calibration, q/esc=quit.")

    while True:
        recv = subscriber.recv_latest_image()
        if recv is None:
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            continue

        stitched, timestamp_ns, remote_frame_idx = recv
        # Keep this log lightweight and safe after None-check.
        # print(f"[ZMQ-Calib] Received frame {remote_frame_idx} at {timestamp_ns} ns")

        try:
            views = split_and_crop_like_crop_py(
                stitched,
                crop_width=args.crop_width,
                crop_height=args.crop_height,
                top_extra_rows=args.top_extra_rows,
            )
        except ValueError as exc:
            print(f"[ZMQ-Calib] Skip frame due to crop/split error: {exc}")
            frame_idx += 1
            continue

        if image_size is None:
            h, w = views["left_eye"].shape[:2]
            image_size = (w, h)

        found_now: Dict[str, Tuple[bool, np.ndarray]] = {}
        found_count = 0
        for name, img in views.items():
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = detect_corners(gray, args.board_cols, args.board_rows)
            found_now[name] = (found, corners)
            if found:
                found_count += 1

        sampled_frames += 1
        frame_idx += 1

        fps_counter += 1
        now = time.perf_counter()
        if now - last_t >= 1.0:
            fps = fps_counter / (now - last_t)
            fps_counter = 0
            last_t = now

        view_vis = build_views_visualization(views, found_now, args.board_cols, args.board_rows)
        text = (
            f"remote={remote_frame_idx} sampled={sampled_frames} found={found_count}/4 fps={fps:.2f} "
            f"captures={capture_events} | s=capture c=compute q=quit"
        )
        view_vis = draw_status_text(view_vis, text, (0, 255, 255))
        cv2.imshow(args.window, view_vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break

        if key == ord("s"):
            added = 0
            for name in ["left_eye", "right_eye", "right_hand", "left_hand"]:
                found, corners = found_now[name]
                if found:
                    per_cam_obj[name].append(objp.copy())
                    per_cam_img[name].append(corners)
                    added += 1

            capture_events += 1
            counts = ", ".join(
                [
                    f"left_eye={len(per_cam_img['left_eye'])}",
                    f"right_eye={len(per_cam_img['right_eye'])}",
                    f"right_hand={len(per_cam_img['right_hand'])}",
                    f"left_hand={len(per_cam_img['left_hand'])}",
                ]
            )
            print(f"[ZMQ-Calib] Capture #{capture_events}: added {added} view(s). counts: {counts}")

        if key == ord("c"):
            print("[ZMQ-Calib] Compute requested by user.")
            break

    cv2.waitKey(1)
    cv2.destroyAllWindows()

    if image_size is None:
        raise RuntimeError("No valid frame was received from ZMQ.")

    print("\nCollected samples:")
    print(f"  sampled_frames = {sampled_frames}")
    for k in ["left_eye", "right_eye", "right_hand", "left_hand"]:
        print(f"  {k}: valid chessboard frames = {len(per_cam_img[k])}")

    for k in ["left_eye", "right_eye", "right_hand", "left_hand"]:
        if len(per_cam_img[k]) < args.min_samples:
            raise RuntimeError(
                f"{k} has only {len(per_cam_img[k])} valid frames (< min_samples={args.min_samples})."
            )

    results: Dict[str, CameraCalibrationResult] = {}
    for k in ["left_eye", "right_eye", "right_hand", "left_hand"]:
        results[k] = calibrate_single_camera(k, per_cam_obj[k], per_cam_img[k], image_size)

        np.savez(
            intrinsic_dir / f"{k}_intrinsics.npz",
            rms=results[k].rms,
            camera_matrix=results[k].camera_matrix,
            dist_coeffs=results[k].dist_coeffs,
            image_width=results[k].image_size[0],
            image_height=results[k].image_size[1],
        )
        save_intrinsic_text(results[k], intrinsic_dir / f"{k}_intrinsics.txt")

    fs = cv2.FileStorage(str(output_dir / "calibration_intrinsics.yaml"), cv2.FILE_STORAGE_WRITE)
    for name, res in results.items():
        fs.write(f"{name}_rms", float(res.rms))
        fs.write(f"{name}_K", res.camera_matrix)
        fs.write(f"{name}_D", res.dist_coeffs)
    fs.release()

    print("\nIntrinsic calibration finished (no extrinsics).")
    print("  Camera model = cv2.fisheye")
    for name in ["left_eye", "right_eye", "right_hand", "left_hand"]:
        print(f"  {name} RMS = {results[name].rms:.6f}")
    print(f"  Outputs in: {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Subscribe stitched image from ZMQ, split/crop using crop.py rule, "
            "and run per-camera fisheye intrinsic calibration only."
        )
    )
    parser.add_argument(
        "--zmq-endpoint",
        default="tcp://192.168.1.15:5556",
        help="ZMQ endpoint, e.g. tcp://192.168.1.10:5556",
    )
    parser.add_argument(
        "--stream-codec",
        choices=["raw", "h264"],
        default="h264",
        help="ZMQ stream codec. Use h264 when publisher is ZmqH264Viewer format.",
    )
    parser.add_argument("--board-cols", type=int, default=8, help="Chessboard inner corners along width.")
    parser.add_argument("--board-rows", type=int, default=11, help="Chessboard inner corners along height.")
    parser.add_argument(
        "--square-size",
        type=float,
        default=35.0,
        help="Chessboard square size (unit for output).",
    )
    parser.add_argument("--crop-width", type=int, default=640, help="Crop width for each single view.")
    parser.add_argument("--crop-height", type=int, default=480, help="Crop height for each single view.")
    parser.add_argument(
        "--top-extra-rows",
        type=int,
        default=0,
        help="Extra rows added to top views before center crop, same as crop.py.",
    )
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum valid samples per camera.")
    parser.add_argument("--output-dir", type=str, default="calib_output", help="Output directory.")
    parser.add_argument("--window", default="zmq-intrinsic-calibration", help="OpenCV window name")
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Show visualization window (required for manual capture).",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Disable visualization window.",
    )
    parser.set_defaults(visualize=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_intrinsic_calibration(args)
        return 0
    except ImportError:
        if args.stream_codec == "h264":
            print("Missing dependency: pyzmq/av. Install with 'pip install pyzmq av'.")
        else:
            print("Missing dependency: pyzmq. Install with 'pip install pyzmq'.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
