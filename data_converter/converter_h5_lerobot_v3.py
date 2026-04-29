from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

h5py = None
np = None
LeRobotV3Writer = None

DEFAULT_CAMERAS: Sequence[str] = ("wrist", "side_1", "side_2")
DEFAULT_STATE_FIELDS: Sequence[str] = ("ee_pose", "q_follower", "gripper_state")
DEFAULT_ACTION_FIELDS: Sequence[str] = ("cmd_ee_pose", "q_cmd", "gripper_action")


def _require_h5py():
    global h5py
    if h5py is not None:
        return h5py
    try:
        import h5py as h5py_module
    except ImportError as exc:
        raise ImportError(
            "h5py is required for .h5 conversion. Please run this script in the project "
            "environment that has h5py installed."
        ) from exc
    h5py = h5py_module
    return h5py_module


def _require_runtime() -> None:
    global np, LeRobotV3Writer
    _require_h5py()
    if np is None:
        try:
            import numpy as np_module
        except ImportError as exc:
            raise ImportError(
                "numpy is required for .h5 conversion. Please run this script in the project "
                "environment that has numpy installed."
            ) from exc
        np = np_module
    if LeRobotV3Writer is None:
        from diffusion_policy.common.lerobot_v3_io import LeRobotV3Writer as writer_cls

        LeRobotV3Writer = writer_cls


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(width * current / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _parse_csv(value: Optional[str], default: Sequence[str]) -> List[str]:
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _episode_paths(input_path: Path, max_episodes: Optional[int]) -> List[Path]:
    input_path = input_path.expanduser()
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(input_path.glob("episode_*.h5"))
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise RuntimeError(f"No episode_*.h5 files found: {input_path}")
    return paths


def _as_1d_float(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected one 3-D image frame, got shape {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _dataset_shape(h5_file, path: str) -> tuple[int, ...]:
    if path not in h5_file:
        raise KeyError(f"Missing HDF5 dataset: {path}")
    dataset = h5_file[path]
    if not isinstance(dataset, h5py.Dataset):
        raise KeyError(f"HDF5 path is not a dataset: {path}")
    return tuple(int(dim) for dim in dataset.shape)


def _feature_dim(h5_file, fields: Sequence[str]) -> int:
    return int(sum(np.prod(_dataset_shape(h5_file, field)[1:], dtype=np.int64) for field in fields))


def _image_shape(h5_file, camera: str) -> tuple[int, int, int]:
    shape = _dataset_shape(h5_file, f"cameras/{camera}/frames")
    if len(shape) != 4:
        raise ValueError(f"Expected cameras/{camera}/frames to be 4-D, got {shape}")
    frame_shape = shape[1:]
    if frame_shape[0] in (1, 3) and frame_shape[-1] not in (1, 3):
        c, h, w = frame_shape
        return (h, w, c)
    h, w, c = frame_shape
    return (h, w, c)


def _field_names(fields: Sequence[str], h5_file) -> List[str]:
    names: List[str] = []
    for field in fields:
        dim = int(np.prod(_dataset_shape(h5_file, field)[1:], dtype=np.int64))
        if dim == 1:
            names.append(field)
        else:
            names.extend([f"{field}.{i}" for i in range(dim)])
    return names


class H5EpisodeReader:
    def __init__(
        self,
        path: Path,
        *,
        cameras: Sequence[str],
        state_fields: Sequence[str],
        action_fields: Sequence[str],
        fps: int,
    ):
        self.path = Path(path)
        self.file = h5py.File(self.path, "r")
        self.cameras = list(cameras)
        self.state_fields = list(state_fields)
        self.action_fields = list(action_fields)
        self.fps = int(fps)

        for camera in self.cameras:
            _dataset_shape(self.file, f"cameras/{camera}/frames")
            _dataset_shape(self.file, f"cameras/{camera}/timestamp_us")
        for field in [*self.state_fields, *self.action_fields, "timestamp_us"]:
            _dataset_shape(self.file, field)

        lengths = [int(self.file[field].shape[0]) for field in [*self.state_fields, *self.action_fields, "timestamp_us"]]
        lengths.extend(int(self.file[f"cameras/{camera}/frames"].shape[0]) for camera in self.cameras)
        self.length = min(lengths)
        if self.length <= 0:
            raise RuntimeError(f"Episode has no frames: {self.path}")

    def close(self) -> None:
        self.file.close()

    def __len__(self) -> int:
        return self.length

    def task_text(self, fallback: str) -> str:
        if "task" in self.file.attrs:
            return _decode_attr(self.file.attrs["task"])
        return fallback

    def state_dim(self) -> int:
        return _feature_dim(self.file, self.state_fields)

    def action_dim(self) -> int:
        return _feature_dim(self.file, self.action_fields)

    def image_shape(self) -> tuple[int, int, int]:
        return _image_shape(self.file, self.cameras[0])

    def state_names(self) -> List[str]:
        return _field_names(self.state_fields, self.file)

    def action_names(self) -> List[str]:
        return _field_names(self.action_fields, self.file)

    def frame(self, index: int, *, episode_index: int, task_index: int, image_color_space: str) -> Dict[str, object]:
        if index < 0 or index >= self.length:
            raise IndexError(f"index out of range: {index}")

        timestamp_us = float(np.asarray(self.file["timestamp_us"][index]).reshape(-1)[0])
        timestamp = timestamp_us / 1_000_000.0
        if not np.isfinite(timestamp):
            timestamp = index / max(self.fps, 1)

        frame: Dict[str, object] = {
            "timestamp": np.asarray([timestamp], dtype=np.float32),
            "episode_index": np.asarray([episode_index], dtype=np.int64),
            "task_index": np.asarray([task_index], dtype=np.int64),
        }

        for camera in self.cameras:
            image = _to_hwc_uint8(self.file[f"cameras/{camera}/frames"][index])
            if image_color_space == "bgr":
                image = image[..., ::-1]
            frame[f"observation.images.{camera}"] = image
            frame[f"observation.images.{camera}.timestamp"] = np.asarray(
                [float(np.asarray(self.file[f"cameras/{camera}/timestamp_us"][index]).reshape(-1)[0]) / 1_000_000.0],
                dtype=np.float32,
            )
            frame[f"observation.images.{camera}.is_valid"] = np.asarray([True], dtype=np.bool_)

        state_parts = [_as_1d_float(self.file[field][index]) for field in self.state_fields]
        action_parts = [_as_1d_float(self.file[field][index]) for field in self.action_fields]
        frame["observation.state"] = np.concatenate(state_parts, axis=0).astype(np.float32, copy=False)
        frame["action"] = np.concatenate(action_parts, axis=0).astype(np.float32, copy=False)
        for field, value in zip(self.state_fields, state_parts):
            frame[f"observation.{field}"] = value
        for field, value in zip(self.action_fields, action_parts):
            frame[f"action.{field}"] = value
        return frame


def _build_feature_spec(
    *,
    image_shape: Sequence[int],
    cameras: Sequence[str],
    state_fields: Sequence[str],
    action_fields: Sequence[str],
    state_dim: int,
    action_dim: int,
    state_names: Sequence[str],
    action_names: Sequence[str],
    first_file,
) -> Dict[str, Dict]:
    features: Dict[str, Dict] = {
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": list(state_names)},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": list(action_names)},
    }
    for field in state_fields:
        dim = int(np.prod(_dataset_shape(first_file, field)[1:], dtype=np.int64))
        features[f"observation.{field}"] = {"dtype": "float32", "shape": (dim,), "names": None}
    for field in action_fields:
        dim = int(np.prod(_dataset_shape(first_file, field)[1:], dtype=np.int64))
        features[f"action.{field}"] = {"dtype": "float32", "shape": (dim,), "names": None}
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": tuple(image_shape),
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": None,
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
        features[f"observation.images.{camera}.timestamp"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        }
        features[f"observation.images.{camera}.is_valid"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        }
    return features


def inspect_episode(input_path: Path, cameras: Sequence[str], state_fields: Sequence[str], action_fields: Sequence[str]) -> None:
    _require_runtime()
    first_path = _episode_paths(input_path, max_episodes=1)[0]
    reader = H5EpisodeReader(
        first_path,
        cameras=cameras,
        state_fields=state_fields,
        action_fields=action_fields,
        fps=30,
    )
    try:
        print(f"episode_file: {first_path}")
        print(f"frames: {len(reader)}")
        print("root_attrs:", sorted(reader.file.attrs.keys()))
        if "format" in reader.file.attrs:
            print("format:", _decode_attr(reader.file.attrs["format"]))
        for key in ("config_yaml", "saved_at_us", "timestamp_us", *state_fields, *action_fields):
            if key in reader.file:
                value = reader.file[key]
                if isinstance(value, h5py.Dataset):
                    print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        for camera in cameras:
            print(f"cameras/{camera}/frames: shape={tuple(reader.file[f'cameras/{camera}/frames'].shape)} dtype={reader.file[f'cameras/{camera}/frames'].dtype}")
            print(f"cameras/{camera}/timestamp_us: shape={tuple(reader.file[f'cameras/{camera}/timestamp_us'].shape)} dtype={reader.file[f'cameras/{camera}/timestamp_us'].dtype}")
        print("state_fields:", list(state_fields), "state_dim:", reader.state_dim())
        print("action_fields:", list(action_fields), "action_dim:", reader.action_dim())
    finally:
        reader.close()


def convert_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    fps: int,
    cameras: Sequence[str],
    state_fields: Sequence[str],
    action_fields: Sequence[str],
    use_videos: bool,
    robot_type: str,
    task: str,
    max_episodes: Optional[int],
) -> None:
    _require_runtime()
    episode_paths = _episode_paths(input_path, max_episodes=max_episodes)
    first_reader = H5EpisodeReader(
        episode_paths[0],
        cameras=cameras,
        state_fields=state_fields,
        action_fields=action_fields,
        fps=fps,
    )
    try:
        image_shape = first_reader.image_shape()
        state_dim = first_reader.state_dim()
        action_dim = first_reader.action_dim()
        features = _build_feature_spec(
            image_shape=image_shape,
            cameras=cameras,
            state_fields=state_fields,
            action_fields=action_fields,
            state_dim=state_dim,
            action_dim=action_dim,
            state_names=first_reader.state_names(),
            action_names=first_reader.action_names(),
            first_file=first_reader.file,
        )
    finally:
        first_reader.close()

    video_keys = [f"observation.images.{camera}" for camera in cameras]
    dataset = LeRobotV3Writer(
        root=str(output_dir),
        fps=fps,
        features=features,
        video_keys=video_keys if use_videos else [],
        robot_type=robot_type,
        image_color_space="bgr" if use_videos else "rgb",
    )

    print(f"input_path: {input_path}")
    print(f"output_dir: {output_dir}")
    print(f"episodes: {len(episode_paths)}")
    print(f"cameras: {list(cameras)}")
    print(f"state_fields: {list(state_fields)} state_dim={state_dim}")
    print(f"action_fields: {list(action_fields)} action_dim={action_dim}")

    total_frames = 0
    start = time.perf_counter()
    print(f"{_progress_bar(0, len(episode_paths))} 0/{len(episode_paths)} 0.0% total_elapsed=0.000s", flush=True)
    for ep_idx, episode_path in enumerate(episode_paths, start=1):
        ep_start = time.perf_counter()
        reader = H5EpisodeReader(
            episode_path,
            cameras=cameras,
            state_fields=state_fields,
            action_fields=action_fields,
            fps=fps,
        )
        try:
            for frame_idx in range(len(reader)):
                dataset.add_frame(
                    reader.frame(
                        frame_idx,
                        episode_index=ep_idx - 1,
                        task_index=0,
                        image_color_space="bgr" if use_videos else "rgb",
                    )
                )
            dataset.save_episode(task=task)
            total_frames += len(reader)
            total_elapsed = time.perf_counter() - start
            progress = (ep_idx / len(episode_paths)) * 100 if episode_paths else 100.0
            print(
                f"{_progress_bar(ep_idx, len(episode_paths))} {ep_idx}/{len(episode_paths)} {progress:.1f}% "
                f"[{episode_path.name}] episode_elapsed={_format_seconds(time.perf_counter() - ep_start)} "
                f"total_elapsed={_format_seconds(total_elapsed)} frames={len(reader)}",
                flush=True,
            )
        finally:
            reader.close()

    dataset.finalize()
    print(f"Done. episodes={len(episode_paths)} frames={total_frames}")
    print(f"LeRobot v3 output: {output_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert wipe_board episode_*.h5 files to the official LeRobot v3 format.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("/home/hirol/code/data/train_episode/wipe_board"),
        help="Input episode_*.h5 file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output LeRobot v3 dataset directory path. Required unless --inspect-only is used.",
    )
    parser.add_argument("--inspect-only", action="store_true", help="Print the first episode schema and exit.")
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS written into meta/info.json.")
    parser.add_argument("--cameras", type=str, default=None, help="Comma-separated camera names.")
    parser.add_argument("--state-fields", type=str, default=None, help="Comma-separated root datasets for observation.state.")
    parser.add_argument("--action-fields", type=str, default=None, help="Comma-separated root datasets for action.")
    parser.add_argument("--robot-type", type=str, default="fr3", help="robot_type written into meta/info.json.")
    parser.add_argument("--task", type=str, default="wipe_board", help="Task text written into metadata.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional limit for quick conversion tests.")
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Disable MP4 packing and keep RGB frames as parquet values.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cameras = _parse_csv(args.cameras, DEFAULT_CAMERAS)
    state_fields = _parse_csv(args.state_fields, DEFAULT_STATE_FIELDS)
    action_fields = _parse_csv(args.action_fields, DEFAULT_ACTION_FIELDS)
    if args.inspect_only:
        inspect_episode(args.input_path, cameras, state_fields, action_fields)
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --inspect-only is used.")
    start = time.perf_counter()
    convert_dataset(
        input_path=args.input_path,
        output_dir=args.output_dir,
        fps=args.fps,
        cameras=cameras,
        state_fields=state_fields,
        action_fields=action_fields,
        use_videos=not args.no_videos,
        robot_type=args.robot_type,
        task=args.task,
        max_episodes=args.max_episodes,
    )
    print(f"elapsed={_format_seconds(time.perf_counter() - start)}")


if __name__ == "__main__":
    main()
