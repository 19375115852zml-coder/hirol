import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNK_INDEX = 0
DEFAULT_FILE_INDEX = 0
DEFAULT_CHUNK_SIZE = 1000


def _path(path_like) -> Path:
    return Path(os.path.expanduser(str(path_like)))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _feature_dtype_to_numpy(dtype_name: str):
    mapping = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "bool": np.bool_,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype in custom LeRobot v3 IO: {dtype_name}")
    return mapping[dtype_name]


def _feature_to_arrow_type(spec: Mapping) -> pa.DataType:
    dtype_name = spec["dtype"]
    if dtype_name == "video":
        base = pa.uint8()
        shape = tuple(spec.get("shape") or ())
        arr_type = base
        for dim in reversed(shape):
            arr_type = pa.list_(arr_type, int(dim))
        return arr_type

    shape = tuple(spec.get("shape") or ())
    if dtype_name == "bool":
        base = pa.bool_()
    elif dtype_name == "float32":
        base = pa.float32()
    elif dtype_name == "float64":
        base = pa.float64()
    elif dtype_name == "int32":
        base = pa.int32()
    elif dtype_name == "int64":
        base = pa.int64()
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    if len(shape) == 0:
        return base

    arr_type = base
    for dim in reversed(shape):
        arr_type = pa.list_(arr_type, int(dim))
    return arr_type


def _array_to_python(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()


def _normalize_parquet_compression(compression: Optional[str]) -> Optional[str]:
    if compression is None:
        return None
    normalized = str(compression).strip().lower()
    return None if normalized in {"", "none"} else normalized


def _numpy_batch_to_arrow(batch: np.ndarray, arrow_type: pa.DataType) -> pa.Array:
    if pa.types.is_fixed_size_list(arrow_type):
        list_size = arrow_type.list_size
        reshaped = np.asarray(batch).reshape(-1, list_size, *np.asarray(batch).shape[2:])
        child = _numpy_batch_to_arrow(reshaped.reshape(-1, *reshaped.shape[2:]), arrow_type.value_type)
        return pa.FixedSizeListArray.from_arrays(child, list_size)
    return pa.array(np.asarray(batch).reshape(-1), type=arrow_type)


def _vector_stats_update(stats: Dict[str, Dict], key: str, array: np.ndarray) -> None:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    current = stats.get(key)
    if current is None:
        stats[key] = {
            "count": int(arr.shape[0]),
            "sum": arr.sum(axis=0),
            "sum_sq": np.square(arr).sum(axis=0),
            "min": arr.min(axis=0),
            "max": arr.max(axis=0),
        }
        return
    current["count"] += int(arr.shape[0])
    current["sum"] += arr.sum(axis=0)
    current["sum_sq"] += np.square(arr).sum(axis=0)
    current["min"] = np.minimum(current["min"], arr.min(axis=0))
    current["max"] = np.maximum(current["max"], arr.max(axis=0))


def _finalize_stats(stats_accumulator: Dict[str, Dict]) -> Dict[str, Dict]:
    finalized: Dict[str, Dict] = {}
    for key, value in stats_accumulator.items():
        count = max(1, int(value["count"]))
        mean = value["sum"] / count
        var = np.maximum(value["sum_sq"] / count - np.square(mean), 0.0)
        std = np.sqrt(var)
        finalized[key] = {
            "count": count,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": value["min"].tolist(),
            "max": value["max"].tolist(),
        }
    return finalized


class CustomLeRobotV3Writer:
    def __init__(
        self,
        root: str,
        fps: int,
        features: Mapping[str, Mapping],
        video_keys: Sequence[str],
        robot_type: str = "unknown",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        codec: str = "mp4v",
        image_color_space: str = "rgb",
        parquet_compression: Optional[str] = None,
    ):
        self.root = _path(root)
        self.fps = int(fps)
        self.features = OrderedDict((key, dict(value)) for key, value in features.items())
        self.video_keys = list(video_keys)
        self.robot_type = str(robot_type or "unknown")
        self.chunk_size = int(chunk_size)
        self.codec = codec
        self.image_color_space = str(image_color_space).lower()
        if self.image_color_space not in {"rgb", "bgr"}:
            raise ValueError(f"Unsupported image_color_space={image_color_space!r}")
        self.parquet_compression = _normalize_parquet_compression(parquet_compression)

        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_dir = self.root / "meta"
        self.data_dir = self.root / "data" / f"chunk-{DEFAULT_CHUNK_INDEX:03d}"
        self.videos_dir = self.root / "videos"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

        self._episode_rows: List[Dict] = []
        self._task_to_index: Dict[str, int] = {}
        self._video_writers: Dict[str, cv2.VideoWriter] = {}
        self._video_paths: Dict[str, Path] = {}
        self._stats_accumulator: Dict[str, Dict] = {}
        self._current_episode_start = 0
        self._current_episode_length = 0
        self._episode_counter = 0
        self._total_frames = 0
        self._data_rel_path = f"data/chunk-{DEFAULT_CHUNK_INDEX:03d}/file-{DEFAULT_FILE_INDEX:03d}.parquet"
        self._data_path = self.root / self._data_rel_path
        self._stored_columns = OrderedDict()
        self._buffer_columns: Dict[str, List[object]] = {}
        self._data_writer: Optional[pq.ParquetWriter] = None

        for video_key in self.video_keys:
            video_path = (
                self.videos_dir
                / video_key
                / f"chunk-{DEFAULT_CHUNK_INDEX:03d}"
                / f"file-{DEFAULT_FILE_INDEX:03d}.mp4"
            )
            _ensure_parent(video_path)
            self._video_paths[video_key] = video_path

        data_fields = []
        for feature_name, spec in self.features.items():
            if spec["dtype"] == "video" and feature_name in self.video_keys:
                file_key = f"{feature_name}.__video_file__"
                frame_key = f"{feature_name}.__frame_index__"
                self._stored_columns[file_key] = {"kind": "video_file"}
                self._stored_columns[frame_key] = {"kind": "video_frame_index"}
                data_fields.append(pa.field(file_key, pa.string()))
                data_fields.append(pa.field(frame_key, pa.int64()))
                continue
            self._stored_columns[feature_name] = {"kind": "feature", "spec": spec}
            data_fields.append(pa.field(feature_name, _feature_to_arrow_type(spec)))
        self._data_schema = pa.schema(data_fields)
        self._buffer_columns = {name: [] for name in self._stored_columns}

    def _get_video_writer(self, video_key: str, frame_shape: Sequence[int]) -> cv2.VideoWriter:
        writer = self._video_writers.get(video_key)
        if writer is not None:
            return writer

        video_path = self._video_paths[video_key]
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*self.codec),
            float(self.fps),
            (frame_w, frame_h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")
        self._video_writers[video_key] = writer
        return writer

    def _default_feature_value(self, feature_name: str, global_frame_idx: int):
        if feature_name == "index":
            return np.asarray([global_frame_idx], dtype=np.int64)
        if feature_name == "frame_index":
            return np.asarray([self._current_episode_length], dtype=np.int64)
        if feature_name == "next.done":
            return np.asarray([False], dtype=np.bool_)
        raise KeyError(f"Missing feature {feature_name!r} in frame payload.")

    def _get_data_writer(self) -> pq.ParquetWriter:
        writer = self._data_writer
        if writer is not None:
            return writer
        _ensure_parent(self._data_path)
        writer = pq.ParquetWriter(
            self._data_path,
            self._data_schema,
            compression=self.parquet_compression,
        )
        self._data_writer = writer
        return writer

    def _write_video_frame(self, writer: cv2.VideoWriter, image: np.ndarray) -> None:
        if self.image_color_space == "bgr":
            writer.write(image)
            return
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def _build_data_table(self, row_count: int) -> pa.Table:
        arrays = []
        for name, meta in self._stored_columns.items():
            field = self._data_schema.field(name)
            values = self._buffer_columns[name][:row_count]
            kind = meta["kind"]
            if kind == "video_file":
                arrays.append(pa.array(values, type=field.type))
                continue
            if kind == "video_frame_index":
                arrays.append(pa.array(values, type=field.type))
                continue

            if values:
                batch = np.stack(values, axis=0)
                arrays.append(_numpy_batch_to_arrow(batch, field.type))
            else:
                arrays.append(pa.array([], type=field.type))
        return pa.Table.from_arrays(arrays, schema=self._data_schema)

    def _flush_buffer(self, keep_tail: int = 0) -> None:
        row_count = len(next(iter(self._buffer_columns.values()), []))
        flush_count = row_count - keep_tail
        if flush_count <= 0:
            return
        table = self._build_data_table(flush_count)
        self._get_data_writer().write_table(table)
        for values in self._buffer_columns.values():
            del values[:flush_count]

    def add_frame(self, frame: Mapping[str, object]) -> None:
        global_frame_idx = self._total_frames

        for feature_name, spec in self.features.items():
            if feature_name in frame:
                value = frame[feature_name]
            else:
                value = self._default_feature_value(feature_name, global_frame_idx)
            if spec["dtype"] == "video":
                image = np.asarray(value)
                if image.ndim != 3:
                    raise ValueError(f"Video feature {feature_name!r} must be 3-D, got {image.shape}")
                if feature_name in self.video_keys:
                    writer = self._get_video_writer(feature_name, image.shape)
                    self._write_video_frame(writer, image)
                    self._buffer_columns[f"{feature_name}.__video_file__"].append(
                        str(self._video_paths[feature_name].relative_to(self.root))
                    )
                    self._buffer_columns[f"{feature_name}.__frame_index__"].append(global_frame_idx)
                else:
                    image = image.astype(np.uint8, copy=False)
                    if self.image_color_space == "bgr":
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self._buffer_columns[feature_name].append(image)
                continue

            np_dtype = _feature_dtype_to_numpy(spec["dtype"])
            arr = np.asarray(value, dtype=np_dtype)
            self._buffer_columns[feature_name].append(arr)
            if spec["dtype"].startswith("float") or spec["dtype"].startswith("int"):
                _vector_stats_update(self._stats_accumulator, feature_name, arr[None, ...])

        self._total_frames += 1
        self._current_episode_length += 1
        if len(next(iter(self._buffer_columns.values()), [])) > self.chunk_size:
            self._flush_buffer(keep_tail=1)

    def save_episode(self, task: Optional[str] = None) -> None:
        if self._current_episode_length <= 0:
            return
        task_text = task or ""
        if task_text not in self._task_to_index:
            self._task_to_index[task_text] = len(self._task_to_index)
        task_index = self._task_to_index[task_text]

        if "next.done" in self.features:
            self._buffer_columns["next.done"][-1] = np.asarray([True], dtype=np.bool_)

        self._episode_rows.append(
            {
                "episode_index": self._episode_counter,
                "start_frame_index": self._current_episode_start,
                "end_frame_index": self._current_episode_start + self._current_episode_length,
                "from_index": self._current_episode_start,
                "to_index": self._current_episode_start + self._current_episode_length,
                "length": self._current_episode_length,
                "task": task_text,
                "tasks": [task_text] if task_text else [],
                "task_index": task_index,
                "chunk_index": DEFAULT_CHUNK_INDEX,
                "file_index": DEFAULT_FILE_INDEX,
            }
        )
        self._current_episode_start += self._current_episode_length
        self._current_episode_length = 0
        self._episode_counter += 1

    def finalize(self) -> None:
        for writer in self._video_writers.values():
            writer.release()
        self._video_writers.clear()

        self._flush_buffer()
        if self._data_writer is None:
            pq.write_table(
                pa.Table.from_arrays(
                    [pa.array([], type=field.type) for field in self._data_schema],
                    schema=self._data_schema,
                ),
                self._data_path,
                compression=self.parquet_compression,
            )
        else:
            self._data_writer.close()
            self._data_writer = None

        features = OrderedDict((key, dict(value)) for key, value in self.features.items())
        for video_key in self.video_keys:
            spec = dict(features[video_key])
            video_info = dict(spec.get("video_info") or {})
            video_info["video.fps"] = self.fps
            video_info["video.codec"] = self.codec
            video_info.setdefault("video.pix_fmt", "yuv420p")
            video_info.setdefault("video.is_depth_map", False)
            video_info.setdefault("has_audio", False)
            spec["video_info"] = video_info
            features[video_key] = spec

        data_rel_path = f"data/chunk-{DEFAULT_CHUNK_INDEX:03d}/file-{DEFAULT_FILE_INDEX:03d}.parquet"
        episodes_rel_path = (
            f"meta/episodes/chunk-{DEFAULT_CHUNK_INDEX:03d}/file-{DEFAULT_FILE_INDEX:03d}.parquet"
        )
        tasks_rel_path = "meta/tasks.parquet"
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": self._episode_counter,
            "total_frames": self._total_frames,
            "total_tasks": len(self._task_to_index),
            "chunks_size": self.chunk_size,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": self.fps,
            "splits": {"train": f"0:{self._episode_counter}"} if self._episode_counter > 0 else {},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "episodes_path": "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "tasks_path": tasks_rel_path,
            "features": features,
            "video_keys": self.video_keys,
        }
        with (self.meta_dir / "info.json").open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        episode_table = pa.Table.from_pylist(self._episode_rows)
        episodes_dir = self.meta_dir / "episodes" / f"chunk-{DEFAULT_CHUNK_INDEX:03d}"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            episode_table,
            self.root / episodes_rel_path,
            compression=self.parquet_compression,
        )

        task_rows = [
            {"task_index": task_index, "task": task}
            for task, task_index in sorted(self._task_to_index.items(), key=lambda item: item[1])
        ]
        pq.write_table(
            pa.Table.from_pylist(task_rows or [{"task_index": -1, "task": ""}]),
            self.root / tasks_rel_path,
            compression=self.parquet_compression,
        )
        with (self.meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
            for row in task_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with (self.meta_dir / "stats.json").open("w", encoding="utf-8") as f:
            json.dump(_finalize_stats(self._stats_accumulator), f, indent=2)


class CustomLeRobotV3Dataset:
    def __init__(self, root: str):
        self.root = _path(root)
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {info_path}")

        with info_path.open("r", encoding="utf-8") as f:
            self.info = json.load(f)

        self.features = OrderedDict((key, dict(value)) for key, value in self.info["features"].items())
        self.video_keys = list(self.info.get("video_keys", []))
        self.fps = int(self.info["fps"])

        data_path = self.root / self._resolve_template_path(
            self.info["data_path"],
            chunk_index=DEFAULT_CHUNK_INDEX,
            file_index=DEFAULT_FILE_INDEX,
        )
        episodes_template = self.info.get(
            "episodes_path",
            "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        )
        episodes_path = self.root / self._resolve_template_path(
            episodes_template,
            chunk_index=DEFAULT_CHUNK_INDEX,
            file_index=DEFAULT_FILE_INDEX,
        )
        self.data_table = pq.read_table(data_path)
        self.episodes_table = pq.read_table(episodes_path)
        self.length = self.data_table.num_rows

        self._column_cache: Dict[str, List] = {}
        self._video_caps: Dict[str, cv2.VideoCapture] = {}
        self._video_file_cache: Dict[str, str] = {}
        self.episode_data_index = {
            "from": self.episodes_table.column("start_frame_index").to_pylist(),
            "to": self.episodes_table.column("end_frame_index").to_pylist(),
        }
        self.frame_to_episode_index = np.empty((self.length,), dtype=np.int64)
        for episode_index, (start, stop) in enumerate(
            zip(self.episode_data_index["from"], self.episode_data_index["to"])
        ):
            self.frame_to_episode_index[int(start) : int(stop)] = episode_index

    def __len__(self) -> int:
        return self.length

    def close(self) -> None:
        for cap in self._video_caps.values():
            cap.release()
        self._video_caps.clear()

    def _get_column(self, name: str) -> List:
        cached = self._column_cache.get(name)
        if cached is not None:
            return cached
        column = self.data_table.column(name).to_pylist()
        self._column_cache[name] = column
        return column

    def get_column(self, name: str) -> List:
        return self._get_column(name)

    def _get_video_capture(self, rel_path: str) -> cv2.VideoCapture:
        cap = self._video_caps.get(rel_path)
        if cap is not None:
            return cap
        abs_path = self.root / rel_path
        cap = cv2.VideoCapture(str(abs_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {abs_path}")
        self._video_caps[rel_path] = cap
        return cap

    @staticmethod
    def _resolve_template_path(path_template: str, **kwargs) -> str:
        if "{" not in path_template:
            return path_template
        return path_template.format(**kwargs)

    def _read_video_frame(self, video_key: str, row_index: int) -> np.ndarray:
        rel_path_key = f"{video_key}.__video_file__"
        frame_key = f"{video_key}.__frame_index__"
        if rel_path_key in self.data_table.column_names:
            rel_path = self._get_column(rel_path_key)[row_index]
        else:
            rel_path = self._resolve_template_path(
                self.info["video_path"],
                video_key=video_key,
                chunk_index=DEFAULT_CHUNK_INDEX,
                file_index=DEFAULT_FILE_INDEX,
            )
        if frame_key in self.data_table.column_names:
            frame_index = int(self._get_column(frame_key)[row_index])
        else:
            frame_index = row_index
        cap = self._get_video_capture(rel_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(
                f"Failed to decode frame {frame_index} from video {self.root / rel_path}"
            )
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for feature_name, spec in self.features.items():
            if spec["dtype"] == "video":
                if feature_name in self.video_keys:
                    result[feature_name] = self._read_video_frame(feature_name, idx)
                else:
                    result[feature_name] = np.asarray(
                        self._get_column(feature_name)[idx],
                        dtype=np.uint8,
                    )
                continue

            if feature_name not in self.data_table.column_names:
                continue
            value = self._get_column(feature_name)[idx]
            np_dtype = _feature_dtype_to_numpy(spec["dtype"])
            arr = np.asarray(value, dtype=np_dtype)
            result[feature_name] = arr
        if "episode_index" not in result:
            result["episode_index"] = np.asarray([self.frame_to_episode_index[idx]], dtype=np.int64)
        return result

    def _episode_index_for_frame(self, frame_index: int) -> int:
        if frame_index < 0 or frame_index >= self.length:
            raise IndexError(f"Frame index {frame_index} is out of bounds for dataset length {self.length}.")
        return int(self.frame_to_episode_index[frame_index])
