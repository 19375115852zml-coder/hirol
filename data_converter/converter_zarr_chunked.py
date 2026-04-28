from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirol_reader import HiROLEpisodeReader


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(width * current / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _require_nested_group(root_group, nested: str):
    group = root_group
    for part in nested.split("/"):
        group = group.require_group(part)
    return group


def _create_dataset(root_group, nested: str, **kwargs):
    parts = nested.split("/")
    parent = _require_nested_group(root_group, "/".join(parts[:-1])) if len(parts) > 1 else root_group
    return parent.create_dataset(parts[-1], **kwargs)


def _build_compressor(name: str):
    if name in {"default", "none"}:
        return None
    if name == "blosc-lz4":
        from numcodecs import Blosc

        return Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
    raise ValueError(f"Unsupported compressor={name}")


def _dataset_kwargs(shape: Tuple[int, ...], chunks: Tuple[int, ...], dtype, compressor_name: str) -> Dict:
    kwargs = {
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "overwrite": True,
    }
    if compressor_name != "default":
        kwargs["compressor"] = _build_compressor(compressor_name)
    return kwargs


def _scan_selected_indices(reader: HiROLEpisodeReader, missing_policy: str) -> Tuple[List[int], int]:
    selected: List[int] = []
    missing_refs_total = 0
    for i, record in enumerate(reader.records):
        missing = reader.missing_cameras(record)
        missing_refs_total += len(missing)
        if missing_policy == "error" and missing:
            raise FileNotFoundError(
                f"Episode {reader.episode_dir.name} step={record.get('idx', i)} has missing cameras: {missing}"
            )
        if missing_policy == "skip" and missing:
            continue
        selected.append(i)
    return selected, missing_refs_total


def _allocate_buffers(camera_keys, chunk_t: int, image_shape: Tuple[int, int, int]) -> Dict:
    buffers = {
        "idx": np.empty((chunk_t,), dtype=np.int32),
        "state": np.empty((chunk_t, 8), dtype=np.float32),
        "state_ee": np.empty((chunk_t, 15), dtype=np.float32),
        "joint_position": np.empty((chunk_t, 7), dtype=np.float32),
        "ee_pose": np.empty((chunk_t, 7), dtype=np.float32),
        "gripper_width": np.empty((chunk_t, 1), dtype=np.float32),
        "action": np.empty((chunk_t, 8), dtype=np.float32),
        "action_joint_position": np.empty((chunk_t, 7), dtype=np.float32),
        "action_ee_pose": np.empty((chunk_t, 7), dtype=np.float32),
        "action_gripper_width": np.empty((chunk_t, 1), dtype=np.float32),
        "joint_timestamp": np.empty((chunk_t,), dtype=np.float64),
        "ee_timestamp": np.empty((chunk_t,), dtype=np.float64),
        "tool_timestamp": np.empty((chunk_t,), dtype=np.float64),
        "action_joint_timestamp": np.empty((chunk_t,), dtype=np.float64),
        "action_ee_timestamp": np.empty((chunk_t,), dtype=np.float64),
        "action_tool_timestamp": np.empty((chunk_t,), dtype=np.float64),
        "images": {},
        "valid": {},
        "ts_images": {},
    }
    for cam in camera_keys:
        buffers["images"][cam] = np.empty((chunk_t, *image_shape), dtype=np.uint8)
        buffers["valid"][cam] = np.empty((chunk_t,), dtype=np.bool_)
        buffers["ts_images"][cam] = np.empty((chunk_t,), dtype=np.float64)
    return buffers


def convert_episode_chunked(
    ep_dir: Path,
    out_root,
    missing_policy: str,
    image_chunk_t: int,
    compressor_name: str,
    chunk_progress_cb: Callable[[Dict], None] | None = None,
) -> Dict:
    reader = HiROLEpisodeReader(ep_dir)
    if missing_policy == "zeros":
        indices = list(range(len(reader)))
        missing_refs_total = 0
        scan_time = 0.0
    else:
        scan_start = time.perf_counter()
        indices, missing_refs_total = _scan_selected_indices(reader, missing_policy)
        scan_time = time.perf_counter() - scan_start

    if not indices:
        return {
            "episode": ep_dir.name,
            "raw_steps": len(reader),
            "written_steps": 0,
            "skipped": len(reader),
            "missing_refs_total": missing_refs_total,
            "scan_time_s": scan_time,
            "decode_time_s": 0.0,
            "write_time_s": 0.0,
            "primary_stream": reader.primary_stream,
        }

    image_shape = reader.infer_image_shape()
    t = len(indices)
    chunk_t = max(1, min(image_chunk_t, t))

    ep_group = _require_nested_group(out_root, f"episodes/{ep_dir.name}")
    obs_group = _require_nested_group(ep_group, "observation")
    images_group = _require_nested_group(obs_group, "images")
    meta_group = _require_nested_group(ep_group, "meta")

    image_ds = {}
    valid_ds = {}
    for cam in reader.camera_keys:
        image_ds[cam] = images_group.create_dataset(
            cam,
            **_dataset_kwargs(
                shape=(t, *image_shape),
                chunks=(chunk_t, *image_shape),
                dtype=np.uint8,
                compressor_name=compressor_name,
            ),
        )
        valid_ds[cam] = images_group.create_dataset(
            f"{cam}_valid",
            **_dataset_kwargs(
                shape=(t,),
                chunks=(chunk_t,),
                dtype=np.bool_,
                compressor_name=compressor_name,
            ),
        )

    obs_datasets = {
        "idx": _create_dataset(
            obs_group, "idx", **_dataset_kwargs((t,), (chunk_t,), np.int32, compressor_name)
        ),
        "state": _create_dataset(
            obs_group, "state", **_dataset_kwargs((t, 8), (chunk_t, 8), np.float32, compressor_name)
        ),
        "state_ee": _create_dataset(
            obs_group, "state_ee", **_dataset_kwargs((t, 15), (chunk_t, 15), np.float32, compressor_name)
        ),
        "joint_position": _create_dataset(
            obs_group, "joint_position", **_dataset_kwargs((t, 7), (chunk_t, 7), np.float32, compressor_name)
        ),
        "ee_pose": _create_dataset(
            obs_group, "ee_pose", **_dataset_kwargs((t, 7), (chunk_t, 7), np.float32, compressor_name)
        ),
        "gripper_width": _create_dataset(
            obs_group, "gripper_width", **_dataset_kwargs((t, 1), (chunk_t, 1), np.float32, compressor_name)
        ),
    }
    action_datasets = {
        "action": _create_dataset(
            ep_group, "action", **_dataset_kwargs((t, 8), (chunk_t, 8), np.float32, compressor_name)
        ),
        "action_joint_position": _create_dataset(
            ep_group, "action_joint_position", **_dataset_kwargs((t, 7), (chunk_t, 7), np.float32, compressor_name)
        ),
        "action_ee_pose": _create_dataset(
            ep_group, "action_ee_pose", **_dataset_kwargs((t, 7), (chunk_t, 7), np.float32, compressor_name)
        ),
        "action_gripper_width": _create_dataset(
            ep_group, "action_gripper_width", **_dataset_kwargs((t, 1), (chunk_t, 1), np.float32, compressor_name)
        ),
    }

    ts_group = _require_nested_group(meta_group, "timestamps")
    ts_img_ds = {}
    for cam in reader.camera_keys:
        ts_img_ds[cam] = _create_dataset(
            ts_group,
            f"image_{cam}",
            **_dataset_kwargs((t,), (chunk_t,), np.float64, compressor_name),
        )
    ts_datasets = {
        "joint_timestamp": _create_dataset(
            ts_group, "joint", **_dataset_kwargs((t,), (chunk_t,), np.float64, compressor_name)
        ),
        "ee_timestamp": _create_dataset(
            ts_group, "ee", **_dataset_kwargs((t,), (chunk_t,), np.float64, compressor_name)
        ),
        "tool_timestamp": _create_dataset(
            ts_group, "tool", **_dataset_kwargs((t,), (chunk_t,), np.float64, compressor_name)
        ),
        "action_joint_timestamp": _create_dataset(
            ts_group, "action_joint", **_dataset_kwargs((t,), (chunk_t,), np.float64, compressor_name)
        ),
        "action_ee_timestamp": _create_dataset(
            ts_group, "action_ee", **_dataset_kwargs((t,), (chunk_t,), np.float64, compressor_name)
        ),
        "action_tool_timestamp": _create_dataset(
            ts_group, "action_tool", **_dataset_kwargs((t,), (chunk_t,), np.float64, compressor_name)
        ),
    }

    fill_missing = "zeros" if missing_policy == "zeros" else "error"
    buffers = _allocate_buffers(reader.camera_keys, chunk_t=chunk_t, image_shape=image_shape)
    decode_time = 0.0
    write_time = 0.0

    for start in range(0, t, chunk_t):
        end = min(start + chunk_t, t)
        batch_len = end - start

        decode_start = time.perf_counter()
        for local_i, src_i in enumerate(indices[start:end]):
            step = reader.get_step(src_i, load_images=True, fill_missing=fill_missing, image_shape=image_shape)
            role = reader.primary_stream

            buffers["idx"][local_i] = step["idx"]
            buffers["state"][local_i] = step["state"]
            buffers["state_ee"][local_i] = step["state_ee"]
            buffers["joint_position"][local_i] = step["joint_position"][role]
            buffers["ee_pose"][local_i] = step["ee_pose"][role]
            buffers["gripper_width"][local_i, 0] = step["gripper_width"][role]
            buffers["action"][local_i] = step["action"]
            buffers["action_joint_position"][local_i] = step["action_joint_position"][role]
            buffers["action_ee_pose"][local_i] = step["action_ee_pose"][role]
            buffers["action_gripper_width"][local_i, 0] = step["action_gripper_width"][role]
            buffers["joint_timestamp"][local_i] = step["joint_timestamps"][role]
            buffers["ee_timestamp"][local_i] = step["ee_timestamps"][role]
            buffers["tool_timestamp"][local_i] = step["tool_timestamps"][role]
            buffers["action_joint_timestamp"][local_i] = step["action_joint_timestamps"][role]
            buffers["action_ee_timestamp"][local_i] = step["action_ee_timestamps"][role]
            buffers["action_tool_timestamp"][local_i] = step["action_tool_timestamps"][role]

            for cam in reader.camera_keys:
                image = step["images"][cam]
                if image is None:
                    buffers["images"][cam][local_i].fill(0)
                else:
                    buffers["images"][cam][local_i] = image
                buffers["valid"][cam][local_i] = step["image_valid"][cam]
                buffers["ts_images"][cam][local_i] = step["image_timestamps"][cam]
        decode_time += time.perf_counter() - decode_start

        write_start = time.perf_counter()
        for name, dataset in obs_datasets.items():
            dataset[start:end] = buffers[name][:batch_len]
        for name, dataset in action_datasets.items():
            dataset[start:end] = buffers[name][:batch_len]
        for name, dataset in ts_datasets.items():
            dataset[start:end] = buffers[name][:batch_len]
        for cam in reader.camera_keys:
            image_ds[cam][start:end] = buffers["images"][cam][:batch_len]
            valid_ds[cam][start:end] = buffers["valid"][cam][:batch_len]
            ts_img_ds[cam][start:end] = buffers["ts_images"][cam][:batch_len]
        write_time += time.perf_counter() - write_start

        if chunk_progress_cb is not None:
            chunk_progress_cb(
                {
                    "episode": ep_dir.name,
                    "chunk_start": start,
                    "chunk_end": end,
                    "total_steps": t,
                    "chunk_index": (start // chunk_t) + 1,
                    "chunk_count": (t + chunk_t - 1) // chunk_t,
                    "decode_time_s": decode_time,
                    "write_time_s": write_time,
                }
            )

    ep_group.attrs["source_episode"] = ep_dir.name
    ep_group.attrs["source_path"] = str(ep_dir)
    ep_group.attrs["text"] = json.dumps(reader.text, ensure_ascii=False)
    ep_group.attrs["info"] = json.dumps(reader.info, ensure_ascii=False)
    ep_group.attrs["raw_steps"] = len(reader)
    ep_group.attrs["written_steps"] = t
    ep_group.attrs["missing_policy"] = missing_policy
    ep_group.attrs["missing_image_refs_total"] = missing_refs_total
    ep_group.attrs["writer"] = "chunked_batch_writer"
    ep_group.attrs["image_chunk_t"] = chunk_t
    ep_group.attrs["compressor"] = compressor_name
    ep_group.attrs["primary_stream"] = reader.primary_stream

    return {
        "episode": ep_dir.name,
        "raw_steps": len(reader),
        "written_steps": t,
        "skipped": len(reader) - t,
        "missing_refs_total": missing_refs_total,
        "scan_time_s": scan_time,
        "decode_time_s": decode_time,
        "write_time_s": write_time,
        "primary_stream": reader.primary_stream,
    }


def convert_dataset(
    input_root: Path,
    output_zarr: Path,
    missing_policy: str,
    image_chunk_t: int,
    compressor_name: str,
    episode_name: str | None = None,
) -> None:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "zarr is required for conversion. Install it first, e.g. `pip install zarr`."
        ) from exc

    episode_dirs = HiROLEpisodeReader.list_episode_dirs(input_root)
    if episode_name:
        episode_dirs = [p for p in episode_dirs if p.name == episode_name]
    if not episode_dirs:
        if episode_name:
            raise RuntimeError(f"Episode not found: {episode_name} in {input_root}")
        raise RuntimeError(f"No episode_* directories found in: {input_root}")

    root = zarr.open_group(str(output_zarr), mode="w", zarr_version=2)
    root.attrs["dataset_name"] = input_root.name
    root.attrs["source_root"] = str(input_root)
    root.attrs["missing_policy"] = missing_policy
    root.attrs["num_episodes"] = len(episode_dirs)
    root.attrs["writer"] = "chunked_batch_writer"
    root.attrs["zarr_format"] = 2
    root.attrs["image_chunk_t"] = image_chunk_t
    root.attrs["compressor"] = compressor_name
    root.attrs["format"] = "hirol_chunked_episode"

    summaries: List[Dict] = []
    dataset_start = time.perf_counter()
    total_episodes = len(episode_dirs)
    print(f"Found {total_episodes} episode(s) under: {input_root}")
    print(f"{_progress_bar(0, total_episodes)} 0/{total_episodes} 0.0% total_elapsed=0.000s", flush=True)

    for idx, ep_dir in enumerate(episode_dirs, start=1):
        episode_start = time.perf_counter()

        def chunk_progress(info: Dict) -> None:
            episode_elapsed = time.perf_counter() - episode_start
            print(
                f"  [{info['episode']}] chunk {info['chunk_index']}/{info['chunk_count']} "
                f"steps={info['chunk_end']}/{info['total_steps']} "
                f"episode_elapsed={_format_seconds(episode_elapsed)} "
                f"decode={_format_seconds(info['decode_time_s'])} "
                f"write={_format_seconds(info['write_time_s'])}",
                flush=True,
            )

        summary = convert_episode_chunked(
            ep_dir,
            root,
            missing_policy=missing_policy,
            image_chunk_t=image_chunk_t,
            compressor_name=compressor_name,
            chunk_progress_cb=chunk_progress,
        )
        episode_elapsed = time.perf_counter() - episode_start
        summaries.append(summary)
        total_elapsed = time.perf_counter() - dataset_start
        progress = (idx / total_episodes) * 100 if total_episodes else 100.0
        print(
            f"{_progress_bar(idx, total_episodes)} {idx}/{total_episodes} {progress:.1f}% "
            f"[{summary['episode']}] episode_elapsed={_format_seconds(episode_elapsed)} "
            f"total_elapsed={_format_seconds(total_elapsed)} raw={summary['raw_steps']} "
            f"written={summary['written_steps']} skipped={summary['skipped']} "
            f"missing_refs={summary['missing_refs_total']} primary_stream={summary['primary_stream']} "
            f"scan={_format_seconds(summary['scan_time_s'])} "
            f"decode={_format_seconds(summary['decode_time_s'])} "
            f"write={_format_seconds(summary['write_time_s'])}",
            flush=True,
        )

    total_raw = sum(s["raw_steps"] for s in summaries)
    total_written = sum(s["written_steps"] for s in summaries)
    total_scan = sum(s["scan_time_s"] for s in summaries)
    total_decode = sum(s["decode_time_s"] for s in summaries)
    total_write = sum(s["write_time_s"] for s in summaries)
    total_time = time.perf_counter() - dataset_start

    root.attrs["total_raw_steps"] = total_raw
    root.attrs["total_written_steps"] = total_written
    root.attrs["total_scan_time_s"] = total_scan
    root.attrs["total_decode_time_s"] = total_decode
    root.attrs["total_write_time_s"] = total_write
    root.attrs["primary_stream"] = summaries[0]["primary_stream"] if summaries else None

    print(
        f"Done. episodes={len(episode_dirs)} raw_steps={total_raw} written_steps={total_written} "
        f"scan={total_scan:.3f}s decode={total_decode:.3f}s write={total_write:.3f}s total={total_time:.3f}s",
        flush=True,
    )
    print(f"Zarr output: {output_zarr}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HIROL episodes to chunked per-episode Zarr.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("dataset/duo_unitree_pick_n_place"),
        help="Input dataset root containing episode_* folders.",
    )
    parser.add_argument(
        "--output-zarr",
        type=Path,
        default=Path("dataset/duo_unitree_pick_n_place_chunked.zarr"),
        help="Output .zarr directory path.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["error", "skip", "zeros"],
        default="zeros",
        help="How to handle steps with missing image files.",
    )
    parser.add_argument(
        "--image-chunk-t",
        type=int,
        default=32,
        help="Chunk size on time axis for image arrays.",
    )
    parser.add_argument(
        "--compressor",
        choices=["default", "none", "blosc-lz4"],
        default="none",
        help="Compression strategy for Zarr arrays.",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Convert only one episode, e.g. episode_0010.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    convert_dataset(
        input_root=args.input_root,
        output_zarr=args.output_zarr,
        missing_policy=args.missing_policy,
        image_chunk_t=args.image_chunk_t,
        compressor_name=args.compressor,
        episode_name=args.episode,
    )


if __name__ == "__main__":
    main()
