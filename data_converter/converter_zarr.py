from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirol_reader import HiROLEpisodeReader
from diffusion_policy.common.replay_buffer import ReplayBuffer


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(width * current / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _selected_indices(reader: HiROLEpisodeReader, missing_policy: str) -> List[int]:
    selected: List[int] = []
    for i, record in enumerate(reader.records):
        missing = reader.missing_cameras(record)
        if missing_policy == "error" and missing:
            raise FileNotFoundError(
                f"Episode {reader.episode_dir.name} step={record.get('idx', i)} has missing cameras: {missing}"
            )
        if missing_policy == "skip" and missing:
            continue
        selected.append(i)
    return selected


def _episode_chunks(episode_data: Dict[str, np.ndarray], image_chunk_t: int) -> Dict[str, tuple]:
    chunks: Dict[str, tuple] = {}
    chunk_t = max(1, image_chunk_t)
    for key, value in episode_data.items():
        chunks[key] = (min(chunk_t, value.shape[0]),) + value.shape[1:]
    return chunks


def convert_episode(
    ep_dir: Path,
    replay_buffer: ReplayBuffer,
    missing_policy: str,
    image_chunk_t: int,
    chunk_progress_cb: Callable[[Dict], None] | None = None,
) -> Dict:
    reader = HiROLEpisodeReader(ep_dir)
    indices = _selected_indices(reader, missing_policy)
    if not indices:
        return {"episode": ep_dir.name, "written_steps": 0, "skipped": len(reader)}

    image_shape = reader.infer_image_shape()
    t = len(indices)
    fill_missing = "zeros" if missing_policy in {"zeros", "skip"} else "error"

    episode_data: Dict[str, np.ndarray] = {
        "idx": np.empty((t,), dtype=np.int32),
        "state": np.empty((t, 8), dtype=np.float32),
        "state_ee": np.empty((t, 15), dtype=np.float32),
        "joint_position": np.empty((t, 7), dtype=np.float32),
        "ee_pose": np.empty((t, 7), dtype=np.float32),
        "gripper_width": np.empty((t, 1), dtype=np.float32),
        "action": np.empty((t, 8), dtype=np.float32),
        "action_joint_position": np.empty((t, 7), dtype=np.float32),
        "action_ee_pose": np.empty((t, 7), dtype=np.float32),
        "action_gripper_width": np.empty((t, 1), dtype=np.float32),
        "joint_timestamp": np.empty((t,), dtype=np.float64),
        "ee_timestamp": np.empty((t,), dtype=np.float64),
        "tool_timestamp": np.empty((t,), dtype=np.float64),
        "action_joint_timestamp": np.empty((t,), dtype=np.float64),
        "action_ee_timestamp": np.empty((t,), dtype=np.float64),
        "action_tool_timestamp": np.empty((t,), dtype=np.float64),
    }
    for cam in reader.camera_keys:
        episode_data[cam] = np.empty((t, *image_shape), dtype=np.uint8)
        episode_data[f"{cam}_valid"] = np.empty((t,), dtype=np.bool_)
        episode_data[f"{cam}_timestamp"] = np.empty((t,), dtype=np.float64)

    decode_time = 0.0
    chunk_t = max(1, min(image_chunk_t, t))

    for start in range(0, t, chunk_t):
        end = min(start + chunk_t, t)
        decode_start = time.perf_counter()
        for out_i, src_i in enumerate(indices[start:end], start=start):
            step = reader.get_step(src_i, load_images=True, fill_missing=fill_missing, image_shape=image_shape)

            episode_data["idx"][out_i] = step["idx"]
            episode_data["state"][out_i] = step["state"]
            episode_data["state_ee"][out_i] = step["state_ee"]
            episode_data["joint_position"][out_i] = step["joint_position"][reader.primary_stream]
            episode_data["ee_pose"][out_i] = step["ee_pose"][reader.primary_stream]
            episode_data["gripper_width"][out_i, 0] = step["gripper_width"][reader.primary_stream]
            episode_data["action"][out_i] = step["action"]
            episode_data["action_joint_position"][out_i] = step["action_joint_position"][reader.primary_stream]
            episode_data["action_ee_pose"][out_i] = step["action_ee_pose"][reader.primary_stream]
            episode_data["action_gripper_width"][out_i, 0] = step["action_gripper_width"][reader.primary_stream]
            episode_data["joint_timestamp"][out_i] = step["joint_timestamps"][reader.primary_stream]
            episode_data["ee_timestamp"][out_i] = step["ee_timestamps"][reader.primary_stream]
            episode_data["tool_timestamp"][out_i] = step["tool_timestamps"][reader.primary_stream]
            episode_data["action_joint_timestamp"][out_i] = step["action_joint_timestamps"][reader.primary_stream]
            episode_data["action_ee_timestamp"][out_i] = step["action_ee_timestamps"][reader.primary_stream]
            episode_data["action_tool_timestamp"][out_i] = step["action_tool_timestamps"][reader.primary_stream]

            for cam in reader.camera_keys:
                episode_data[cam][out_i] = step["images"][cam]
                episode_data[f"{cam}_valid"][out_i] = step["image_valid"][cam]
                episode_data[f"{cam}_timestamp"][out_i] = step["image_timestamps"][cam]
        decode_time += time.perf_counter() - decode_start

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
                }
            )

    replay_buffer.add_episode(
        episode_data,
        chunks=_episode_chunks(episode_data, image_chunk_t=image_chunk_t),
    )

    missing_refs_total = sum(len(reader.missing_cameras(rec)) for rec in reader.records)
    return {
        "episode": ep_dir.name,
        "raw_steps": len(reader),
        "written_steps": t,
        "skipped": len(reader) - t,
        "missing_refs_total": missing_refs_total,
        "primary_stream": reader.primary_stream,
        "decode_time_s": decode_time,
    }


def convert_dataset(
    input_root: Path,
    output_zarr: Path,
    missing_policy: str,
    image_chunk_t: int,
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

    root = zarr.open_group(str(output_zarr), mode="w")
    root.attrs["dataset_name"] = input_root.name
    root.attrs["source_root"] = str(input_root)
    root.attrs["missing_policy"] = missing_policy
    root.attrs["num_episodes"] = len(episode_dirs)
    root.attrs["format"] = "diffusion_policy_replay_buffer"

    replay_buffer = ReplayBuffer.create_empty_zarr(root=root)

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
                f"decode={_format_seconds(info['decode_time_s'])}",
                flush=True,
            )

        summary = convert_episode(
            ep_dir,
            replay_buffer=replay_buffer,
            missing_policy=missing_policy,
            image_chunk_t=image_chunk_t,
            chunk_progress_cb=chunk_progress,
        )
        summaries.append(summary)
        episode_elapsed = time.perf_counter() - episode_start
        total_elapsed = time.perf_counter() - dataset_start
        progress = (idx / total_episodes) * 100 if total_episodes else 100.0
        print(
            f"{_progress_bar(idx, total_episodes)} {idx}/{total_episodes} {progress:.1f}% "
            f"[{summary['episode']}] episode_elapsed={_format_seconds(episode_elapsed)} "
            f"total_elapsed={_format_seconds(total_elapsed)} raw={summary.get('raw_steps', 0)} "
            f"written={summary.get('written_steps', 0)} skipped={summary.get('skipped', 0)} "
            f"missing_refs={summary.get('missing_refs_total', 0)} "
            f"primary_stream={summary.get('primary_stream')} "
            f"decode={_format_seconds(summary.get('decode_time_s', 0.0))}"
        )

    total_raw = sum(s.get("raw_steps", 0) for s in summaries)
    total_written = sum(s.get("written_steps", 0) for s in summaries)
    root.attrs["total_raw_steps"] = total_raw
    root.attrs["total_written_steps"] = total_written
    root.attrs["primary_stream"] = summaries[0]["primary_stream"] if summaries else None

    print(f"Done. episodes={len(episode_dirs)} raw_steps={total_raw} written_steps={total_written}")
    print(f"Zarr output: {output_zarr}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HIROL episodes to a Diffusion Policy replay-buffer Zarr.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("dataset/duo_unitree_pick_n_place"),
        help="Input dataset root containing episode_* folders.",
    )
    parser.add_argument(
        "--output-zarr",
        type=Path,
        default=Path("dataset/duo_unitree_pick_n_place_replay.zarr"),
        help="Output replay-buffer .zarr directory path.",
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
        default=8,
        help="Chunk size on time axis for image arrays.",
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
    start_time = time.perf_counter()
    convert_dataset(
        input_root=args.input_root,
        output_zarr=args.output_zarr,
        missing_policy=args.missing_policy,
        image_chunk_t=args.image_chunk_t,
        episode_name=args.episode,
    )
    read_time = time.perf_counter() - start_time
    print(f"read time : {read_time:.4f}")


if __name__ == "__main__":
    main()
