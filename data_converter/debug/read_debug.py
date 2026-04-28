from data_converter.hirol_reader import HiROLEpisodeReader as HirolReader
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from diffusion_policy.common.lerobot_v3_io import CustomLeRobotV3Writer

import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HIROL episodes to a LeRobot v3 dataset.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("dataset/duo_unitree_pick_n_place"),
        help="Input dataset root containing episode_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/duo_unitree_pick_n_place_lerobot_v3"),
        help="Output LeRobot v3 dataset directory path.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["error", "skip", "zeros"],
        default="zeros",
        help="How to handle steps with missing image files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Dataset FPS. Defaults to inference from timestamps.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="fr3",
        help="robot_type written into meta/info.json.",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Disable MP4 packing and keep RGB frames as parquet values.",
    )
    return parser

def debuger(
    input_root: Path,
    output_dir: Path,
    missing_policy: str,
    fps: int | None,
    uese_videos: bool,
    robot_type: str,
) -> None:
    episode_dirs = HirolReader.list_episode_dirs(input_root)
    if not episode_dirs:
        raise RuntimeError(f"No episode_* directories found in: {input_root}")
    first_reader = HirolReader(episode_dirs[0])
    image_shape = first_reader.infer_image_shape()
    camera_keys = first_reader.camera_keys
    video_keys = [f"observation.images.{camera_key}" for camera_key in camera_keys]
    indices = _selected_indices(reader, missing_policy)
    fill_missing = "zeros" if missing_policy in {"zeros", "skip"} else "error"
    image_shape = reader.infer_image_shape()
    task_text = ""
    if isinstance(reader.text, dict):
            task_text = str(
                reader.text.get("task")
                or reader.text.get("description")
                or reader.text.get("instruction")
                or ""
            )
    if task_text not in task_to_index:
            task_to_index[task_text] = len(task_to_index)
    task_index = task_to_index[task_text]

    written_steps = 0
    for local_step_idx, raw_idx in enumerate(indices):
            fallback_timestamp = local_step_idx / max(fps, 1)
            frame = reader.get_lerobot_frame(
                raw_idx,
                fallback_timestamp=fallback_timestamp,
                episode_index=ep_idx - 1,
                task_index=task_index,
                fill_missing=fill_missing,
                image_shape=image_shape,
                image_color_space="bgr" if use_videos else "rgb",
            )
    written_steps += 1
    for ep_idx, ep_dir in enumerate(episode_dirs, start=1):
        episode_start = time.perf_counter()
        reader = HirolReader(ep_dir)
        image_shape = reader.infer_image_shape()
        key = reader.iter_steps
        print(f"{key}")


def main() -> None:
    args = build_argparser().parse_args()
    start = time.perf_counter()
    debuger(
        input_root=args.input_root,
        output_dir=args.output_dir,
        missing_policy=args.missing_policy,
        fps=args.fps,
        use_videos=not args.no_videos,
        robot_type=args.robot_type,
    )

if __name__ == "__main__":
    main()