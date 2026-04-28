from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.common.lerobot_v3_io import CustomLeRobotV3Dataset
from diffusion_policy.dataset.hirol_lerobot_v3_dataset import HirolLeRobotV3Dataset


def _assert_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def _load_task_dataset_config(task_config_path: Path, dataset_path: str):
    with task_config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataset_cfg = dict(config["dataset"])
    dataset_cfg.pop("_target_", None)
    dataset_cfg["dataset_path"] = dataset_path
    defaults = {
        "horizon": 1,
        "pad_before": 0,
        "pad_after": 0,
        "n_obs_steps": 1,
        "n_latency_steps": 0,
    }
    for key, default in defaults.items():
        value = dataset_cfg.get(key, default)
        if isinstance(value, str) and "${" in value:
            dataset_cfg[key] = default
    return dataset_cfg


def validate_dataset(dataset_path: Path, task_config: Path | None) -> None:
    dataset_path = dataset_path.expanduser().resolve()
    _assert_file(dataset_path / "meta" / "info.json")
    _assert_file(dataset_path / "meta" / "stats.json")
    _assert_file(dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    _assert_file(dataset_path / "meta" / "tasks.parquet")
    _assert_file(dataset_path / "data" / "chunk-000" / "file-000.parquet")

    with (dataset_path / "meta" / "info.json").open("r", encoding="utf-8") as f:
        info = json.load(f)

    print("dataset_path:", dataset_path)
    print("codebase_version:", info.get("codebase_version"))
    print("robot_type:", info.get("robot_type"))
    print("fps:", info.get("fps"))
    print("feature_count:", len(info.get("features", {})))
    print("video_keys:", info.get("video_keys", []))
    print("data_path:", info.get("data_path"))
    print("video_path:", info.get("video_path"))

    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected codebase_version=v3.0, got {info.get('codebase_version')!r}")
    if "chunk-{chunk_index:03d}" not in str(info.get("data_path", "")):
        raise ValueError("info.json data_path does not use the standard chunk/file template.")
    if "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4" != info.get("video_path"):
        raise ValueError("info.json video_path does not match the standard LeRobot v3 template.")

    for rel_path in info.get("video_keys", []):
        video_file = dataset_path / "videos" / rel_path / "chunk-000" / "file-000.mp4"
        _assert_file(video_file)
        print("video_ok:", video_file)

    custom_dataset = CustomLeRobotV3Dataset(str(dataset_path))
    print("frames:", len(custom_dataset))
    print("episodes:", len(custom_dataset.episode_data_index["from"]))
    expected_frames = sum(
        int(stop) - int(start)
        for start, stop in zip(
            custom_dataset.episode_data_index["from"],
            custom_dataset.episode_data_index["to"],
        )
    )
    if expected_frames != len(custom_dataset):
        raise ValueError(
            f"Frame count mismatch: dataset has {len(custom_dataset)} rows but "
            f"episodes metadata sums to {expected_frames}."
        )
    print("frame_count_check: ok")

    if len(custom_dataset) > 0:
        sample = custom_dataset[0]
        print("sample_keys:", sorted(sample.keys()))
        for key in info.get("video_keys", [])[:1]:
            print("sample_video_shape:", key, sample[key].shape)
        if "observation.state" in sample:
            print("sample_state_shape:", sample["observation.state"].shape)
        if "action" in sample:
            print("sample_action_shape:", sample["action"].shape)
    custom_dataset.close()

    if task_config is not None:
        task_cfg = _load_task_dataset_config(task_config.expanduser().resolve(), str(dataset_path))
        dataset = HirolLeRobotV3Dataset(**task_cfg)
        print("adapter_len:", len(dataset))
        if len(dataset) > 0:
            batch = dataset[0]
            print("adapter_obs_keys:", sorted(batch["obs"].keys()))
            for key, tensor in batch["obs"].items():
                print("adapter_obs_shape:", key, tuple(tensor.shape))
            print("adapter_action_shape:", tuple(batch["action"].shape))
            expected_obs = task_cfg["shape_meta"]["obs"]
            for key, tensor in batch["obs"].items():
                expected_shape = tuple(expected_obs[key]["shape"])
                if tuple(tensor.shape[1:]) != expected_shape:
                    raise ValueError(
                        f"Adapter obs shape mismatch for {key}: got {tuple(tensor.shape[1:])}, "
                        f"expected {expected_shape}"
                    )
            expected_action_shape = tuple(task_cfg["shape_meta"]["action"]["shape"])
            if tuple(batch["action"].shape[1:]) != expected_action_shape:
                raise ValueError(
                    f"Adapter action shape mismatch: got {tuple(batch['action'].shape[1:])}, "
                    f"expected {expected_action_shape}"
                )
            print("adapter_shape_check: ok")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a LeRobot v3 dataset for the local training path.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to converted LeRobot v3 dataset.")
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help="Optional task config yaml. If provided, also validate the training adapter contract.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    validate_dataset(dataset_path=args.dataset_path, task_config=args.task_config)


if __name__ == "__main__":
    main()
