from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hirol_reader import HiROLEpisodeReader


DEFAULT_OUTPUT_DIR = Path("data/inference_visualized/ee_visualizer")
DEFAULT_OVERVIEW_NAME = "all_episodes.png"
VALID_IMAGE_SUFFIXES = {
    ".eps",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".pgf",
    ".png",
    ".ps",
    ".raw",
    ".rgba",
    ".svg",
    ".tif",
    ".tiff",
    ".webp",
}


def resolve_episode_dirs(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    if (root / "data.json").is_file():
        return [root]
    episode_dirs = HiROLEpisodeReader.list_episode_dirs(root)
    if not episode_dirs:
        raise RuntimeError(f"No episode_* directories found under: {root}")
    return episode_dirs


def iter_ee_xyz_and_tool_data(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    reader = HiROLEpisodeReader(episode_dir)
    if reader.primary_stream is None:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    xyz_points: list[np.ndarray] = []
    tool_values: list[np.float32] = []

    for step in reader.iter_steps(load_images=False):
        role = step["primary_stream"] or reader.primary_stream
        ee_pose = step["ee_pose"].get(role)
        tool_value = step["tool_position"].get(role, np.float32(np.nan))

        if ee_pose is None:
            continue

        ee_pose = np.asarray(ee_pose, dtype=np.float32)
        if ee_pose.shape[0] < 3:
            continue

        xyz = ee_pose[:3]
        if not np.all(np.isfinite(xyz)):
            continue

        xyz_points.append(xyz)
        tool_values.append(np.float32(tool_value))

    if not xyz_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.stack(xyz_points, axis=0), np.asarray(tool_values, dtype=np.float32)


def iter_ee_xyz_and_tool_infer(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data_path = episode_dir / "data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"data.json not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"Invalid predictions field in {data_path}")

    xyz_points: list[np.ndarray] = []
    tool_values: list[np.float32] = []

    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue

        action = prediction.get("first_action")
        if action is None:
            action_chunk = prediction.get("action_chunk")
            if isinstance(action_chunk, list) and action_chunk:
                action = action_chunk[0]

        if action is None:
            continue

        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 1 or action.shape[0] < 4:
            continue

        xyz = action[:3]
        if not np.all(np.isfinite(xyz)):
            continue

        xyz_points.append(xyz)
        tool_value = action[-1]
        tool_values.append(np.float32(tool_value) if np.isfinite(tool_value) else np.float32(np.nan))

    if not xyz_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.stack(xyz_points, axis=0), np.asarray(tool_values, dtype=np.float32)


def iter_ee_xyz_and_tool_franka_infer(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data_path = episode_dir / "data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"data.json not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    steps = payload.get("data")
    if not isinstance(steps, list):
        raise ValueError(f"Invalid data field in {data_path}")

    xyz_points: list[np.ndarray] = []
    tool_values: list[np.float32] = []

    for step in steps:
        if not isinstance(step, dict):
            continue

        ee_states = step.get("ee_states")
        actions = step.get("actions")
        if not isinstance(ee_states, dict) or not isinstance(actions, dict):
            continue

        single_state = ee_states.get("single")
        if not isinstance(single_state, dict):
            continue

        pose = single_state.get("pose")
        if pose is None:
            continue

        pose = np.asarray(pose, dtype=np.float32)
        if pose.ndim != 1 or pose.shape[0] < 3:
            continue

        xyz = pose[:3]
        if not np.all(np.isfinite(xyz)):
            continue

        tool = actions.get("tool", np.nan)
        tool_array = np.asarray(tool, dtype=np.float32).reshape(-1)
        tool_value = tool_array[0] if tool_array.size > 0 and np.isfinite(tool_array[0]) else np.float32(np.nan)

        xyz_points.append(xyz)
        tool_values.append(np.float32(tool_value))

    if not xyz_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.stack(xyz_points, axis=0), np.asarray(tool_values, dtype=np.float32)


def set_equal_axes(ax, all_points: np.ndarray) -> None:
    if all_points.size == 0:
        return
    mins = np.nanmin(all_points, axis=0)
    maxs = np.nanmax(all_points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if not np.isfinite(radius) or radius == 0:
        radius = 0.1
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_group(
    ax,
    episode_dirs: Iterable[Path],
    color: str,
    linewidth: float,
    alpha: float,
    label: str,
    tool_threshold: float,
    reader_fn,
    closed_label: str | None = None,
    closed_color: str | None = None,
    closed_linewidth: float | None = None,
    closed_alpha: float = 1.0,
    closed_marker_size: float = 10.0,
) -> tuple[list[np.ndarray], int]:
    all_xyz: list[np.ndarray] = []
    black_point_count = 0
    first_line = True
    first_closed = True
    closed_linewidth = closed_linewidth or (linewidth * 3.0)
    closed_color = closed_color or color

    for episode_dir in episode_dirs:
        try:
            xyz, tool = reader_fn(episode_dir)
        except Exception as exc:
            print(f"Skip episode {episode_dir}: {exc}")
            continue

        if xyz.shape[0] == 0:
            continue

        all_xyz.append(xyz)
        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label if first_line else None,
        )
        first_line = False

        closed_mask = np.isfinite(tool) & (tool < tool_threshold)
        if not np.any(closed_mask):
            continue

        for i in range(len(xyz) - 1):
            if closed_mask[i] or closed_mask[i + 1]:
                seg_xyz = xyz[i : i + 2]
                ax.plot(
                    seg_xyz[:, 0],
                    seg_xyz[:, 1],
                    seg_xyz[:, 2],
                    color=closed_color,
                    alpha=closed_alpha,
                    linewidth=closed_linewidth,
                )

        ax.scatter(
            xyz[closed_mask, 0],
            xyz[closed_mask, 1],
            xyz[closed_mask, 2],
            color=closed_color,
            alpha=closed_alpha,
            s=closed_marker_size,
            depthshade=False,
            label=closed_label if first_closed else None,
        )
        first_closed = False
        black_point_count += int(np.count_nonzero(closed_mask))

    return all_xyz, black_point_count


def _sorted_episode_dirs(path: str | Path) -> list[Path]:
    return sorted(resolve_episode_dirs(path), key=lambda p: p.name)


def _create_figure() -> tuple[plt.Figure, object]:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def _finalize_figure(
    fig: plt.Figure,
    ax,
    title: str,
    xyz_groups: Iterable[np.ndarray],
    output_path: Path,
    dpi: int,
) -> None:
    xyz_arrays = [arr for arr in xyz_groups if arr.size > 0]
    if xyz_arrays:
        set_equal_axes(ax, np.concatenate(xyz_arrays, axis=0))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_episode_pair(
    data_episode_dir: Path | None,
    infer_episode_dir: Path | None,
    output_path: Path,
    *,
    data_label: str = "data",
    infer_label: str,
    infer_reader_fn,
    data_tool_threshold: float,
    infer_tool_threshold: float,
    data_alpha: float,
    infer_alpha: float,
    dpi: int,
) -> tuple[int, int]:
    fig, ax = _create_figure()
    xyz_a: list[np.ndarray] = []
    xyz_b: list[np.ndarray] = []
    closed_a = 0
    closed_b = 0

    if data_episode_dir is not None:
        xyz_a, closed_a = plot_group(
            ax,
            [data_episode_dir],
            color="blue",
            linewidth=1.0,
            alpha=data_alpha,
            label=data_label,
            tool_threshold=data_tool_threshold,
            reader_fn=iter_ee_xyz_and_tool_data,
            closed_label="closed gripper",
            closed_alpha=1.0,
            closed_linewidth=1.4,
            closed_marker_size=1.0,
        )

    if infer_episode_dir is not None:
        xyz_b, closed_b = plot_group(
            ax,
            [infer_episode_dir],
            color="green",
            linewidth=1.0,
            alpha=infer_alpha,
            label=infer_label,
            tool_threshold=infer_tool_threshold,
            reader_fn=infer_reader_fn,
            closed_alpha=infer_alpha,
            closed_linewidth=1.5,
            closed_marker_size=1.0,
        )

    if data_episode_dir is not None and infer_episode_dir is not None:
        title = f"{data_episode_dir.name} vs {infer_episode_dir.name}"
    elif data_episode_dir is not None:
        title = f"{data_label}: {data_episode_dir.name}"
    elif infer_episode_dir is not None:
        title = f"{infer_label}: {infer_episode_dir.name}"
    else:
        raise ValueError("At least one episode source must be provided.")

    _finalize_figure(
        fig,
        ax,
        title,
        [*xyz_a, *xyz_b],
        output_path,
        dpi,
    )
    return closed_a, closed_b


def render_overview(
    data_episode_dirs: list[Path],
    infer_episode_dirs: list[Path],
    output_path: Path,
    *,
    title: str,
    data_label: str = "data",
    infer_label: str,
    infer_reader_fn,
    data_tool_threshold: float,
    infer_tool_threshold: float,
    data_alpha: float,
    infer_alpha: float,
    dpi: int,
) -> tuple[int, int]:
    fig, ax = _create_figure()
    xyz_a: list[np.ndarray] = []
    xyz_b: list[np.ndarray] = []
    black_a = 0
    black_b = 0

    if data_episode_dirs:
        xyz_a, black_a = plot_group(
            ax,
            data_episode_dirs,
            color="blue",
            linewidth=1.0,
            label=data_label,
            alpha=data_alpha,
            tool_threshold=data_tool_threshold,
            reader_fn=iter_ee_xyz_and_tool_data,
            closed_alpha=1.0,
            closed_linewidth=1.4,
            closed_marker_size=1.0,
        )

    if infer_episode_dirs:
        xyz_b, black_b = plot_group(
            ax,
            infer_episode_dirs,
            color="green",
            linewidth=1.0,
            alpha=infer_alpha,
            label=infer_label,
            tool_threshold=infer_tool_threshold,
            reader_fn=infer_reader_fn,
            closed_alpha=infer_alpha,
            closed_linewidth=1.5,
            closed_marker_size=1.0,
        )

    _finalize_figure(fig, ax, title, [*xyz_a, *xyz_b], output_path, dpi)
    return black_a, black_b


def _episode_output_path(
    output_dir: Path,
    index: int,
    data_episode_dir: Path | None,
    infer_episode_dir: Path | None,
    infer_label: str,
) -> Path:
    if data_episode_dir is not None and infer_episode_dir is not None:
        filename = f"{index + 1:03d}_{data_episode_dir.name}_vs_{infer_episode_dir.name}.png"
    elif data_episode_dir is not None:
        filename = f"{index + 1:03d}_{data_episode_dir.name}.png"
    elif infer_episode_dir is not None:
        safe_label = infer_label.replace(" ", "_")
        filename = f"{index + 1:03d}_{safe_label}_{infer_episode_dir.name}.png"
    else:
        raise ValueError("At least one episode source must be provided.")
    return output_dir / filename


def _resolve_output_targets(output_dir: Path, save_path: Path | None, overview_name: str) -> tuple[Path, Path]:
    if save_path is None:
        return output_dir, output_dir / overview_name

    if save_path.is_dir():
        return save_path, save_path / overview_name

    if save_path.suffix.lower() in VALID_IMAGE_SUFFIXES:
        return output_dir, save_path

    return save_path, save_path / overview_name


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render per-episode and all-episode HIROL EE trajectory visualizations into one folder."
    )
    parser.add_argument("--data", default=None, help="HIROL dataset root or one episode dir for data episodes.")
    parser.add_argument(
        "--infer",
        default=None,
        help="Inference dataset root or one episode dir for infer episodes in predictions[] format.",
    )
    parser.add_argument(
        "--franka-infer",
        default=None,
        help="Inference dataset root or one episode dir for Franka-style infer episodes in data[].actions format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the overview image and each per-episode image.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Deprecated: explicit output path for the all-episode overview image.",
    )
    parser.add_argument(
        "--overview-name",
        default=DEFAULT_OVERVIEW_NAME,
        help="Filename for the all-episode overview image inside --output-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of paired episodes to render. Default: render all pairs.",
    )
    parser.add_argument(
        "--data-tool-threshold",
        type=float,
        default=70.0,
        help="Closed-gripper threshold for data episodes.",
    )
    parser.add_argument(
        "--infer-tool-threshold",
        type=float,
        default=0.5,
        help="Closed-gripper threshold for infer episodes.",
    )
    parser.add_argument(
        "--tool-threshold",
        type=float,
        default=None,
        help="Optional shared threshold override for both data and infer.",
    )
    parser.add_argument(
        "--data-alpha",
        type=float,
        default=1.0,
        help="Alpha for data trajectories.",
    )
    parser.add_argument(
        "--infer-alpha",
        type=float,
        default=0.5,
        help="Alpha for infer trajectories.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--title",
        default="HIROL EE States XYZ Trajectories",
        help="Plot title for the all-episode overview image.",
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Do not render the combined all-episode overview image.",
    )
    parser.add_argument(
        "--skip-episodes",
        action="store_true",
        help="Do not render per-episode images.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.skip_overview and args.skip_episodes:
        raise ValueError("Nothing to render: both --skip-overview and --skip-episodes are set.")
    if args.infer and args.franka_infer:
        raise ValueError("Specify at most one of --infer or --franka-infer.")
    if not any([args.data, args.infer, args.franka_infer]):
        raise ValueError("Specify at least one source: --data, --infer, or --franka-infer.")

    if args.tool_threshold is not None:
        args.data_tool_threshold = args.tool_threshold
        args.infer_tool_threshold = args.tool_threshold

    data_episode_dirs = _sorted_episode_dirs(args.data) if args.data else []
    infer_input = args.franka_infer or args.infer
    infer_episode_dirs = _sorted_episode_dirs(infer_input) if infer_input else []
    infer_reader_fn = iter_ee_xyz_and_tool_franka_infer if args.franka_infer else iter_ee_xyz_and_tool_infer
    infer_label = "franka infer" if args.franka_infer else "infer"
    has_data = bool(data_episode_dirs)
    has_infer = bool(infer_episode_dirs)

    if has_data and has_infer:
        item_count = min(len(data_episode_dirs), len(infer_episode_dirs))
        mode_desc = "episode pairs"
    elif has_data:
        item_count = len(data_episode_dirs)
        mode_desc = "data episodes"
    else:
        item_count = len(infer_episode_dirs)
        mode_desc = f"{infer_label} episodes"

    if args.limit is not None:
        item_count = min(item_count, args.limit)
    if item_count <= 0:
        raise RuntimeError("No episodes to visualize.")

    paired_data_episode_dirs = data_episode_dirs[:item_count] if has_data else []
    paired_infer_episode_dirs = infer_episode_dirs[:item_count] if has_infer else []

    output_dir, overview_output_path = _resolve_output_targets(
        args.output_dir,
        args.save_path,
        args.overview_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Rendering {item_count} {mode_desc} "
        f"(data={len(data_episode_dirs)}, infer={len(infer_episode_dirs)}) "
        f"into {output_dir}"
    )

    if not args.skip_episodes:
        if has_data and has_infer:
            episode_iter = zip(paired_data_episode_dirs, paired_infer_episode_dirs)
        elif has_data:
            episode_iter = ((episode_dir, None) for episode_dir in paired_data_episode_dirs)
        else:
            episode_iter = ((None, episode_dir) for episode_dir in paired_infer_episode_dirs)

        for idx, (data_episode_dir, infer_episode_dir) in enumerate(episode_iter):
            output_path = _episode_output_path(output_dir, idx, data_episode_dir, infer_episode_dir, infer_label)
            closed_a, closed_b = render_episode_pair(
                data_episode_dir,
                infer_episode_dir,
                output_path,
                data_label="data",
                infer_label=infer_label,
                infer_reader_fn=infer_reader_fn,
                data_tool_threshold=args.data_tool_threshold,
                infer_tool_threshold=args.infer_tool_threshold,
                data_alpha=args.data_alpha,
                infer_alpha=args.infer_alpha,
                dpi=args.dpi,
            )
            print(
                f"[{idx + 1:03d}/{item_count:03d}] saved {output_path} "
                f"(data_closed={closed_a}, infer_closed={closed_b})"
            )

    if not args.skip_overview:
        if not has_data:
            overview_title = args.title if args.title != "HIROL EE States XYZ Trajectories" else infer_label.title() + " EE States XYZ Trajectories"
        elif not has_infer:
            overview_title = args.title if args.title != "HIROL EE States XYZ Trajectories" else "Data EE States XYZ Trajectories"
        else:
            overview_title = args.title
        black_a, black_b = render_overview(
            paired_data_episode_dirs,
            paired_infer_episode_dirs,
            overview_output_path,
            title=overview_title,
            data_label="data",
            infer_label=infer_label,
            infer_reader_fn=infer_reader_fn,
            data_tool_threshold=args.data_tool_threshold,
            infer_tool_threshold=args.infer_tool_threshold,
            data_alpha=args.data_alpha,
            infer_alpha=args.infer_alpha,
            dpi=args.dpi,
        )
        print(f"Saved overview to: {overview_output_path}")
        if has_data:
            print(f"data episodes: {len(paired_data_episode_dirs)}, black points: {black_a}")
        if has_infer:
            print(f"{infer_label} episodes: {len(paired_infer_episode_dirs)}, black points: {black_b}")

    if has_data and has_infer and len(data_episode_dirs) != len(infer_episode_dirs):
        print(
            "Warning: data/infer episode counts differ; "
            f"only rendered the first {item_count} paired episodes."
        )


if __name__ == "__main__":
    main()
