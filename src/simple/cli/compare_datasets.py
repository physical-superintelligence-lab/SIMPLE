"""
Compare LeRobot datasets: replayed episodes vs original teleoperated episodes.

Validates that all non-image data fields in the replayed dataset match the original,
since the replay pipeline reuses source row fields directly.
"""

from __future__ import annotations

import json
import numpy as np
import typer
from pathlib import Path
from typing_extensions import Annotated
from tqdm import tqdm

from simple.cli.render_decoupled_wbc import _load_episodes


def _load_dataset_info(data_dir: str) -> dict:
    """Load dataset info (features list)."""
    info_path = Path(data_dir) / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def _get_common_features(replay_info: dict, teleop_info: dict) -> list[str]:
    """Identify common non-image, non-metadata features between datasets."""
    replay_features = set(replay_info["features"].keys())
    teleop_features = set(teleop_info["features"].keys())

    # Common features
    common = replay_features & teleop_features

    # Exclude image features and metadata
    exclude = {
        "index",
        "timestamp",
        "episode_index",
        "frame_index",
        "task_index",
    }
    # Also exclude any image features
    common = {f for f in common if not f.startswith("observation.images")}
    common = common - exclude

    return sorted(list(common))


def _compare_feature_value(replay_val, teleop_val) -> tuple[float, float]:
    """Compare a single feature value.

    Returns:
        (mean_l1_error, max_error)
    """
    replay_arr = np.asarray(replay_val).flatten()
    teleop_arr = np.asarray(teleop_val).flatten()

    if replay_arr.shape != teleop_arr.shape:
        # Shape mismatch — report large error
        return float("inf"), float("inf")

    abs_diff = np.abs(replay_arr - teleop_arr)
    return float(np.mean(abs_diff)), float(np.max(abs_diff))


def main(
    replay_dir: Annotated[str, typer.Argument()] = "data/replay_decoupled_wbc/simple/G1WholebodyBendPick-v1/level-0",
    teleop_dir: Annotated[str, typer.Argument()] = "data/teleop_decoupled_wbc/simple/G1WholebodyBendPick-v1/level-0",
    num_episodes: Annotated[int, typer.Option()] = -1,
    verbose: Annotated[bool, typer.Option()] = False,
):
    """Compare replay dataset with original teleop dataset, frame-by-frame."""

    # Load dataset info
    print(f"Loading dataset info...")
    replay_info = _load_dataset_info(replay_dir)
    teleop_info = _load_dataset_info(teleop_dir)

    # Find common features
    common_features = _get_common_features(replay_info, teleop_info)
    print(f"Common non-image features: {len(common_features)}")
    print(f"  {', '.join(common_features[:5])}..." if len(common_features) > 5 else f"  {', '.join(common_features)}")

    # Load episodes
    print(f"\nLoading replay dataset from {replay_dir}...")
    replay_episodes = _load_episodes(replay_dir)
    print(f"  Loaded {len(replay_episodes)} episodes")

    print(f"Loading teleop dataset from {teleop_dir}...")
    teleop_episodes = _load_episodes(teleop_dir)
    print(f"  Loaded {len(teleop_episodes)} episodes")

    # Determine episodes to compare
    episode_indices = sorted(set(replay_episodes.keys()) & set(teleop_episodes.keys()))
    if num_episodes > 0:
        episode_indices = episode_indices[:num_episodes]

    if not episode_indices:
        print("ERROR: No common episodes found!")
        return

    # Per-episode comparison
    overall_stats = {}  # feature -> list of (frame_l1, frame_max)

    for ep_idx in episode_indices:
        print(f"\n{'='*80}")
        print(f"Episode {ep_idx}")
        print(f"{'='*80}")

        replay_df = replay_episodes[ep_idx]
        teleop_df = teleop_episodes[ep_idx]

        n_replay = len(replay_df)
        n_teleop = len(teleop_df)
        n_compare = min(n_replay, n_teleop)

        print(f"Frames: teleop={n_teleop}, replay={n_replay}, comparing={n_compare}")

        # Initialize stats for this episode
        episode_stats = {feat: {"l1_errors": [], "max_errors": []} for feat in common_features}

        # Frame-by-frame comparison
        for frame_idx in tqdm(range(n_compare), desc="Frames", unit="frame"):
            replay_row = replay_df.iloc[frame_idx]
            teleop_row = teleop_df.iloc[frame_idx]

            for feat in common_features:
                try:
                    replay_val = replay_row[feat]
                    teleop_val = teleop_row[feat]

                    l1_err, max_err = _compare_feature_value(replay_val, teleop_val)

                    episode_stats[feat]["l1_errors"].append(l1_err)
                    episode_stats[feat]["max_errors"].append(max_err)

                    if verbose and (l1_err > 1e-6 or max_err > 1e-6):
                        print(f"  Frame {frame_idx} [{feat}]: L1={l1_err:.8f}, max={max_err:.8f}")

                    if feat == "observation.eef_state":
                        print("replay:", replay_val)
                        print("teleop:", teleop_val)
                        exit(0)
                except Exception as e:
                    if verbose:
                        print(f"  Frame {frame_idx} [{feat}]: ERROR - {e}")
                    episode_stats[feat]["l1_errors"].append(float("nan"))
                    episode_stats[feat]["max_errors"].append(float("nan"))

        # Print per-episode summary
        print(f"\n{'Feature':<40} {'L1 (mean)':<15} {'Max error':<15} {'Status':<10}")
        print("-" * 80)

        for feat in common_features:
            l1_errors = np.array(episode_stats[feat]["l1_errors"])
            max_errors = np.array(episode_stats[feat]["max_errors"])

            # Filter out NaNs and infs
            valid_l1 = l1_errors[np.isfinite(l1_errors)]
            valid_max = max_errors[np.isfinite(max_errors)]

            if len(valid_l1) == 0:
                mean_l1 = float("nan")
                max_max = float("nan")
                status = "✗ ERROR"
            else:
                mean_l1 = float(np.mean(valid_l1))
                max_max = float(np.max(valid_max))

                # Status
                if mean_l1 < 1e-6:
                    status = "✓ MATCH"
                elif mean_l1 < 1e-3:
                    status = "≈ CLOSE"
                else:
                    status = "✗ MISMATCH"

            print(f"{feat:<40} {mean_l1:<15.8e} {max_max:<15.8e} {status:<10}")

            # Track overall stats
            if feat not in overall_stats:
                overall_stats[feat] = []
            overall_stats[feat].extend(valid_l1)

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Feature':<40} {'L1 (mean)':<15} {'Status':<10}")
    print("-" * 65)

    all_match = True
    for feat in sorted(overall_stats.keys()):
        errors = np.array(overall_stats[feat])
        if len(errors) == 0:
            status = "✗ NO DATA"
            mean_l1 = float("nan")
        else:
            mean_l1 = float(np.mean(errors))
            if mean_l1 < 1e-6:
                status = "✓ MATCH"
            else:
                status = "✗ MISMATCH"
                all_match = False

        print(f"{feat:<40} {mean_l1:<15.8e} {status:<10}")

    if all_match:
        print(f"\n✓ SUCCESS: All features match between replay and teleop datasets!")
    else:
        print(f"\n✗ FAILURE: Some features do not match!")


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
