"""
SignBridge AI — Evaluate V2 BiLSTM Inference on a 10-Class Test Subset

Selects 10 distinct classes (0, 7, 14, 21, 28, 35, 42, 49, 56, 63) from
split_date.csv where split == "test", runs predict_video() from infer.py
using existing checkpoint and standardization artifacts, prints a summary table,
and saves test_subset_results.csv.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from infer import predict_video, DEFAULT_CHECKPOINT, DEFAULT_STATS_PATH

SPLIT_CSV = Path("split_date.csv")
OUTPUT_CSV = Path("test_subset_results.csv")
TARGET_CLASSES = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]


def select_test_subset(split_csv: Path, target_classes: list[int]) -> list[tuple[int, Path]]:
    """Select one existing test video per target class from split_date.csv."""
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    df = pd.read_csv(split_csv, encoding="utf-8-sig")
    if not {"path", "y", "split"}.issubset(df.columns):
        raise ValueError("split_date.csv must contain columns: path, y, split")

    test_df = df[df["split"] == "test"].copy()
    test_df["y"] = pd.to_numeric(test_df["y"], errors="raise").astype(int)

    selected: list[tuple[int, Path]] = []
    for cls_id in target_classes:
        cls_rows = test_df[test_df["y"] == cls_id]
        if cls_rows.empty:
            raise ValueError(f"No test rows found for class {cls_id} in {split_csv}")

        chosen_path = None
        for raw_path in cls_rows["path"].astype(str):
            candidate = Path(raw_path.replace("\\", "/"))
            if candidate.exists() and candidate.is_file():
                chosen_path = candidate
                break

        if chosen_path is None:
            raise FileNotFoundError(
                f"No existing video file found on disk for test class {cls_id}"
            )
        selected.append((cls_id, chosen_path))

    return selected


def main() -> None:
    if not DEFAULT_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {DEFAULT_CHECKPOINT}")
    if not DEFAULT_STATS_PATH.exists():
        raise FileNotFoundError(f"Standardization file not found: {DEFAULT_STATS_PATH}")

    selected = select_test_subset(SPLIT_CSV, TARGET_CLASSES)

    rows = []
    print("\nRunning inference on 10-class test subset...\n")
    header = (
        f"{'Video Filename':<36} | {'GT':>3} | {'Pred':>4} | "
        f"{'Status':<9} | {'Confidence':>10} | {'Time (s)':>8}"
    )
    print(header)
    print("-" * len(header))

    for gt_class, video_path in selected:
        filename = video_path.name
        try:
            res = predict_video(
                video_path=video_path,
                checkpoint_path=DEFAULT_CHECKPOINT,
                stats_path=DEFAULT_STATS_PATH,
                device="cpu",
            )
            pred_class = int(res["class_index"])
            confidence = float(res["confidence"])
            infer_time = float(res["inference_time_seconds"])
            is_correct = pred_class == gt_class
            status = "Correct" if is_correct else "Incorrect"
            error_msg = ""

            print(
                f"{filename:<36} | {gt_class:>3} | {pred_class:>4} | "
                f"{status:<9} | {confidence * 100:>9.2f}% | {infer_time:>8.2f}"
            )
        except Exception as exc:
            pred_class = -1
            confidence = 0.0
            infer_time = 0.0
            is_correct = False
            status = "Error"
            error_msg = f"{type(exc).__name__}: {exc}"
            print(
                f"{filename:<36} | {gt_class:>3} | {'ERR':>4} | "
                f"{status:<9} | {'N/A':>10} | {'N/A':>8}  ({error_msg})"
            )

        rows.append({
            "video_filename": filename,
            "video_path": str(video_path),
            "ground_truth_class": gt_class,
            "predicted_class": pred_class,
            "status": status,
            "correct": is_correct,
            "confidence": confidence,
            "inference_time_seconds": infer_time,
            "error": error_msg,
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_CSV, index=False)

    total_tested = len(results_df)
    total_correct = int(results_df["correct"].sum())
    accuracy = (total_correct / total_tested) if total_tested > 0 else 0.0
    valid_times = results_df.loc[results_df["status"] != "Error", "inference_time_seconds"]
    avg_time = float(valid_times.mean()) if not valid_times.empty else 0.0

    print("-" * len(header))
    print(f"Total Correct        : {total_correct}")
    print(f"Total Tested         : {total_tested}")
    print(f"Accuracy             : {accuracy * 100:.2f}% ({total_correct}/{total_tested})")
    print(f"Avg Inference Time   : {avg_time:.2f}s")
    print(f"Saved Results To     : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
