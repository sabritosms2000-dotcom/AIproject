# SignBridge AI — Dataset & Error Analysis Summary (Read-Only)

## 1. Split & Class Distribution (70 Classes, 560 Videos)
- **Train (285 videos):** `3` videos (21 classes), `4` videos (24 classes), `5` videos (24 classes), `6` videos (1 class: class `20` = `200`).
- **Validation (70 videos):** `1` video per class across all 70 classes.
- **Test (205 videos):** `1` video (1 class: class `20` = `200`), `2` videos (23 classes), `3` videos (26 classes), `4` videos (20 classes).
- **Total per class:** `8` videos for 68 classes, `7` videos for 1 class, `9` videos for 1 class.

## 2. Session, Date, and Signer Distribution
- **Recording dates (`split_date.csv`):**
  - `51` classes: Train + Val on `15-11-16`, Test on `16-11-16`.
  - `19` classes: Train + Val on `16-11-16`, Test on `15-11-16`.
  - Zero date overlap between Train/Val and Test in any class.
- **Signers:** `01_02` performs classes `0..30` (31 number classes); `02_02` performs classes `31..69` (39 letter/hamza classes). No class has multiple signers.

## 3. Train-to-Test Feature Shift & Poor Test Representation
- **Poor test representation:** Class `20` (`200`) has only `1` test video; `23` classes have only `2` test videos; `21` classes have only `3` training videos.
- **Feature shift (`keypoints/train_v2.npz` vs `keypoints/test_v2.npz`, L2 distance on 80 hand coords):**
  - Classes with `acc_seed_42 > 0`: mean L2 shift = `5.5079`.
  - Classes with `acc_seed_42 == 0`: mean L2 shift = `7.3123`.
  - Largest shifts: class `2` (`2`, L2 `17.68`), class `30` (`10000000`, L2 `16.54`), class `54` (`miim`, L2 `15.05`), class `67` (`Alif maqsoura`, L2 `14.99`) — all `0.00%` across all 3 seeds.

## 4. Full 205-Video Test Predictions (Seed 42)
- **Overall:** `56 / 205` correct (`27.32%`).
- **Per-class spread:** `10 / 70` classes achieved `100%` (e.g., `Daad` 4/4, `ALif with maad` 4/4, `90` 4/4); `40 / 70` classes scored `0%`.
- **Top off-diagonal confusions:** `5 -> 9` (4), `daal -> raa` (4), `2 -> 3` (3), `6 -> 50` (3), `siin -> Waaw with hamza` (3), `Ayn -> zay` (3).

## 5. Generated Artifacts
- `analysis_reports/per_class_analysis.csv`
- `analysis_reports/confusion_matrix_v2_seed_42_labeled.csv`
- `analysis_reports/top_confusions_seed_42.csv`
- `analysis_reports/SUMMARY.md`
