# Data

`manifest.csv` (FlyBot's photo export, git-ignored) columns:

`snapshot_id, device_serial, provider, captured_at, image_path, s3_key, farm_id, plot_id, lat, lon, trap_kind, ai_count, ai_counts_json, review_status, human_count, quality_issues, previous_snapshot_id, card_replaced_at, treatment_ids`

Raw exports live in `data/export/` and `data/raw/` (git-ignored). Faces are blurred before anything leaves the production side. Roboflow dataset credit: camos/fruit-fly (CC BY 4.0).

## Count labels (T0.7b, git-ignored)

`data/labels/gt_sample.csv` (+ `.json` allocation) from `eval/select_gt_sample.py`; human counts in
`data/labels/counts.csv` from `eval/label_counts.py`; pre-label cache `data/labels/prelabel__*.json`.
Protocol: `eval/GROUND_TRUTH.md`. Rows with `labeler=agent-test` are tool checks, not ground truth.

## Excluded cameras

`data/excluded_cameras.csv` (committed) lists cameras that every eval, replay and training script must skip. AC6527557 "Gravity01" was excluded on 2026-10-15: it films a building facade, not a trap.
