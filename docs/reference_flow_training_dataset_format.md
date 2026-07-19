# Flow Training Dataset Format

> **Reference-only document:** this was copied because it informed the GUAVA
> ingestion decision. The standalone project provides `smr-complete-guava`;
> other general Flow dataset commands shown below remain SOKE utilities and
> are not part of SignMotionRefinement.

This repository trains the Flow models from compact upper-body SMPL-X sequence files plus JSONL manifests. The prepared datasets live under `/media/cvpr/haomian/data/SOKE_FLOW`.

## Directory Layout

Each dataset root must have this structure:

```text
<dataset_root>/
  train/
    <sample_name>.npz
  val/
    <sample_name>.npz
  test/
    <sample_name>.npz
  meta/
    manifest_train.jsonl
    manifest_val.jsonl
    manifest_test.jsonl
    mean.npy
    std.npy
```

Optional files may be created on demand:

```text
meta/mean_rot6d.npy
meta/std_rot6d.npy
```

## Motion Files

Each sample is a NumPy `.npz` file. `motion` is required; the hand-validity arrays are strongly recommended and are present in the prepared datasets:

```text
motion:      float32 [T, 133]
left_valid:  float32 [T]
right_valid: float32 [T]
```

`motion` is the compact 133-D upper-body signing representation:

```text
upper body: 30
left hand:  45
right hand: 45
jaw:         3
expression: 10
total:     133
```

The values are axis-angle SMPL-X pose/expression features. The Flow dataset loader normalizes them using `meta/mean.npy` and `meta/std.npy`.

### Completed / Fitted Frame Extensions

Datasets reconstructed from sparse or confidence-filtered tracking may include
additional arrays while keeping the required Flow fields above:

```text
observed_mask:             bool  [T]  # frame came from the tracker
filled_mask:               bool  [T]  # frame was fitted; inverse of observed_mask
original_frame_index:      int32 [T]  # index on the native source-video timeline
tracked_frame_index:       int32 [T]  # retained tracker index, or -1 when fitted
missing_run_length:        int32 [T]  # size of the missing span containing each frame
nearest_observed_distance: int32 [T]  # distance to the closest tracker observation
rot6d:                     float32 [T, 256]  # optional cached compact rot6D motion
```

The standard loader ignores these extra arrays, so the dataset remains
compatible with `UpperSMPLXFlowDataset`. Consumers that evaluate fitted frames
or need confidence-aware losses should read `observed_mask`/`filled_mask`
directly from the NPZ.

For completed GUAVA data, `left_valid` and `right_valid` retain the tracking
quality semantics: a hand is zero on a discarded frame when its confidence was
low or the hands were too close to disambiguate. The fitted motion is still
present at that frame. Native source FPS is retained in the manifest so frame
indices continue to correspond exactly to the original video.

## Manifest Files

Each manifest is JSONL: one JSON object per sample. Required fields:

```json
{
  "name": "sample_name",
  "motion_path": "train/sample_name.npz",
  "text": "spoken language sentence",
  "fps": 20.0,
  "num_frames": 120,
  "duration": 6.0
}
```

`motion_path` should usually be relative to the dataset root. Absolute paths are also accepted by the loader and are useful for manifest-only joint datasets that reference already-prepared motion files.

Optional annotation fields:

```json
{
  "gloss": "gloss sequence, when the source dataset has gloss annotations"
}
```

Recommended provenance fields:

```json
{
  "dataset": "how2sign|csl_daily|phoenix14t|chatsign",
  "source_name": "original source id",
  "source_split": "train|val|test|dev",
  "source_fps": 25.0,
  "source_pose_frames": 150
}
```

Dataset-specific metadata such as `signer`, `video_id`, `sentence_id`, `start_realigned`, and `end_realigned` can also be included. The loader ignores unknown fields, so extra metadata is safe.

## Word / Lexicon Datasets

Word-prior datasets can contain one or more pose clips for the same word or gloss. A one-clip-per-key dictionary can use the lexicon key directly:

```text
ASK.npz
HAVE-SEEN.npz
```

For multiple clips of the same key, use this filename convention:

```text
<lexicon_key>-<variant_id>.npz
```

Examples:

```text
ASK-0001.npz
ASK-0002.npz
THANK_YOU-0001.npz
```

For automatic variant parsing, use letters, digits, or underscores for `<lexicon_key>`, then one final hyphen followed by one or more digits for the variant suffix. The matcher strips that suffix before language matching, so `ASK-0001` and `ASK-0002` both match the lexicon key `ASK`; the numeric suffix is not treated as a word token.

Hyphenated phrase names without a numeric suffix are valid, for example `HAVE-SEEN.npz`. If a hyphenated phrase also needs variants, prefer underscore keys such as `HAVE_SEEN-0001.npz`, or set `lexicon_key` and `variant_id` explicitly in the manifest.

Minimal one-clip word manifest row, accepted by the prior:

```json
{
  "name": "ASK",
  "motion_path": "train/ASK.npz",
  "text": "ASK",
  "fps": 20.0,
  "num_frames": 48,
  "duration": 2.4
}
```

Recommended manifest row for variant word datasets:

```json
{
  "name": "ASK-0001",
  "lexicon_key": "ASK",
  "variant_id": "0001",
  "word": "ASK",
  "motion_path": "train/ASK-0001.npz",
  "text": "ASK",
  "fps": 20.0,
  "num_frames": 48,
  "duration": 2.4
}
```

The current `chatsign_175_word` dataset uses this explicit variant form, even for one clip per key. For example, `SHIP` is stored as `SHIP-0001.npz`, and hyphenated labels use ASCII-safe keys such as `HAVE_SEEN-0001.npz`. The original label is preserved in `word` and `gloss`.

If `lexicon_key` is omitted, `flow.residual_prior.WordMotionPrior` derives it from `name` or the `motion_path` stem by removing a final `-\d+` suffix. Explicit fields have priority in this order:

```text
lexicon_key -> word -> gloss -> label -> stripped name / file stem
```

The matcher stores all variants:

```text
entries_by_key[("ASK",)] = [ASK-0001, ASK-0002, ...]
```

Legacy concat priors still use the first variant deterministically through `match_text`. The soft word arranger uses `match_text_variants`, so every matched variant can become a positive candidate, together with random negative word clips.

When preparing a word dataset with `flow.dataset.prepare_dataset`, pass:

```bash
python -m flow.dataset.prepare_dataset \
  ... \
  --parse_word_variant_names
```

This adds `lexicon_key`, `variant_id`, and `word` to manifest rows for samples whose names follow the suffix convention.

## Normalization Stats

`meta/mean.npy` and `meta/std.npy` must be:

```text
float32 [133]
```

They are computed from the training split only:

```text
x_norm = (x - mean) / std
```

The standard deviation should be clamped to at least `1e-4`.

## Prepared Datasets

The currently prepared datasets are:

```text
/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx
/media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx
/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx
/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_word
/media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_sentence_word_joint
/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word
```

Current counts:

```text
how2sign_soke_upper_smplx:
  train: 30685
  val:    1717
  test:   2308

csl_daily_upper_smplx:
  train: 18399
  val:    1077
  test:   1176

phoenix_upper_smplx:
  train: 7092
  val:    519
  test:   642

chatsign_175_word:
  train: 538
  val:   538
  test:  538

chatsign_175_sentence_word_joint:
  train: 1076
  val:   1076
  test:  1076

phoenix_upper_smplx_word:
  train: 55213
  val:    3748
  test:   4264
```

Notes:

- How2Sign CSV annotations do not include glosses, so `gloss` is stored as an empty string.
- CSL-Daily and Phoenix include gloss annotations, and those are preserved in the manifest.
- Phoenix `dev` is mapped to Flow `val`.
- All prepared datasets listed here are stored at `fps: 20.0`.
- `chatsign_175_word` stores the same 538-word dictionary in train/val/test. It uses explicit variant names such as `SHIP-0001` and `HAVE_SEEN-0001`, with `lexicon_key`, `variant_id`, `word`, and `gloss` fields in every manifest row.
- `chatsign_175_sentence_word_joint` is a manifest-only joint view with 538 balanced sentence rows plus 538 word rows per split. Its manifest rows use absolute `motion_path` values that point to the source sentence and word datasets.
- `phoenix_upper_smplx_word` is derived from `phoenix_upper_smplx` by evenly splitting each sentence motion over its gloss tokens. It uses explicit variant names such as `REGEN-0001` and `ES_BEDEUTET-0001`; the original gloss token is preserved in `word` and `gloss`.

## Loader Contract

`flow.dataset.UpperSMPLXFlowDataset` expects this layout. It returns items with:

```text
name
text
gloss
motion
length
left_valid
right_valid
rotation_rep
```

If `gloss` is missing from a manifest row, the loader returns an empty string for that item.

`collate_upper_smplx` batches these into:

```text
name
text
gloss
motion
length
mask
left_valid
right_valid
```

When `rotation_rep="rot6d"` is requested, the loader converts the stored 133-D axis-angle motion to 256-D rot6d at load time and caches `mean_rot6d.npy` / `std_rot6d.npy`.

## Regeneration Scripts

The prepared datasets can be regenerated with:

```bash
python -m flow.dataset.prepare_how2sign_soke_dataset \
  --how2sign_root /media/cvpr/haomian/data/SOKE/How2Sign \
  --out_dir /media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx \
  --target_fps 20 \
  --max_duration 30 \
  --num_workers 20 \
  --overwrite

python -m flow.dataset.prepare_csl_daily_dataset \
  --csl_root /media/cvpr/haomian/data/SOKE/CSL-Daily \
  --out_dir /media/cvpr/haomian/data/SOKE_FLOW/csl_daily_upper_smplx \
  --source_fps 25 \
  --target_fps 20 \
  --num_workers 8 \
  --overwrite

python -m flow.dataset.prepare_phoenix_dataset \
  --phoenix_root /media/cvpr/haomian/data/SOKE/Phoenix_2014T \
  --out_dir /media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx \
  --source_fps 25 \
  --target_fps 20 \
  --num_workers 8 \
  --overwrite

python -m flow.dataset.prepare_tracked_guard_dataset \
  --tracked_dir /media/cvpr/haomian/data/word_lib_sent_smpl \
  --out_dir /media/cvpr/haomian/data/SOKE_FLOW/chatsign_175_word \
  --all_dirs \
  --missing_text name \
  --word_variant_format \
  --fps 20 \
  --overwrite

python -m flow.dataset.prepare_phoenix_word_dataset \
  --source_dir /media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx \
  --out_dir /media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx_word \
  --overwrite

smr-complete-guava \
  --tracked_dir /media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx_GUAVA/guava_tracked \
  --out_dir /media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx_GUAVA/guava_completed_flow \
  --source_manifest_dir /media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx/meta \
  --overwrite
```
