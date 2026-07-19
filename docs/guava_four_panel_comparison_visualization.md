# GUAVA Four-Panel Comparison Visualization

Last reproduced: 2026-07-17

This runbook documents how to turn compact GUAVA completion fits into the
photorealistic, frame-aligned four-panel videos used for visual comparison:

```text
original RGB | sparse GUAVA | dense method A | dense method B
```

The usual completion comparison is:

```text
original RGB | sparse GUAVA | SO(3) SLERP | learned completion
```

This is a two-stage pipeline:

1. render each dense `[T, 133]` motion sequence with the tracked `women001`
   GUAVA identity;
2. compose the original RGB, sparse GUAVA state, and two dense GUAVA renders
   on the original video timeline.

The procedure below produces the photorealistic GUAVA avatar. It is separate
from this project's software-mesh renderer
`src/sign_motion_refinement/visualization/meta_jerk.py`,
which renders SMPL-X meshes and an animated jerk chart.

---

## 1. Implementation Files

Dense compact-motion renderer:

```text
/media/cvpr/haomian/SignMotionRefinement/integrations/guava/render_guava_compact_motion.py
```

Four-panel compositor:

```text
/media/cvpr/haomian/SignMotionRefinement/integrations/guava/compose_guava_motion_comparison.py
```

The renderer owns GUAVA model loading and photorealistic animation. The
compositor owns timeline reconstruction, labels, placeholders, exact output
FPS, per-video metadata, and the aggregate manifest.

Set `GUAVA_ROOT` so the renderer can import GUAVA-local modules. The extracted
scripts can be launched from the SignMotionRefinement root:

```bash
cd /media/cvpr/haomian/SignMotionRefinement
```

---

## 2. Panel Contract

The four panels have the following meanings.

| Panel | Content |
|---|---|
| 1 | Original RGB frame from the source video recorded in `frame_trace.json` |
| 2 | Sparse GUAVA pose on observed frames; explicit `NO TRACKED POSE` placeholder on discarded frames |
| 3 | Dense method A, normally the SO(3) SLERP scaffold |
| 4 | Dense method B, normally a learned completion |

The footer timeline uses gold for observed frames and green for
discarded/filled frames. A discarded placeholder includes the original reason,
for example `left hand low confidence`.

### Critical invariant

The compositor uses the method-A render as the sparse GUAVA image on observed
frames. Therefore all of the following must be true before composition:

1. method A and method B both have exactly `T` frames;
2. the original RGB video also has exactly `T` frames;
3. method A is an exact copy of the source GUAVA motion at observed indices;
4. method B is an exact copy of the source GUAVA motion at observed indices;
5. the fit and source-completion `observed_mask` arrays are identical.

Do not use this compositor for methods that modify observed anchors unless its
sparse-panel and status logic is changed first.

---

## 3. Environment and Assets

Set the common paths once per shell:

```bash
export GUAVA_ROOT=/media/cvpr/haomian/GUAVA
export SMR_ROOT=/media/cvpr/haomian/SignMotionRefinement
export GUAVA_PYTHON=/media/cvpr/haomian/python_envs/GUAVA/bin/python
export GUAVA_MODEL=/media/cvpr/haomian/GUAVA/assets/GUAVA
export GUAVA_SOURCE=/media/cvpr/haomian/GUAVA/outputs/tracked_source_images/women001
export RENDER_SCRIPT="$SMR_ROOT/integrations/guava/render_guava_compact_motion.py"
export COMPOSE_SCRIPT="$SMR_ROOT/integrations/guava/compose_guava_motion_comparison.py"

cd "$SMR_ROOT"
```

Required local assets:

```text
/media/cvpr/haomian/GUAVA/assets/GUAVA/checkpoints/
/media/cvpr/haomian/GUAVA/assets/SMPLX/
/media/cvpr/haomian/GUAVA/assets/FLAME/
/media/cvpr/haomian/GUAVA/outputs/tracked_source_images/women001/
```

The ready `women001` tracking bundle contains files such as:

```text
base_tracking.pkl
id_share_params.pkl
optim_tracking_ehm.pkl
optim_tracking_flame.pkl
videos_info.json
img_lmdb/
images/women001.png
```

If this tracked bundle is unavailable, process the source image with the
GUAVA-recommended EHM-Tracker first. A plain PNG is not enough for the renderer.

Check the runtime before starting a long job:

```bash
test -x "$GUAVA_PYTHON"
test -d "$GUAVA_MODEL/checkpoints"
test -d "$GUAVA_SOURCE"
command -v ffmpeg
command -v ffprobe
nvidia-smi
```

The compositor also uses `jq` in the validation commands below, although `jq`
is not required to create the videos.

---

## 4. Input NPZ Contract

A fit file must contain at least:

| Key | Expected shape | Meaning |
|---|---:|---|
| `source_completion` | scalar string | Path to the timeline-restored GUAVA completion NPZ |
| `observed_mask` | `[T]` bool | Original frames retained by GUAVA confidence filtering |
| method-A motion | `[T, 133]` float | First dense method |
| method-B motion | `[T, 133]` float | Second dense method |

Common motion-key-to-video-suffix mappings are built into the renderer:

| Fit motion key | Rendered suffix | Compositor method name |
|---|---|---|
| `linear_motion` | `linear` | `linear` |
| `siren_motion` | `siren` | `siren` |
| `slerp_motion` | `slerp` | `slerp` |
| `frozen_soft_motion` | `frozen_soft` | `frozen_soft` |
| `finetuned_motion` | `finetuned` | `finetuned` |

The compact 133-D layout is:

| Slice | Parameters |
|---|---|
| `0:30` | ten upper-body SMPL-X joints, written to `body_pose[:, 11:21]` |
| `30:75` | 15 left-hand axis-angle joints |
| `75:120` | 15 right-hand axis-angle joints |
| `120:123` | FLAME jaw axis-angle |
| `123:133` | first ten SMPL-X and FLAME expression coefficients |

Identity, shape, global pose, lower-body pose, and camera remain fixed from the
tracked source image. The visualized differences therefore come from the
compact motion arrays.

The source completion must contain `motion`, `observed_mask`,
`tracked_frame_index`, `missing_run_length`, `nearest_observed_distance`, and
`source_frame_trace`. The frame trace supplies the original video path and the
discard reason for each removed native frame.

---

## 5. Choose a Run Preset

Choose one preset, then use the common commands in Sections 7 and 8.

### 5.1 Mask-aware meta v1

```bash
export PILOT_ROOT="$SMR_ROOT/artifacts/visualizations/guava_mask_aware_meta_pilot"
export FITS="$PILOT_ROOT/fits"
export RENDER_ROOT="$PILOT_ROOT/guava_renders/women001"
export COMPARE_ROOT="$RENDER_ROOT/four_panel_comparisons"

export METHOD_A_KEY=slerp_motion
export METHOD_B_KEY=finetuned_motion
export METHOD_A=slerp
export METHOD_B=finetuned
export METHOD_A_LABEL='SO(3) SLERP result'
export METHOD_B_LABEL='Mask-aware meta result'
export METHOD_A_STATUS='SO(3) SLERP'
export METHOD_B_STATUS='mask-aware meta'
export OUTPUT_TAG=original_guava_slerp_finetuned
```

### 5.2 Improved C2/FK-temporal meta

```bash
export PILOT_ROOT="$SMR_ROOT/artifacts/visualizations/guava_mask_aware_meta_c2_fk_jerk_pilot"
export FITS="$PILOT_ROOT/fits"
export RENDER_ROOT="$PILOT_ROOT/guava_renders/women001"
export COMPARE_ROOT="$RENDER_ROOT/four_panel_comparisons"

export METHOD_A_KEY=slerp_motion
export METHOD_B_KEY=finetuned_motion
export METHOD_A=slerp
export METHOD_B=finetuned
export METHOD_A_LABEL='SO(3) SLERP result'
export METHOD_B_LABEL='Improved C2/FK result'
export METHOD_A_STATUS='SO(3) SLERP'
export METHOD_B_STATUS='C2/FK-temporal meta'
export OUTPUT_TAG=original_guava_slerp_c2_fk_meta
```

### 5.3 Linear versus direct SIREN

```bash
export PILOT_ROOT="$SMR_ROOT/artifacts/visualizations/guava_linear_siren_jerk_compare"
export FITS="$PILOT_ROOT/fits"
export RENDER_ROOT="$PILOT_ROOT/guava_renders/women001"
export COMPARE_ROOT="$RENDER_ROOT/four_panel_comparisons"

export METHOD_A_KEY=linear_motion
export METHOD_B_KEY=siren_motion
export METHOD_A=linear
export METHOD_B=siren
export METHOD_A_LABEL='Linear motion result'
export METHOD_B_LABEL='SIREN motion result'
export METHOD_A_STATUS='linear rot6D'
export METHOD_B_STATUS='direct SIREN'
export OUTPUT_TAG=original_guava_linear_siren
```

### 5.4 Retained-GUAVA-only diagnostic

This preset visualizes the forced alpha-1 audit result. It is a rejected
diagnostic, not the deployable model. The selected deployment remains alpha 0,
which is exactly the SO(3) SLERP scaffold.

```bash
export PILOT_ROOT="$SMR_ROOT/artifacts/visualizations/guava_self_only_meta_c2_fk_jerk_soke_free_diagnostic_pilot"
export FITS="$PILOT_ROOT/fits"
export RENDER_ROOT="$PILOT_ROOT/guava_renders/women001"
export COMPARE_ROOT="$RENDER_ROOT/four_panel_comparisons"

export METHOD_A_KEY=slerp_motion
export METHOD_B_KEY=finetuned_motion
export METHOD_A=slerp
export METHOD_B=finetuned
export METHOD_A_LABEL='SO(3) SLERP result'
export METHOD_B_LABEL='GUAVA-only diagnostic (rejected)'
export METHOD_A_STATUS='SO(3) SLERP'
export METHOD_B_STATUS='alpha 1 diagnostic; deployed alpha 0'
export OUTPUT_TAG=original_guava_slerp_guava_only_diagnostic_meta
```

---

## 6. Preflight Validation

Run this before loading GUAVA. It checks keys, shapes, finite values, source
video length, observed masks, and exact observed-anchor copy-back.

The preset variables `FITS`, `METHOD_A_KEY`, and `METHOD_B_KEY` must be exported.

```bash
"$GUAVA_PYTHON" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

import numpy as np

fits_dir = Path(os.environ["FITS"])
method_a_key = os.environ["METHOD_A_KEY"]
method_b_key = os.environ["METHOD_B_KEY"]
files = sorted(fits_dir.glob("*.npz"))
if not files:
    raise FileNotFoundError(f"No fit NPZ files under {fits_dir}")

def scalar_text(value):
    value = np.asarray(value).reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)

def max_delta(left, right, mask):
    return float(np.max(np.abs(left[mask] - right[mask]))) if mask.any() else 0.0

total_frames = 0
total_observed = 0
for fit_path in files:
    with np.load(fit_path, allow_pickle=False) as fit:
        for key in (method_a_key, method_b_key, "observed_mask", "source_completion"):
            if key not in fit.files:
                raise KeyError(f"{fit_path} does not contain {key!r}")
        method_a = fit[method_a_key].astype(np.float32)
        method_b = fit[method_b_key].astype(np.float32)
        fit_mask = fit["observed_mask"].astype(bool)
        completion_path = Path(scalar_text(fit["source_completion"]))

    with np.load(completion_path, allow_pickle=False) as completion:
        source_motion = completion["motion"].astype(np.float32)
        observed_mask = completion["observed_mask"].astype(bool)
        original_index = completion["original_frame_index"].astype(np.int64)
        trace_path = Path(scalar_text(completion["source_frame_trace"]))

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    source_video = Path(trace["source_video"])
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate", "-of", "json",
        str(source_video),
    ]))["streams"][0]
    native_frames = int(probe["nb_frames"])

    expected_shape = (native_frames, 133)
    assert method_a.shape == expected_shape, (fit_path, method_a.shape, expected_shape)
    assert method_b.shape == expected_shape, (fit_path, method_b.shape, expected_shape)
    assert source_motion.shape == expected_shape
    assert fit_mask.shape == observed_mask.shape == (native_frames,)
    assert np.array_equal(fit_mask, observed_mask)
    assert np.array_equal(original_index, np.arange(native_frames))
    assert np.isfinite(method_a).all() and np.isfinite(method_b).all()

    delta_a = max_delta(method_a, source_motion, observed_mask)
    delta_b = max_delta(method_b, source_motion, observed_mask)
    assert delta_a == 0.0, (fit_path, method_a_key, delta_a)
    assert delta_b == 0.0, (fit_path, method_b_key, delta_b)

    observed = int(observed_mask.sum())
    print(
        f"{fit_path.name}\tframes={native_frames}\tobserved={observed}"
        f"\tdiscarded={native_frames - observed}\tfps={probe['r_frame_rate']}"
        f"\tanchor_delta=({delta_a:.1f},{delta_b:.1f})"
    )
    total_frames += native_frames
    total_observed += observed

print(
    f"TOTAL\tclips={len(files)}\tframes={total_frames}"
    f"\tobserved={total_observed}\tdiscarded={total_frames-total_observed}"
)
PY
```

For both 12-clip mask-aware pilots, the expected totals are:

```text
clips=12  frames=2455  observed=1589  discarded=866
```

---

## 7. Stage 1: Render Both Dense Methods with GUAVA

### 7.1 Optional smoke test

The model takes roughly two to three minutes to initialize even for a smoke
test. This command renders three frames from the first fit for each method:

```bash
export SMOKE_ROOT="/tmp/guava_compact_render_smoke_$(date +%Y%m%d_%H%M%S)_$$"

CUDA_VISIBLE_DEVICES=0 "$GUAVA_PYTHON" "$RENDER_SCRIPT" \
  --fits "$FITS" \
  --source_data_path "$GUAVA_SOURCE" \
  --model_path "$GUAVA_MODEL" \
  --save_path "$SMOKE_ROOT" \
  --devices 0 \
  --motion_keys "$METHOD_A_KEY" "$METHOD_B_KEY" \
  --limit 1 \
  --max_frames 3 \
  --overwrite
```

Check the smoke outputs:

```bash
find "$SMOKE_ROOT" -name '*.mp4' -print
find "$SMOKE_ROOT" -name '*.mp4' -print0 | \
  xargs -0 -n1 ffprobe -v error \
    -show_entries stream=codec_name,width,height,nb_frames \
    -of default=noprint_wrappers=1
```

Do not run the compositor on this truncated smoke test. It intentionally fails
the native frame-count contract.

### 7.2 Full GPU render

```bash
CUDA_VISIBLE_DEVICES=0 "$GUAVA_PYTHON" "$RENDER_SCRIPT" \
  --fits "$FITS" \
  --source_data_path "$GUAVA_SOURCE" \
  --model_path "$GUAVA_MODEL" \
  --save_path "$RENDER_ROOT" \
  --devices 0 \
  --motion_keys "$METHOD_A_KEY" "$METHOD_B_KEY"
```

When `CUDA_VISIBLE_DEVICES` exposes only one GPU, use `--devices 0` because the
visible device is remapped to `cuda:0` inside the process.

For 12 fits and two methods, expect 24 MP4s and 24 JSONL records:

```bash
wc -l "$RENDER_ROOT/render_manifest.jsonl"
jq -r '.output' "$RENDER_ROOT/render_manifest.jsonl" | wc -l

while IFS= read -r video; do
  test -f "$video" || { printf 'MISSING %s\n' "$video"; exit 1; }
done < <(jq -r '.output' "$RENDER_ROOT/render_manifest.jsonl")
```

Intermediate structure:

```text
guava_renders/women001/
  render_manifest.jsonl
  <sequence>/
    <sequence>_<method_a>.mp4
    <sequence>_<method_b>.mp4
```

The intermediate method-video FPS is not authoritative for final comparison
timing. The compositor reads frames by index and writes the final MP4 at the
exact rational FPS probed from the original RGB video, such as `24000/1001`.

---

## 8. Stage 2: Compose the Four Panels

The compositor is CPU/FFmpeg work and does not require a GPU:

```bash
"$GUAVA_PYTHON" "$COMPOSE_SCRIPT" \
  --fits "$FITS" \
  --render_root "$RENDER_ROOT" \
  --save_path "$COMPARE_ROOT" \
  --method_a "$METHOD_A" \
  --method_b "$METHOD_B" \
  --method_a_label "$METHOD_A_LABEL" \
  --method_b_label "$METHOD_B_LABEL" \
  --method_a_status "$METHOD_A_STATUS" \
  --method_b_status "$METHOD_B_STATUS" \
  --output_tag "$OUTPUT_TAG"
```

Default panel size is 512x512. The header is 72 pixels and the footer is 44
pixels, so the final four-panel resolution is 2048x628.

Output structure:

```text
four_panel_comparisons/
  comparison_manifest.json
  <sequence>_<output_tag>.mp4
  <sequence>_<output_tag>.json
```

Each per-video JSON records source paths, method videos, panel order, native
frame count, observed/discarded counts, exact source FPS, resolution, and the
sparse-panel contract.

Use `--overwrite` on either stage only when final files must be regenerated.
Without it, complete existing outputs are reused.

---

## 9. Validate the Finished Batch

### 9.1 Aggregate metadata

```bash
jq '{
  count: (.renders | length),
  total_frames: (.renders | map(.native_frames) | add),
  total_observed: (.renders | map(.observed_frames) | add),
  total_discarded: (.renders | map(.discarded_frames) | add),
  description,
  output_tags: (.renders | map(.output_tag) | unique)
}' "$COMPARE_ROOT/comparison_manifest.json"
```

### 9.2 Exact frame count, FPS, and resolution

```bash
fail=0
count=0
total=0
while IFS=$'\t' read -r output expected_frames expected_fps; do
  IFS=$'\t' read -r width height actual_fps actual_frames < <(
    ffprobe -v error -select_streams v:0 \
      -show_entries stream=width,height,r_frame_rate,nb_frames \
      -of json "$output" |
      jq -r '.streams[0] | [.width,.height,.r_frame_rate,.nb_frames] | @tsv'
  )
  count=$((count + 1))
  total=$((total + actual_frames))
  if [[ "$width" != 2048 || "$height" != 628 || \
        "$actual_frames" != "$expected_frames" || \
        "$actual_fps" != "$expected_fps" ]]; then
    printf 'MISMATCH\t%s\t%sx%s\t%s/%s frames\t%s/%s fps\n' \
      "$output" "$width" "$height" \
      "$actual_frames" "$expected_frames" \
      "$actual_fps" "$expected_fps"
    fail=1
  fi
done < <(
  jq -r '.renders[] | [.output,.native_frames,.source_fps_fraction] | @tsv' \
    "$COMPARE_ROOT/comparison_manifest.json"
)
printf 'comparison_count=%d total_frames=%d mismatches=%d\n' \
  "$count" "$total" "$fail"
test "$fail" -eq 0
```

### 9.3 Decode every video end to end

Method videos:

```bash
while IFS= read -r video; do
  ffmpeg -v error -i "$video" -f null - || exit 1
done < <(jq -r '.output' "$RENDER_ROOT/render_manifest.jsonl")
```

Comparison videos:

```bash
while IFS= read -r video; do
  ffmpeg -v error -i "$video" -f null - || exit 1
done < <(jq -r '.renders[].output' "$COMPARE_ROOT/comparison_manifest.json")
```

### 9.4 Check for incomplete temporary files

```bash
find "$RENDER_ROOT" -type f -name '*.saving*' -print
```

No output is expected after a successful run.

### 9.5 Inspect observed and discarded visual states

Read the first output and fit paths:

```bash
VIDEO=$(jq -r '.renders[0].output' "$COMPARE_ROOT/comparison_manifest.json")
FIT=$(jq -r '.renders[0].fit' "$COMPARE_ROOT/comparison_manifest.json")
printf 'video=%s\nfit=%s\n' "$VIDEO" "$FIT"
```

Find representative zero-based frame indices:

```bash
"$GUAVA_PYTHON" - "$FIT" <<'PY'
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as fit:
    mask = fit["observed_mask"].astype(bool)
print("first_observed_zero_based", int(np.flatnonzero(mask)[0]))
print("first_discarded_zero_based", int(np.flatnonzero(~mask)[0]))
PY
```

Extract frames with FFmpeg. Here `FRAME` is zero-based, while the label drawn
inside the output video is one-based:

```bash
FRAME=3
ffmpeg -y -v error -i "$VIDEO" \
  -vf "select=eq(n\\,$FRAME)" -frames:v 1 \
  /tmp/guava_four_panel_observed.png

FRAME=95
ffmpeg -y -v error -i "$VIDEO" \
  -vf "select=eq(n\\,$FRAME)" -frames:v 1 \
  /tmp/guava_four_panel_discarded.png
```

On an observed frame, panels 2, 3, and 4 must show the same GUAVA pose. On a
discarded frame, panel 2 must show the reason-specific placeholder while panels
3 and 4 remain populated.

---

## 10. Cancellation and Resume Behavior

Both tools use atomic output names:

```text
<final_stem>.saving.<pid>.<uuid>.mp4
```

The temporary file is renamed to the final name only after the writer closes
successfully. JSON and manifest files use the same temporary-suffix strategy.

If a job is canceled:

1. already completed final MP4s remain valid;
2. an in-progress file may remain with `.saving.` in its name;
3. rerunning the same command without `--overwrite` skips completed finals and
   renders missing outputs;
4. stale `.saving.` files can be removed after confirming that no renderer or
   compositor process is active.

Check before cleanup:

```bash
pgrep -af 'render_guava_compact_motion|compose_guava_motion_comparison'
find "$RENDER_ROOT" -type f -name '*.saving*' -print
```

Only when no matching process is active:

```bash
find "$RENDER_ROOT" -type f -name '*.saving*' -delete
```

Do not delete finalized MP4s when resuming. The skip-existing behavior is what
makes continuation inexpensive.

---

## 11. Troubleshooting

### `KeyError: ... does not contain <motion_key>`

Inspect the fit keys and choose the correct pair:

```bash
"$GUAVA_PYTHON" -c \
  'import numpy as np,sys; z=np.load(sys.argv[1],allow_pickle=False); print(z.files)' \
  "$(find "$FITS" -maxdepth 1 -name '*.npz' | sort | head -1)"
```

Update both `METHOD_*_KEY` and the matching compositor suffix `METHOD_*`.

### Output sequence name still includes an experiment suffix

Sequence names are recovered from `config_json.sequence` when present;
otherwise known fit suffixes are removed. Add a new suffix to `FIT_SUFFIXES` in
both GUAVA tools if a new experiment naming convention is introduced.

### `Frame mismatch`

At least one original, method-A, or method-B video does not have the fit's
native frame count. Common causes are a `--max_frames` smoke output in the full
render directory, a fit paired with the wrong source completion, or a stale
method render. Remove the affected output or rerun that stage with
`--overwrite` after fixing the input.

### Missing original source video or frame trace

The lookup chain is:

```text
fit[source_completion]
  -> completion[source_frame_trace]
  -> frame_trace[source_video]
```

Repair the stored source paths or regenerate the completion metadata. Do not
guess the source video from the filename when authoritative metadata exists.

### GPU appears idle for two to three minutes

GUAVA loads DINO, FLAME, SMPL-X, and the avatar checkpoint before frame
progress begins. This startup delay is normal. Check `nvidia-smi` and the
process rather than restarting immediately.

### Common warnings

These warnings were non-fatal in the reproduced runs:

```text
Using generic FLAME model
xFormers is not available
pynvml package is deprecated
torch.sparse.SparseTensor ... is deprecated
```

Treat a nonzero process exit, missing final files, frame mismatch, or FFmpeg
decode error as a real failure.

### Existing MP4 but missing per-video JSON

This should not occur with atomic saves. If it does, rerun the compositor with
`--overwrite` for a consistent MP4/JSON pair.

---

## 12. Reproduced Output Locations

Mask-aware v1:

```text
/media/cvpr/haomian/SignMotionRefinement/artifacts/visualizations/guava_mask_aware_meta_pilot/guava_renders/women001/four_panel_comparisons
```

Improved C2/FK-temporal:

```text
/media/cvpr/haomian/SignMotionRefinement/artifacts/visualizations/guava_mask_aware_meta_c2_fk_jerk_pilot/guava_renders/women001/four_panel_comparisons
```

Linear versus direct SIREN:

```text
/media/cvpr/haomian/SignMotionRefinement/artifacts/visualizations/guava_linear_siren_jerk_compare/guava_renders/women001/four_panel_comparisons
```

Retained-GUAVA-only rejected diagnostic:

```text
/media/cvpr/haomian/SignMotionRefinement/artifacts/visualizations/guava_self_only_meta_c2_fk_jerk_soke_free_diagnostic_pilot/guava_renders/women001/four_panel_comparisons
```

Each reproduced 12-clip mask-aware batch has:

```text
12 comparison MP4s
2455 total native frames
1589 observed frames
866 discarded/filled frames
2048x628 final resolution
exact original-video rational FPS
```

On the current machine, a 24-video GUAVA method batch takes about four minutes,
including roughly two to three minutes of model startup. Four-panel composition
takes about 80 to 90 seconds. Times depend on GPU and storage load.

---

## 13. Short Recall Checklist

1. Work from `/media/cvpr/haomian/GUAVA` with the GUAVA Python environment.
2. Select a preset and export all path, key, suffix, label, and output-tag
   variables.
3. Run the preflight check; never proceed with frame or observed-anchor
   mismatches.
4. Optionally render a three-frame smoke test outside the final output root.
5. Render both full dense methods on GPU.
6. Confirm 24 method MP4s and no `.saving.` files.
7. Compose the 12 four-panel videos.
8. Validate manifest totals, exact frame counts, rational FPS, 2048x628
   resolution, and full decoding.
9. Inspect at least one observed frame and one discarded frame.
10. Keep `comparison_manifest.json` with the videos; it is the provenance and
    timing record for the batch.
