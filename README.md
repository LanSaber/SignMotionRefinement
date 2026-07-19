# SignMotionRefinement

Standalone extraction of the GUAVA SMPL-X missing-frame completion work that
was developed inside SOKE. The source SOKE tree was not modified or deleted.

The current pipeline is:

```text
sparse GUAVA poses + observed mask
  -> per-joint SO(3) SLERP scaffold
  -> shared meta-implicit residual field
  -> bounded, C2-tapered SO(3) corrections inside bracketed gaps
  -> exact restoration of every observed GUAVA frame
  -> complete SMPL-X sequence
```

The retained-GUAVA-only diagnostic checkpoint did **not** pass the deployment
gate. The official selected strength is `alpha=0`, so the production-safe
result remains pure SO(3) SLERP. The `alpha=1` checkpoint is archived for
analysis and visualization, not as the accepted fitter.

## Canonical current artifacts

| Role | Path |
|---|---|
| Deployable fail-safe checkpoint (`alpha=0`) | `artifacts/experiments/guava_self_only_meta_c2_fk_jerk_stable_so3_trunk5e5/checkpoints/best.pt` |
| Rejected analysis checkpoint (`alpha=1`) | `artifacts/experiments/guava_self_only_meta_c2_fk_jerk_stable_so3_trunk5e5/checkpoints/best_safe_diagnostic.pt` |
| Diagnostic pilot metrics | `artifacts/visualizations/guava_self_only_meta_c2_fk_jerk_soke_free_diagnostic_pilot/evaluation_summary.json` |
| Four-panel videos and jerk charts | `artifacts/visualizations/guava_self_only_meta_c2_fk_jerk_soke_free_diagnostic_pilot/render_guava_only_diagnostic_four_panel/` |
| Migration audit | `MIGRATION_MANIFEST.json` |

## Project layout

| Path | Contents |
|---|---|
| `src/sign_motion_refinement/` | Standalone fitting, evaluation, and rendering package |
| `configs/` | Four GUAVA training configurations with new artifact paths |
| `tests/` | Focused completion, mask, temporal, and retained-GUAVA-only tests |
| `docs/` | Pipeline explanation, visualization runbook, and migration record |
| `integrations/guava/` | Photorealistic GUAVA renderer and four-panel compositor used by archived renders |
| `assets/blank_text_tokens.npz` | Archived blank FLAN-T5 embedding used by the current pipeline |
| `artifacts/experiments/` | GUAVA experiment lineage and required parent checkpoints |
| `artifacts/visualizations/` | All GUAVA fitting/evaluation/render outputs |

## Environment

The existing SOKE Python environment can run the extraction directly, without
installing anything into that environment:

```bash
cd /media/cvpr/haomian/SignMotionRefinement
export SMR_PYTHON=/media/cvpr/haomian/python_envs/soke/bin/python
$SMR_PYTHON smr.py --help
```

`pyproject.toml` also exposes normal `smr-*` console scripts after package
installation. The source-checkout dispatcher above is preferable on this
network-mounted workspace because it avoids editable-install metadata writes.

SMPL-X model files and, only when the cached blank embedding is unavailable,
FLAN-T5 are external licensed dependencies. Their defaults point to the
existing machine assets. Override them without editing a config:

```bash
export SMR_SMPLX_MODEL_DIR=/path/to/smpl_models
export SMR_FLAN_T5_DIR=/path/to/flan-t5-base
```

Input GUAVA/SOKE datasets remain in `/media/cvpr/haomian/data`; they were not
duplicated as project outputs.

## Main commands

Create the exact SO(3)-SLERP completion dataset:

```bash
$SMR_PYTHON smr.py complete \
  --tracked_dir /media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx_GUAVA/guava_tracked \
  --out_dir /media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx_GUAVA/guava_completed_flow
```

Start a new retained-GUAVA-only run. Use a new output directory; archived runs
are deliberately not overwritten:

```bash
$SMR_PYTHON smr.py train-guava-only \
  --config configs/guava_self_only_meta_c2_fk_jerk.yaml \
  --out_dir artifacts/experiments/runs/guava_only_new
```

Evaluate a checkpoint on selected completion files:

```bash
$SMR_PYTHON smr.py evaluate-meta \
  --input /path/to/completions/train/CLIP_A.npz /path/to/completions/train/CLIP_B.npz \
  --checkpoint artifacts/experiments/guava_self_only_meta_c2_fk_jerk_stable_so3_trunk5e5/checkpoints/best_safe_diagnostic.pt \
  --alpha 1 \
  --out_dir artifacts/visualizations/runs/guava_only_diagnostic
```

Render the four-panel comparison with animated jerk curves:

```bash
$SMR_PYTHON smr.py visualize-meta-jerk \
  --input_dir artifacts/visualizations/guava_self_only_meta_c2_fk_jerk_soke_free_diagnostic_pilot/fits \
  --evaluation_summary artifacts/visualizations/guava_self_only_meta_c2_fk_jerk_soke_free_diagnostic_pilot/evaluation_summary.json \
  --out_dir artifacts/visualizations/runs/four_panel
```

See [the pipeline document](docs/guava_mask_aware_meta_implicit_finetuning.md)
for the full objective and results, and
[the migration record](docs/migration.md) for exact scope and provenance.
