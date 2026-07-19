# Migration from SOKE

Date: 2026-07-19

## Outcome

The GUAVA frame-completion work was extracted additively from:

```text
/media/cvpr/haomian/SOKE
```

into:

```text
/media/cvpr/haomian/SignMotionRefinement
```

Nothing was removed from or rewritten in the source SOKE repository. The
source fitting tree was untracked in SOKE at extraction time, so this record
and checksum verification are the migration audit trail rather than a Git
commit reference.

## Scope

Included:

- irregular-timeline GUAVA ingestion and per-joint SO(3) SLERP completion;
- direct linear/SIREN comparison used during early fitting experiments;
- bounded meta-implicit residual-field inference;
- dense-SOKE pseudo-target mask-aware fine-tuning history;
- C2 envelope and reference-free FK jerk/boundary regularization;
- retained-GUAVA-only masked self-supervision and deployment gating;
- missing-frame/FK/jerk evaluation;
- software-mesh four-panel and animated jerk-curve visualization;
- all GUAVA experiment and visualization artifacts;
- the two upstream meta-implicit checkpoints required by historical GUAVA
  pilots;
- the four focused GUAVA tests and the relevant documentation.

Excluded:

- input datasets under `/media/cvpr/haomian/data`;
- licensed SMPL-X and FLAN-T5 assets;
- unrelated Phoenix, retrieval, trajectory-field, flow-generation, and
  text-to-sign code or experiments;
- GUAVA's full photorealistic renderer repository and model weights. The two
  fitting-specific integration scripts are retained under
  `integrations/guava/` and still load the external GUAVA runtime.

## Code mapping

| Original SOKE path | Standalone path |
|---|---|
| `NIAF/continuous_sign_field/guava_gap.py` | `src/sign_motion_refinement/pipeline/gap.py` |
| `NIAF/continuous_sign_field/guava_self_supervision.py` | `src/sign_motion_refinement/pipeline/self_supervision.py` |
| `NIAF/continuous_sign_field/guava_temporal.py` | `src/sign_motion_refinement/pipeline/temporal.py` |
| `NIAF/continuous_sign_field/models/meta_implicit.py` | `src/sign_motion_refinement/models/meta_implicit.py` |
| `NIAF/continuous_sign_field/scripts/complete_guava_missing_frames.py` | `src/sign_motion_refinement/cli/complete.py` |
| `NIAF/continuous_sign_field/scripts/train_guava_mask_aware_meta.py` | `src/sign_motion_refinement/cli/train_mask_aware.py` |
| `NIAF/continuous_sign_field/scripts/train_guava_self_only_meta.py` | `src/sign_motion_refinement/cli/train_guava_only.py` |
| `NIAF/continuous_sign_field/scripts/evaluate_guava_mask_aware_meta.py` | `src/sign_motion_refinement/cli/evaluate_meta.py` |
| `NIAF/continuous_sign_field/scripts/evaluate_guava_completion_jerk.py` | `src/sign_motion_refinement/cli/evaluate_jerk.py` |
| `NIAF/continuous_sign_field/scripts/run_guava_bounded_meta_pilot.py` | `src/sign_motion_refinement/cli/run_bounded_pilot.py` |
| GUAVA visualization scripts | `src/sign_motion_refinement/visualization/` |
| `flow/smplx_features.py` | `src/sign_motion_refinement/features.py` |
| required SOKE dataset-reader functions | `src/sign_motion_refinement/data/guava.py` |
| required oracle rotation/FK helpers | `src/sign_motion_refinement/geometry/` |

The refactored Python package has no imports from `flow` or `NIAF`. Shared
factory functions and GUAVA readers were reduced to the functionality used by
this pipeline rather than copying unrelated training stacks.

## Artifact mapping

The copied binary artifacts were not edited. This preserves checkpoint state
dicts, NPZ contents, historical absolute provenance fields, and result JSON.

| Source | Destination | Directories | Files | Bytes |
|---|---|---:|---:|---:|
| `SOKE/experiments/NIAF/continuous_sign_field/{guava*, two parent runs}` | `artifacts/experiments/` | 13 | 78 | 306,712,077 |
| `SOKE/visualize/NIAF/continuous_sign_field/guava*` | `artifacts/visualizations/` | 12 | 671 | 496,253,530 |

Every one of the 25 copied artifact directories was checked with recursive
file checksums (`rsync -rcn --delete`) against its source. The verification
reported no missing, extra, or changed artifact file.

## Functional verification

- 33 package modules, two GUAVA integration scripts, the source launcher, and
  four tests (40 Python files total) parsed and byte-compiled successfully.
- 19 core, CLI, and visualization modules imported in the existing SOKE
  Python environment without importing SOKE code.
- All 16 focused GUAVA numerical tests passed.
- Project-aware config expansion resolves archived checkpoint, fit, output,
  and blank-text-cache paths under this project.
- All 14 archived `best*.pt` checkpoints instantiated the refactored model and
  loaded their state dictionaries with `strict=True` (including both parent
  architectures and the current safe diagnostic).
- A 199-frame archived pilot fit ran through the refactored diagnostic model:
  all outputs were finite, alpha 0 matched SLERP within `1.20e-7`, and all
  observed frames were restored exactly at alpha 1.

## External paths

The dataset paths in the four configs remain absolute because the data was not
requested as part of the project extraction. Model assets are configurable:

```bash
export SMR_SMPLX_MODEL_DIR=/path/to/smpl_models
export SMR_FLAN_T5_DIR=/path/to/flan-t5-base
```

The archived blank-text token file normally makes FLAN-T5 unnecessary for the
current retained-GUAVA-only training and evaluation path.
