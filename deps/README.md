# External model assets

This directory intentionally does not duplicate licensed model weights.

The fitting/evaluation runtime needs an SMPL-X model directory. Set:

```bash
export SMR_SMPLX_MODEL_DIR=/path/to/smpl_models
```

The current pipeline reads `assets/blank_text_tokens.npz`, so it does not need
to load FLAN-T5. Legacy runs can regenerate that embedding by setting:

```bash
export SMR_FLAN_T5_DIR=/path/to/flan-t5-base
```

On the machine where this extraction was created, the defaults are:

```text
/media/cvpr/haomian/SOKE/deps/smpl_models
/media/cvpr/haomian/SOKE/deps/flan-t5-base
```
