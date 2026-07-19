# GUAVA photorealistic rendering integration

These two scripts generated the photorealistic four-panel results archived
under `artifacts/visualizations/*/guava_renders`.

`compose_guava_motion_comparison.py` is self-contained. The renderer imports
the separately installed GUAVA repository and its model assets. Point it to
that repository with:

```bash
export GUAVA_ROOT=/media/cvpr/haomian/GUAVA
```

The scripts were copied from:

```text
/media/cvpr/haomian/GUAVA/tools/render_guava_compact_motion.py
/media/cvpr/haomian/GUAVA/tools/compose_guava_motion_comparison.py
```

The renderer's repository lookup and default model path were refactored to use
`GUAVA_ROOT`; its motion/rendering behavior was otherwise retained.
