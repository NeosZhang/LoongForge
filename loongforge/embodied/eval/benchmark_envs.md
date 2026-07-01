# Benchmark Environments

This document records the benchmark runtime environments used by the LoongForge-VLA eval module under `loongforge/embodied/eval`.

## Scope

The benchmark side uses conda environments for isolation. The training/inference base image is outside the scope of this document, so this file only lists benchmark client environments and their key dependency versions.

User-editable benchmark configs live under `examples/embodied/pi05/eval/configs`.

Current benchmark environments:

```text
LIBERO      /workspace/miniconda3/envs/libero
CALVIN      /workspace/miniconda3/envs/calvin
SimplerEnv  /workspace/miniconda3/envs/simplerenv
RoboTwin    /workspace/miniconda3/envs/robotwin
```

No benchmark currently uses uv for environment isolation.

## Tool Version

```text
conda 26.3.2
```

## LIBERO

Runtime Python:

```text
/workspace/miniconda3/envs/libero/bin/python
```

Selected dependency versions:

```text
python          3.8.13
torch           2.1.2
numpy           1.24.4
libero          0.1.0 dev
robosuite       1.4.0
mujoco          3.2.3
gym             0.25.2
imageio         2.35.1
imageio-ffmpeg  0.5.1
opencv-python   4.6.0.66
websockets      13.1
msgpack         1.1.1
pyyaml          6.0.3
```

Used by:

```text
examples/embodied/pi05/eval/configs/libero/*.yaml
```

## CALVIN

Runtime Python:

```text
/workspace/miniconda3/envs/calvin/bin/python
```

CALVIN evaluation uses the original-format CALVIN validation dataset and config assets, including `validation/.hydra/merged_config.yaml`, `calvin_models/conf`, and `eval_sequences.json`. LeRobot-format CALVIN datasets are useful for training/statistics but are not sufficient by themselves for official online long-horizon rollout.

Current internal CALVIN status:

```text
benchmark env:       /workspace/miniconda3/envs/calvin
repo/config assets:  /ssd1/sunyuehang/calvin
validation dataset:  /ssd1/sunyuehang/calvin_debug_dataset
smoke status:        pass; 1 sequence, first subtask capped at 30 steps, trace and summary written
model score status:  smoke/debug only with current LIBERO-domain pi05 checkpoint
```

Used by:

```text
examples/embodied/pi05/eval/configs/calvin/*.yaml
```

## SimplerEnv

Runtime Python:

```text
/workspace/miniconda3/envs/simplerenv/bin/python
```

Selected dependency versions:

```text
python                 3.10.20
numpy                  1.24.4
mani-skill2-real2sim   0.5.3
sapien                 2.2.2
gymnasium              0.29.1
imageio                2.37.3
imageio-ffmpeg         0.6.0
opencv-python          4.13.0.92
websockets             16.0
msgpack                1.1.2
pyyaml                 6.0.3
```

Runtime variables used by the current eval command:

```text
LD_LIBRARY_PATH=/path/to/nvidia_lib:/usr/lib64:$LD_LIBRARY_PATH
VK_ICD_FILENAMES=/path/to/nvidia_icd.json
XDG_RUNTIME_DIR=/tmp/runtime-<uid>
```

Internal SimplerEnv configs use:

```text
LD_LIBRARY_PATH=/ssd1/opt/nvidia_lib:/usr/lib64:$LD_LIBRARY_PATH
VK_ICD_FILENAMES=/ssd1/opt/nvidia_lib/10_nvidia.json
```

SAPIEN/svulkan2 needs these variables before Python imports the renderer. The SimplerEnv runner therefore prepares `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, and `XDG_RUNTIME_DIR`, then re-execs the benchmark Python process once before constructing the environment. Setting `LD_LIBRARY_PATH` only after Python has already started is not sufficient for this renderer stack.

Current SimplerEnv status:

```text
Bridge tasks configured: eggplant, carrot_on_plate, stack_cube, spoon_on_towel
Runtime smoke:         pass
Fixed-action sanity:   pass; small meter-scale actions move the WidowX controller
Model score status:    smoke/debug only until Bridge/WidowX pi05 checkpoint and dataset_statistics.json are available
```

The current internal pi05 SimplerEnv configs use LIBERO-domain weights or no Bridge/WidowX `dataset_statistics_path`. They validate benchmark plumbing, trace writing, and GIF output, but they are not credible SimplerEnv benchmark scores.

Used by:

```text
examples/embodied/pi05/eval/configs/simplerenv/*.yaml
```

## RoboTwin

Runtime Python:

```text
/workspace/miniconda3/envs/robotwin/bin/python
```

Selected dependency versions:

```text
python            3.10.20
torch             2.4.1
numpy             1.26.4
sapien            3.0.0b1
imageio           2.34.2
imageio-ffmpeg    0.6.0
opencv-python     4.11.0.86
websockets        16.0
msgpack           1.1.2
msgpack-numpy     0.4.8
pyyaml            6.0.3
```

Video logging dependency:

```text
/workspace/miniconda3/envs/robotwin/bin/ffmpeg
ffmpeg version 7.0.2-static
```

The `ffmpeg` executable is provided by the installed `imageio-ffmpeg` package and linked into the `robotwin` env `bin` directory so RoboTwin official video logging can launch `ffmpeg` directly.

Used by:

```text
examples/embodied/pi05/eval/configs/robotwin/*.yaml
```

## Config-to-Environment Mapping

```text
examples/embodied/pi05/eval/configs/libero/*.yaml
  benchmark env: /workspace/miniconda3/envs/libero
  runtime:       /workspace/miniconda3/envs/libero/bin/python

examples/embodied/pi05/eval/configs/calvin/*.yaml
  benchmark env: /workspace/miniconda3/envs/calvin
  runtime:       /workspace/miniconda3/envs/calvin/bin/python

examples/embodied/pi05/eval/configs/simplerenv/*.yaml
  benchmark env: /workspace/miniconda3/envs/simplerenv
  runtime:       /workspace/miniconda3/envs/simplerenv/bin/python

examples/embodied/pi05/eval/configs/robotwin/*.yaml
  benchmark env: /workspace/miniconda3/envs/robotwin
  runtime:       /workspace/miniconda3/envs/robotwin/bin/python
```

## Re-check Commands

Use these commands to refresh version information after environment changes:

```bash
/workspace/miniconda3/bin/conda list -n libero
/workspace/miniconda3/bin/conda list -n calvin
/workspace/miniconda3/bin/conda list -n simplerenv
/workspace/miniconda3/bin/conda list -n robotwin
```
