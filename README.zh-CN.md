# MixFormerV2 (Extended README)

- [MixFormerV2 (Extended README)](#mixformerv2-extended-readme)
  - [Exporting ONNX Models](#exporting-onnx-models)
  - [Video Demo Execution And Visualization](#video-demo-execution-and-visualization)

## Exporting ONNX Models

Script: `export_onnx.py`. Supported features:

- Automatically reads template and search sizes from the experiment config
- Static or dynamic batch export (`--batch_mode static|dynamic`)
- Automatic post-export validation plus ONNX Runtime inference verification

Minimal example:

```bash
python export_onnx.py \
  --checkpoint ./models/MixFormerV2/mixformerv2_small.pth.tar \
  --config_name 224_depth4_mlp1_score \
  --output ./models/mixformerv2_online_small.onnx \
  --batch_mode static \
  --batch_size 1 \
  --opset_version 17
```

Common arguments:

- `--tracker_name`: maps to `lib/config/<tracker_name>/config.py`, default is `mixformer2_vit_online`
- `--config_name`: experiment config name from `experiments/<tracker_name>/<config_name>.yaml`
- `--config_path`: explicitly provide a YAML file and override `--config_name`
- `--template_size / --search_size / --online_template_size`: auto-resolved from config if omitted
- `--batch_mode dynamic`: export with a dynamic batch dimension; make sure your runtime supports it

Post-export validation:

1. The script runs `onnx.checker.check_model` automatically.
2. It performs one ONNX Runtime forward pass and prints the output shapes.

Deployment notes:

- If the backend is TensorRT and some operators are unsupported, try lowering `--opset_version` to `16` or upgrading TRT.
- You can also force a fixed input size with `--search_size` to reduce backend optimization variability.

---

## Video Demo Execution And Visualization

Script: `tracking/video_demo.py`

Features:

- Track a single video with either an initial box or manual interaction
- Adjust online template update frequency, search-area scale, attention visualization (`vis_attn`), and more
- Save the tracking result video (`--save_video`) and YOLO-format labels (`--save_yolo`)

![Basketball tracking example](resources/track_basketball.gif)

Example (a `video_demo` VSCode launch config is already provided):

```bash
python tracking/video_demo.py \
  mixformer2_vit_online \
  224_depth4_mlp1_score \
  /path/to/video.mp4 \
  --params__model models/MixFormerV2/mixformerv2_small.pth.tar \
  --params__search_area_scale 5.0 \
  --zoomin \
  --save_video
```

Parameter forwarding: any argument in the form `--params__<key>` is parsed into
`tracker_params["<key>"]` and then passed into the `Tracker` instance. For example:

- `--params__model`: model checkpoint path
- `--params__search_area_scale`: search-area scale factor
- `--params__update_interval`: online update interval
- `--params__vis_attn 1`: visualize attention maps (requires tracker support)

Optional initial bbox:

```bash
--optional_box X Y W H
```

Output:

- Results may be generated under `debug/` or the current run directory, depending on the implementation.
- `--save_video` produces a video file with the predicted boxes overlaid.
