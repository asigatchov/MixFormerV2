# MixFormerV2
The official implementation of the NeurIPS 2023 paper: [**MixFormerV2: Efficient Fully Transformer Tracking**](https://arxiv.org/abs/2305.15896).

## Model Framework
![model](tracking/model.png)

## Distillation Training Pipeline
![training](tracking/training.png)


## News

- **[May 31, 2023]** We released two versions of the pretrained model, which can be accessed on [Google Driver](https://drive.google.com/drive/folders/1soQMZyvIcY7YrYrGdk6MCstTPlMXNd30?usp=sharing) and [NJU Box](https://box.nju.edu.cn/d/aba770262d984b1594d2/).

- **[May 26, 2023]** Code is available now!


## Highlights

### :sparkles: Efficient Fully Transformer Tracking Framework

MixFormerV2 is a well unified fully transformer tracking model, without any dense convolutional operation and complex score prediction module. We propose four key prediction tokens to capture the correlation between target template and search area.

### :sparkles: A New Distillation-based Model Reduction Paradigm

To further improve efficiency, we present a new distillation paradigm for tracking model, including dense-to-sparse stage and deep-to-shallow stage.

### :sparkles: Strong Performance and Fast Inference Speed

MixFormerV2 works well for different benchmarks and can achieve **70.6%** AUC on LaSOT and **57.4%** AUC on TNL2k, while keeping 165fps on GPU. To our best knowledge, MixFormerV2-S is the **first** transformer-based one-stream tracker which achieves real-time running on CPU.


## Install the environment
Use `uv` with Python 3.12.
```bash
uv python pin 3.12
bash install_requirements.sh
```

After installation, run commands directly with `uv run`, for example:
```bash
uv run python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
```
   ${MixFormerV2_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- train2017
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
```

## Set project paths
Run the following command to set paths for this project
```
uv run python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Exporting an ONNX Model

Script: `export_onnx.py`. It supports:

- Auto-reading template / search sizes from an experiment config
- Static or dynamic batch (`--batch_mode static|dynamic`)
- Automatic post-export validation + ONNX Runtime inference check

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

Common argument notes:

- `--tracker_name`: Corresponds to `lib/config/<tracker_name>/config.py`, default: `mixformer2_vit_online`
- `--config_name`: Experiment name found at `experiments/<tracker_name>/<config_name>.yaml`
- `--config_path`: Explicit YAML path (overrides `--config_name`)
- `--template_size / --search_size / --online_template_size`: If not set, they are parsed automatically from the config
- `--batch_mode dynamic`: Export with dynamic batch dimension (ensure runtime/backend support)

Post-export checks:

1. The script calls `onnx.checker.check_model`
2. Runs a forward pass with onnxruntime and prints output shapes

Deployment tips:

- If TensorRT reports unsupported ops, try lowering `--opset_version 16` or upgrading TRT.
- You can add `--search_size` to force a fixed size and reduce backend optimization uncertainty.

---

## Running the Video Demo & Visualization

Script: `tracking/video_demo.py`

Features:

- Track a single video with a provided initial box or interactive selection (this script example uses arguments)
- Adjustable online template update interval, search area scale, attention visualization (`vis_attn`), etc.
- Supports saving the tracking result video (`--save_video`) and YOLO annotation format (`--save_yolo`)

![track demo](resources/track_basketball.gif)

Example (a VSCode launch config named `video_demo` may already exist):

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

Argument passing mechanism: any flag of the form `--params__<key>` is parsed into `tracker_params["<key>"]` and passed to the `Tracker` instance. Examples:

- `--params__model`: Model weight path
- `--params__search_area_scale`: Search area scale factor
- `--params__update_interval`: Online update interval
- `--params__vis_attn 1`: Visualize attention maps (tracker must support it)

Optional: Initial bbox

```bash
--optional_box X Y W H
```

Outputs:

 - Results are written to `debug/` or the working directory (depends on implementation details).
 - With `--save_video`, a video file with overlaid predicted boxes is generated.

## Train MixFormerV2

Training with multiple GPUs using DDP. 
You can follow instructions (in Chinese now) in [training.md](tutorials/training_zh.md).
Example scripts can be found in `tracking/train_mixformer.sh`.

``` bash
uv run bash tracking/train_mixformer.sh
```

## Test and evaluate MixFormerV2 on benchmarks
- LaSOT/GOT10k-test/TrackingNet/OTB100/UAV123/TNL2k. More details of test settings can be found in `tracking/test_mixformer.sh`.

``` bash
uv run bash tracking/test_mixformer.sh

```


## TODO
- [x] Progressive eliminating version of training.
- [ ] Fast version of test forwarding.

## Contant
Tianhui Song: 191098194@smail.nju.edu.cn

Yutao Cui: cuiyutao@smail.nju.edu.cn 


## Citiation
``` bibtex
@misc{mixformerv2,
      title={MixFormerV2: Efficient Fully Transformer Tracking}, 
      author={Yutao Cui and Tianhui Song and Gangshan Wu and Limin Wang},
      year={2023},
      eprint={2305.15896},
      archivePrefix={arXiv}
}
```
