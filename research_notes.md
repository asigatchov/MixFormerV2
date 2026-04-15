# Download Models
According to the upstream project, download the model weights below. These
two models were mainly used for export:
https://box.nju.edu.cn/d/aba770262d984b1594d2/files/?p=%2Fmixformerv2_small.pth.tar
The config file for `mixformerv2_small.pth.tar` is `experiments/mixformer2_vit_online/224_depth4_mlp1_score.yaml`.
https://box.nju.edu.cn/d/aba770262d984b1594d2/files/?p=%2Fmixformerv2_base.pth.tar
The config file for `mixformerv2_base.pth.tar` is `experiments/mixformer2_vit_online/288_depth8_score.yaml`.

# Environment Setup
The original project uses Python 3.6 and Torch 1.7, which is too old.
This setup uses Python 3.10, and `install_requirements.sh` was adjusted for it.
Note that some `apt` packages may still need to be installed manually with `sudo`.
Some dependencies are only available from the official `pip` index, while others
need to be installed through `conda`.
```
conda create -n mixformer2 python=3.10.18
conda activate mixformer2
bash install_requirements.sh
```
# Export ONNX Models
The opset version should be lower than 19 because `rknn-toolkit2` does not
support opset 19 or newer. Here it is set to 11, and the export also applies
model simplification.
```
python export_onnx_4.py \
    --checkpoint ./models/mixformerv2_small.pth.tar \
    --output_dir out_4 \
    --template_size 112 \
    --search_size 224 \
    --tracker_name mixformer2_vit_online \
    --config_name 224_depth4_mlp1_score \
    --opset_version 11
```
# Test The ONNX Model
The default parameters are already suitable.
```
python onnx_inference.py 
```
# Attempt To Split The Model
The split-model attempt did not succeed. A likely reason is that the attention
mechanism breaks after splitting, so inference results are incorrect.
One important detail is that MixFormerV2 hardcodes the head to run on CUDA, so
`lib/models/mixformer2_vit/head.py` must be adjusted when exporting split models.
```
self.indice = torch.arange(0, feat_sz).unsqueeze(0).cuda() * stride # (1, feat_sz)
# self.indice = torch.arange(0, feat_sz).unsqueeze(0) * stride # (1, feat_sz)
```

```
python export_onnx_4.py \
    --checkpoint ./models/mixformerv2_small.pth.tar \
    --output_dir out_4 \
    --template_size 112 \
    --search_size 224 \
    --tracker_name mixformer2_vit_online \
    --config_name 224_depth4_mlp1_score \
    --opset_version 11
```
```
python onnx_inference_4.py \
    --template_encoder_path out_4/template_encoder.onnx \
    --online_template_encoder_path out_4/online_template_encoder.onnx \
    --search_encoder_path out_4/search_encoder.onnx \
    --tracking_head_path out_4/tracking_head.onnx \
    --video videos/track-car4.mp4 \
    --embed_dim 768
```
