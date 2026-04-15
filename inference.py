#!/usr/bin/env python3
import argparse
import importlib
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lib.config.mixformer2_vit_online.config import update_new_config_from_file
from lib.models.mixformer2_vit import build_mixformer2_vit_online
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


MODEL_PRESETS = {
    "small": {
        "checkpoint": ROOT_DIR / "mixformerv2_small.pth.tar",
        "config": ROOT_DIR / "experiments" / "mixformer2_vit_online" / "224_depth4_mlp1_score.yaml",
    },
    "base": {
        "checkpoint": ROOT_DIR / "mixformerv2_base.pth.tar",
        "config": ROOT_DIR / "experiments" / "mixformer2_vit_online" / "288_depth8_score.yaml",
    },
}
DEFAULT_MODEL = "small"
WINDOW_NAME = "MixFormerV2"


def parse_args():
    parser = argparse.ArgumentParser(description="MixFormerV2 video inference demo")
    parser.add_argument("--video_path", required=True, help="path to input video")
    parser.add_argument(
        "--model",
        choices=["auto", "small", "base"],
        default="auto",
        help="model preset: auto resolves from checkpoint name, default fallback is small",
    )
    parser.add_argument("--checkpoint", default="", help="path to model checkpoint")
    parser.add_argument("--config", default="", help="path to yaml config")
    parser.add_argument(
        "--output_path",
        default="",
        help="path to output demo video, default is <video>_mixformerv2.mp4",
    )
    parser.add_argument("--init_rect", default="", type=str, help="initial bbox as x,y,w,h")
    parser.add_argument("--max_frames", default=0, type=int, help="process at most this many frames, 0 means full video")
    parser.add_argument("--cpu", action="store_true", help="force CPU mode")
    parser.add_argument("--headless", action="store_true", help="run without OpenCV UI")
    parser.add_argument("--search_area_scale", default=None, type=float, help="override search factor")
    parser.add_argument("--online_size", default=None, type=int, help="override online template size")
    parser.add_argument("--update_interval", default=None, type=int, help="override template update interval")
    return parser.parse_args()


def detect_model_from_checkpoint(checkpoint_path):
    checkpoint_name = Path(checkpoint_path).name.lower()
    for model_name in MODEL_PRESETS:
        if model_name in checkpoint_name:
            return model_name
    return None


def resolve_model_assets(args):
    explicit_model = None if args.model == "auto" else args.model
    checkpoint_model = detect_model_from_checkpoint(args.checkpoint) if args.checkpoint else None

    if explicit_model and checkpoint_model and explicit_model != checkpoint_model:
        raise ValueError(
            f"--model={explicit_model} conflicts with checkpoint '{args.checkpoint}', "
            f"which looks like model '{checkpoint_model}'"
        )

    model_name = explicit_model or checkpoint_model or DEFAULT_MODEL
    preset = MODEL_PRESETS[model_name]

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else preset["checkpoint"]
    config_path = Path(args.config) if args.config else preset["config"]
    return model_name, checkpoint_path, config_path


def parse_init_rect(value):
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 4:
        raise ValueError("--init_rect must be x,y,w,h")
    rect = tuple(float(item) for item in parts)
    if rect[2] <= 0 or rect[3] <= 0:
        raise ValueError("--init_rect width and height must be positive")
    return rect


def resolve_output_path(video_path, output_path):
    if output_path:
        return Path(output_path)
    video = Path(video_path)
    return video.with_name(f"{video.stem}_mixformerv2.mp4")


def create_writer(output_path, fps, frame_size):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)


def ensure_display_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)


def close_display_window():
    try:
        cv2.destroyWindow(WINDOW_NAME)
    except cv2.error:
        pass


def seek_to_frame(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return frame


def draw_browse_overlay(frame, frame_index, paused):
    vis = frame.copy()
    status = "pause" if paused else "play"
    cv2.putText(vis, f"frame={frame_index} [{status}]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(vis, "Enter: select ROI  Space: play/pause", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, "d:+1  a:-1  w:+15  s:-15", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def browse_for_init_rect(capture, first_frame):
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_index = 0
    frame = first_frame
    paused = True

    while True:
        cv2.imshow(WINDOW_NAME, draw_browse_overlay(frame, frame_index, paused))
        key = cv2.waitKey(0 if paused else 30) & 0xFF

        if key in (13, 10):
            rect = cv2.selectROI(WINDOW_NAME, frame, False, False)
            if rect[2] > 0 and rect[3] > 0:
                return frame.copy(), frame_index, rect
            continue
        if key in (27, ord("q")):
            raise KeyboardInterrupt("Inference cancelled by user")
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("d"):
            next_index = frame_index + 1
        elif key == ord("a"):
            next_index = frame_index - 1
        elif key == ord("w"):
            next_index = frame_index + 15
        elif key == ord("s"):
            next_index = frame_index - 15
        elif not paused and key == 255:
            next_index = frame_index + 1
        else:
            continue

        if total_frames > 0:
            next_index = max(0, min(total_frames - 1, next_index))
        else:
            next_index = max(0, next_index)

        next_frame = seek_to_frame(capture, next_index)
        if next_frame is None:
            paused = True
            continue
        frame_index = next_index
        frame = next_frame


class Preprocessor:
    def __init__(self, device):
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, 3, 1, 1))
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, 3, 1, 1))

    def process(self, img_arr):
        img_tensor = torch.tensor(img_arr, device=self.device).float().permute((2, 0, 1)).unsqueeze(dim=0)
        return ((img_tensor / 255.0) - self.mean) / self.std


class MixFormerV2Runner:
    def __init__(self, cfg, checkpoint_path, device, search_area_scale=None, online_size=None, update_interval=None):
        self.cfg = cfg
        self.device = device
        self.network = build_mixformer2_vit_online(cfg, train=False)
        checkpoint = load_checkpoint_weights(checkpoint_path)
        self.network.load_state_dict(checkpoint["net"], strict=True)
        self.network.to(device)
        self.network.eval()

        self.preprocessor = Preprocessor(device)
        self.template_factor = cfg.TEST.TEMPLATE_FACTOR
        self.template_size = cfg.TEST.TEMPLATE_SIZE
        self.search_factor = search_area_scale if search_area_scale is not None else cfg.TEST.SEARCH_FACTOR
        self.search_size = cfg.TEST.SEARCH_SIZE
        self.update_interval = update_interval if update_interval is not None else self._default_from_cfg(cfg.TEST.UPDATE_INTERVALS, 0, 200)
        self.online_size = online_size if online_size is not None else self._default_from_cfg(cfg.TEST.ONLINE_SIZES, 0, 1)
        self.max_score_decay = 1.0
        self.state = None
        self.template = None
        self.online_template = None
        self.online_max_template = None
        self.online_forget_id = 0
        self.max_pred_score = -1.0
        self.frame_id = 0

    @staticmethod
    def _default_from_cfg(values, index, fallback):
        for _, value in values.items():
            if isinstance(value, (list, tuple)) and len(value) > index:
                return value[index]
        return fallback

    def initialize(self, image, init_bbox):
        z_patch_arr, _, _ = sample_target(image, init_bbox, self.template_factor, output_sz=self.template_size)
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.online_template = template
        if self.online_size > 1:
            with torch.no_grad():
                self.network.set_online(self.template, self.online_template)

        self.online_max_template = template
        self.max_pred_score = -1.0
        self.online_forget_id = 0
        self.state = list(init_bbox)
        self.frame_id = 0

    def track(self, image):
        height, width, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, _ = sample_target(image, self.state, self.search_factor, output_sz=self.search_size)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            out_dict = self.network(self.template, self.online_template, search, softmax=True, run_score_head=True)

        pred_boxes = out_dict["pred_boxes"].view(-1, 4)
        pred_score = out_dict["pred_scores"].view(1).sigmoid().item()
        pred_box = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()
        self.state = clip_box(self._map_box_back(pred_box, resize_factor), height, width, margin=10)

        self.max_pred_score *= self.max_score_decay
        if pred_score > 0.5 and pred_score > self.max_pred_score:
            z_patch_arr, _, _ = sample_target(image, self.state, self.template_factor, output_sz=self.template_size)
            self.online_max_template = self.preprocessor.process(z_patch_arr)
            self.max_pred_score = pred_score

        if self.frame_id % self.update_interval == 0:
            if self.online_size == 1:
                self.online_template = self.online_max_template
            elif self.online_template.shape[0] < self.online_size:
                self.online_template = torch.cat([self.online_template, self.online_max_template])
            else:
                self.online_template[self.online_forget_id:self.online_forget_id + 1] = self.online_max_template
                self.online_forget_id = (self.online_forget_id + 1) % self.online_size

            if self.online_size > 1:
                with torch.no_grad():
                    self.network.set_online(self.template, self.online_template)

            self.max_pred_score = -1.0
            self.online_max_template = self.template

        return {"target_bbox": self.state, "conf_score": pred_score}

    def _map_box_back(self, pred_box, resize_factor):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, width, height = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * width, cy_real - 0.5 * height, width, height]


def load_checkpoint_weights(checkpoint_path):
    environment_settings_cls = ensure_train_local_module()
    safe_types = []

    try:
        from torch.serialization import safe_globals
        from lib.train.admin.settings import Settings
        from lib.train.admin.stats import AverageMeter, StatValue

        safe_types = [AverageMeter, StatValue, Settings, environment_settings_cls]
    except Exception:
        safe_globals = None

    try:
        if safe_globals is None:
            raise RuntimeError("safe_globals is unavailable")
        with safe_globals(safe_types):
            return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def ensure_train_local_module():
    module_name = "lib.train.admin.local"
    try:
        module = importlib.import_module(module_name)
        return module.EnvironmentSettings
    except ModuleNotFoundError:
        module = types.ModuleType(module_name)

        class EnvironmentSettings:
            def __init__(self):
                self.workspace_dir = str(ROOT_DIR)
                self.tensorboard_dir = str(ROOT_DIR / "tensorboard")
                self.pretrained_networks = str(ROOT_DIR / "pretrained_networks")
                self.lasot_dir = ""
                self.got10k_dir = ""
                self.lasot_lmdb_dir = ""
                self.got10k_lmdb_dir = ""
                self.trackingnet_dir = ""
                self.trackingnet_lmdb_dir = ""
                self.coco_dir = ""
                self.coco_lmdb_dir = ""
                self.lvis_dir = ""
                self.sbd_dir = ""
                self.imagenet_dir = ""
                self.imagenet_lmdb_dir = ""
                self.imagenetdet_dir = ""
                self.ecssd_dir = ""
                self.hkuis_dir = ""
                self.msra10k_dir = ""
                self.davis_dir = ""
                self.youtubevos_dir = ""

        module.EnvironmentSettings = EnvironmentSettings
        sys.modules[module_name] = module
        return EnvironmentSettings


def draw_overlay(frame, bbox=None, score=None, label="MixFormerV2", frame_index=None, paused=False, needs_reinit=False):
    vis = frame.copy()

    if bbox is not None:
        x, y, w, h = [int(round(value)) for value in bbox]
        shade = vis.copy()
        cv2.rectangle(shade, (x, y), (x + w, y + h), (20, 190, 255), -1)
        vis = cv2.addWeighted(shade, 0.15, vis, 0.85, 0.0)

        cv2.rectangle(vis, (x, y), (x + w, y + h), (20, 190, 255), 3)
        cv2.circle(vis, (x + w // 2, y + h // 2), 3, (20, 190, 255), -1)

        score_text = f"{score:.3f}" if score is not None else "init"
        text = f"{label}  score={score_text}"
        text_origin = (max(10, x), max(30, y - 12))
        cv2.putText(vis, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(vis, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 190, 255), 2)

    status = "pause" if paused else "play"
    frame_text = f"frame={frame_index} [{status}]" if frame_index is not None else status
    cv2.putText(vis, frame_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, "Enter: reselect ROI  Space: play/pause  Esc/Q: quit", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(vis, "d:+1  a:-1  w:+15  s:-15", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if needs_reinit:
        cv2.putText(vis, "Navigation changed frame. Press Enter to set a new ROI.", (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)
    return vis


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    model_name, checkpoint_path, config_path = resolve_model_assets(args)
    capture = None
    writer = None

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    headless = args.headless or not os.environ.get("DISPLAY")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    try:
        cfg = update_new_config_from_file(str(config_path))
        print(f"Using model preset: {model_name}")
        print(f"Using checkpoint: {checkpoint_path}")
        print(f"Using config: {config_path}")
        tracker = MixFormerV2Runner(
            cfg=cfg,
            checkpoint_path=str(checkpoint_path),
            device=device,
            search_area_scale=args.search_area_scale,
            online_size=args.online_size,
            update_interval=args.update_interval,
        )

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise RuntimeError(f"Failed to read first frame from video: {video_path}")

        if not headless:
            ensure_display_window()

        start_frame = first_frame
        start_frame_index = 0
        if args.init_rect:
            x, y, w, h = parse_init_rect(args.init_rect)
        elif headless:
            raise RuntimeError("Headless mode requires --init_rect x,y,w,h")
        else:
            start_frame, start_frame_index, (x, y, w, h) = browse_for_init_rect(capture, first_frame)

        init_bbox = [x, y, w, h]
        tracker.initialize(start_frame, init_bbox)

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or first_frame.shape[1]
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or first_frame.shape[0]
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        output_path = resolve_output_path(video_path, args.output_path)
        writer = create_writer(output_path, fps, (width, height))

        current_frame = start_frame
        current_bbox = init_bbox
        current_score = None
        frame_index = start_frame_index
        paused = False
        needs_reinit = False

        first_vis = draw_overlay(
            current_frame,
            current_bbox,
            score=current_score,
            label="MixFormerV2",
            frame_index=frame_index,
            paused=paused,
            needs_reinit=needs_reinit,
        )
        writer.write(first_vis)
        if not headless:
            cv2.imshow(WINDOW_NAME, first_vis)

        if start_frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index + 1)

        processed_frames = 1
        tick_start = cv2.getTickCount()
        while True:
            if args.max_frames and processed_frames >= args.max_frames:
                break

            if not headless:
                current_vis = draw_overlay(
                    current_frame,
                    current_bbox,
                    score=current_score,
                    label="MixFormerV2",
                    frame_index=frame_index,
                    paused=paused,
                    needs_reinit=needs_reinit,
                )
                cv2.imshow(WINDOW_NAME, current_vis)
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord(" "):
                    if needs_reinit:
                        paused = True
                    else:
                        paused = not paused
                    continue
                if key in (13, 10):
                    rect = cv2.selectROI(WINDOW_NAME, current_frame, False, False)
                    if rect[2] > 0 and rect[3] > 0:
                        tracker.initialize(current_frame.copy(), list(rect))
                        current_bbox = list(rect)
                        current_score = None
                        needs_reinit = False
                        current_vis = draw_overlay(
                            current_frame,
                            current_bbox,
                            score=current_score,
                            label="MixFormerV2",
                            frame_index=frame_index,
                            paused=paused,
                            needs_reinit=needs_reinit,
                        )
                        writer.write(current_vis)
                        cv2.imshow(WINDOW_NAME, current_vis)
                        cv2.waitKey(1)
                    continue
                if paused and key in (ord("d"), ord("a"), ord("w"), ord("s")):
                    if key == ord("d"):
                        next_index = frame_index + 1
                    elif key == ord("a"):
                        next_index = frame_index - 1
                    elif key == ord("w"):
                        next_index = frame_index + 15
                    else:
                        next_index = frame_index - 15

                    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                    if total_frames > 0:
                        next_index = max(0, min(total_frames - 1, next_index))
                    else:
                        next_index = max(0, next_index)

                    next_frame = seek_to_frame(capture, next_index)
                    if next_frame is not None:
                        current_frame = next_frame
                        frame_index = next_index
                        current_bbox = None
                        current_score = None
                        needs_reinit = True
                    continue
                if paused:
                    continue

            ok, frame = capture.read()
            if not ok or frame is None:
                break

            out = tracker.track(frame)
            frame_index += 1
            processed_frames += 1
            current_frame = frame
            current_bbox = out["target_bbox"]
            current_score = out.get("conf_score")
            current_vis = draw_overlay(
                current_frame,
                current_bbox,
                score=current_score,
                label="MixFormerV2",
                frame_index=frame_index,
                paused=paused,
                needs_reinit=needs_reinit,
            )
            writer.write(current_vis)

        elapsed = (cv2.getTickCount() - tick_start) / cv2.getTickFrequency()
        effective_frames = max(processed_frames - 1, 1)
        print(f"Saved output to: {output_path}")
        print(f"Processed frames: {processed_frames}")
        print(f"Tracking FPS: {effective_frames / max(elapsed, 1e-6):.2f}")
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if not headless:
            close_display_window()


if __name__ == "__main__":
    main()
