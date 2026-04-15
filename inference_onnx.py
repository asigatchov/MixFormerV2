#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime


WINDOW_NAME = "MixFormerV2 ONNX"
MODEL_DEFAULTS = {
    "small": {"template_factor": 2.0, "search_factor": 4.5},
    "base": {"template_factor": 2.0, "search_factor": 4.5},
}


class PreprocessorXOnnx:
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1))

    def process(self, img_arr, amask_arr):
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2).astype(np.float32)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std
        amask_arr_3d = amask_arr[np.newaxis, :, :].astype(np.bool_)
        return img_arr_4d, amask_arr_3d


def sample_target(image, target_bb, search_area_factor, output_sz=None):
    x, y, w, h = [float(v) for v in target_bb]
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    if crop_sz < 1:
        raise ValueError("Too small bounding box.")

    x1 = int(round(x + 0.5 * w - 0.5 * crop_sz))
    y1 = int(round(y + 0.5 * h - 0.5 * crop_sz))
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    y1_pad = max(0, -y1)
    x2_pad = max(x2 - image.shape[1] + 1, 0)
    y2_pad = max(y2 - image.shape[0] + 1, 0)

    image_crop = image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    image_crop = cv2.copyMakeBorder(image_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)

    crop_h, crop_w = image_crop.shape[:2]
    att_mask = np.ones((crop_h, crop_w), dtype=np.float32)
    end_x = None if x2_pad == 0 else -x2_pad
    end_y = None if y2_pad == 0 else -y2_pad
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    if output_sz is None:
        return image_crop, 1.0, att_mask.astype(np.bool_)

    resize_factor = output_sz / crop_sz
    image_crop = cv2.resize(image_crop, (output_sz, output_sz))
    att_mask = cv2.resize(att_mask, (output_sz, output_sz), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
    return image_crop, resize_factor, att_mask


def clip_box(box, height, width, margin=0):
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h
    x1 = min(max(0, x1), width - margin)
    x2 = min(max(margin, x2), width)
    y1 = min(max(0, y1), height - margin)
    y2 = min(max(margin, y2), height)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]


def parse_args():
    parser = argparse.ArgumentParser(description="MixFormerV2 ONNX video inference demo")
    parser.add_argument("--model_path", required=True, help="path to ONNX model")
    parser.add_argument("--video_path", required=True, help="path to input video")
    parser.add_argument("--output_path", default="", help="path to output demo video, default is <video>_onnx.mp4")
    parser.add_argument("--init_rect", default="", type=str, help="initial bbox as x,y,w,h")
    parser.add_argument("--max_frames", default=0, type=int, help="process at most this many frames, 0 means full video")
    parser.add_argument("--headless", action="store_true", help="run without OpenCV UI")
    parser.add_argument("--template_size", default=None, type=int, help="override template size")
    parser.add_argument("--search_size", default=None, type=int, help="override search size")
    parser.add_argument("--template_factor", default=None, type=float, help="override template factor")
    parser.add_argument("--search_factor", default=None, type=float, help="override search factor")
    parser.add_argument("--providers", nargs="*", default=None, help="optional ONNX Runtime providers list")
    return parser.parse_args()


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
    return video.with_name(f"{video.stem}_onnx.mp4")


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


def draw_overlay(frame, bbox=None, score=None, label="MixFormerV2 ONNX", frame_index=None, paused=False, needs_reinit=False):
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


def infer_spatial_size(input_meta, fallback_name):
    shape = input_meta.shape
    if len(shape) != 4:
        raise ValueError(f"Unexpected input shape for {fallback_name}: {shape}")
    height, width = shape[2], shape[3]
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError(f"Dynamic spatial size for {fallback_name} is not supported, use --template_size/--search_size")
    if height != width:
        raise ValueError(f"Expected square spatial input for {fallback_name}, got {height}x{width}")
    return height


def detect_model_name(model_path):
    model_name = Path(model_path).name.lower()
    for candidate in MODEL_DEFAULTS:
        if candidate in model_name:
            return candidate
    return "small"


class MixFormerV2OnnxRunner:
    def __init__(self, session, template_size, search_size, template_factor, search_factor):
        self.session = session
        self.preprocessor = PreprocessorXOnnx()
        self.input_names = [item.name for item in session.get_inputs()]
        if len(self.input_names) != 3:
            raise ValueError(f"Expected 3 ONNX inputs, got {len(self.input_names)}")

        self.template_size = template_size
        self.search_size = search_size
        self.template_factor = template_factor
        self.search_factor = search_factor
        self.state = None
        self.template = None
        self.online_template = None

    def initialize(self, image_bgr, init_bbox):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        z_patch_arr, _, z_amask_arr = sample_target(image_rgb, init_bbox, self.template_factor, output_sz=self.template_size)
        template_input, _ = self.preprocessor.process(z_patch_arr, np.asarray(z_amask_arr))
        self.template = template_input
        self.online_template = template_input.copy()
        self.state = list(init_bbox)

    def track(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_bgr.shape[:2]
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image_rgb, self.state, self.search_factor, output_sz=self.search_size)
        search_input, _ = self.preprocessor.process(x_patch_arr, np.asarray(x_amask_arr))

        ort_inputs = {
            self.input_names[0]: self.template.astype(np.float32),
            self.input_names[1]: self.online_template.astype(np.float32),
            self.input_names[2]: search_input.astype(np.float32),
        }
        pred_boxes, pred_scores = self.session.run(None, ort_inputs)

        pred_boxes = np.asarray(pred_boxes)
        if pred_boxes.ndim == 3:
            mean_box = pred_boxes.mean(axis=1).reshape(-1)
        elif pred_boxes.ndim == 2 and pred_boxes.shape[1] % 4 == 0:
            mean_box = pred_boxes.reshape(pred_boxes.shape[0], -1, 4).mean(axis=1).reshape(-1)
        else:
            mean_box = pred_boxes.reshape(-1)[:4]

        pred_box = (mean_box * self.search_size / resize_factor).tolist()
        pred_score = float(np.asarray(pred_scores).reshape(-1)[0])
        self.state = clip_box(self._map_box_back(pred_box, resize_factor), height, width, margin=10)
        return {"target_bbox": self.state, "conf_score": pred_score}

    def _map_box_back(self, pred_box, resize_factor):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, width, height = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * width, cy_real - 0.5 * height, width, height]


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    video_path = Path(args.video_path)
    capture = None
    writer = None

    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    headless = args.headless
    providers = args.providers or onnxruntime.get_available_providers()
    session = onnxruntime.InferenceSession(str(model_path), providers=providers)

    input_metas = session.get_inputs()
    if len(input_metas) != 3:
        raise RuntimeError(f"Expected 3 inputs in ONNX model, got {len(input_metas)}")

    model_name = detect_model_name(model_path)
    template_size = args.template_size or infer_spatial_size(input_metas[0], input_metas[0].name)
    search_size = args.search_size or infer_spatial_size(input_metas[2], input_metas[2].name)
    template_factor = args.template_factor or MODEL_DEFAULTS[model_name]["template_factor"]
    search_factor = args.search_factor or MODEL_DEFAULTS[model_name]["search_factor"]

    print(f"Using model: {model_path}")
    print(f"Using providers: {providers}")
    print(f"Template size: {template_size}")
    print(f"Search size: {search_size}")
    print(f"Template factor: {template_factor}")
    print(f"Search factor: {search_factor}")

    try:
        runner = MixFormerV2OnnxRunner(
            session=session,
            template_size=template_size,
            search_size=search_size,
            template_factor=template_factor,
            search_factor=search_factor,
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
        runner.initialize(start_frame, init_bbox)

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

        first_vis = draw_overlay(current_frame, current_bbox, score=current_score, frame_index=frame_index, paused=paused, needs_reinit=needs_reinit)
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
                current_vis = draw_overlay(current_frame, current_bbox, score=current_score, frame_index=frame_index, paused=paused, needs_reinit=needs_reinit)
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
                        runner.initialize(current_frame.copy(), list(rect))
                        current_bbox = list(rect)
                        current_score = None
                        needs_reinit = False
                        current_vis = draw_overlay(current_frame, current_bbox, score=current_score, frame_index=frame_index, paused=paused, needs_reinit=needs_reinit)
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

            out = runner.track(frame)
            frame_index += 1
            processed_frames += 1
            current_frame = frame
            current_bbox = out["target_bbox"]
            current_score = out.get("conf_score")
            current_vis = draw_overlay(current_frame, current_bbox, score=current_score, frame_index=frame_index, paused=paused, needs_reinit=needs_reinit)
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
