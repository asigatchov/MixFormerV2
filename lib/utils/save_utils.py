import cv2
import numpy as np
import os


def save_yolo_annotation(image, box, root_dir, image_name):
    """
    Save an image and its YOLO annotation into the target directory.

    Args:
        image: OpenCV BGR image.
        box: Bounding box in [x, y, w, h, class_id] format.
        root_dir: Root output directory.
        image_name: Image filename including extension.
    """
    # Ensure the root directory exists.
    os.makedirs(root_dir, exist_ok=True)

    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Save the image.
    img_save_path = os.path.join(images_dir, image_name)
    cv2.imwrite(img_save_path, image)

    # Convert to normalized YOLO format: class_id cx cy w h.
    h_img, w_img = image.shape[:2]
    x, y, w, h, class_id = box
    cx = (x + w / 2) / w_img
    cy = (y + h / 2) / h_img
    bw = w / w_img
    bh = h / h_img

    label_line = f"{int(class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_save_path = os.path.join(labels_dir, label_name)
    with open(label_save_path, "w") as f:
        f.write(label_line + "\n")


def resize_and_save(
    target_crop,
    save_dir,
    img_idx,
    target_size=(224, 224),
    keep_aspect_ratio=True  # Controls whether aspect ratio is preserved.
):
    """
    Resize an image to the target size and save it.

    Args:
        target_crop: Input image in OpenCV BGR format.
        save_dir: Output directory.
        img_idx: Image index or name.
        target_size: Target size as (width, height). Defaults to (224, 224).
        keep_aspect_ratio: Whether to preserve the aspect ratio.
            True uses resize + padding, False stretches directly.
    """
    # Get the original size.
    h, w = target_crop.shape[:2]
    target_w, target_h = target_size

    if keep_aspect_ratio:
        # Resize while preserving aspect ratio, then pad.
        # Compute the scale factor while preserving aspect ratio.
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize proportionally.
        resized = cv2.resize(target_crop, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)

        # Create a blank canvas.
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Place the resized image at the center of the canvas.
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    else:
        # Direct resize without preserving aspect ratio.
        canvas = cv2.resize(target_crop, (target_w, target_h),
                            interpolation=cv2.INTER_AREA)

    # Save the image.
    cv2.imwrite(os.path.join(save_dir, f"{img_idx}.jpg"), canvas)


def crop_save(
    target_crop,
    save_dir,
    img_idx,
):
    """
    Save the crop without resizing.

    Args:
        target_crop: Input image in OpenCV BGR format.
        save_dir: Output directory.
        img_idx: Image index or name.
    """

    # Save the image.
    cv2.imwrite(os.path.join(save_dir, f"{img_idx}.jpg"), target_crop)
