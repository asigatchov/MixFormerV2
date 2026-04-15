import os
import sys
import argparse
import cv2
import numpy as np

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker

# Try to import YOLO, but make it optional
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. YOLOv11 detection will be disabled.")


def run_video(tracker_name, 
              tracker_param, 
              videofile='', 
              optional_box=None, 
              debug=None,
              save_results=False, 
              tracker_params=None, 
              zoomin=False, 
              expansion_ratio=1.0,
              max_per_folder=60, 
              save_yolo=False, 
              yolo_label=0,
              save_cls=False,
              save_video=False,
              yolo_model=None,
              yolo_conf=0.25,
              yolo_classes=None,
              use_yolo_init=False,
              save_roi=False,
              roi_size=64,
              roi_output_dir=None,
              no_display=False):
    """Run the tracker on your video with optional YOLO detection.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
        save_video: Whether to save the video with tracking results.
        yolo_model: Path to YOLOv11 model file (e.g., yolo11n.pt).
        yolo_conf: Confidence threshold for YOLO detection.
        yolo_classes: List of class IDs to detect (None for all classes).
        use_yolo_init: If True, use YOLO to detect initial box automatically.
        save_roi: Whether to save ROI crops centered on target.
        roi_size: Size of the ROI crop (default 64x64).
        roi_output_dir: Output directory for ROI crops.
        no_display: If True, do not display video frames (faster processing).
    """
    tracker = Tracker(tracker_name, tracker_param, "LASOT",
                      tracker_params=tracker_params)
    tracker.run_video(videofilepath=videofile,
                      optional_box=optional_box,
                      debug=debug,
                      save_results=save_results,
                      is_zoomin=zoomin,
                      expansion_ratio=expansion_ratio,
                      max_per_folder=max_per_folder,
                      save_yolo=save_yolo,
                      yolo_label=yolo_label,
                      save_cls=save_cls,
                      save_video=save_video,
                      yolo_model=yolo_model,
                      yolo_conf=yolo_conf,
                      yolo_classes=yolo_classes,
                      use_yolo_init=use_yolo_init,
                      save_roi=save_roi,
                      roi_size=roi_size,
                      roi_output_dir=roi_output_dir,
                      no_display=no_display)


def get_video_files(path):
    """Return a list of video files.
    
    Args:
        path: Path to a video file or a directory containing videos.
        
    Returns:
        A list of video file paths.
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg']
    
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        video_files = []
        for root, dirs, files in os.walk(path):
            for file in sorted(files):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))
        return video_files
    else:
        raise ValueError(f"Path does not exist: {path}")


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on a video file or a directory of videos')
    parser.add_argument('tracker_name', type=str, help='Tracking method name')
    parser.add_argument('tracker_param', type=str, help='Parameter file name')
    parser.add_argument('videofile', type=str, help='Path to a video file or a directory containing videos')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='Optional initial box in x y w h format')
    parser.add_argument('--debug', type=int, default=0, help='Debug level')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding box results')

    parser.add_argument('--params__model', type=str, default=None, help="Tracking model path")
    parser.add_argument('--params__update_interval', type=int, default=None, help="Update interval for online tracking")
    parser.add_argument('--params__online_size', type=int, default=None)
    parser.add_argument('--params__search_area_scale', type=float, default=None)
    parser.add_argument('--params__max_score_decay', type=float, default=1.0)
    parser.add_argument('--params__vis_attn', type=int, choices=[0, 1], default=0, help="Whether to visualize attention maps")
    parser.add_argument('--zoomin', action='store_true', help='Zoom in on the video')
    parser.add_argument('--expansion_ratio', type=float, default=1.0, help="Expansion ratio used when enlarging the tracked box")
    parser.add_argument('--max_per_folder', type=int, default=60, help="Maximum number of saved frames per folder")
    parser.add_argument('--save_yolo', action='store_true', help="Save results in YOLO format")
    parser.add_argument('--yolo_label', type=int, default=0, help="Class label to use for YOLO output")
    parser.add_argument('--save_cls', action='store_true', help="Save classification crops (currently unused in video_demo.py)")
    parser.add_argument('--save_video', action='store_true', help="Save the video with tracking overlays")
    
    # YOLOv11 detection parameters.
    parser.add_argument('--yolo_model', type=str, default=None, help="Path to the YOLOv11 model file (for example: yolo11n.pt)")
    parser.add_argument('--yolo_conf', type=float, default=0.25, help="Confidence threshold for YOLO detection")
    parser.add_argument('--yolo_classes', type=int, nargs='+', default=None, help="List of class IDs to detect (for example: --yolo_classes 0 2 3)")
    parser.add_argument('--use_yolo_init', action='store_true', help="Use YOLO to automatically detect and initialize the tracking target")
    
    # ROI crop parameters.
    parser.add_argument('--save_roi', action='store_true', help="Save ROI crops centered on the target")
    parser.add_argument('--roi_size', type=int, default=64, help="ROI crop size (default: 64x64)")
    parser.add_argument('--roi_output_dir', type=str, default=None, help="Output directory for ROI images (defaults to an output folder next to the video)")
    
    # Display control parameters.
    parser.add_argument('--no_display', action='store_true', help="Do not display video frames (faster for batch processing)")
    
    args = parser.parse_args()

    tracker_params = {}
    for param in list(filter(lambda s: s.split('__')[0] == 'params' and getattr(args, s) is not None, args.__dir__())):
        tracker_params[param.split('__')[1]] = getattr(args, param)
    print(tracker_params)

    # Collect the input video files.
    try:
        video_files = get_video_files(args.videofile)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if not video_files:
        print(f"No video files found in {args.videofile}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video file.
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing video [{idx}/{len(video_files)}]: {video_path}")
        print(f"{'='*60}")
        
        try:
            run_video(args.tracker_name,
                      args.tracker_param,
                      video_path,
                      args.optional_box,
                      args.debug,
                      args.save_results,
                      tracker_params=tracker_params,
                      zoomin=args.zoomin,
                      expansion_ratio=args.expansion_ratio,
                      max_per_folder=args.max_per_folder,
                      save_yolo=args.save_yolo,
                      yolo_label=args.yolo_label,
                      save_cls=args.save_cls,
                      save_video=args.save_video,
                      yolo_model=args.yolo_model,
                      yolo_conf=args.yolo_conf,
                      yolo_classes=args.yolo_classes,
                      use_yolo_init=args.use_yolo_init,
                      save_roi=args.save_roi,
                      roi_size=args.roi_size,
                      roi_output_dir=args.roi_output_dir,
                      no_display=args.no_display)
            print(f"✓ Video processed successfully: {video_path}")
        except Exception as e:
            print(f"✗ Video processing failed: {video_path}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Finished processing all videos. Total processed: {len(video_files)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
