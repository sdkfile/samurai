import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import logging
import traceback
from datetime import datetime
import subprocess
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]

def setup_logging():
    log_filename = f"samurai_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def crop_and_upscale_smooth(img, bbox, prev_center, prev_bbox, logger, alpha=0.3, target_size=(2160, 3840)):
    x, y, w, h = bbox

    # bbox가 급격하게 작아진 경우 이전 bbox 사용
    if prev_bbox is not None:
        prev_x, prev_y, prev_w, prev_h = prev_bbox
        # bbox 크기가 이전 프레임의 70% 미만으로 작아진 경우
        if (w * h) < (prev_w * prev_h * 0.7):
            logger.info(f"Detected sudden bbox size reduction: {w}x{h} -> using previous bbox: {prev_w}x{prev_h}")
            x, y, w, h = prev_bbox

    cx, cy = x + w // 2, y + h // 2

    if prev_center is not None:
        cx = int(alpha * cx + (1 - alpha) * prev_center[0])
        cy = int(alpha * cy + (1 - alpha) * prev_center[1])

    # 크롭 세로 길이 계산 (bbox 세로의 2.0배, 원본 높이 제한)
    crop_height = min(int(h * 2.0), img.shape[0])
    crop_width = int(crop_height * 9/16)

    # 크롭 영역이 유효한지 확인
    if crop_width <= 0 or crop_height <= 0:
        logger.warning(f"Invalid crop dimensions: {crop_width}x{crop_height}")
        return img, (cx, cy)  # 원본 이미지 반환

    # 중심을 기준으로 크롭 영역 계산
    x1 = max(0, cx - crop_width // 2)
    y1 = max(0, cy - crop_height // 2)
    x2 = min(img.shape[1], x1 + crop_width)
    y2 = min(img.shape[0], y1 + crop_height)

    # 크롭 영역이 너무 작은 경우 처리
    if x2 - x1 < 10 or y2 - y1 < 10:  # 최소 크기 설정
        logger.warning(f"Crop region too small: {x2-x1}x{y2-y1}")
        return img, (cx, cy)  # 원본 이미지 반환

    # 크롭
    cropped = img[y1:y2, x1:x2]

    # 크롭된 이미지가 비어있지 않은지 확인
    if cropped.size == 0:
        logger.warning("Empty cropped image")
        return img, (cx, cy)  # 원본 이미지 반환

    # 타겟 사이즈로 업스케일
    try:
        if cropped.shape[:2] != target_size[::-1]:
            cropped = cv2.resize(cropped, target_size[::-1], interpolation=cv2.INTER_CUBIC)
    except cv2.error as e:
        logger.error(f"Resize error: {str(e)}, crop shape: {cropped.shape}")
        return img, (cx, cy)  # 원본 이미지 반환

    return cropped, (cx, cy)

def main(args):
    logger = setup_logging()
    try:
        logger.info(f"Starting process with arguments: {args}")
        
        # CUDA 메모리 초기화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
        
        logger.info("Building model...")
        model_cfg = determine_model_cfg(args.model_path)
        try:
            predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
        except Exception as e:
            logger.error(f"Failed to build predictor: {str(e)}")
            raise
        
        logger.info("Preparing frames...")
        frames_or_path = prepare_frames_or_path(args.video_path)
        
        # 운딩 박스 형식을 (x1, y1, x2, y2)로 변환
        bbox = (args.x, args.y, args.x + args.width, args.y + args.height)
        logger.info(f"Using bbox: {bbox}")
        
        # 프레임 로딩 전 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        frame_rate = 30
        if args.save_to_video:
            if osp.isdir(args.video_path):
                logger.info("Processing directory of frames...")
                frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
                logger.info(f"Found {len(frames)} frames")
                loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
                height, width = loaded_frames[0].shape[:2]
            else:
                logger.info("Processing video file...")
                cap = cv2.VideoCapture(args.video_path)
                if not cap.isOpened():
                    raise ValueError(f"Failed to open video file: {args.video_path}")
                
                frame_rate = cap.get(cv2.CAP_PROP_FPS)
                logger.info(f"Video frame rate: {frame_rate}")
                
                loaded_frames = []
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    loaded_frames.append(frame)
                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.info(f"Loaded {frame_count} frames...")
                cap.release()
                
                if len(loaded_frames) == 0:
                    raise ValueError("No frames were loaded from the video. The video might be corrupted or in an unsupported format.")
                
                logger.info(f"Total frames loaded: {len(loaded_frames)}")
                height, width = loaded_frames[0].shape[:2]
                logger.info(f"Frame dimensions: {width}x{height}")
            
            # 입력 해상도에 따른 세로형 타겟 사이즈 결정 (16:9 비율)
            if height >= 2160:  # 4K
                target_size = (3840, 2160)  # 16:9
            elif height >= 1440:  # 2K
                target_size = (2560, 1440)
            elif height >= 1080:  # FHD
                target_size = (1920, 1080)
            else:  # HD
                target_size = (1280, 720)
            logger.info(f"Using target size: {target_size}")

        try:
            logger.info("Initializing video writers...")
            
            # 출력 파일명에서 특수문자만 제거하고 공백은 유지
            output_filename = os.path.basename(args.video_output_path)
            output_filename = ''.join(c for c in output_filename if c.isalnum() or c in ('_', '.', '-', ' '))
            args.video_output_path = os.path.join(os.path.dirname(args.video_output_path), output_filename)
            
            # 시각화 영상 경로도 같은 방식으로 처리
            vis_output_path = os.path.join(
                os.path.dirname(args.video_output_path),
                f"{os.path.splitext(output_filename)[0]}_visualization.mp4"
            )
            
            # 출력 디렉토리 확인 및 생성
            output_dir = os.path.dirname(args.video_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Ensuring output directory exists: {output_dir}")
            
            # mp4v 코덱만 사용
            logger.info("Using mp4v codec")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (target_size[1], target_size[0]))
            out_vis = cv2.VideoWriter(vis_output_path, fourcc, frame_rate, (width, height))
            
            if not out.isOpened() or not out_vis.isOpened():
                raise ValueError("Failed to create VideoWriter with mp4v codec")
            
            logger.info("Successfully created VideoWriters with mp4v codec")

        except Exception as e:
            logger.error(f"Error creating VideoWriter: {str(e)}")
            raise

        logger.info("Starting inference...")
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                try:
                    logger.info("Initializing model state...")
                    state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
                    
                    logger.info("Adding initial box...")
                    try:
                        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
                        if masks is None or len(masks) == 0:
                            raise ValueError("Failed to generate initial masks")
                        logger.info(f"Initial masks generated successfully")
                    except Exception as e:
                        logger.error(f"Error adding initial box: {str(e)}")
                        raise

                    if args.save_to_video:
                        os.makedirs('./video/outputs', exist_ok=True)
                        prev_center = None
                        prev_bbox = None  # 이전 프레임의 bbox 저장용
                        
                        logger.info("Starting frame processing...")
                        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                            if frame_idx % 10 == 0:
                                logger.info(f"Processing frame {frame_idx}")
                            
                            try:
                                mask_to_vis = {}
                                bbox_to_vis = {}

                                for obj_id, mask in zip(object_ids, masks):
                                    mask = mask[0].cpu().numpy()
                                    mask = mask > 0.0
                                    non_zero_indices = np.argwhere(mask)
                                    if len(non_zero_indices) == 0:
                                        bbox = prev_bbox if prev_bbox is not None else [0, 0, 0, 0]  # 마스크가 비어있을 때도 이전 bbox 사용
                                        logger.warning(f"Empty mask detected in frame {frame_idx}")
                                    else:
                                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                                        
                                        bbox_to_vis[obj_id] = bbox
                                        mask_to_vis[obj_id] = mask

                                img = loaded_frames[frame_idx]
                                
                                # 크롭된 영상 생성
                                cropped_frame, prev_center = crop_and_upscale_smooth(
                                    img, bbox, prev_center, prev_bbox,
                                    logger=logger,
                                    target_size=target_size
                                )
                                prev_bbox = bbox  # 현재 bbox를 다음 프레임을 위해 저장

                                out.write(cropped_frame)
                                
                                # 시각화 영상 생성
                                vis_frame = img.copy()
                                # 마스크 시각화
                                for obj_id, mask in mask_to_vis.items():
                                    mask_overlay = np.zeros_like(vis_frame)
                                    mask_overlay[mask] = [0, 0, 255]  # 파란색
                                    vis_frame = cv2.addWeighted(vis_frame, 1.0, mask_overlay, 0.5, 0)
                                
                                # 바운딩 박스 그리기
                                for obj_id, bbox in bbox_to_vis.items():
                                    x, y, w, h = bbox
                                    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                
                                out_vis.write(vis_frame)

                            except Exception as e:
                                logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                                logger.error(traceback.format_exc())
                                raise

                except Exception as e:
                    logger.error(f"Error during inference: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
        except Exception as e:
            logger.error(f"Error in inference context: {str(e)}")
            raise
        finally:
            # 리소스 정리
            if 'out' in locals() and out is not None:
                out.release()
            if 'out_vis' in locals() and out_vis is not None:
                out_vis.release()
            if 'predictor' in locals():
                del predictor
            if 'state' in locals():
                del state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # 비디오 작성 완료 후 오디오 처리
        if 'args' in locals() and hasattr(args, 'video_output_path'):
            if os.path.exists(args.video_output_path) and os.path.getsize(args.video_output_path) > 0:
                logger.info(f"Successfully created output file: {args.video_output_path}")
                
                # 원본 오디오를 새 영상에 입히기
                try:
                    logger.info("Adding original audio to the output video...")
                    temp_output = args.video_output_path.replace('.mp4', '_temp.mp4')
                    
                    # 먼저 오디오 스트림이 있는지 확인
                    probe_cmd = ['ffmpeg', '-i', args.video_path]
                    probe_result = subprocess.run(
                        probe_cmd, 
                        stderr=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        text=True
                    )
                    
                    # 오디오 스트림이 있는지 확인
                    if 'Stream' in probe_result.stderr and 'Audio' in probe_result.stderr:
                        # FFmpeg 명령어로 오디오 합성
                        cmd = [
                            'ffmpeg', '-y',
                            '-i', args.video_output_path,  # 생성된 비디오
                            '-i', args.video_path,         # 원본 비디오 (오디오 소스)
                            '-c:v', 'copy',                # 비디오는 그대로 복사
                            '-c:a', 'aac',                 # 오디오는 AAC 코덱 사용
                            '-map', '0:v:0',               # 첫 번째 입력에서 비디오 스트림 사용
                            '-map', '1:a:0',               # 두 번째 입력에서 오디오 스트림 사용
                            temp_output
                        ]
                        
                        process = subprocess.run(
                            cmd,
                            stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            text=True
                        )
                        
                        if process.returncode == 0:
                            # 성공적으로 오디오가 합성되면 파일 교체
                            os.replace(temp_output, args.video_output_path)
                            logger.info("Successfully added original audio to the output video")
                        else:
                            logger.error(f"Failed to add audio: {process.stderr}")
                            if os.path.exists(temp_output):
                                os.remove(temp_output)  # 임시 파일 제거
                    else:
                        logger.info("No audio stream found in the original video. Skipping audio processing.")
                        
                except Exception as e:
                    logger.error(f"Error while adding audio: {str(e)}")
                    if os.path.exists(temp_output):
                        os.remove(temp_output)  # 임시 파일 제거
            else:
                logger.error(f"Output file was not created or is empty: {args.video_output_path}")
                raise ValueError("Failed to create output video file")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
        parser.add_argument("--model_path", default="./sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
        parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
        parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
        
        # 바운딩 박스 좌표를 위한 새로운 인자들
        parser.add_argument("--x", type=int, required=True, help="X coordinate of bounding box")
        parser.add_argument("--y", type=int, required=True, help="Y coordinate of bounding box")
        parser.add_argument("--width", type=int, required=True, help="Width of bounding box")
        parser.add_argument("--height", type=int, required=True, help="Height of bounding box")
        
        args = parser.parse_args()
        main(args)
    except Exception as e:
        logging.error(f"Error in main script: {str(e)}")
        logging.error(traceback.format_exc())
        raise
