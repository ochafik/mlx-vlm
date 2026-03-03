from __future__ import annotations

import argparse
import base64
import logging
import math
import os
import subprocess
import time
from io import BytesIO
from typing import List

import cv2
import mlx.core as mx
import numpy as np
import requests
from PIL import Image

from .generate import generate
from .utils import load, load_image, process_inputs_with_fallback

# This is a beta version of the video generation script.
# It is not fully tested and may not work as expected.

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

logger.info(
    "This is a beta version of the video understanding. It may not work as expected."
)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# Set the maximum number of video token inputs.
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(
            pil_image, mask=pil_image.split()[3]
        )  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(
    ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR
) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        )
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """Calculate the number of frames for the video to be used as model inputs.

    Either a fixed 'nframes' is provided in ele or 'fps' is used to calculate how many frames to sample.
    """
    assert not (
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should be in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


def extract_keyframes(video_path: str, video_fps: float) -> list[int]:
    """Extract I-frame (keyframe) indices from a video using ffprobe.

    Returns a list of frame indices corresponding to keyframes, or an empty
    list if ffprobe is unavailable or fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-print_format",
                "csv",
                "-show_entries",
                "packet=pts_time,flags",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"ffprobe failed (rc={result.returncode}), skipping keyframes")
            return []

        indices = []
        for line in result.stdout.strip().splitlines():
            # Format: packet,<pts_time>,<flags>
            parts = line.split(",")
            if len(parts) < 3:
                continue
            if parts[0] != "packet":
                continue
            pts_time_str, flags = parts[1], parts[2]
            if "K" not in flags:
                continue
            try:
                pts_time = float(pts_time_str)
            except (ValueError, TypeError):
                continue
            frame_idx = int(round(pts_time * video_fps))
            indices.append(frame_idx)

        logger.info(f"extract_keyframes: found {len(indices)} keyframes")
        return indices
    except FileNotFoundError:
        logger.warning("ffprobe not found, skipping keyframe extraction")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timed out, skipping keyframe extraction")
        return []
    except Exception:
        logger.warning("ffprobe error, skipping keyframe extraction")
        return []


def extract_audio_transcript(
    video_path: str,
    frame_indices: list[int],
    video_fps: float,
) -> str | None:
    """Extract audio transcript with timestamps aligned to sampled frames.

    Uses mlx-whisper to transcribe the audio track, then tags each transcript
    segment with the nearest frame index. Returns a formatted string to embed
    in the prompt, or None if transcription fails.
    """
    try:
        import mlx_whisper
    except ImportError:
        logger.warning("mlx-whisper not installed, skipping audio transcription")
        return None

    import tempfile

    # Extract audio to a temp WAV file
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                wav_path,
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg audio extraction failed")
            return None

        # Transcribe with word-level timestamps
        logger.info("Transcribing audio with mlx-whisper...")
        transcript = mlx_whisper.transcribe(
            wav_path,
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            word_timestamps=True,
        )

        if not transcript or not transcript.get("segments"):
            logger.info("No speech detected in audio")
            return None

        # Convert frame indices to timestamps
        frame_times = [idx / video_fps for idx in frame_indices]

        def nearest_frame(t: float) -> int:
            """Find the nearest frame index for a given timestamp."""
            best_i = 0
            best_dist = abs(frame_times[0] - t)
            for i, ft in enumerate(frame_times):
                d = abs(ft - t)
                if d < best_dist:
                    best_dist = d
                    best_i = i
            return best_i

        # Build transcript lines tagged with frame numbers
        lines = []
        for seg in transcript["segments"]:
            start_t = seg["start"]
            end_t = seg["end"]
            text = seg["text"].strip()
            if not text:
                continue
            f_start = nearest_frame(start_t)
            f_end = nearest_frame(end_t)
            mm_s = int(start_t) // 60
            ss_s = int(start_t) % 60
            mm_e = int(end_t) // 60
            ss_e = int(end_t) % 60
            if f_start == f_end:
                lines.append(f"[{mm_s}:{ss_s:02d}-{mm_e}:{ss_e:02d}, frame {f_start}] {text}")
            else:
                lines.append(f"[{mm_s}:{ss_s:02d}-{mm_e}:{ss_e:02d}, frames {f_start}-{f_end}] {text}")

        if not lines:
            return None

        logger.info(f"Transcribed {len(lines)} segments")
        return "\n".join(lines)

    except FileNotFoundError:
        logger.warning("ffmpeg not found, skipping audio transcription")
        return None
    except Exception as e:
        logger.warning(f"Audio transcription failed: {e}")
        return None
    finally:
        if wav_path:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def load_video(
    ele: dict,
    use_keyframes: bool = False,
) -> tuple[np.ndarray, float, list[int], int, float]:
    """
    Read video using cv2.VideoCapture.

    The video is read as a NumPy array with shape (T, C, H, W) where T is the number of frames,
    C is the number of channels, and H, W are the frame dimensions.

    Returns:
        video_np: Array of shape (T, C, H, W)
        sample_fps: Effective sampling fps
        frame_indices: List of frame indices that were sampled
        total_frames: Total number of frames in the video
        video_fps: Original video fps
    """
    video_path = ele["video"]
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0  # default to 1.0 if fps returns 0
    st = time.time()
    logger.info(
        f"numpy reader: video_path={video_path}, total_frames={total_frames}, video_fps={video_fps}, time={time.time()-st:.3f}s"
    )

    if use_keyframes:
        keyframe_indices = extract_keyframes(video_path, video_fps)
        if keyframe_indices:
            # Filter to valid range
            keyframe_indices = [i for i in keyframe_indices if 0 <= i < total_frames]
            available = len(keyframe_indices)
            # Cap to max_frames if specified
            max_frames = ele.get("max_frames")
            if max_frames and len(keyframe_indices) > max_frames:
                step = len(keyframe_indices) / max_frames
                keyframe_indices = [
                    keyframe_indices[int(i * step)]
                    for i in range(max_frames)
                ]
            # Ensure divisible by FRAME_FACTOR
            nframes = floor_by_factor(len(keyframe_indices), FRAME_FACTOR)
            nframes = max(nframes, FRAME_FACTOR)
            keyframe_indices = keyframe_indices[:nframes]
            indices = np.array(keyframe_indices, dtype=int)
            logger.info(
                f"Using {len(indices)} keyframes (from {available} available)"
            )
        else:
            logger.info("No keyframes found, falling back to uniform sampling")
            nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
            indices = np.linspace(0, total_frames - 1, nframes).round().astype(int)
    else:
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        indices = np.linspace(0, total_frames - 1, nframes).round().astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError("No frames read from the video.")
    # Stack frames into a numpy array: (T, H, W, C)
    video_np = np.stack(frames, axis=0)
    # Rearrange to (T, C, H, W)
    video_np = np.transpose(video_np, (0, 3, 1, 2))
    sample_fps = len(frames) / max(total_frames, 1e-6) * video_fps
    return video_np, sample_fps, indices.tolist(), total_frames, video_fps


def fetch_video(
    ele: dict,
    image_factor: int = IMAGE_FACTOR,
    return_video_sample_fps: bool = False,
    use_keyframes: bool = False,
) -> np.ndarray | list[Image.Image]:
    if isinstance(ele["video"], str):
        video, sample_fps, frame_indices, total_frames, orig_fps = load_video(
            ele, use_keyframes=use_keyframes
        )
        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
            int(min_pixels * 1.05),
        )
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(
                f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
            )
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        # Resize each frame using OpenCV (similar to torchvision.transforms.functional.resize with BICUBIC)
        resized_frames = []
        # video is (T, C, H, W) so we need to process each frame
        for frame in video:
            # Rearrange from (C, H, W) to (H, W, C)
            frame_np = np.transpose(frame, (1, 2, 0))
            # cv2.resize expects size as (width, height)
            resized = cv2.resize(
                frame_np, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC
            )
            # Convert back to (C, H, W)
            resized = np.transpose(resized, (2, 0, 1))
            resized_frames.append(resized)
        video = np.stack(resized_frames, axis=0).astype(np.float32)
        if return_video_sample_fps:
            return video, sample_fps, frame_indices, total_frames, orig_fps
        return video
    else:
        # Assume video is provided as a list/tuple of image objects.
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image(
                {"image": video_element, **process_info}, size_factor=image_factor
            )
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            fps = process_info.pop("fps", 2.0)
            frame_indices = list(range(len(images)))
            return images, fps, frame_indices, len(images), fps
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
    use_keyframes: bool = False,
) -> tuple[
    list[Image.Image] | None, list[np.ndarray | list[Image.Image]] | None, dict | None
]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    video_metadata_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps, frame_indices, total_frames, orig_fps = (
                fetch_video(
                    vision_info,
                    return_video_sample_fps=True,
                    use_keyframes=use_keyframes,
                )
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
            video_metadata_list.append(
                {
                    "total_num_frames": total_frames,
                    "fps": orig_fps,
                    "frames_indices": frame_indices,
                }
            )
        else:
            raise ValueError("Content must include image, image_url, or video.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {
            "fps": video_sample_fps_list,
            "video_metadata": video_metadata_list,
        }
    return image_inputs, video_inputs


class VideoFrameExtractor:
    def __init__(self, max_frames: int = 50):
        self.max_frames = max_frames

    def resize_and_center_crop(
        self, image: Image.Image, target_size: int
    ) -> Image.Image:
        # Get current dimensions
        width, height = image.size

        # Calculate new dimensions keeping aspect ratio
        if width < height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))

        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size

        return image.crop((left, top, right, bottom))

    def extract_frames(self, video_path: str) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Calculate frame indices to extract (1fps)
        frame_indices = list(range(0, total_frames, fps))

        # If we have more frames than max_frames, sample evenly
        if len(frame_indices) > self.max_frames:
            indices = np.linspace(0, len(frame_indices) - 1, self.max_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices]

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                pil_image = self.resize_and_center_crop(pil_image, 384)
                frames.append(pil_image)

        cap.release()
        return frames


def is_video_model(model):
    return hasattr(model.config, "video_token_id") or hasattr(
        model.config, "video_token_index"
    )


def is_video_file(video_path: List[str]) -> bool:
    video_extensions = [".mp4", ".avi", ".mov"]
    for path in video_path:
        if not any(path.lower().endswith(ext) for ext in video_extensions):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Video Description CLI")
    parser.add_argument(
        "--video", type=str, nargs="+", required=True, help="Path to the video file"
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        nargs=2,
        default=[448, 448],
        help="Maximum resolution as two integers (height width)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum number of frames"
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second")
    parser.add_argument(
        "--prompt", default="Describe this video.", help="Text prompt for the model"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        help="Select the model to use",
    )
    parser.add_argument(
        "--use-keyframes",
        action="store_true",
        default=False,
        help="Use ffprobe to extract keyframes (I-frames) instead of uniform sampling",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        default=False,
        help="Extract audio transcript with timestamps and include in prompt (requires mlx-whisper)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write generated text to this file",
    )
    parser.add_argument("--verbose", action="store_false", help="Print verbose output")

    args = parser.parse_args()

    print(f"\033[32mLoading model:\033[0m {args.model}")
    model, processor = load(args.model)

    # Validate the model
    if not is_video_model(model):
        logger.warning(
            "Warning: The model selected doesn't natively support video inputs. Performance may be degraded."
        )

    if isinstance(args.max_pixels, tuple) or isinstance(args.max_pixels, list):
        max_pixels = args.max_pixels[0] * args.max_pixels[1]
    else:
        max_pixels = args.max_pixels

    kwargs = {}
    video_kwargs = {}
    if is_video_model(model):

        # Check if video is image or video
        if is_video_file(args.video):
            video_info = {
                "type": "video",
                "video": args.video[0],
                "max_pixels": max_pixels,
                "fps": args.fps,
            }
            if args.max_frames is not None:
                video_info["max_frames"] = args.max_frames
            messages = [
                {
                    "role": "user",
                    "content": [
                        video_info,
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in args.video],
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True, use_keyframes=args.use_keyframes
        )

        if args.max_frames is not None:
            video_inputs = video_inputs[: args.max_frames]

        # Optionally transcribe audio and inject into prompt
        if args.transcribe and is_video_file(args.video) and video_kwargs.get("video_metadata"):
            meta = video_kwargs["video_metadata"][0]
            transcript_text = extract_audio_transcript(
                args.video[0], meta["frames_indices"], meta["fps"]
            )
            if transcript_text:
                # Inject transcript before the user's question in the message
                transcript_block = (
                    f"Audio transcript (with timestamps and frame references):\n"
                    f"{transcript_text}\n\n"
                )
                for msg in messages:
                    if msg["role"] == "user" and isinstance(msg["content"], list):
                        for i, part in enumerate(msg["content"]):
                            if part.get("type") == "text":
                                msg["content"][i] = {
                                    "type": "text",
                                    "text": transcript_block + part["text"],
                                }
                                break
                        break

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Pass video_metadata to prevent the processor from re-sampling frames
        processor_kwargs = {}
        if video_kwargs.get("video_metadata"):
            processor_kwargs["videos_kwargs"] = {
                "video_metadata": video_kwargs["video_metadata"],
                "do_sample_frames": False,
            }

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **processor_kwargs,
        )

        input_ids = mx.array(inputs["input_ids"])
        pixel_values = inputs.get(
            "pixel_values_videos", inputs.get("pixel_values", None)
        )
        if pixel_values is None:
            raise ValueError("Please provide a valid video or image input.")
        pixel_values = mx.array(pixel_values)

        mask = mx.array(inputs["attention_mask"])
        if inputs.get("video_grid_thw", None) is not None:
            kwargs["video_grid_thw"] = mx.array(inputs["video_grid_thw"])
        if inputs.get("image_grid_thw", None) is not None:
            kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])

    else:
        if is_video_file(args.video):
            if len(args.video) > 1:
                raise ValueError("Only one video is supported for video models.")
            else:
                frame_extractor = VideoFrameExtractor(args.max_frames)
                frames = frame_extractor.extract_frames(args.video[0])
        else:
            frames = [load_image(image) for image in args.video]

        # Create prompt with frames
        image_tokens = [{"type": "image"} for _ in range(len(frames))]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    *image_tokens,
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Configure processor for video frames
        processor.image_processor.size = (
            tuple(args.max_pixels)
            if isinstance(args.max_pixels, (tuple, list))
            else (args.max_pixels, args.max_pixels)
        )
        if hasattr(processor.image_processor, "do_resize"):
            processor.image_processor.do_resize = False
        if hasattr(processor.image_processor, "do_image_splitting"):
            processor.image_processor.do_image_splitting = False

        # Process inputs
        inputs = process_inputs_with_fallback(
            processor,
            images=[img for img in frames],
            prompts=text,
        )

        input_ids = mx.array(inputs["input_ids"])
        pixel_values = mx.array(inputs["pixel_values"])
        mask = mx.array(inputs["attention_mask"])
        for key, value in inputs.items():
            if key not in [
                "input_ids",
                "pixel_values",
                "attention_mask",
            ] and not isinstance(value, (str, list)):
                kwargs[key] = mx.array(value)

    logger.info("\033[32mGenerating response...\033[0m")

    kwargs["video"] = args.video
    kwargs["input_ids"] = input_ids
    kwargs["pixel_values"] = pixel_values
    kwargs["mask"] = mask
    kwargs["temperature"] = args.temperature
    kwargs["max_tokens"] = args.max_tokens

    gen_start = time.time()
    response = generate(
        model,
        processor,
        prompt=text,
        verbose=args.verbose,
        **kwargs,
    )
    gen_elapsed = time.time() - gen_start

    if not args.verbose:
        print(response)

    if args.output:
        # Extract text from GenerationResult if needed
        if hasattr(response, "text"):
            response_text = response.text
        else:
            response_text = str(response)
        # Build frontmatter for .md files
        if args.output.endswith(".md"):
            video_path = args.video[0]
            frontmatter_lines = ["---"]
            frontmatter_lines.append(f"source: \"{video_path}\"")
            frontmatter_lines.append(f"model: \"{args.model}\"")
            frontmatter_lines.append(f"max_pixels: {args.max_pixels}")
            frontmatter_lines.append(f"use_keyframes: {args.use_keyframes}")
            frontmatter_lines.append(f"transcribe: {args.transcribe}")
            video_duration = 0.0
            try:
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "quiet",
                        "-show_entries", "format=duration",
                        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                        "-of", "json",
                        video_path,
                    ],
                    capture_output=True, text=True, timeout=10,
                )
                if probe.returncode == 0:
                    import json as _json
                    info = _json.loads(probe.stdout)
                    if info.get("streams"):
                        s = info["streams"][0]
                        frontmatter_lines.append(f"width: {s.get('width', '?')}")
                        frontmatter_lines.append(f"height: {s.get('height', '?')}")
                        frontmatter_lines.append(f"fps: \"{s.get('r_frame_rate', '?')}\"")
                        frontmatter_lines.append(f"total_frames: {s.get('nb_frames', '?')}")
                    if info.get("format"):
                        video_duration = float(info["format"].get("duration", 0))
                        mm, ss = divmod(int(video_duration), 60)
                        frontmatter_lines.append(f"duration: \"{mm}m{ss:02d}s\"")
            except Exception:
                pass
            sampled = 0
            if video_kwargs.get("video_metadata"):
                meta = video_kwargs["video_metadata"][0]
                sampled = len(meta["frames_indices"])
                frontmatter_lines.append(f"sampled_frames: {sampled}")
            # Generation stats from GenerationResult
            if hasattr(response, "prompt_tokens"):
                frontmatter_lines.append(f"prompt_tokens: {response.prompt_tokens}")
                frontmatter_lines.append(f"generation_tokens: {response.generation_tokens}")
                frontmatter_lines.append(f"prompt_tps: \"{response.prompt_tps:.1f}\"")
                frontmatter_lines.append(f"generation_tps: \"{response.generation_tps:.1f}\"")
            frontmatter_lines.append(f"generation_time: \"{gen_elapsed:.1f}s\"")
            if sampled > 0:
                frontmatter_lines.append(f"time_per_frame: \"{gen_elapsed / sampled:.2f}s\"")
            if video_duration > 0:
                frontmatter_lines.append(f"realtime_factor: \"{gen_elapsed / video_duration:.2f}x\"")
            peak_mem = mx.metal.get_peak_memory() / 1e9
            if hasattr(response, "peak_memory") and response.peak_memory > peak_mem:
                peak_mem = response.peak_memory / 1e9
            frontmatter_lines.append(f"peak_memory: \"{peak_mem:.2f} GB\"")
            frontmatter_lines.append("---")
            response_text = "\n".join(frontmatter_lines) + "\n\n" + response_text
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(response_text)
        logger.info(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
