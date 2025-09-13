import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import base64
import io
import tempfile
import os
import logging
from typing import List, Tuple, Union, Optional, Dict

# ロガーの設定
log = logging.getLogger(__name__)

# 既存のinsert_image_to_video.pyから必要な関数をインポート
from .insert_image_to_video import (
    create_blank_frames,
    insert_multiple_images_to_frames,
    calculate_blend_custom,
    build_timeline,
    apply_layers_to_frame
)

def tensor_to_pil(tensor):
    """ComfyUIのテンソル形式をPIL Imageに変換"""
    # tensor shape: [B, H, W, C] (0-1 range)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 0-1 range to 0-255
    numpy_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(numpy_image)

def pil_to_tensor(pil_image):
    """PIL ImageをComfyUIのテンソル形式に変換"""
    numpy_image = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(numpy_image)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    return tensor

def frames_to_tensor(frames):
    """numpy配列のフレームリストをComfyUIテンソルに変換"""
    if not frames:
        return torch.zeros((1, 64, 64, 3))
    
    # フレームをRGBに変換（OpenCVはBGR）
    rgb_frames = []
    for frame in frames:
        if frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame[:, :, :3]  # アルファチャンネルを除去
        rgb_frames.append(rgb_frame)
    
    # numpy配列に変換して正規化
    frames_array = np.stack(rgb_frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(frames_array)

def tensor_to_frames(tensor):
    """ComfyUIテンソルをnumpy配列のフレームリストに変換"""
    # tensor shape: [B, H, W, C] (0-1 range)
    numpy_frames = (tensor.cpu().numpy() * 255).astype(np.uint8)
    
    bgr_frames = []
    for frame in numpy_frames:
        # RGBからBGRに変換（OpenCVで使用するため）
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr_frames.append(bgr_frame)
    
    return bgr_frames

def base64_to_pil_images(base64_strings):
    """base64文字列のリストをPIL Imageのリストに変換"""
    pil_images = []
    for b64_string in base64_strings:
        try:
            # base64デコード
            image_data = base64.b64decode(b64_string)
            # PIL Imageに変換
            pil_image = Image.open(io.BytesIO(image_data))
            # RGBに変換（必要に応じて）
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            pil_images.append(pil_image)
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            # エラーの場合はダミー画像を作成
            pil_images.append(Image.new('RGB', (64, 64), color=(0, 0, 0)))
    
    return pil_images

class CreateBlankFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 640, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 416, "min": 64, "max": 4096, "step": 8}),
                "frame_count": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "color_r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "color_g": ("INT", {"default": 0, "min": 0, "max": 255}),
                "color_b": ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "create_frames"
    CATEGORY = "Video/Frames"

    def create_frames(self, width, height, frame_count, color_r, color_g, color_b):
        # BGRフォーマットでフレーム作成（OpenCV用）
        color_bgr = (color_b, color_g, color_r)
        frames = create_blank_frames(width, height, frame_count, color_bgr)
        
        # ComfyUIテンソルに変換
        tensor_frames = frames_to_tensor(frames)
        
        return (tensor_frames,)

class ImageFrameSelector:
    @classmethod
    def INPUT_TYPES(s):
        # 基本の必須入力
        inputs = {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 1, "max": 1000, "step": 1}),
                "output_type": (["list", "tensor"], {"default": "list"}),
                # tensor出力時のリサイズ・パディングモード
                "resize_mode": ([
                    "pad",              # リサイズなし中央パディング
                    "fit_short_side",   # 最大サイズ内に収まるよう縮小（レターボックス）
                    "cover_fill"        # 全体を埋めるよう拡大（センタークロップ）
                ], {"default": "pad"}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            }
        }
        
        return inputs

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "select_images_and_frames"
    CATEGORY = "Video/Frames"

    def select_images_and_frames(self, inputcount, output_type, resize_mode, **kwargs):
        images = []
        
        for i in range(1, inputcount + 1):
            image_key = f"image_{i}"
            
            # 画像が存在する場合のみ処理
            if image_key in kwargs and kwargs[image_key] is not None:
                images.append(kwargs[image_key])
        
        # 出力タイプに応じて処理
        if output_type == "list":
            # リスト形式で返す（サイズ違いもそのまま）
            if images:
                return (images,)
            else:
                # 空の場合はダミー画像リストを作成
                dummy_image = torch.zeros((1, 64, 64, 3))
                return ([dummy_image],)
        else:
            # tensor形式で返す（モード切替: pad / fit_short_side / cover_fill）
            if not images:
                # 空の場合はダミー画像を作成
                combined_images = torch.zeros((1, 64, 64, 3))
                return (combined_images,)

            # 目標（最大）サイズを算出
            target_h = max(img.shape[1] for img in images)
            target_w = max(img.shape[2] for img in images)

            processed = []
            for img in images:
                b, h, w, c = img.shape  # B,H,W,C

                if resize_mode == "pad":
                    # リサイズなしの中央パディング
                    pad_h = target_h - h
                    pad_w = target_w - w
                    if pad_h == 0 and pad_w == 0:
                        processed.append(img)
                        continue
                    top = max(pad_h // 2, 0)
                    bottom = max(pad_h - top, 0)
                    left = max(pad_w // 2, 0)
                    right = max(pad_w - left, 0)
                    padded = F.pad(
                        img.permute(0, 3, 1, 2),  # BHWC -> BCHW
                        (left, right, top, bottom),
                        mode="constant",
                        value=0.0
                    ).permute(0, 2, 3, 1)
                    processed.append(padded)
                    continue

                # 以降はリサイズあり
                # BCHWにしてinterpolateを使用
                nchw = img.permute(0, 3, 1, 2)

                if resize_mode == "fit_short_side":
                    # 画像を目標サイズに収まるよう縮小（レターボックス）
                    scale = min(target_h / h, target_w / w)
                    new_h = max(1, int(round(h * scale)))
                    new_w = max(1, int(round(w * scale)))
                    resized = F.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)

                    # 中央パディングして目標サイズに合わせる
                    pad_h = target_h - new_h
                    pad_w = target_w - new_w
                    top = max(pad_h // 2, 0)
                    bottom = max(pad_h - top, 0)
                    left = max(pad_w // 2, 0)
                    right = max(pad_w - left, 0)
                    padded = F.pad(resized, (left, right, top, bottom), mode="constant", value=0.0)
                    processed.append(padded.permute(0, 2, 3, 1))  # BCHW->BHWC

                elif resize_mode == "cover_fill":
                    # 画像を目標サイズを覆うよう拡大（センタークロップ）
                    scale = max(target_h / h, target_w / w)
                    new_h = max(1, int(round(h * scale)))
                    new_w = max(1, int(round(w * scale)))
                    resized = F.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)

                    # 余分をセンタークロップして目標サイズに合わせる
                    start_y = max((new_h - target_h) // 2, 0)
                    start_x = max((new_w - target_w) // 2, 0)
                    end_y = start_y + target_h
                    end_x = start_x + target_w
                    cropped = resized[:, :, start_y:end_y, start_x:end_x]
                    processed.append(cropped.permute(0, 2, 3, 1))  # BCHW->BHWC

                else:
                    # 未知のモード（フォールバック: pad）
                    pad_h = target_h - h
                    pad_w = target_w - w
                    top = max(pad_h // 2, 0)
                    bottom = max(pad_h - top, 0)
                    left = max(pad_w // 2, 0)
                    right = max(pad_w - left, 0)
                    padded = F.pad(
                        img.permute(0, 3, 1, 2),
                        (left, right, top, bottom),
                        mode="constant",
                        value=0.0
                    ).permute(0, 2, 3, 1)
                    processed.append(padded)

            # 結合
            combined_images = torch.cat(processed, dim=0)
            return (combined_images,)

class MultiImageInserter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "images": ("IMAGE",),
                "frame_indices": ("STRING", {"default": "10,20,30", "multiline": True}),
                "fade_width": ("INT", {"default": 2, "min": 0, "max": 20}),
                "min_opacity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "blend_mode": (["auto", "additive", "override"], {"default": "auto"}),
            },
            "optional": {
                "background_frames": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_frames",)
    FUNCTION = "insert_images"
    CATEGORY = "Video/Frames"

    def parse_frame_indices(self, indices_string):
        """カンマ区切り文字列をフレームインデックスのリストに変換"""
        if not indices_string.strip():
            return []
        
        indices = []
        parts = indices_string.replace(" ", "").split(",")
        
        for part in parts:
            try:
                indices.append(int(part))
            except ValueError:
                print(f"Invalid frame index: {part}")
        
        return indices

    def insert_images(self, frames, images, frame_indices, fade_width, min_opacity, blend_mode, **kwargs):
        # フレームをnumpy配列に変換
        frame_list = tensor_to_frames(frames)
        
        # フレームインデックスを解析
        indices = self.parse_frame_indices(frame_indices)
        
        # 画像を分割してリストに変換
        image_list = []
        
        # imagesがリストかテンソルかを判定
        if isinstance(images, list):
            # リスト形式の場合
            for image_tensor in images:
                # PIL経由でnumpy配列に変換
                pil_image = tensor_to_pil(image_tensor)
                numpy_image = np.array(pil_image)
                
                # BGRAフォーマットに変換（アルファチャンネル付きで処理）
                if numpy_image.shape[2] == 3:
                    # アルファチャンネルを追加
                    alpha = np.ones((numpy_image.shape[0], numpy_image.shape[1], 1), dtype=np.uint8) * 255
                    numpy_image = np.concatenate([numpy_image, alpha], axis=2)
                
                # RGBAからBGRAに変換
                bgra_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2BGRA)
                image_list.append(bgra_image)
        else:
            # テンソル形式の場合（従来通り）
            if images.dim() == 4:  # バッチ次元がある場合
                for i in range(images.shape[0]):
                    image_tensor = images[i:i+1]
                    
                    # PIL経由でnumpy配列に変換
                    pil_image = tensor_to_pil(image_tensor)
                    numpy_image = np.array(pil_image)
                    
                    # BGRAフォーマットに変換（アルファチャンネル付きで処理）
                    if numpy_image.shape[2] == 3:
                        # アルファチャンネルを追加
                        alpha = np.ones((numpy_image.shape[0], numpy_image.shape[1], 1), dtype=np.uint8) * 255
                        numpy_image = np.concatenate([numpy_image, alpha], axis=2)
                    
                    # RGBAからBGRAに変換
                    bgra_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2BGRA)
                    image_list.append(bgra_image)
        
        # 画像挿入リストを構築
        image_insertions = []
        min_count = min(len(indices), len(image_list))
        for i in range(min_count):
            image_insertions.append((indices[i], image_list[i]))
        
        if not image_insertions:
            # 画像が指定されていない場合、元のフレームを返す
            return (frames,)
        
        # 背景フレーム処理
        background_frame_list = None
        if "background_frames" in kwargs and kwargs["background_frames"] is not None:
            background_frame_list = tensor_to_frames(kwargs["background_frames"])
        
        try:
            # 画像挿入処理を実行
            processed_frames = insert_multiple_images_to_frames(
                frame_list,
                image_insertions,
                fade_width=fade_width,
                min_opacity=min_opacity,
                blend_mode=blend_mode,
                background_frames=background_frame_list,
                verbose=False
            )
            
            # 結果をComfyUIテンソルに変換
            result_tensor = frames_to_tensor(processed_frames)
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"MultiImageInserter Error: {e}")
            # エラーの場合、元のフレームを返す
            return (frames,)

class ImagesToBase64Video:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "output_format": (["mp4", "webm", "avi"], {"default": "mp4"}),
                "video_quality": (["high", "medium", "low"], {"default": "medium"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base64_video",)
    FUNCTION = "convert_to_base64_video"
    CATEGORY = "Video/Export"

    def convert_to_base64_video(self, images, fps, output_format, video_quality):
        try:
            # 一時ファイルを作成
            with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as temp_file:
                temp_video_path = temp_file.name

            # 画像テンソルをnumpy配列に変換
            if isinstance(images, list):
                # リスト形式の場合
                frames = []
                for image_tensor in images:
                    frame = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    frames.append(frame)
            else:
                # テンソル形式の場合
                frames = (images.cpu().numpy() * 255).astype(np.uint8)

            if len(frames) == 0:
                return ("",)

            # 最初のフレームからビデオのサイズを取得
            first_frame = frames[0] if isinstance(frames, list) else frames[0]
            height, width = first_frame.shape[:2]

            # ビデオ品質設定
            quality_settings = {
                "high": {"crf": 18, "preset": "slow"},
                "medium": {"crf": 23, "preset": "medium"},
                "low": {"crf": 28, "preset": "fast"}
            }
            
            # OpenCVでビデオライターを初期化（ブラウザ互換のH.264コーデック）
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264コーデックを使用
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            # H.264が利用できない場合のフォールバック
            if not out.isOpened():
                print("H264 codec not available, trying MJPG...")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEGフォールバック
                out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            # フレームを書き込み
            if isinstance(frames, list):
                for frame in frames:
                    # RGBからBGRに変換（OpenCVはBGR）
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)
            else:
                for i in range(frames.shape[0]):
                    frame = frames[i]
                    # RGBからBGRに変換（OpenCVはBGR）
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)

            out.release()

            # ビデオファイルをbase64にエンコード
            with open(temp_video_path, 'rb') as video_file:
                video_data = video_file.read()
                base64_video = base64.b64encode(video_data).decode('utf-8')

            # 一時ファイルを削除
            os.unlink(temp_video_path)

            return (base64_video,)

        except Exception as e:
            print(f"ImagesToBase64Video Error: {e}")
            return ("",)

class Base64ListToImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_list": ("STRING", {"default": "", "multiline": True}),
                "separator": (["newline", "comma"], {"default": "newline"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "convert_base64_list"
    CATEGORY = "Image/Convert"

    def convert_base64_list(self, base64_list, separator):
        try:
            # 区切り文字で分割
            if separator == "newline":
                base64_strings = [s.strip() for s in base64_list.split('\n') if s.strip()]
            else:  # comma
                base64_strings = [s.strip() for s in base64_list.split(',') if s.strip()]

            if not base64_strings:
                # 空の場合はダミー画像を返す
                dummy_image = torch.zeros((1, 64, 64, 3))
                return ([dummy_image],)

            # Base64文字列を画像テンソルのリストに変換
            image_tensors = []
            for b64_string in base64_strings:
                try:
                    # Data URIの場合はヘッダーを除去
                    if b64_string.startswith('data:'):
                        b64_string = b64_string.split(',', 1)[1]
                    
                    # base64デコード
                    image_data = base64.b64decode(b64_string)
                    
                    # PIL Imageに変換
                    pil_image = Image.open(io.BytesIO(image_data))
                    
                    # RGBに変換（必要に応じて）
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # ComfyUIテンソル形式に変換
                    tensor = pil_to_tensor(pil_image)
                    image_tensors.append(tensor)
                    
                except Exception as e:
                    print(f"Error decoding base64 image: {e}")
                    # エラーの場合はダミー画像を作成
                    dummy_image = torch.zeros((1, 64, 64, 3))
                    image_tensors.append(dummy_image)
            
            # リスト形式で返す（MultiImageInserterがリストを受け入れるため）
            return (image_tensors,)
            
        except Exception as e:
            print(f"Base64ListToImages Error: {e}")
            # エラーの場合はダミー画像を返す
            dummy_image = torch.zeros((1, 64, 64, 3))
            return ([dummy_image],)

class Base64VideoToImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_video": ("STRING", {"default": "", "multiline": True}),
                "frame_step": ("INT", {"default": 1, "min": 1, "max": 100}),
            },
            "optional": {
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "convert_base64_video"
    CATEGORY = "Video/Import"

    def convert_base64_video(self, base64_video, frame_step, **kwargs):
        try:
            max_frames = kwargs.get("max_frames", 0)
            
            if not base64_video.strip():
                # 空の場合はダミー画像を返す
                dummy_image = torch.zeros((1, 64, 64, 3))
                return (dummy_image,)

            # Data URIの場合はヘッダーを除去
            if base64_video.startswith('data:'):
                base64_video = base64_video.split(',', 1)[1]
            
            # base64デコード
            video_data = base64.b64decode(base64_video)
            
            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name
                temp_file.write(video_data)
            
            # OpenCVでビデオを読み込み
            cap = cv2.VideoCapture(temp_video_path)
            
            if not cap.isOpened():
                print("Error: Could not open video file")
                os.unlink(temp_video_path)
                dummy_image = torch.zeros((1, 64, 64, 3))
                return (dummy_image,)
            
            frames = []
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # フレームステップに従ってフレームを抽出
                if frame_count % frame_step == 0:
                    # BGRからRGBに変換
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)
                    extracted_count += 1
                    
                    # 最大フレーム数に達した場合は終了
                    if max_frames > 0 and extracted_count >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            os.unlink(temp_video_path)
            
            if not frames:
                # フレームが取得できない場合はダミー画像を返す
                dummy_image = torch.zeros((1, 64, 64, 3))
                return (dummy_image,)
            
            # ComfyUIテンソル形式に変換
            frames_tensor = frames_to_tensor(frames)
            
            return (frames_tensor,)
            
        except Exception as e:
            print(f"Base64VideoToImages Error: {e}")
            # エラーの場合はダミー画像を返す
            dummy_image = torch.zeros((1, 64, 64, 3))
            return (dummy_image,)

class WanVideoLatentZeroFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "samples": ("LATENT",),
                    "frame_indices": ("STRING", {"default": "0", "tooltip": "Comma-separated frame indices to zero out (e.g., '0,1,2,6-9,12')"}),
                }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "zero_frames"
    CATEGORY = "image/video"
    DESCRIPTION = "Zero out specific frames in latent tensor by indices"

    def parse_indices(self, indices_str, max_frames=None):
        """Parse comma-separated indices string, supporting ranges and negative indices.
        
        Supported formats:
        - Single index: '5'
        - Negative index: '-1' (last frame), '-2' (second to last)
        - Range: '5-10'
        - Range to negative: '5--1' (5 to last), '0--5' (0 to 5th from end)
        - Negative range: '-5--1' (last 5 frames)
        """
        indices = []
        parts = indices_str.replace(" ", "").split(",")
        
        for part in parts:
            if "--" in part:
                # Handle range to negative index (e.g., '5--1', '0--5')
                if max_frames is None:
                    raise ValueError("Cannot use negative indices without knowing total frame count")
                start, neg_end = part.split("--")
                try:
                    if start.startswith("-"):
                        # '-5--1' means from 5th from end to last
                        start_idx = max_frames + int(start)
                    else:
                        # '5--1' means from index 5 to last
                        start_idx = int(start)
                    
                    # Convert negative end to actual index
                    end_idx = max_frames + int("-" + neg_end)
                    
                    if start_idx < 0 or start_idx >= max_frames:
                        raise ValueError(f"Start index {start} is out of bounds for {max_frames} frames")
                    if end_idx < 0 or end_idx >= max_frames:
                        raise ValueError(f"End index -{neg_end} is out of bounds for {max_frames} frames")
                    
                    indices.extend(range(start_idx, end_idx + 1))
                except ValueError as e:
                    raise ValueError(f"Invalid range format: {part}. {str(e)}")
            elif "-" in part and not part.startswith("-"):
                # Handle normal range notation (e.g., '5-10')
                start, end = part.split("-")
                try:
                    start_idx = int(start)
                    end_idx = int(end)
                    indices.extend(range(start_idx, end_idx + 1))
                except ValueError:
                    raise ValueError(f"Invalid range format: {part}")
            elif part.startswith("-"):
                # Handle negative index (e.g., '-1' for last frame)
                if max_frames is None:
                    raise ValueError("Cannot use negative indices without knowing total frame count")
                try:
                    neg_idx = int(part)
                    actual_idx = max_frames + neg_idx
                    if actual_idx < 0 or actual_idx >= max_frames:
                        raise ValueError(f"Negative index {part} is out of bounds for {max_frames} frames")
                    indices.append(actual_idx)
                except ValueError:
                    raise ValueError(f"Invalid negative index format: {part}")
            else:
                # Handle single positive index
                try:
                    indices.append(int(part))
                except ValueError:
                    raise ValueError(f"Invalid index format: {part}")
        
        return sorted(set(indices))  # Remove duplicates and sort

    def zero_frames(self, samples, frame_indices):
        samples = samples.copy()
        latents = samples["samples"].clone()
        
        # Get tensor dimensions (B, C, T, H, W)
        B, C, T, H, W = latents.shape
        
        # Parse frame indices with max_frames
        indices = self.parse_indices(frame_indices, max_frames=T)
        
        # Check if any index is out of bounds
        for idx in indices:
            if idx < 0 or idx >= T:
                raise ValueError(f"Frame index {idx} is out of bounds. Valid range is 0-{T-1}")
        
        # Zero out specified frames
        for idx in indices:
            latents[:, :, idx, :, :] = 0
        
        log.info(f"WanVideoLatentZeroFrames: Zeroed frames at indices {indices} in latent shape {latents.shape}")
        
        return ({"samples": latents, "noise_mask": samples.get("noise_mask")},)

class WanVideoLatentInsertFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "frame_indices": ("STRING", {"default": "0", "tooltip": "Comma-separated frame indices to insert frames (e.g., '0,1,2,6-9,12')"}),
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 100, "step": 1}),
                "latent_1": ("LATENT",),
                "latent_2": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "insert_frames"
    CATEGORY = "image/video"
    DESCRIPTION = """
Insert single-frame latents at specified positions in latent tensor.  
You can set how many inputs the node has,  
with the **inputcount** and clicking update.
"""

    def parse_indices(self, indices_str, max_frames=None):
        """Parse comma-separated indices string, supporting ranges and negative indices.
        
        Supported formats:
        - Single index: '5'
        - Negative index: '-1' (last frame), '-2' (second to last)
        - Range: '5-10'
        - Range to negative: '5--1' (5 to last), '0--5' (0 to 5th from end)
        - Negative range: '-5--1' (last 5 frames)
        """
        indices = []
        parts = indices_str.replace(" ", "").split(",")
        
        for part in parts:
            if "--" in part:
                # Handle range to negative index (e.g., '5--1', '0--5')
                if max_frames is None:
                    raise ValueError("Cannot use negative indices without knowing total frame count")
                start, neg_end = part.split("--")
                try:
                    if start.startswith("-"):
                        # '-5--1' means from 5th from end to last
                        start_idx = max_frames + int(start)
                    else:
                        # '5--1' means from index 5 to last
                        start_idx = int(start)
                    
                    # Convert negative end to actual index
                    end_idx = max_frames + int("-" + neg_end)
                    
                    if start_idx < 0 or start_idx >= max_frames:
                        raise ValueError(f"Start index {start} is out of bounds for {max_frames} frames")
                    if end_idx < 0 or end_idx >= max_frames:
                        raise ValueError(f"End index -{neg_end} is out of bounds for {max_frames} frames")
                    
                    indices.extend(range(start_idx, end_idx + 1))
                except ValueError as e:
                    raise ValueError(f"Invalid range format: {part}. {str(e)}")
            elif "-" in part and not part.startswith("-"):
                # Handle normal range notation (e.g., '5-10')
                start, end = part.split("-")
                try:
                    start_idx = int(start)
                    end_idx = int(end)
                    indices.extend(range(start_idx, end_idx + 1))
                except ValueError:
                    raise ValueError(f"Invalid range format: {part}")
            elif part.startswith("-"):
                # Handle negative index (e.g., '-1' for last frame)
                if max_frames is None:
                    raise ValueError("Cannot use negative indices without knowing total frame count")
                try:
                    neg_idx = int(part)
                    actual_idx = max_frames + neg_idx
                    if actual_idx < 0 or actual_idx >= max_frames:
                        raise ValueError(f"Negative index {part} is out of bounds for {max_frames} frames")
                    indices.append(actual_idx)
                except ValueError:
                    raise ValueError(f"Invalid negative index format: {part}")
            else:
                # Handle single positive index
                try:
                    indices.append(int(part))
                except ValueError:
                    raise ValueError(f"Invalid index format: {part}")
        
        return sorted(set(indices))  # Remove duplicates and sort

    def insert_frames(self, samples, frame_indices, inputcount, **kwargs):
        samples = samples.copy()
        latents = samples["samples"].clone()
        
        # Get tensor dimensions (B, C, T, H, W)
        B, C, T, H, W = latents.shape
        
        # Parse frame indices with max_frames
        indices = self.parse_indices(frame_indices, max_frames=T)
        
        # Check if any index is out of bounds
        for idx in indices:
            if idx < 0 or idx >= T:
                raise ValueError(f"Frame index {idx} is out of bounds. Valid range is 0-{T-1}")
        
        # Collect input latents (following ImageConcatMulti pattern)
        input_latents = []
        for c in range(1, inputcount + 1):
            input_latent = kwargs[f"latent_{c}"]["samples"]
            # Check if input latent has T=1
            if input_latent.shape[2] != 1:
                raise ValueError(f"Input latent_{c} must have exactly 1 frame (T=1), got T={input_latent.shape[2]}")
            # Check dimensions match
            if input_latent.shape[0] != B or input_latent.shape[1] != C or input_latent.shape[3] != H or input_latent.shape[4] != W:
                raise ValueError(f"Input latent_{c} dimensions don't match. Expected B={B}, C={C}, H={H}, W={W}")
            input_latents.append(input_latent)
        
        # Check if we have enough input latents for the specified indices
        if len(indices) > len(input_latents):
            raise ValueError(f"Number of indices ({len(indices)}) exceeds number of input latents ({len(input_latents)})")
        
        # Insert frames at specified positions
        for i, idx in enumerate(indices):
            if i < len(input_latents):
                latents[:, :, idx:idx+1, :, :] = input_latents[i]
        
        log.info(f"WanVideoLatentInsertFrames: Inserted {min(len(indices), len(input_latents))} frames at indices {indices[:min(len(indices), len(input_latents))]} in latent shape {latents.shape}")
        
        return ({"samples": latents, "noise_mask": samples.get("noise_mask")},)

# ComfyUIノード登録
NODE_CLASS_MAPPINGS = {
    "CreateBlankFrames": CreateBlankFrames,
    "ImageFrameSelector": ImageFrameSelector,
    "MultiImageInserter": MultiImageInserter,
    "ImagesToBase64Video": ImagesToBase64Video,
    "Base64ListToImages": Base64ListToImages,
    "Base64VideoToImages": Base64VideoToImages,
    "WanVideoLatentZeroFrames": WanVideoLatentZeroFrames,
    "WanVideoLatentInsertFrames": WanVideoLatentInsertFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateBlankFrames": "Create Blank Frames",
    "ImageFrameSelector": "Image Frame Selector",
    "MultiImageInserter": "Multi Image Inserter",
    "ImagesToBase64Video": "Images to Base64 Video",
    "Base64ListToImages": "Base64 List to Images",
    "Base64VideoToImages": "Base64 Video to Images",
    "WanVideoLatentZeroFrames": "WanVideo Latent Zero Frames",
    "WanVideoLatentInsertFrames": "WanVideo Latent Insert Frames",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
