import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Union, Optional, Dict

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

class MultiImageInserter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fade_width": ("INT", {"default": 2, "min": 0, "max": 20}),
                "min_opacity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "blend_mode": (["auto", "additive", "override"], {"default": "auto"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "frame_1": ("INT", {"default": 10, "min": 0, "max": 10000}),
                "image_2": ("IMAGE",),
                "frame_2": ("INT", {"default": 20, "min": 0, "max": 10000}),
                "image_3": ("IMAGE",),
                "frame_3": ("INT", {"default": 30, "min": 0, "max": 10000}),
                "image_4": ("IMAGE",),
                "frame_4": ("INT", {"default": 40, "min": 0, "max": 10000}),
                "image_5": ("IMAGE",),
                "frame_5": ("INT", {"default": 50, "min": 0, "max": 10000}),
                "background_frames": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_frames",)
    FUNCTION = "insert_images"
    CATEGORY = "Video/Frames"

    def insert_images(self, frames, fade_width, min_opacity, blend_mode, **kwargs):
        # フレームをnumpy配列に変換
        frame_list = tensor_to_frames(frames)
        
        # 画像挿入リストを構築
        image_insertions = []
        for i in range(1, 6):  # image_1 から image_5 まで
            image_key = f"image_{i}"
            frame_key = f"frame_{i}"
            
            if image_key in kwargs and kwargs[image_key] is not None:
                frame_idx = kwargs.get(frame_key, i * 10)
                
                # ComfyUIテンソルをnumpy配列に変換
                image_tensor = kwargs[image_key]
                if image_tensor.dim() == 4:
                    image_tensor = image_tensor.squeeze(0)
                
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
                
                image_insertions.append((frame_idx, bgra_image))
        
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

# ComfyUIノード登録
NODE_CLASS_MAPPINGS = {
    "CreateBlankFrames": CreateBlankFrames,
    "MultiImageInserter": MultiImageInserter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateBlankFrames": "Create Blank Frames",
    "MultiImageInserter": "Multi Image Inserter",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']