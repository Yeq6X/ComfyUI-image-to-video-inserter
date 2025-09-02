import torch
import numpy as np
from PIL import Image
import cv2
import base64
import io
import tempfile
import os
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
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            }
        }
        
        return inputs

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "select_images_and_frames"
    CATEGORY = "Video/Frames"

    def select_images_and_frames(self, inputcount, output_type, **kwargs):
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
            # tensor形式で返す（従来通り、サイズを揃えてパディング）
            if images:
                # 最大サイズを取得
                max_height = max(img.shape[1] for img in images)
                max_width = max(img.shape[2] for img in images)
                
                # 全ての画像を同じサイズにパディング
                padded_images = []
                for img in images:
                    # 現在のサイズ
                    batch, height, width, channels = img.shape
                    
                    # パディングが必要な場合
                    if height != max_height or width != max_width:
                        # パディングサイズを計算（中央配置）
                        pad_height = max_height - height
                        pad_width = max_width - width
                        pad_top = pad_height // 2
                        pad_bottom = pad_height - pad_top
                        pad_left = pad_width // 2
                        pad_right = pad_width - pad_left
                        
                        # パディング実行（黒で埋める）
                        padded_img = torch.nn.functional.pad(
                            img.permute(0, 3, 1, 2),  # BHWC -> BCHW
                            (pad_left, pad_right, pad_top, pad_bottom),
                            mode='constant', 
                            value=0
                        ).permute(0, 2, 3, 1)  # BCHW -> BHWC
                        
                        padded_images.append(padded_img)
                    else:
                        padded_images.append(img)
                
                # 結合
                combined_images = torch.cat(padded_images, dim=0)
                return (combined_images,)
            else:
                # 空の場合はダミー画像を作成
                combined_images = torch.zeros((1, 64, 64, 3))
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
            
            # OpenCVでビデオライターを初期化
            fourcc_map = {
                "mp4": cv2.VideoWriter_fourcc(*'mp4v'),
                "webm": cv2.VideoWriter_fourcc(*'VP80'),
                "avi": cv2.VideoWriter_fourcc(*'XVID')
            }
            
            fourcc = fourcc_map.get(output_format, cv2.VideoWriter_fourcc(*'mp4v'))
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

# ComfyUIノード登録
NODE_CLASS_MAPPINGS = {
    "CreateBlankFrames": CreateBlankFrames,
    "ImageFrameSelector": ImageFrameSelector,
    "MultiImageInserter": MultiImageInserter,
    "ImagesToBase64Video": ImagesToBase64Video,
    "Base64ListToImages": Base64ListToImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateBlankFrames": "Create Blank Frames",
    "ImageFrameSelector": "Image Frame Selector",
    "MultiImageInserter": "Multi Image Inserter",
    "ImagesToBase64Video": "Images to Base64 Video",
    "Base64ListToImages": "Base64 List to Images",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']