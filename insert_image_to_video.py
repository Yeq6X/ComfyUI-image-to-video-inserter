import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional


def calculate_blend_custom(A: float, B: float) -> Tuple[float, float, float]:
    """
    2つの画像の重みから背景・画像A・画像Bのブレンド率を計算
    
    Args:
        A: 画像Aの重み (0.0-1.0)
        B: 画像Bの重み (0.0-1.0)
    
    Returns:
        (background_weight, A_weight, B_weight) のタプル
    """
    max_val = max(A, B)
    
    # 1.0なら完全優先
    if max_val >= 1.0:
        if A >= 1.0 and B >= 1.0:
            return (0.0, 0.5, 0.5)
        elif A >= 1.0:
            return (0.0, 1.0, 0.0)
        else:
            return (0.0, 0.0, 1.0)
    
    # 背景の計算（カスタム式）
    if max_val <= 0.5:
        # 0.5以下：背景多め（約30%）
        background = 0.3 * (1 - max_val * 0.6)
    elif max_val <= 0.8:
        # 0.5-0.8：急激に減少
        background = 0.3 * ((0.8 - max_val) / 0.3) ** 2
    else:
        # 0.8-1.0：ほぼゼロに
        background = 0.01 * ((1.0 - max_val) / 0.2) ** 3
    
    # 残りを配分
    remaining = 1.0 - background
    total = A + B
    
    if total > 0:
        A_weight = remaining * (A / total)
        B_weight = remaining * (B / total)
    else:
        A_weight = B_weight = 0.0
    
    return (background, A_weight, B_weight)


def create_blank_frames(width, height, num_frames, color=(0, 0, 0)):
    """
    指定サイズ・フレーム数で塗りつぶし色のフレーム配列を作成する
    
    Args:
        width: フレーム幅
        height: フレーム高さ
        num_frames: フレーム数
        color: 塗りつぶし色 (B, G, R) デフォルトは黒
    
    Returns:
        フレームのリスト
    """
    frames = []
    # BGRフォーマットのフレームを作成
    blank_frame = np.full((height, width, 3), color, dtype=np.uint8)
    
    for _ in range(num_frames):
        # 各フレームは独立したコピーとして作成
        frames.append(blank_frame.copy())
    
    return frames


def center_crop_and_resize(image, target_width, target_height):
    """
    画像をcenter cropして指定サイズにリサイズする
    
    Args:
        image: 入力画像（numpy array）
        target_width: 目標幅
        target_height: 目標高さ
    
    Returns:
        リサイズされた画像
    """
    h, w = image.shape[:2]
    
    # アスペクト比を計算
    target_aspect = target_width / target_height
    image_aspect = w / h
    
    if image_aspect > target_aspect:
        # 画像が横長の場合、高さに合わせてリサイズ
        new_height = target_height
        new_width = int(target_height * image_aspect)
        resized = cv2.resize(image, (new_width, new_height))
        
        # 左右をクロップ
        crop_x = (new_width - target_width) // 2
        cropped = resized[:, crop_x:crop_x + target_width]
    else:
        # 画像が縦長の場合、幅に合わせてリサイズ
        new_width = target_width
        new_height = int(target_width / image_aspect)
        resized = cv2.resize(image, (new_width, new_height))
        
        # 上下をクロップ
        crop_y = (new_height - target_height) // 2
        cropped = resized[crop_y:crop_y + target_height, :]
    
    return cropped


def process_frames_with_image(frames, insert_img, frame_blend_dict, verbose=True):
    """
    フレーム配列の指定したフレームに画像を挿入する
    
    Args:
        frames: フレームのリストまたは配列
        insert_img: 挿入する画像（numpy array, BGRまたはBGRA形式）
        frame_blend_dict: フレームインデックスとブレンド率の辞書 {frame_index: blend_ratio}
        verbose: 詳細出力を表示するか
    
    Returns:
        処理されたフレームのリスト
    """
    processed_frames = []
    total_frames = len(frames)
    
    # フレームインデックスの妥当性チェック
    for idx in frame_blend_dict.keys():
        if idx < 0 or idx >= total_frames:
            raise ValueError(f"フレームインデックスが範囲外です: {idx} (範囲: 0-{total_frames-1})")
    
    # フレームサイズを取得
    if len(frames) == 0:
        raise ValueError("フレーム配列が空です")
    
    height, width = frames[0].shape[:2]
    
    # 画像をフレームサイズに合わせる
    insert_img_resized = center_crop_and_resize(insert_img, width, height)
    
    # アルファチャンネルの処理
    if insert_img_resized.shape[2] == 4:
        bgr_img = insert_img_resized[:, :, :3]
        alpha = insert_img_resized[:, :, 3] / 255.0
    else:
        bgr_img = insert_img_resized
        alpha = None
    
    if verbose:
        print(f"処理中: 総フレーム数 {total_frames}")
        print(f"フレーム挿入設定:")
        for idx, blend in sorted(frame_blend_dict.items()):
            print(f"  フレーム {idx}: ブレンド率 {blend * 100:.1f}%")
    
    # 各フレームを処理
    for frame_count, frame in enumerate(frames):
        frame_copy = frame.copy()
        
        if frame_count in frame_blend_dict:
            blend_ratio = frame_blend_dict[frame_count]
            # 指定フレームの場合、画像をブレンド
            if alpha is not None:
                # アルファチャンネルとブレンド率を組み合わせる
                effective_alpha = alpha * blend_ratio
                for c in range(3):
                    frame_copy[:, :, c] = effective_alpha * bgr_img[:, :, c] + (1 - effective_alpha) * frame_copy[:, :, c]
            else:
                # アルファチャンネルがない場合、ブレンド率のみで合成
                frame_copy = cv2.addWeighted(bgr_img, blend_ratio, frame_copy, 1 - blend_ratio, 0)
            
            if verbose:
                print(f"フレーム {frame_count} に画像を挿入しました (ブレンド率: {blend_ratio * 100:.1f}%)")
        
        processed_frames.append(frame_copy)
        
        # 進捗表示
        if verbose and frame_count % 100 == 0 and frame_count > 0:
            print(f"処理済み: {frame_count}/{total_frames} フレーム")
    
    return processed_frames


def insert_image_to_video(video_path, image_path, frame_blend_dict, output_path):
    """
    MP4動画の指定したフレームにPNG画像を挿入する
    
    Args:
        video_path: 入力動画ファイルパス
        image_path: 挿入するPNG画像ファイルパス
        frame_blend_dict: フレームインデックスとブレンド率の辞書 {frame_index: blend_ratio}
        output_path: 出力動画ファイルパス
    """
    # 動画を開く
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"動画ファイルを開けません: {video_path}")
    
    # 動画のプロパティを取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # PNG画像を読み込む（アルファチャンネル付き）
    insert_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if insert_img is None:
        raise ValueError(f"画像ファイルを読み込めません: {image_path}")
    
    # 動画からすべてのフレームを読み込む
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    # フレーム処理を実行
    processed_frames = process_frames_with_image(frames, insert_img, frame_blend_dict)
    
    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 処理済みフレームを書き込む
    for frame in processed_frames:
        out.write(frame)
    
    out.release()
    cv2.destroyAllWindows()
    
    print(f"処理完了: {output_path} に保存しました")


def build_timeline(
    image_insertions: List[Tuple[int, Union[str, np.ndarray]]],
    fade_width: int,
    total_frames: int
) -> Dict[int, List[Dict]]:
    """
    画像挿入情報からフレームごとのタイムラインを構築
    
    Args:
        image_insertions: [(frame_idx, image_path_or_array), ...]
        fade_width: フェードイン/アウトのフレーム幅
        total_frames: 総フレーム数
    
    Returns:
        {frame_idx: [{"image": img, "weight": weight}, ...]} の辞書
    """
    timeline = {}
    
    for frame_idx, image in image_insertions:
        # 画像を読み込み（パスの場合）
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"画像ファイルを読み込めません: {image}")
        else:
            img = image
        
        # フェード範囲を計算
        fade_in_start = max(0, frame_idx - fade_width)
        fade_out_end = min(total_frames - 1, frame_idx + fade_width)
        
        # 各フレームに重みを設定
        for f in range(fade_in_start, fade_out_end + 1):
            if f not in timeline:
                timeline[f] = []
            
            # 重みを計算
            if f < frame_idx:
                # フェードイン
                weight = 0.5 + 0.5 * (f - fade_in_start) / fade_width if fade_width > 0 else 1.0
            elif f > frame_idx:
                # フェードアウト
                weight = 0.5 + 0.5 * (fade_out_end - f) / fade_width if fade_width > 0 else 1.0
            else:
                # メインフレーム
                weight = 1.0
            
            timeline[f].append({"image": img, "weight": weight})
    
    return timeline


def calculate_multi_blend(layers: List[Dict]) -> Dict:
    """
    複数レイヤーの重みを計算
    
    Args:
        layers: [{"image": img, "weight": weight}, ...]
    
    Returns:
        {"background": bg_weight, "layers": [layer_weights...]}
    """
    if len(layers) == 0:
        return {"background": 1.0, "layers": []}
    elif len(layers) == 1:
        weight = layers[0]["weight"]
        if weight >= 1.0:
            return {"background": 0.0, "layers": [1.0]}
        else:
            # 単一画像の場合、背景とのブレンド
            return {"background": 1.0 - weight, "layers": [weight]}
    elif len(layers) == 2:
        # 2画像の場合：calculate_blend_custom使用
        A_val = layers[0]["weight"]
        B_val = layers[1]["weight"]
        bg_w, A_w, B_w = calculate_blend_custom(A_val, B_val)
        return {"background": bg_w, "layers": [A_w, B_w]}
    else:
        # 3画像以上：最大値を見つけて処理
        weights = [l["weight"] for l in layers]
        max_weight = max(weights)
        
        if max_weight >= 1.0:
            # 1.0のものだけを均等配分
            result_weights = []
            count_max = sum(1 for w in weights if w >= 1.0)
            for w in weights:
                if w >= 1.0:
                    result_weights.append(1.0 / count_max)
                else:
                    result_weights.append(0.0)
            return {"background": 0.0, "layers": result_weights}
        else:
            # 全て1.0未満：正規化して配分
            total = sum(weights)
            if total > 0:
                normalized = [w / total for w in weights]
                # 背景を計算（最大値に基づく）
                bg_weight = (1 - max_weight) ** 10  # 簡易版
                remaining = 1.0 - bg_weight
                return {"background": bg_weight, "layers": [w * remaining for w in normalized]}
            else:
                return {"background": 1.0, "layers": [0.0] * len(layers)}


def apply_layers_to_frame(
    base_frame: np.ndarray,
    layers_info: List[Dict],
    background_frame: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    フレームに複数レイヤーを適用
    
    Args:
        base_frame: ベースフレーム
        layers_info: レイヤー情報のリスト
        background_frame: 背景フレーム（Noneの場合はbase_frameを使用）
    
    Returns:
        合成されたフレーム
    """
    if not layers_info:
        return base_frame
    
    if background_frame is None:
        background_frame = base_frame
    
    # ブレンド重みを計算
    blend_weights = calculate_multi_blend(layers_info)
    
    height, width = base_frame.shape[:2]
    result = np.zeros_like(base_frame, dtype=np.float32)
    
    # 背景を追加
    if blend_weights["background"] > 0:
        result += background_frame.astype(np.float32) * blend_weights["background"]
    
    # 各レイヤーを追加
    for layer_info, weight in zip(layers_info, blend_weights["layers"]):
        if weight > 0:
            img = layer_info["image"]
            # 画像をリサイズ
            img_resized = center_crop_and_resize(img, width, height)
            
            # アルファチャンネル処理
            if img_resized.shape[2] == 4:
                bgr = img_resized[:, :, :3].astype(np.float32)
                alpha = img_resized[:, :, 3].astype(np.float32) / 255.0
                # アルファと重みを組み合わせる
                for c in range(3):
                    result[:, :, c] += bgr[:, :, c] * alpha * weight
            else:
                result += img_resized.astype(np.float32) * weight
    
    return np.clip(result, 0, 255).astype(np.uint8)


def load_frames_from_video(video_path: str) -> Tuple[List[np.ndarray], int, int, int]:
    """
    動画ファイルからフレームを読み込む
    
    Args:
        video_path: 動画ファイルパス
    
    Returns:
        (frames, fps, width, height) のタプル
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"動画ファイルを開けません: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps, width, height


def save_frames_to_video(frames: List[np.ndarray], output_path: str, fps: int = 30) -> None:
    """
    フレーム配列を動画ファイルとして保存
    
    Args:
        frames: フレーム配列
        output_path: 出力動画ファイルパス
        fps: フレームレート
    """
    if not frames:
        raise ValueError("フレーム配列が空です")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    cv2.destroyAllWindows()


def insert_multiple_images_to_frames(
    frames: List[np.ndarray],
    image_insertions: List[Tuple[int, Union[str, np.ndarray]]],
    fade_width: int = 1,
    blend_mode: str = "auto",
    background_frames: Optional[List[np.ndarray]] = None,
    verbose: bool = True
) -> List[np.ndarray]:
    """
    複数画像を自動的にフェード付きで挿入
    
    Args:
        frames: フレーム配列
        image_insertions: [(frame_idx, image_path_or_array), ...]
        fade_width: フェードイン/アウトのフレーム幅
        blend_mode: "auto"（calculate_blend_custom使用）, "additive", "override"
        background_frames: 背景フレーム配列（オプション）
        verbose: 詳細出力
    
    Returns:
        処理されたフレームのリスト
    """
    if not frames:
        raise ValueError("フレーム配列が空です")
    
    total_frames = len(frames)
    
    if background_frames is None:
        background_frames = frames
    
    # タイムライン構築
    timeline = build_timeline(image_insertions, fade_width, total_frames)
    
    if verbose:
        print(f"処理中: 総フレーム数 {total_frames}")
        print(f"画像挿入数: {len(image_insertions)}")
        print(f"フェード幅: ±{fade_width}フレーム")
        print(f"ブレンドモード: {blend_mode}")
        
        # 重なり検出
        overlaps = [f for f, layers in timeline.items() if len(layers) > 1]
        if overlaps:
            print(f"重なりフレーム: {overlaps}")
    
    # フレーム処理
    result_frames = []
    for frame_idx, frame in enumerate(frames):
        if frame_idx in timeline:
            # レイヤー適用
            if blend_mode == "auto":
                processed_frame = apply_layers_to_frame(
                    frame,
                    timeline[frame_idx],
                    background_frames[frame_idx]
                )
            else:
                # TODO: 他のブレンドモード実装
                processed_frame = apply_layers_to_frame(
                    frame,
                    timeline[frame_idx],
                    background_frames[frame_idx]
                )
            result_frames.append(processed_frame)
            
            if verbose and frame_idx % 10 == 0:
                layers_count = len(timeline[frame_idx])
                if layers_count > 0:
                    print(f"フレーム {frame_idx}: {layers_count}枚の画像を合成")
        else:
            result_frames.append(frame.copy())
    
    if verbose:
        print(f"処理完了: {len(result_frames)}フレーム")
    
    return result_frames


def parse_frame_indices(indices_str):
    """
    フレームインデックスの文字列をパースして辞書に変換
    例: "1,5,10" -> {1: 1.0, 5: 1.0, 10: 1.0}
    例: "1-5" -> {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
    例: "1-5,10(0.5),15-17(0.3)" -> {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 10: 0.5, 15: 0.3, 16: 0.3, 17: 0.3}
    """
    frame_blend_dict = {}
    parts = indices_str.split(',')
    
    for part in parts:
        part = part.strip()
        blend_ratio = 1.0  # デフォルトは100%
        
        # ブレンド率が指定されているか確認
        if '(' in part and ')' in part:
            idx_part, blend_part = part.split('(')
            blend_ratio = float(blend_part.rstrip(')'))
            part = idx_part
        
        # 範囲指定か単一指定かを判定
        if '-' in part:
            # 範囲指定の場合
            start, end = part.split('-')
            for frame_idx in range(int(start), int(end) + 1):
                frame_blend_dict[frame_idx] = blend_ratio
        else:
            # 単一の値
            frame_blend_dict[int(part)] = blend_ratio
    
    return frame_blend_dict


def main():
    parser = argparse.ArgumentParser(description='MP4動画の指定フレームにPNG画像を挿入')
    parser.add_argument('video', help='入力MP4動画ファイルパス')
    parser.add_argument('image', help='挿入するPNG画像ファイルパス')
    parser.add_argument('frame_indices', help='画像を挿入するフレームインデックス（例: "5,10(0.5),15-20(0.3)" - 5は100%、10は50%、15-20は30%でブレンド）')
    parser.add_argument('-o', '--output', help='出力動画ファイルパス（デフォルト: output_[元のファイル名].mp4）')
    parser.add_argument('--overlay', action='store_true', help='画像をオーバーレイ（透過合成）する')
    
    args = parser.parse_args()
    
    # フレームインデックスをパース
    try:
        frame_blend_dict = parse_frame_indices(args.frame_indices)
    except ValueError as e:
        print(f"エラー: フレームインデックスの形式が不正です: {args.frame_indices}")
        print("形式例: \"5,10(0.5),15-20(0.3)\"")
        return
    
    # 出力ファイル名の生成
    if args.output:
        output_path = args.output
    else:
        video_name = Path(args.video).stem
        output_path = f"output_{video_name}.mp4"
    
    # ファイル存在チェック
    if not os.path.exists(args.video):
        print(f"エラー: 動画ファイルが見つかりません: {args.video}")
        return
    
    if not os.path.exists(args.image):
        print(f"エラー: 画像ファイルが見つかりません: {args.image}")
        return
    
    try:
        insert_image_to_video(args.video, args.image, frame_blend_dict, output_path)
    except Exception as e:
        print(f"エラー: {e}")
        return


if __name__ == "__main__":
    main()