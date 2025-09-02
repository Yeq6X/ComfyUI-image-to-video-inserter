import cv2
import numpy as np
import argparse
import os
from pathlib import Path


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
    
    # フレームインデックスの妥当性チェック
    for idx in frame_blend_dict.keys():
        if idx < 0 or idx >= total_frames:
            raise ValueError(f"フレームインデックスが範囲外です: {idx} (範囲: 0-{total_frames-1})")
    
    # PNG画像を読み込む（アルファチャンネル付き）
    insert_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if insert_img is None:
        raise ValueError(f"画像ファイルを読み込めません: {image_path}")
    
    # Center cropして動画フレームサイズに合わせる
    insert_img = center_crop_and_resize(insert_img, width, height)
    
    # アルファチャンネルがない場合はBGRに変換
    if insert_img.shape[2] == 4:
        # アルファチャンネルありの場合
        bgr_img = insert_img[:, :, :3]
        alpha = insert_img[:, :, 3] / 255.0
    else:
        # アルファチャンネルなしの場合
        bgr_img = insert_img
        alpha = None
    
    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print(f"処理中: 総フレーム数 {total_frames}")
    print(f"フレーム挿入設定:")
    for idx, blend in sorted(frame_blend_dict.items()):
        print(f"  フレーム {idx}: ブレンド率 {blend * 100:.1f}%")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count in frame_blend_dict:
            blend_ratio = frame_blend_dict[frame_count]
            # 指定フレームの場合、PNG画像をブレンド
            if alpha is not None:
                # アルファチャンネルとブレンド率を組み合わせる
                effective_alpha = alpha * blend_ratio
                for c in range(3):
                    frame[:, :, c] = effective_alpha * bgr_img[:, :, c] + (1 - effective_alpha) * frame[:, :, c]
            else:
                # アルファチャンネルがない場合、ブレンド率のみで合成
                frame = cv2.addWeighted(bgr_img, blend_ratio, frame, 1 - blend_ratio, 0)
            
            print(f"フレーム {frame_count} に画像を挿入しました (ブレンド率: {blend_ratio * 100:.1f}%)")
        
        out.write(frame)
        frame_count += 1
        
        # 進捗表示
        if frame_count % 100 == 0:
            print(f"処理済み: {frame_count}/{total_frames} フレーム")
    
    # クリーンアップ
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"処理完了: {output_path} に保存しました")


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