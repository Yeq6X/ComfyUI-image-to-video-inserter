import os
import sys
import cv2
import numpy as np
from pathlib import Path

# insert_image_to_videoモジュールをインポート
from insert_image_to_video import (
    create_blank_frames,
    insert_multiple_images_to_frames,
    save_frames_to_video
)

def create_test_images():
    """
    テスト用の画像を生成（image1.png, image2.png, image3.png）
    """
    # 画像1: 赤い四角
    img1 = np.zeros((416, 640, 4), dtype=np.uint8)
    img1[:, :, 2] = 255  # Red channel
    img1[100:316, 100:540, 3] = 255  # Alpha channel (中央に四角)
    cv2.imwrite("image1.png", img1)
    
    # 画像2: 緑の円
    img2 = np.zeros((416, 640, 4), dtype=np.uint8)
    img2[:, :, 1] = 255  # Green channel
    center = (320, 208)
    radius = 100
    # 円形のアルファチャンネル
    y, x = np.ogrid[:416, :640]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
    img2[mask, 3] = 255
    cv2.imwrite("image2.png", img2)
    
    # 画像3: 青い三角形
    img3 = np.zeros((416, 640, 4), dtype=np.uint8)
    img3[:, :, 0] = 255  # Blue channel
    # 三角形のポイント
    pts = np.array([[320, 100], [220, 316], [420, 316]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    # アルファチャンネル用のマスク
    mask = np.zeros((416, 640), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    img3[:, :, 3] = mask
    cv2.imwrite("image3.png", img3)
    
    print("テスト用画像を生成しました: image1.png, image2.png, image3.png")


def run_test(test_name, frames, image_insertions, fade_width=1, min_opacity=0.0, fps=24, description=""):
    """
    個別のテストを実行
    """
    print(f"\n{'='*60}")
    print(f"テスト: {test_name}")
    print(f"説明: {description}")
    print(f"画像挿入: {image_insertions}")
    print(f"フェード幅: {fade_width}, 最小透明度: {min_opacity}")
    print('='*60)
    
    try:
        # 画像挿入処理
        result_frames = insert_multiple_images_to_frames(
            frames.copy(),  # フレームのコピーを渡す
            image_insertions,
            fade_width=fade_width,
            min_opacity=min_opacity,
            blend_mode="auto",
            verbose=True
        )
        
        # 出力ファイル名
        output_path = f"test_output/{test_name}.mp4"
        
        # 動画として保存
        save_frames_to_video(result_frames, output_path, fps=fps)
        
        print(f"✓ テスト成功: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ テスト失敗: {e}")
        return False


def main(output_fps=24):
    """
    メインテスト関数
    
    Args:
        output_fps: 出力動画のFPS（デフォルト24、デバッグ用には6など）
    """
    # test_outputディレクトリ作成
    os.makedirs("test_output", exist_ok=True)
    
    # テスト用画像を生成
    create_test_images()
    
    # ベースとなる黒い動画フレームを作成
    print("\nベースフレームを作成中...")
    base_frames = create_blank_frames(640, 416, 100, color=(0, 0, 0))
    print(f"作成完了: 640x416, 100フレーム, 出力FPS={output_fps}")
    
    # テストケース一覧
    test_cases = [
        # 1. 基本的なケース
        {
            "name": "01_single_image",
            "insertions": [(50, "image1.png")],
            "fade_width": 2,
            "min_opacity": 0.0,
            "description": "単一画像の挿入（フレーム50）"
        },
        
        # 1.5. 最小透明度テスト
        {
            "name": "01b_min_opacity_test",
            "insertions": [(50, "image1.png")],
            "fade_width": 3,
            "min_opacity": 0.5,
            "description": "最小透明度テスト（min_opacity=0.5）"
        },
        
        # 1.6. 完全透明テスト
        {
            "name": "01c_full_fade",
            "insertions": [(50, "image1.png")],
            "fade_width": 5,
            "min_opacity": 0.0,
            "description": "完全フェードテスト（min_opacity=0.0）"
        },
        
        # 2. 複数画像（重ならない）
        {
            "name": "02_multiple_no_overlap",
            "insertions": [(20, "image1.png"), (50, "image2.png"), (80, "image3.png")],
            "fade_width": 2,
            "min_opacity": 0.0,
            "description": "複数画像の挿入（重なりなし）"
        },
        
        # 3. 隣接する画像
        {
            "name": "03_adjacent_images",
            "insertions": [(30, "image1.png"), (33, "image2.png")],
            "fade_width": 2,
            "description": "隣接する画像（フェード領域が重なる）"
        },
        
        # 4. 完全に重なる画像
        {
            "name": "04_same_frame",
            "insertions": [(50, "image1.png"), (50, "image2.png")],
            "fade_width": 2,
            "description": "同じフレームに複数画像"
        },
        
        # 5. 連続する画像
        {
            "name": "05_consecutive_frames",
            "insertions": [(40, "image1.png"), (41, "image2.png"), (42, "image3.png")],
            "fade_width": 1,
            "description": "連続するフレームに画像挿入"
        },
        
        # 6. フェード幅0（カット）
        {
            "name": "06_no_fade",
            "insertions": [(30, "image1.png"), (31, "image2.png")],
            "fade_width": 0,
            "description": "フェードなし（ハードカット）"
        },
        
        # 7. 大きなフェード幅
        {
            "name": "07_large_fade",
            "insertions": [(50, "image1.png")],
            "fade_width": 10,
            "description": "大きなフェード幅（±10フレーム）"
        },
        
        # 8. 境界付近（開始）
        {
            "name": "08_near_start",
            "insertions": [(2, "image1.png")],
            "fade_width": 3,
            "description": "動画開始付近での挿入"
        },
        
        # 9. 境界付近（終了）
        {
            "name": "09_near_end",
            "insertions": [(98, "image1.png")],
            "fade_width": 3,
            "description": "動画終了付近での挿入"
        },
        
        # 10. 境界エッジケース
        {
            "name": "10_boundary_edge",
            "insertions": [(0, "image1.png"), (99, "image2.png")],
            "fade_width": 2,
            "description": "最初と最後のフレーム"
        },
        
        # 11. 3つの画像が重なる
        {
            "name": "11_triple_overlap",
            "insertions": [(50, "image1.png"), (51, "image2.png"), (52, "image3.png")],
            "fade_width": 2,
            "description": "3つの画像のフェードが重なる"
        },
        
        # 12. パターン的な配置
        {
            "name": "12_pattern",
            "insertions": [(10, "image1.png"), (20, "image2.png"), (30, "image3.png"),
                          (40, "image1.png"), (50, "image2.png"), (60, "image3.png")],
            "fade_width": 2,
            "description": "パターン的な画像配置"
        },
        
        # 13. 高密度配置
        {
            "name": "13_high_density",
            "insertions": [(45, "image1.png"), (46, "image2.png"), (47, "image3.png"),
                          (48, "image1.png"), (49, "image2.png"), (50, "image3.png")],
            "fade_width": 1,
            "description": "高密度での画像配置"
        },
        
        # 14. フェード幅が異なる組み合わせ（同じfade_widthだが位置による影響をテスト）
        {
            "name": "14_variable_overlap",
            "insertions": [(30, "image1.png"), (34, "image2.png"), (35, "image3.png")],
            "fade_width": 3,
            "description": "様々な重なり具合のテスト"
        },
        
        # 15. ストレステスト
        {
            "name": "15_stress_test",
            "insertions": [(i, f"image{(i % 3) + 1}.png") for i in range(10, 90, 5)],
            "fade_width": 2,
            "description": "多数の画像挿入（5フレームごとに16枚）"
        }
    ]
    
    # テスト実行
    success_count = 0
    failed_count = 0
    
    for test_case in test_cases:
        min_opacity = test_case.get("min_opacity", 0.0)  # デフォルトは0.0
        success = run_test(
            test_case["name"],
            base_frames,
            test_case["insertions"],
            test_case["fade_width"],
            min_opacity,
            output_fps,
            test_case["description"]
        )
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # 結果サマリー
    print("\n" + "="*60)
    print("テスト結果サマリー")
    print("="*60)
    print(f"成功: {success_count}/{len(test_cases)}")
    print(f"失敗: {failed_count}/{len(test_cases)}")
    
    if failed_count == 0:
        print("\n✓ すべてのテストが成功しました！")
    else:
        print(f"\n✗ {failed_count}個のテストが失敗しました")
    
    print(f"\n生成された動画ファイルは test_output/ フォルダに保存されています")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='複数画像挿入機能のテスト')
    parser.add_argument('--fps', type=int, default=24, help='出力動画のFPS（デフォルト24、デバッグ用に6など）')
    
    args = parser.parse_args()
    
    main(output_fps=args.fps)