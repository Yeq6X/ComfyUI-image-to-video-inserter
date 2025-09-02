# insert_image_to_video.py

動画の指定フレームに画像を挿入し、自動的にフェードイン・フェードアウト効果を適用するPythonモジュール。

## 主な機能

- **複数画像の一括挿入**: 複数の画像を異なるフレームに同時挿入
- **自動フェード処理**: フェードイン・フェードアウトの自動計算
- **重なり検出と合成**: 複数画像が重なった場合の自動ブレンド
- **アルファチャンネル対応**: PNG画像の透明度を活用
- **柔軟な入力形式**: 動画ファイル、フレーム配列、新規作成に対応

## 基本的な使用方法

### 1. 動画ファイルから処理

```python
from insert_image_to_video import load_frames_from_video, insert_multiple_images_to_frames, save_frames_to_video

# 動画を読み込み
frames, fps, width, height = load_frames_from_video("input.mp4")

# 画像を挿入
result = insert_multiple_images_to_frames(
    frames,
    [(30, "image1.png"), (50, "image2.png")]  # フレーム30とフレーム50に挿入
)

# 動画として保存
save_frames_to_video(result, "output.mp4", fps)
```

### 2. 新規動画作成

```python
from insert_image_to_video import create_blank_frames, insert_multiple_images_to_frames, save_frames_to_video

# 黒い背景の動画フレームを作成
frames = create_blank_frames(1920, 1080, 100)  # 1920x1080, 100フレーム

# 画像を挿入
result = insert_multiple_images_to_frames(
    frames,
    [(25, "logo.png"), (75, "title.png")]
)

# 動画として保存
save_frames_to_video(result, "generated.mp4", fps=24)
```

## 高度な設定

### フェード設定

```python
result = insert_multiple_images_to_frames(
    frames,
    image_insertions=[(50, "image.png")],
    fade_width=3,        # ±3フレームでフェード
    min_opacity=0.0      # 完全透明まで下がる (0.0-1.0)
)
```

### 複数画像の重なり処理

重なった画像は`calculate_blend_custom`関数によって自動的に合成されます：

- **1.0の重みを持つ画像**: 優先表示
- **1.0未満の重み**: 背景との混合比率で表示
- **複数画像の重なり**: 重み比率に応じて自動配分

## 主要関数

### コア関数

- `insert_multiple_images_to_frames()`: メイン処理関数
- `build_timeline()`: フレームごとのタイムライン構築
- `calculate_blend_custom()`: 複数画像の重み計算
- `apply_layers_to_frame()`: レイヤー合成処理

### ユーティリティ関数

- `create_blank_frames()`: 新規フレーム配列作成
- `load_frames_from_video()`: 動画読み込み
- `save_frames_to_video()`: 動画保存
- `center_crop_and_resize()`: 画像リサイズ

## パラメータ詳細

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `fade_width` | 2 | フェードイン・アウトのフレーム幅 |
| `min_opacity` | 0.0 | フェード時の最小透明度 (0.0=完全透明, 1.0=不透明) |
| `blend_mode` | "auto" | ブレンドモード ("auto", "additive", "override") |

## フェード計算の例

**fade_width=2, min_opacity=0.0の場合:**

```
フレーム48: weight = 0.0  (完全透明)
フレーム49: weight = 0.5  (50%透明)
フレーム50: weight = 1.0  (完全不透明) ← メインフレーム
フレーム51: weight = 0.5  (50%透明)
フレーム52: weight = 0.0  (完全透明)
```

## 対応形式

- **入力画像**: PNG (アルファチャンネル対応), JPG
- **動画形式**: MP4 (OpenCV対応形式)
- **フレーム形式**: numpy配列 (BGR, BGRA)

## エラーハンドリング

- 存在しないファイルパスのチェック
- フレームインデックス範囲外の検出
- 画像サイズの自動調整
- アルファチャンネルの有無を自動判定

## 使用例とテスト

`test_multiple_images.py`には15種類のエッジケースを含むテストが含まれています：

```bash
# 通常実行
python test_multiple_images.py

# デバッグ用（低FPS）
python test_multiple_images.py --fps 6
```

## 依存関係

- OpenCV (`cv2`)
- NumPy
- Python 3.6+

## 注意事項

- メモリ使用量は動画サイズとフレーム数に比例します
- 大量の画像挿入時はメモリ不足に注意
- アルファチャンネル付き画像の処理は若干重くなります