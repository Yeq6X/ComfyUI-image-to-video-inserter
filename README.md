# ComfyUI Image to Video Inserter

動画に画像を挿入するComfyUIカスタムノード

## ComfyUIへの導入

1. ComfyUIの `custom_nodes` フォルダにこのプロジェクトをクローン
2. ComfyUIを再起動
3. ノードメニューから以下のノードが利用可能:
   - `Create Blank Frames`: 空白フレーム作成
   - `Multi Image Inserter`: 複数画像の挿入

## スクリプト使用方法

### insert_image_to_video.py
```bash
python insert_image_to_video.py
```
動画フレーム処理のメイン機能を含むスクリプト

### test_multiple_images.py
```bash
python test_multiple_images.py
```
複数画像挿入のテスト・デモ用スクリプト