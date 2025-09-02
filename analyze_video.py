#!/usr/bin/env python3
"""
動画フォーマット詳細分析ツール
使用例: python analyze_video.py path/to/video.mp4
"""

import sys
import os
from pathlib import Path
import subprocess
import json

def analyze_with_ffprobe(video_path):
    """ffprobeを使用した詳細分析"""
    print(f"📹 動画分析: {video_path}")
    print("=" * 80)
    
    # 基本情報
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # ファイル情報
        file_stat = os.stat(video_path)
        print(f"📁 ファイル情報:")
        print(f"  - ファイル名: {Path(video_path).name}")
        print(f"  - ファイルサイズ: {file_stat.st_size / 1024 / 1024:.2f} MB")
        print(f"  - 拡張子: {Path(video_path).suffix}")
        print()
        
        # フォーマット情報
        format_info = data.get('format', {})
        print(f"📺 コンテナ情報:")
        print(f"  - フォーマット: {format_info.get('format_name', 'N/A')}")
        print(f"  - フォーマット(詳細): {format_info.get('format_long_name', 'N/A')}")
        print(f"  - 継続時間: {float(format_info.get('duration', 0)):.3f} 秒")
        print(f"  - ビットレート: {format_info.get('bit_rate', 'N/A')} bps")
        print()
        
        # ストリーム情報
        streams = data.get('streams', [])
        video_streams = [s for s in streams if s.get('codec_type') == 'video']
        audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
        
        print(f"🎬 映像ストリーム情報:")
        if video_streams:
            for i, stream in enumerate(video_streams):
                print(f"  [映像ストリーム {i}]")
                print(f"    - コーデック: {stream.get('codec_name', 'N/A')}")
                print(f"    - コーデック(詳細): {stream.get('codec_long_name', 'N/A')}")
                print(f"    - プロファイル: {stream.get('profile', 'N/A')}")
                print(f"    - レベル: {stream.get('level', 'N/A')}")
                print(f"    - 解像度: {stream.get('width', 'N/A')}x{stream.get('height', 'N/A')}")
                print(f"    - アスペクト比: {stream.get('display_aspect_ratio', 'N/A')}")
                print(f"    - フレームレート: {stream.get('r_frame_rate', 'N/A')}")
                print(f"    - 平均フレームレート: {stream.get('avg_frame_rate', 'N/A')}")
                print(f"    - ピクセルフォーマット: {stream.get('pix_fmt', 'N/A')}")
                print(f"    - ビットレート: {stream.get('bit_rate', 'N/A')} bps")
                print(f"    - フレーム数: {stream.get('nb_frames', 'N/A')}")
                print()
        else:
            print("    映像ストリームが見つかりません")
            print()
        
        print(f"🎵 音声ストリーム情報:")
        if audio_streams:
            for i, stream in enumerate(audio_streams):
                print(f"  [音声ストリーム {i}]")
                print(f"    - コーデック: {stream.get('codec_name', 'N/A')}")
                print(f"    - コーデック(詳細): {stream.get('codec_long_name', 'N/A')}")
                print(f"    - サンプルレート: {stream.get('sample_rate', 'N/A')} Hz")
                print(f"    - チャンネル数: {stream.get('channels', 'N/A')}")
                print(f"    - ビットレート: {stream.get('bit_rate', 'N/A')} bps")
                print()
        else:
            print("    音声ストリームが見つかりません")
            print()
        
        # ブラウザ互換性判定
        analyze_browser_compatibility(video_streams, audio_streams, format_info)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ffprobe実行エラー: {e}")
        print(f"標準エラー: {e.stderr}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析エラー: {e}")
    except FileNotFoundError:
        print("❌ ffprobeが見つかりません。FFmpegをインストールしてください。")

def analyze_browser_compatibility(video_streams, audio_streams, format_info):
    """ブラウザ互換性を分析"""
    print(f"🌐 ブラウザ互換性分析:")
    
    if not video_streams:
        print("    ❌ 映像ストリームがないため、ブラウザで再生不可")
        return
    
    video_stream = video_streams[0]  # 最初の映像ストリームを分析
    codec_name = video_stream.get('codec_name', '').lower()
    profile = video_stream.get('profile', '').lower()
    container = format_info.get('format_name', '').lower()
    
    print(f"  🎬 映像コーデック判定:")
    if codec_name == 'h264':
        if profile in ['baseline', 'main', 'high']:
            print(f"    ✅ H.264 ({profile}) - 広くサポート")
        else:
            print(f"    ⚠️ H.264 ({profile}) - プロファイルによっては制限あり")
    elif codec_name == 'hevc' or codec_name == 'h265':
        print(f"    ⚠️ H.265/HEVC - 新しいブラウザのみサポート")
    elif codec_name == 'vp8':
        print(f"    ✅ VP8 - WebMコンテナでサポート")
    elif codec_name == 'vp9':
        print(f"    ✅ VP9 - 新しいブラウザでサポート")
    elif codec_name == 'av1':
        print(f"    ✅ AV1 - 最新ブラウザでサポート")
    elif codec_name == 'mpeg4':
        print(f"    ❌ MPEG-4 Part 2 - 多くのブラウザで非サポート")
    else:
        print(f"    ❓ {codec_name} - サポート状況不明")
    
    print(f"  📦 コンテナフォーマット判定:")
    if 'mp4' in container:
        print(f"    ✅ MP4 - 広くサポート")
    elif 'webm' in container:
        print(f"    ✅ WebM - 広くサポート")
    elif 'avi' in container:
        print(f"    ❌ AVI - ブラウザで制限的")
    else:
        print(f"    ❓ {container} - サポート状況不明")
    
    # 総合判定
    print(f"  📋 総合判定:")
    if codec_name == 'h264' and 'mp4' in container:
        print(f"    ✅ 高い互換性 - ほぼ全てのブラウザで再生可能")
    elif codec_name in ['vp8', 'vp9'] and 'webm' in container:
        print(f"    ✅ 良い互換性 - 多くのブラウザで再生可能")
    elif codec_name == 'mpeg4' or codec_name == 'mp4v':
        print(f"    ❌ 低い互換性 - 多くのブラウザで再生不可")
        print(f"    💡 推奨: H.264コーデックに再エンコード")
    else:
        print(f"    ⚠️ 要確認 - ブラウザでのテストが必要")

def main():
    if len(sys.argv) != 2:
        print("使用法: python analyze_video.py <動画ファイルパス>")
        print("例: python analyze_video.py sample.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ ファイルが見つかりません: {video_path}")
        sys.exit(1)
    
    analyze_with_ffprobe(video_path)

if __name__ == "__main__":
    main()