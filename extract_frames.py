import cv2
import os
from pathlib import Path
import glob


def extract_frames_from_video(video_path, output_base_dir="output_frames", interval_seconds=1, image_format="jpg"):
    """
    動画から1秒ごとに画像を切り取る
    
    Args:
        video_path (str): 入力動画ファイルパス
        output_base_dir (str): 出力ベースディレクトリ。デフォルトは output_frames
        interval_seconds (float): 切り取り間隔（秒）。デフォルトは1秒
        image_format (str): 画像フォーマット。"jpg"または"png"。デフォルトは"jpg"
    """
    
    # 入力動画を開く
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"エラー: {video_path} を開けません")
        return
    
    # 動画情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        fps = 30
    
    # フレーム間隔をフレーム数で計算
    frame_interval = int(fps * interval_seconds)
    
    # 動画の名前を取得（拡張子なし）
    video_name = Path(video_path).stem
    
    # 出力ディレクトリを作成
    os.makedirs(output_base_dir, exist_ok=True)
    video_output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    print(f"動画処理開始: {video_path}")
    print(f"  - フレームレート: {fps} fps")
    print(f"  - 総フレーム数: {total_frames}")
    print(f"  - 切り取り間隔: {interval_seconds}秒（{frame_interval}フレーム）")
    print(f"  - 出力ディレクトリ: {video_output_dir}")
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # フレーム間隔で画像を保存
        if frame_count % frame_interval == 0:
            extracted_count += 1
            time_seconds = frame_count / fps
            
            # ファイル名を生成（フレーム番号と時間を含める）
            filename = os.path.join(
                video_output_dir,
                f"frame_{extracted_count:05d}_{time_seconds:.2f}s.{image_format}"
            )
            
            # 画像を保存
            cv2.imwrite(filename, frame)
            
            if extracted_count % 10 == 0:
                print(f"  進捗: {extracted_count}フレーム抽出済み ({time_seconds:.2f}秒)")
        
        frame_count += 1
    
    cap.release()
    
    print(f"処理完了: {extracted_count}枚の画像を抽出しました")
    print()


def batch_process_videos(video_directory, output_base_dir="output_frames", interval_seconds=1, image_format="jpg"):
    """
    フォルダ内のすべての動画を処理
    
    Args:
        video_directory (str): 動画ファイルが入っているディレクトリ
        output_base_dir (str): 出力ベースディレクトリ
        interval_seconds (float): 切り取り間隔（秒）
        image_format (str): 画像フォーマット
    """
    
    # サポートされる動画形式
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MOV', '*.AVI', '*.MP4']
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_directory, ext)))
    
    if not video_files:
        print(f"エラー: {video_directory} に動画ファイルが見つかりません")
        return
    
    print(f"見つかった動画: {len(video_files)}個\n")
    
    for video_file in video_files:
        extract_frames_from_video(video_file, output_base_dir, interval_seconds, image_format)


if __name__ == "__main__":
    # ================================
    # 使用例
    # ================================
    
    # 方法1: 単一の動画を処理
    # extract_frames_from_video("test_movie/webcam_recording_20260421_140949.mp4")
    # extract_frames_from_video("output_mediapipe/webcam_recording_20260421_140949_mediapipe.mp4")
    # extract_frames_from_video("output_yolov8/webcam_recording_20260421_140949_yolov8.mp4")
    # extract_frames_from_video("output_yolov8/Video Project_yolov8_1280.mp4")
    # Extract frames from all videos in the divid_move folder.
    batch_process_videos("divid_move")
    
    
    # 方法2: フォルダ内のすべての動画を処理
    # batch_process_videos("test_movie")
    
    # 方法3: カスタム設定で処理
    # extract_frames_from_video(
    #     "test_movie/webcam_recording_20260421_192130.mp4",
    #     output_base_dir="output_frames",
    #     interval_seconds=0.5,  # 0.5秒ごと
    #     image_format="png"      # PNG形式
    # )
    
    print("extract_frames.py - 動画フレーム抽出ツール")
    print("=" * 50)
    print()
    
    # # デフォルト: test_movieフォルダ内のすべての動画を処理
    # if os.path.exists("test_movie"):
    #     batch_process_videos("test_movie", output_base_dir="output_frames", interval_seconds=1)
    # else:
    #     print("test_movieフォルダが見つかりません")
    #     print("使用方法:")
    #     print("  from extract_frames import extract_frames_from_video, batch_process_videos")
    #     print()
    #     print("  # 単一の動画を処理")
    #     print('  extract_frames_from_video("path/to/video.mp4")')
    #     print()
    #     print("  # フォルダ内のすべての動画を処理")
    #     print('  batch_process_videos("path/to/video_folder")')
