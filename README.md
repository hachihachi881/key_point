# YOLOv8 Pose Sitting Posture Assessment

YOLOv8 Poseで動画から人体キーポイントを抽出し、座位姿勢を簡易評価する研究用プロトタイプです。

現在は、横向きカメラを主な前提として、頭部前方突出と体幹傾きのルールベース評価を行います。各人物ごとのキーポイントCSV、姿勢評価CSV、姿勢スコアを重ねた動画を出力できます。

## 研究目的

座っている人の姿勢を動画から自動推定し、姿勢の崩れを定量的に確認することを目的としています。

YOLOv8 Poseの2Dキーポイントだけでは、骨盤の向き、かかとの接地、椅子に深く座れているかなどを直接評価することは難しいため、見える関節点から近似指標を作って評価します。

## 現在できること

- 動画からYOLOv8 Poseで17点の人体キーポイントを検出
- 検出結果を人物ごとのCSVとして保存
- キーポイントconfidenceをCSVに保存
- 頭部前方突出をスコア化
- 体幹の垂直方向からの傾きをスコア化
- 姿勢評価結果をCSVとして保存
- 元動画に骨格と姿勢スコアを重ねた動画を生成
- 動画から一定間隔でフレーム画像を抽出

## ディレクトリ構成

```text
.
├── process_video_yolov8.py      # YOLOv8 Poseによる動画処理と姿勢評価
├── extract_frames.py            # 動画からフレーム画像を抽出
├── posture_assessment_notes.md  # 姿勢評価ルールと研究メモ
├── input_movie/                 # 処理対象動画
├── test_movie/                  # テスト用動画
├── output_yolov8/               # YOLOv8処理結果
├── output_frames/               # 抽出フレーム
└── divid_move/                  # 分割・比較用動画
```

`input_movie/`, `test_movie/`, `output_yolov8/`, `output_frames/`, `divid_move/`, `*.pt` は容量が大きくなりやすいため、Git管理対象から除外しています。

## セットアップ

Python環境を用意し、必要なライブラリをインストールします。

```bash
pip install ultralytics opencv-python numpy torch
```

CUDA対応GPUが使える場合は、PyTorchの公式手順に従ってCUDA対応版を入れてください。

## 使い方

1. 処理したい動画を `input_movie/` に置きます。
2. YOLOv8 Poseの重みファイルを用意します。
3. 次のコマンドを実行します。

```bash
python process_video_yolov8.py
```

デフォルトでは `input_movie/` 内の動画を処理し、`yolov8x-pose.pt` を使います。

処理結果は `output_yolov8/` に保存されます。

## 出力

動画ごとに、次のファイルが出力されます。

```text
output_yolov8/
├── <video_name>_yolov8.mp4
└── <video_name>_keypoints/
    ├── <video_name>0.csv
    └── <video_name>0_posture.csv
```

キーポイントCSVには、各フレームの17点キーポイント座標とconfidenceが保存されます。

姿勢評価CSVには、現在以下の列が保存されます。

```text
time
side
head_forward_ratio
trunk_angle
head_forward_score
trunk_forward_score
bad_reasons
```

## 現在の評価指標

### 頭部前方突出

耳と肩のx方向のずれを体幹長で正規化します。

```text
head_forward_ratio = abs(ear_x - shoulder_x) / torso_length
```

### 体幹傾き

肩と腰を結ぶ線が、垂直方向からどれだけ傾いているかを角度で計算します。

```text
trunk_angle = angle(shoulder - hip, vertical)
```

## 今後の課題

- 膝角度の評価
- 股関節角度の評価
- 肩の左右差の評価
- 総合姿勢スコアの計算
- 時系列平滑化
- 5秒以上続く悪姿勢の警告判定
- キャリブレーションによる個人差対応
- 手動ラベルとの比較による評価精度検証
- YOLO `track()` などを使った人物IDの安定化

## 注意

このプログラムによる姿勢判定は、医療診断ではありません。研究・実験・姿勢への注意喚起を目的とした推定結果として扱ってください。
