import json
import os
import sys

import whisper
from dotenv import load_dotenv

load_dotenv()  # 環境変数を読み込む


def check_args():
    # コマンドライン引数の数が正しいか確認
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_file_path>")
        sys.exit(1)


def main():
    model_path = os.getenv("MODEL_PATH")  # モデルのパス
    input_path = sys.argv[1]  # コマンドライン引数からファイルパスを取得
    output_path = os.getenv("OUTPUT_PATH")  # 出力先のパス

    output_path = output_path + input_path.split("/")[-1].replace(".m4a", ".json")

    model = whisper.load_model(model_path)  # モデルの読み込み
    result = model.transcribe(
        input_path, verbose=True, fp16=False, language="ja"
    )  # ファイル指定

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    check_args()
    main()
