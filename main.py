import whisper
import json
import sys

def main():
    model_path = "large"    #モデルのパス. small or medium or large
    output_path = "output/transcription.txt"    #出力先のパス

    # コマンドライン引数の数が正しいか確認
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_file_path>")
        sys.exit(1)
    audio_file_path = sys.argv[1]    # コマンドライン引数からファイルパスを取得

    model = whisper.load_model(model_path) #モデルの読み込み
    result = model.transcribe(audio_file_path, verbose=True, fp16=False, language="ja") #ファイル指定
    print(result['text'])

    f = open(output_path, 'w', encoding='UTF-8')
    f.write(json.dumps(result['text'], sort_keys=True, indent=4, ensure_ascii=False))
    f.close()

if __name__ == '__main__':
    main()