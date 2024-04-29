import whisper
from yaml import safe_load
import json


def main():
    with open("config.yaml", "r") as f:
        config = safe_load(f)

    model = config["model"]
    input_path = config["input_path"]
    output_path = config["output_path"]

    output_path = output_path + input_path.split("/")[-1].replace(".m4a", ".json")

    model = whisper.load_model(model)  # モデルの読み込み
    result = model.transcribe(
        input_path, verbose=True, fp16=False, language="ja"
    )  # ファイル指定

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
