"""alkana の読み辞書を取得して TTS 用 CSV(`english,カタカナ`) を生成するユーティリティ。

alkana(https://github.com/zomysan/alkana.py) の辞書本体は Python ファイル
``alkana/data.py`` にある約 49,000 語の dict で、ライセンスは **GPLv2** です。
本スクリプトは pip 依存を増やさず、その data.py を取得して CSV へ変換するだけです。

  重要: 生成される CSV(alkanadict.csv) は GPLv2 の派生物です。本家へ PR するリポジトリには
  コミットせず、稼働環境の tts_dictionaries/ に置いて .gitignore してください。

使い方:
    python tools/fetch_alkana_dict.py [出力先CSV]
既定の出力先は  <プロジェクト>/tts_dictionaries/alkanadict.csv
"""
import sys
import urllib.request
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/zomysan/alkana.py/master/alkana/data.py"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "tts_dictionaries" / "alkanadict.csv"


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"fetching: {DATA_URL}")
    source = urllib.request.urlopen(DATA_URL, timeout=60).read().decode("utf-8")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # data.py は `data = {..}` のみ
    data = namespace["data"]
    if not isinstance(data, dict):
        raise SystemExit("unexpected alkana data format")

    lines = ["# Generated from alkana (zomysan/alkana.py) alkana/data.py",
             "# License: GPLv2. Do NOT commit this file to an upstream PR branch.",
             f"# entries: {len(data)}"]
    # 英字キーだけを対象（カンマを含む値は保険で除去）。
    for word, kana in data.items():
        w = str(word).strip().lower()
        k = str(kana).strip().replace(",", "")
        if w and k:
            lines.append(f"{w},{k}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(data)} entries -> {out_path}")
    print("Set TTS_KANA_DICT_FILE=\"alkanadict.csv;tts_kana_dict.csv\" so your")
    print("custom terms in tts_kana_dict.csv override alkana's phonetic readings.")


if __name__ == "__main__":
    main()
