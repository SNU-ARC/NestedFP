# save as: extract_throughput_csv.py
import json
import argparse
from pathlib import Path
import csv
import re

def get_model_type_from_filename(filename: str) -> str:
    """
    파일명에서 _True/_False 패턴을 찾아 NestedFP/FP16으로 변환.
    예) throughput_sweep_Llama-3.1-8B_False.json -> FP16
        throughput_sweep_Llama-3.1-8B_True.json  -> NestedFP
    """
    stem = Path(filename).stem  # 확장자 제거
    m = re.search(r'_(True|False)$', stem)
    if not m:
        # 확장자 포함한 전체 이름에서 한 번 더 시도(안전)
        m = re.search(r'_(True|False)\.json$', filename)
    if m:
        return "NestedFP" if m.group(1) == "True" else "FP16"
    return "Unknown"

def main():
    parser = argparse.ArgumentParser(description="Extract throughput info as CSV")
    parser.add_argument("--dir", default="max_batched_tokens=8192",
                        help="directory containing JSON files")
    parser.add_argument("--input", type=int, default=1024,
                        help="input_length to filter (default: 1024)")
    parser.add_argument("--output", type=int, default=512,
                        help="output_length to filter (default: 512)")
    parser.add_argument("--output_csv", default="throughput_summary3.csv",
                        help="output CSV filename")
    parser.add_argument("--recursive", action="store_true",
                        help="search JSON files recursively")
    args = parser.parse_args()

    base = Path(args.dir)
    pattern = "**/*.json" if args.recursive else "*.json"
    files = sorted(base.glob(pattern))

    rows = []
    for fp in files:
        try:
            with fp.open("r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Skip {fp.name}: {e}")
            continue

        model_name = data.get("model_short_name") or data.get("model") or ""
        model_type = get_model_type_from_filename(fp.name)
        results = data.get("results", [])

        for r in results:
            #if (r.get("input_length") == args.input and
            #    r.get("output_length") == args.output and
            #    "throughput_tokens_per_sec" in r):
            rows.append([
                model_type,
                model_name.split("/")[-1],  # 혹시 전체 경로면 마지막 이름만
                r.get("batch_size"),
                r.get("throughput_tokens_per_sec"),
            ])

    # 정렬: model_short_name, model_type, batch_size 순
    rows.sort(key=lambda x: (x[1], x[0], x[2] if x[2] is not None else -1))

    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model_type", "model_short_name", "batch_size", "throughput_tokens_per_sec"])
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} entries to {args.output_csv}")

if __name__ == "__main__":
    main()
