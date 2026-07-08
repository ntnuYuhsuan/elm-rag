import csv
import time
import os
import logging
from app import run_query

logging.getLogger().setLevel(logging.WARNING)

def main():
    input_file = "benchmark_queries.txt"
    output_file = "evaluation_results.csv"

    if not os.path.exists(input_file):
        print(f"❌ 找不到測試檔 {input_file}，請確認是否位於根目錄。")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    results = []
    print(f"🚀 開始執行批次評估，共 {len(queries)} 題...\n")

    for i, query in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] 測試中: {query}")
        start_time = time.time()

        # 呼叫 app.py 的核心邏輯
        response, docs = run_query(query)

        end_time = time.time()
        latency = round(end_time - start_time, 2)

        # 結構化輸出解析
        if isinstance(response, dict):
            results.append({
                "query": query,
                "title": response.get("title", ""),
                "authors": response.get("authors", ""),
                "key_conclusions": response.get("key_conclusions", "").replace("\n", " "),
                "relevance_score": response.get("relevance_score", 0),
                "latency_sec": latency,
                "status": "Success (JSON)"
            })
        else:
            # 降級純文字輸出解析
            results.append({
                "query": query,
                "title": "N/A",
                "authors": "N/A",
                "key_conclusions": str(response).replace("\n", " "),
                "relevance_score": "N/A",
                "latency_sec": latency,
                "status": "Fallback (Text)"
            })
            
        print(f"✅ 完成! 耗時 {latency} 秒\n{'-'*40}")

    # 輸出成 CSV 供實驗比較
    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["query", "title", "authors", "key_conclusions", "relevance_score", "latency_sec", "status"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"🎉 評估完成！結果已儲存至 {output_file}")

if __name__ == "__main__":
    main()