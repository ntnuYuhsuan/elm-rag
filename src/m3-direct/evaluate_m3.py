import csv
import time
import os
import logging
from app_m3 import run_query_m3

logging.getLogger().setLevel(logging.WARNING)

def main():
    input_file = "benchmark_queries.txt"
    output_file = "evaluation_results_m3.csv"

    if not os.path.exists(input_file):
        return

    with open(input_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    results = []
    print(f"🚀 開始執行 BGE-M3 批次評估，共 {len(queries)} 題...\n")

    for i, query in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] 測試中: {query}")
        start_time = time.time()

        response, docs = run_query_m3(query)

        end_time = time.time()
        latency = round(end_time - start_time, 2)
        
        retrieved_info = ""
        for idx, d in enumerate(docs):
            retrieved_info += f"[{idx+1}] {d.metadata.get('source_file', 'Unknown')} - {d.page_content[:150].replace(chr(10), ' ')}...\n"
        if not retrieved_info: retrieved_info = "查無符合門檻之資料"

        row_data = {
            "query": query, "latency_sec": latency,
            "retrieved_chunks": retrieved_info,
            "status": "Success (JSON)" if isinstance(response, dict) else "Fallback (Text)"
        }

        if isinstance(response, dict):
            row_data.update({"title": response.get("title", ""), "authors": response.get("authors", ""), "key_conclusions": response.get("key_conclusions", "").replace("\n", " "), "relevance_score": response.get("relevance_score", 0)})
        else:
            row_data.update({"title": "N/A", "authors": "N/A", "key_conclusions": str(response).replace("\n", " "), "relevance_score": "N/A"})
            
        results.append(row_data)
        print(f"✅ 完成! 耗時 {latency} 秒\n{'-'*40}")

    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["query", "title", "authors", "key_conclusions", "relevance_score", "latency_sec", "status", "retrieved_chunks"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results: writer.writerow(row)

    print(f"🎉 M3 評估完成！結果已儲存至 {output_file}")

if __name__ == "__main__":
    main()