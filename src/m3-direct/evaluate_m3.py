import csv
import time
import os
import logging
from app_m3 import run_query_m3

# logging.getLogger().setLevel(logging.WARNING)

def main():
    input_file = "benchmark_queries.txt"
    output_file = "evaluation_results.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    results = []
    
    print(f"\nBatch Inference ({len(queries)} Queries Found...)\n")

    for i, query in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] Query: {query}")
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
            title = response.get("title", "N/A")
            authors = response.get("authors", "N/A")
            conclusion = response.get("key_conclusions", "").replace("\n", " ")
            
            row_data.update({
                "title": title, 
                "authors": authors, 
                "key_conclusions": conclusion, 
                "relevance_score": response.get("relevance_score", 0)
            })
            
            print(f"   ↳ Conclusion: {conclusion}")
            if title != "未知" and title != "N/A":
                print(f"   ↳ Citation: {title} ({authors})")
            else:
                print("   ↳ Citation: N/A (No relevant data found)")
                
        else:
            fallback_text = str(response).replace("\n", " ")
            row_data.update({
                "title": "N/A", "authors": "N/A", 
                "key_conclusions": fallback_text, 
                "relevance_score": "N/A"
            })
            print(f"   ↳ Output: {fallback_text}")
            
        results.append(row_data)
        # 優化：英文版的延遲時間輸出
        print(f"✅ Completed in {latency}s\n{'-'*40}")

    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["query", "title", "authors", "key_conclusions", "relevance_score", "latency_sec", "status", "retrieved_chunks"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results: writer.writerow(row)

    print(f"🎉 Batch evaluation finished! Results saved to {output_file}")

if __name__ == "__main__":
    main()