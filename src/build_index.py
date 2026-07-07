import os
import re
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. 論文 Metadata 對應表 (對應你 data 目錄下的實際檔名)
PAPERS_META = {
    "Mobile Information Systems - 2020 - Pan - Human Falling Detection Algorithm Based on Multisensor Data Fusion with SVM.pdf": {
        "title": "Human Falling Detection Algorithm Based on Multisensor Data Fusion with SVM",
        "authors": "Pan et al.",
        "year": 2020,
        "topic": "IMU_fall_detection"
    },
    "TROIKA_A_General_Framework_for_Heart_Rate_Monitoring_Using_Wrist-Type_Photoplethysmographic_Signals_During_Intensive_Physical_Exercise.pdf": {
        "title": "TROIKA: A General Framework for Heart Rate Monitoring",
        "authors": "Zhang et al.",
        "year": 2015,
        "topic": "PPG_motion_artifact"
    },
    "Validity of detrended fluctuation analysis of heart rate variability to determine intensity thresholds in elite cyclists.pdf": {
        "title": "Validity of DFA of HRV to Determine Intensity Thresholds",
        "authors": "Mateo-March et al.",
        "year": 2022,
        "topic": "LTHR_estimation"
    }
}

# 2. 自動化 Section 提取邏輯優化
def parse_markdown_sections(md_text):
    """
    透過 Markdown 標題 (#) 動態切割段落，而非寫死單字，
    並正規化成 abstract/methodology/results/conclusion/others。
    """
    # 擷取 # 或 ## 或 ### 開頭的標題及其內容
    pattern = r'\n#{1,3}\s+(.*?)\n'
    # re.split 會保留匹配到的標題在列表中
    parts = re.split(pattern, '\n' + md_text) 
    
    sections = []
    # parts[0] 通常是開頭無標題的文字 (Title, Authors)
    if parts[0].strip():
        sections.append(("metadata", parts[0]))
        
    for i in range(1, len(parts)-1, 2):
        header = parts[i].strip().lower()
        content = parts[i+1]
        
        # 標準化標籤
        tag = "others"
        if any(kw in header for kw in ["abstract"]): tag = "abstract"
        elif any(kw in header for kw in ["method", "experiment", "approach"]): tag = "methodology"
        elif any(kw in header for kw in ["result", "discussion"]): tag = "results"
        elif any(kw in header for kw in ["conclusion"]): tag = "conclusion"
        
        sections.append((tag, header + "\n" + content))
        
    return sections

def main():
    DATA_DIR = "./data"
    DB_DIR = "./chroma_db"
    
    print("啟動 BAAI/bge-large-en-v1.5 模型 (MPS 晶片加速)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, 
        chunk_overlap=50,
        separators=["\n\n", ".\n", "\n", "。", "！", "？", ".", " ", ". ", "\n", " "] # 專為英文學術 PDF 萃取的切割符
    )
    
    all_documents = []
    
    for filename, meta in PAPERS_META.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"找不到檔案: {filename}，請確認是否在 data/ 目錄下。")
            continue
            
        print(f"正在解析與向量化: {meta['title']}...")
        
        # 1. 轉為 Markdown
        md_text = pymupdf4llm.to_markdown(filepath)
        
        # 2. Section 切割
        sections = parse_markdown_sections(md_text)
        
        # 3. 針對每個 Section 進行 Chunking 並注入豐富的 Metadata
        for section_idx, (section_tag, content) in enumerate(sections):
            if not content.strip(): continue
            
            chunks = text_splitter.split_text(content)
            for chunk_idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source_file": filename,
                        "title": meta["title"],
                        "authors": meta["authors"],
                        "year": meta["year"],
                        "topic": meta["topic"],
                        "section": section_tag,
                        "chunk_index": f"{section_idx}_{chunk_idx}"
                    }
                )
                all_documents.append(doc)

    print(f"\n總共建立 {len(all_documents)} 個 Chunks。")
    print("正在將向量寫入 ChromaDB...")
    
    # 4. 存入 ChromaDB
    Chroma.from_documents(
        documents=all_documents, 
        embedding=embeddings, 
        persist_directory=DB_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("✅ 向量資料庫建置完成！")

if __name__ == "__main__":
    main()