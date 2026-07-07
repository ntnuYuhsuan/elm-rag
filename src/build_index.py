import os
import re
import yaml
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. 讀取可控變數
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 2. 完整 7 篇文獻的 Metadata 對應表 (精確對應 ls data 檔名)
PAPERS_META = {
    "Mobile Information Systems - 2020 - Pan - Human Falling Detection Algorithm Based on Multisensor Data Fusion with SVM.pdf": {
        "title": "Human Falling Detection Algorithm Based on Multisensor Data Fusion with SVM",
        "authors": "Pan et al.",
        "year": 2020,
        "topic": "IMU_fall_detection",
        "doi": "10.1155/2020/8826088"
    },
    "TROIKA_A_General_Framework_for_Heart_Rate_Monitoring_Using_Wrist-Type_Photoplethysmographic_Signals_During_Intensive_Physical_Exercise.pdf": {
        "title": "TROIKA: A General Framework for Heart Rate Monitoring",
        "authors": "Zhang et al.",
        "year": 2015,
        "topic": "PPG_motion_artifact",
        "doi": "10.1109/TBME.2014.2359372"
    },
    "Validity of detrended fluctuation analysis of heart rate variability to determine intensity thresholds in elite cyclists.pdf": {
        "title": "Validity of DFA of HRV to Determine Intensity Thresholds",
        "authors": "Mateo-March et al.",
        "year": 2022,
        "topic": "LTHR_estimation",
        "doi": "10.1080/17461391.2022.2047228"
    },
    "SPARE A Spectral Peak Recovery Algorithm for PPG Signals Pulsewave Reconstruction in Multimodal Wearable Devices.pdf": {
        "title": "SPARE: A Spectral Peak Recovery Algorithm for PPG Signals Corrupted by Motion Artifacts",
        "authors": "Masinelli et al.",
        "year": 2021,
        "topic": "PPG_motion_artifact",
        "doi": "10.3390/s21082725"
    },
    "Motion artifact removal from photoplethysmographic signals by combining temporally constrained independent component analysis and adaptive filter.pdf": {
        "title": "Motion artifact removal from photoplethysmographic signals by combining temporally constrained independent component analysis and adaptive filter",
        "authors": "Peng et al.",
        "year": 2014,
        "topic": "PPG_motion_artifact",
        "doi": "10.1186/1475-925X-13-50"
    },
    "A comparison of accuracy of fall detection algorithms (threshold-based vs. machine learning) using waist-mounted tri-axial accelerometer signals from a comprehensive set of falls and non-fall trials..pdf": {
        "title": "A comparison of accuracy of fall detection algorithms (threshold-based vs. machine learning) using waist-mounted tri-axial accelerometer signals",
        "authors": "Aziz et al.",
        "year": 2017,
        "topic": "IMU_fall_detection",
        "doi": "10.1007/s11517-016-1504-y"
    },
    "Heart rate variability-derived thresholds for exercise intensity prescription in endurance sports a systematic review of interrelations and agreement with different ventilatory and blood lactate thresholds..pdf": {
        "title": "Heart Rate Variability-Derived Thresholds for Exercise Intensity Prescription: A Systematic Review",
        "authors": "Kaufmann et al.",
        "year": 2023,
        "topic": "LTHR_estimation",
        "doi": "10.1186/s40798-023-00607-2"
    }
}

def parse_markdown_sections(md_text):
    pattern = r'\n#{1,3}\s+(.*?)\n'
    parts = re.split(pattern, '\n' + md_text) 
    
    sections = []
    if parts[0].strip():
        sections.append(("metadata", parts[0]))
        
    for i in range(1, len(parts)-1, 2):
        header = parts[i].strip().lower()
        content = parts[i+1]
        
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
    DEVICE = "mps"
    
    model_name = config['models']['embedding']
    print(f"啟動 {model_name} 模型 ({DEVICE} 晶片加速)...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, 
        chunk_overlap=50,
        separators=["\n\n", ".\n", ". ", "\n", " "] 
    )
    
    all_documents = []
    
    for filename, meta in PAPERS_META.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[Warning] 找不到檔案: {filename}，已略過。")
            continue
            
        print(f"正在解析與向量化: {meta['title']}...")
        md_text = pymupdf4llm.to_markdown(filepath)
        sections = parse_markdown_sections(md_text)
        
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
                        "doi": meta.get("doi", "N/A"), # 注入 DOI
                        "topic": meta["topic"],
                        "section": section_tag,
                        "chunk_index": f"{section_idx}_{chunk_idx}"
                    }
                )
                all_documents.append(doc)

    print(f"\n總共建立 {len(all_documents)} 個 Chunks。")
    print("正在將向量寫入 ChromaDB...")
    
    distance_metric = config['retrieval']['distance_metric']
    Chroma.from_documents(
        documents=all_documents, 
        embedding=embeddings, 
        persist_directory=DB_DIR,
        collection_metadata={"hnsw:space": distance_metric} 
    )
    print(f"✅ 向量資料庫建置完成！(Distance Metric: {distance_metric})")

if __name__ == "__main__":
    main()