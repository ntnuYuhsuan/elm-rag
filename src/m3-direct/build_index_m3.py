import os
import yaml
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from __init__ import PAPERS_META, parse_markdown_sections

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def main():
    DATA_DIR = "./data"
    DB_DIR = "./chroma_db_m3" # 獨立的 DB
    DEVICE = "mps"
    
    print(f"啟動 BAAI/bge-m3 模型 ({DEVICE} 晶片加速)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50,
        separators=["\n\n", ".\n", ". ", "\n", " "] 
    )
    
    all_documents = []
    for filename, meta in PAPERS_META.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath): continue
            
        print(f"正在解析與向量化: {meta['title']}...")
        md_text = pymupdf4llm.to_markdown(filepath)
        sections = parse_markdown_sections(md_text)
        
        for section_idx, section in enumerate(sections):
            section_tag = section["section"]
            content = section["content"]

            if not content.strip():
                continue

            chunks = text_splitter.split_text(content)
            for chunk_idx, chunk_text in enumerate(chunks):
                doc = Document(
                    page_content=chunk_text,
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

    distance_metric = config['retrieval']['distance_metric']
    Chroma.from_documents(documents=all_documents, embedding=embeddings, persist_directory=DB_DIR, collection_metadata={"hnsw:space": distance_metric})
    print(f"✅ M3 向量資料庫建置完成！")

if __name__ == "__main__":
    main()