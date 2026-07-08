import os
import re
import json
import yaml
import logging
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logging.basicConfig(
    filename='rag_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

DEVICE = "mps"
DB_DIR = "./chroma_db"

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

embeddings = HuggingFaceEmbeddings(
    model_name=config['models']['embedding'],
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

class PaperResponse(BaseModel):
    title: str = Field(description="論文的主題或標題簡述。若查無資料，請填 '未知'")
    authors: str = Field(description="論文作者。若查無資料，請填 '未知'")
    key_conclusions: str = Field(description="繁體中文回答，包含關鍵結論與方法細節。若查無資料，請填 '資料中查無'")
    relevance_score: int = Field(description="評估資料與問題的相關性分數 (1-10分)。若查無資料，請填 0")

def get_retriever():
    vectorstore = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": config['retrieval']['distance_metric']}
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "score_threshold": config['retrieval']['score_threshold'], 
            "k": config['retrieval']['top_k']
        }
    )

# 翻譯/Fallback用的純文字 LLM
llm_chat = OllamaLLM(
    model=config['models']['llm'], 
    temperature=config['models']['llm_temperature']
)

# 生成答案用的 JSON LLM (加入 format="json" 可在多數模型中有效抑制 <think> 特殊 token 的生成，大幅提升速度)
llm_json = OllamaLLM(
    model=config['models']['llm'], 
    temperature=config['models']['llm_temperature'],
    format="json" 
)

translate_prompt = PromptTemplate.from_template(config['prompts']['translation'])
parser = JsonOutputParser(pydantic_object=PaperResponse)
qa_prompt = PromptTemplate(
    template=config['prompts']['qa_system'],
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def format_docs(docs):
    if not docs:
        return ""
    formatted_str = ""
    for idx, doc in enumerate(docs):
        source = doc.metadata.get('source_file', 'Unknown')
        authors = doc.metadata.get('authors', 'Unknown')
        formatted_str += f"[Doc {idx+1} | Source: {source} | Authors: {authors}]\nContent: {doc.page_content}\n\n"
    return formatted_str

def run_query(query: str):
    if not os.path.exists(DB_DIR):
        print("請先執行 python src/build_index.py 建立向量庫！")
        return None, []
        
    retriever = get_retriever()
    
    # 1. 翻譯階段
    english_query = (translate_prompt | llm_chat).invoke({"question": query}).strip()
    logging.info(f"[Translate Rewrite] : {english_query}")
    
    # 2. 檢索階段 (將 docs 攔截下來)
    docs = retriever.invoke(english_query)
    context_str = format_docs(docs)
    logging.info(f"[Retrieval Hit]     : {len(docs)} chunks passing threshold.")

    # 3. 生成階段
    try:
        # 強制使用 JSON LLM 輸出
        response_dict = (qa_prompt | llm_json | parser).invoke({"context": context_str, "question": query})
        return response_dict, docs
    except Exception as e:
        logging.error(f"[JSON Parsing Error]: {e}")
        # 降級使用一般 LLM 輸出文字
        fallback_response = (qa_prompt | llm_chat).invoke({"context": context_str, "question": query})
        # 簡易的正則過濾，確保萬一還有 think 標籤時不顯示
        fallback_response = re.sub(r'<think>.*?</think>', '', fallback_response, flags=re.DOTALL).strip()
        return fallback_response, docs

if __name__ == "__main__":
    test_query = "PPG訊號估計心率時，常見的雜訊抑制處理方式有哪些？"
    res, _ = run_query(test_query)
    print(res)