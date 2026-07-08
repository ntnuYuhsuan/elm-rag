import os
import re
import yaml
import logging
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

DEVICE = "mps"
DB_DIR = "./chroma_db_m3" # 指向 M3 資料庫

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 直接載入 BGE-M3
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

class PaperResponse(BaseModel):
    title: str = Field(description="論文的主題或標題簡述。若查無資料，請填 '未知'")
    authors: str = Field(description="論文作者。若查無資料，請填 '未知'")
    cross_lingual_analysis: str = Field(description="請簡短分析【中文問題】中的關鍵字，對應到【英文參考資料】中的哪些專有名詞 (例如：雜訊抑制對應 motion artifacts)。若真的完全無關才填 '無'")
    key_conclusions: str = Field(description="繁體中文回答，包含關鍵結論與方法細節。若查無資料，請填 '資料中查無'")
    relevance_score: int = Field(description="評估資料與問題的相關性分數 (1-10分)。若查無資料，請填 0")

def get_retriever():
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": config['retrieval']['score_threshold'], "k": config['retrieval']['top_k']}
    )

llm_chat = OllamaLLM(model=config['models']['llm'], temperature=config['models']['llm_temperature'])
llm_json = OllamaLLM(model=config['models']['llm'], temperature=config['models']['llm_temperature'], format="json")

parser = JsonOutputParser(pydantic_object=PaperResponse)
qa_prompt = PromptTemplate(
    template=config['prompts']['qa_system'],
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def format_docs(docs):
    if not docs: return ""
    return "".join([f"[Doc {idx+1} | Source: {d.metadata.get('source_file')} | Authors: {d.metadata.get('authors')}]\nContent: {d.page_content}\n\n" for idx, d in enumerate(docs)])

def run_query_m3(query: str):
    if not os.path.exists(DB_DIR):
        print("請先執行 python src/build_index_m3.py 建立向量庫！")
        return None, []
        
    retriever = get_retriever()
    
    # 移除翻譯階段，直接拿使用者的中文 Query 去檢索 M3 英文資料庫
    logging.info(f"[M3 Direct Retrieval] Query : {query}")
    docs = retriever.invoke(query)
    context_str = format_docs(docs)
    logging.info(f"[Retrieval Hit]     : {len(docs)} chunks passing threshold.")

    try:
        response_dict = (qa_prompt | llm_json | parser).invoke({"context": context_str, "question": query})
        return response_dict, docs
    except Exception as e:
        fallback_response = (qa_prompt | llm_chat).invoke({"context": context_str, "question": query})
        fallback_response = re.sub(r'<think>.*?</think>', '', fallback_response, flags=re.DOTALL).strip()
        return fallback_response, docs