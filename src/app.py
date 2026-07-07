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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# 1. 初始化日誌系統 (專注紀錄實驗數據與錯誤)
logging.basicConfig(
    filename='rag_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# 2. 固定系統參數
DEVICE = "mps"
DB_DIR = "./chroma_db"

# 3. 讀取可控變數
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 4. 初始化模型
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
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": config['retrieval']['score_threshold'], 
            "k": config['retrieval']['top_k']
        }
    )

llm = OllamaLLM(
    model=config['models']['llm'], 
    temperature=config['models']['llm_temperature']
)

# 5. Prompts
translate_prompt = PromptTemplate.from_template(config['prompts']['translation'])
parser = JsonOutputParser(pydantic_object=PaperResponse)
qa_prompt = PromptTemplate(
    template=config['prompts']['qa_system'],
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 6. 工具函式
def format_docs(docs):
    if not docs:
        return ""
    formatted_str = ""
    for idx, doc in enumerate(docs):
        source = doc.metadata.get('source_file', 'Unknown')
        authors = doc.metadata.get('authors', 'Unknown')
        formatted_str += f"[Doc {idx+1} | Source: {source} | Authors: {authors}]\nContent: {doc.page_content}\n\n"
    return formatted_str

def strip_think(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# 7. 核心執行邏輯
def run_query(query: str):
    if not os.path.exists(DB_DIR):
        print("請先執行 python src/build_index.py 建立向量庫！")
        return
        
    logging.info(f"--- [Query Input] --- : {query}")

    retriever = get_retriever()
    translation_chain = {"question": RunnablePassthrough()} | translate_prompt | llm | StrOutputParser()
    
    def retrieve_context(english_query):
        english_query = english_query.strip()
        logging.info(f"[Translate Rewrite] : {english_query}")
        
        docs = retriever.invoke(english_query)
        logging.info(f"[Retrieval Hit]     : {len(docs)} chunks passing threshold.")
        return format_docs(docs)

    base_generation_chain = (
        {
            "context": translation_chain | retrieve_context, 
            "question": RunnablePassthrough()
        }
        | qa_prompt
        | llm
        | StrOutputParser()
        | strip_think
    )
    
    json_rag_chain = base_generation_chain | parser
    
    try:
        response_dict = json_rag_chain.invoke(query)
        # 紀錄成功的 JSON 輸出
        logging.info(f"[Final Output]      : {json.dumps(response_dict, ensure_ascii=False)}")
        print(json.dumps(response_dict, ensure_ascii=False, indent=2))
        return response_dict
    except Exception as e:
        logging.error(f"[JSON Parsing Error]: {e}")
        fallback_response = base_generation_chain.invoke(query)
        logging.info(f"[Fallback Output]   : {fallback_response}")
        print(fallback_response)
        return fallback_response

if __name__ == "__main__":
    # 單筆測試用
    test_query = "PPG訊號估計心率時，常見的雜訊抑制處理方式有哪些？"
    run_query(test_query)