import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 確保使用與建置時相同的英文 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': True}
)

DB_DIR = "./chroma_db"

def get_retriever():
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. 初始化 Qwen3-8B
llm = OllamaLLM(model="qwen3:8b", temperature=0.0)

# 3. 建立 Query Rewrite (翻譯) Prompt
translate_template = """
You are a professional medical and engineering translator. 
Translate the following user query from Traditional Chinese to English to help with database retrieval.
ONLY output the translated English text. Do not add any explanations, notes, or conversational text.

Query: {question}
Translation:
"""
translate_prompt = PromptTemplate.from_template(translate_template)

# 4. 建立最終 QA Prompt (要求模型看英文 Context，回繁體中文)
qa_template = """
你是一個專業的生醫工程助理。請「嚴格」根據以下提供的【英文參考資料】來回答使用者的問題。
請務必以「繁體中文」回答。如果【英文參考資料】中沒有足夠的資訊來回答問題，請明確回覆「資料中查無」，絕對不要編造答案。
回答時，請在句末附上參考資料的來源（如：[來源: 論文檔名]）。

【英文參考資料】：
{context}

使用者問題：{question}

繁體中文答案：
"""
qa_prompt = PromptTemplate.from_template(qa_template)

def format_docs(docs):
    if not docs:
        return "No relevant information found."
    return "\n\n".join(f"Content: {doc.page_content}\n[Source: {doc.metadata.get('source_file', 'Unknown')}]" for doc in docs)

def main():
    retriever = get_retriever()
    
    # 5. 建構兩階段 LCEL Chain
    
    # 階段 A：將使用者的中文問題翻譯成英文
    translation_chain = (
        {"question": RunnablePassthrough()}
        | translate_prompt
        | llm
        | StrOutputParser()
    )
    
    # 階段 B：定義一個檢索函數，接收英文 Query，回傳格式化後的 Context
    def retrieve_context(english_query):
        print(f"[Debug] 翻譯後的檢索 Query: {english_query.strip()}")
        docs = retriever.invoke(english_query)
        return format_docs(docs)

    # 階段 C：最終生成 Chain (將原始問題與檢索到的 Context 送入 QA Prompt)
    rag_chain = (
        {
            "context": translation_chain | retrieve_context, 
            "question": RunnablePassthrough() # 保留原始中文問題
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    # 6. 測試查詢
    query = "PPG訊號估計心率時,常見的雜訊抑制處理方式有哪些?"
    print(f"原始問題: {query}\n")
    print("生成答案中...")
    response = rag_chain.invoke(query)
    print("\n最終回覆:\n", response)

if __name__ == "__main__":
    main()