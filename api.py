from fastapi import FastAPI
from pydantic import BaseModel
from main import load_docs,split_docs,build_vectorstore,answer

app = FastAPI(
    title='后客服 RAG 问答助手',
    description="基于 LangChain + Chroma + DeepSeek 的知识库问答接口",
    version="1.0.0",
)

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        print(">>> 初始化向量库...")
        docs = load_docs()
        chunks = split_docs(docs)
        _vectorstore = build_vectorstore(chunks)
        print(">>> 向量库就绪")
    return _vectorstore

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    rejected: bool

@app.get("/")
def root():
    return {"status": "ok","docs":"/docs"}

@app.post("/chat",response_model=ChatResponse)
def chat(request: ChatRequest):
    vs = get_vectorstore()
    ans=answer(vx=vs, question=request.question)
    rejected ="没有相关信息"in ans or"未提及" in ans
    return {"answer":ans,"rejected":rejected}