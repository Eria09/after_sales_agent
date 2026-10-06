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
    sources: list[str] = []

@app.get("/")
def root():
    return {"status": "ok","docs":"/docs"}

@app.post("/chat",response_model=ChatResponse)
def chat(request: ChatRequest):
    vs = get_vectorstore()
    ans, sources = answer(vx=vs, question=request.question)
    # 拒答的结构化特征就是"没有引用片段"——不要去猜答案文案。
    # 否则 main.py 里把拒答话术改一个字，这里的 rejected 就静默失效。
    rejected = not sources
    return {"answer":ans,"rejected":rejected,"sources":sources}