
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import  Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- 控制台编码兜底 ----------
# 本文件顶层就会打印 ❌（Key 没读到 / 目录不存在），Windows cp936 控制台在
# 输出被重定向成管道时会抛 UnicodeEncodeError，导致连报错都看不到。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
MODEL=os.getenv("LLM_MODEL", "deepseek-chat")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "policies"
DB_DIR   = BASE_DIR / "chroma_db"

def print_env_banner():
    """只在实际运行本文件时打印环境诊断。

    这几行绝不能留在模块顶层：api.py 会 `from main import ...`，
    模块级打印会在 uvicorn 启动时刷一屏无关日志——分层没做干净。
    """
    print("脚本位置:", BASE_DIR)
    print("数据目录:", DATA_DIR, "存在" if DATA_DIR.exists() else "❌ 不存在")
    print("DeepSeek Key:", "已读取" if DEEPSEEK_API_KEY else "❌ 没读到")
    print("硅基流动 Key:", "已读取" if SILICONFLOW_API_KEY else "❌ 没读到")

def load_docs():
    docs=[]
    for md in sorted( DATA_DIR.glob("*.md")):
        print(f"  加载: {md.name}")
        docs.extend(TextLoader((str(md)),encoding="utf-8").load())
    print(f"共加载 {len(docs)} 个文档")
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=50,
        chunk_size = 300,
        separators=["\n\n", "\n", "。", "！", "？", "，", ""],
    )
    chunks=splitter.split_documents(docs)
    print(f'切分成{len(chunks)}个片段')
    return chunks


def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=SILICONFLOW_API_KEY,
        base_url="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=False,
    )
    if DB_DIR.exists():
        print("已有向量库，直接加载")
        return Chroma(persist_directory=str(DB_DIR),embedding_function=embeddings)


    print("首次运行，建库中（可能要 1-2 分钟）...")
    vs=Chroma.from_documents(chunks,embeddings,persist_directory=str(DB_DIR))
    print("建库完成")
    return vs


def answer(vx, question, k=3):
    # ---- 1. 一次检索同时取回「文本 + 距离」----
    # Chroma 1.1.0 的封装没有 similarity_search_with_score，所以走原生接口；
    # 但不要再"先 similarity_search 再单独 embed_query"：那是两次 embedding、
    # 两次独立查询，成本翻倍，而且两份结果只能靠"顺序碰巧一致"这个隐式约定对齐。
    query_vector = vx._embedding_function.embed_query(question)
    raw = vx._collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "distances"],
    )
    texts = raw["documents"][0]
    distances = raw["distances"][0]

    # ---- 2. 调试打印：距离越小越相关 ----
    print("\n[调试] 检索到的片段及距离：")
    for t, s in zip(texts, distances):
        one_line = t[:35].replace("\n", " ")
        print(f"   {s:.4f}  |  {one_line}...")

    # ---- 3. 拒答判断 ----
    top_score = distances[0]
    # 0.80 来自 44 条评测集扫 7 档阈值 0.75~1.05（见 README「拒答阈值调优」）：
    # 该档拒答准确率 100%、误拒 2/36。
    # 取舍理由：漏放（模型编造答案）比误拒（说"不知道"）严重得多。
    THRESHOLD = 0.80

    if top_score > THRESHOLD:
        print(f"\n[拒答] 最近距离 {top_score:.4f} > 阈值 {THRESHOLD}")
        msg = "抱歉，知识库中没有相关信息，建议转人工客服。"
        print("回答： ",msg)
        return msg, []

    # ---- 4. 拼 Prompt 问大模型 ----
    context = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))

    llm = ChatOpenAI(
        model=MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0,
    )
    prompt = f"""你是售后客服助手。请【只】根据下面的资料回答问题，
并在每个结论末尾用 [编号] 标注来源。
资料中没有的信息，回答"资料中未提及"，不要编造。

资料：
{context}

问题：{question}
"""
    resp = llm.invoke(prompt)

    print("\n" + "=" * 55)
    print("回答：", resp.content)
    print("-" * 55)
    print("参考来源：")
    for i, t in enumerate(texts):
        print(f"  [{i+1}] {t[:70]}...")
    print("=" * 55)

    # 返回 (答案, 引用片段)：API 层要把来源一起吐给调用方，不能只打印在控制台
    return resp.content, list(texts)

if __name__ == "__main__":
    print_env_banner()
    print("✅ 进入主流程了")
    docs = load_docs()
    chunks = split_docs(docs)
    vx = build_vectorstore(chunks)


    print("\n可以提问了（输入 q 退出）")
    while True:
            q =input("\n你问:").strip()
            if q.lower() in ('q','quit','exit'):
                break
            if not q:
                continue
            try:
                answer(vx,q)
            except Exception as e:
                print("❌ 出错：", type(e).__name__, e)
