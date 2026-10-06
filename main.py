
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
    # ---- 1. 取文档（用验证过可用的方法）----
    docs = vx.similarity_search(question, k=k)

    # ---- 2. 取距离（绕开坏掉的封装，用 Chroma 原生接口）----
    query_vector = vx._embedding_function.embed_query(question)
    raw = vx._collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["distances"],
    )
    distances = raw["distances"][0]

    # ---- 3. 调试打印：距离越小越相关 ----
    print("\n[调试] 检索到的片段及距离：")
    for d, s in zip(docs, distances):
        one_line = d.page_content[:35].replace("\n", " ")
        print(f"   {s:.4f}  |  {one_line}...")

    # ---- 4. 拒答判断 ----
    top_score = distances[0]
    # 0.80 来自 44 条评测集扫 6 档（见 README「拒答阈值调优」）：拒答准确率 100%、误拒 2/36。
    # 取舍理由：漏放（模型编造答案）比误拒（说"不知道"）严重得多。
    THRESHOLD = 0.80

    if top_score > THRESHOLD:
        print(f"\n[拒答] 最近距离 {top_score:.4f} > 阈值 {THRESHOLD}")
        msg = "抱歉，知识库中没有相关信息，建议转人工客服。"
        print("回答： ",msg)
        return msg, []

    # ---- 5. 拼 Prompt 问大模型 ----
    context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

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
    for i, d in enumerate(docs):
        print(f"  [{i+1}] {d.page_content[:70]}...")
    print("=" * 55)

    # 返回 (答案, 引用片段)：API 层要把来源一起吐给调用方，不能只打印在控制台
    return resp.content, [d.page_content for d in docs]

if __name__ == "__main__":
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

