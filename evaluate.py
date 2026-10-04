"""
评测脚本：自动跑测试集，输出检索命中率与拒答准确率
运行：python evaluate.py
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- 配置 ----------
load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "policies"
QUESTIONS_FILE = BASE_DIR / "tests" / "questions.json"

THRESHOLD = 0.80         # 拒答阈值

# 读取 API Key（两个变量名都试，避免不一致）
SILICON_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("SILICON_API_KEY")

if not SILICON_KEY:
    raise SystemExit(
        "❌ 没读到硅基流动的 API Key\n"
        "   请检查 .env 里是否有 SILICONFLOW_API_KEY=sk-xxx\n"
        "   并确认 .env 在项目根目录"
    )

print("✅ 硅基流动 Key 已读取:", SILICON_KEY[:8] + "...")

EMBEDDINGS = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=SILICON_KEY,
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
)


# ---------- 工具函数 ----------
def normalize(text: str) -> str:
    """去掉所有空白字符，避免 '7 个自然日' 和 '7个自然日' 匹配失败"""
    return re.sub(r"\s+", "", text)


def load_docs():
    docs = []
    for md in sorted(DATA_DIR.glob("*.md")):
        docs.extend(TextLoader(str(md), encoding="utf-8").load())
    return docs


def build_vectorstore(chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", ""],
    )
    chunks = splitter.split_documents(load_docs())
    return Chroma.from_documents(chunks, EMBEDDINGS)


def retrieve(vs, question: str, k: int):
    """返回 [(Document, 距离), ...]，距离越小越相关"""
    docs = vs.similarity_search(question, k=k)
    qv = vs._embedding_function.embed_query(question)
    raw = vs._collection.query(
        query_embeddings=[qv], n_results=k, include=["distances"]
    )
    return list(zip(docs, raw["distances"][0]))


# ---------- 评测 ----------
def evaluate(vs, questions, k):
    hit = should = false_reject = correct_reject = noanswer = 0
    details = []

    for q in questions:
        results = retrieve(vs, q["question"], k)
        top_score = results[0][1]
        all_text = normalize("".join(d.page_content for d, _ in results))

        if q["should_answer"]:
            should += 1
            matched = any(normalize(kw) in all_text for kw in q["answer_keywords"])
            if matched:
                hit += 1
            is_false_reject = top_score > THRESHOLD
            if is_false_reject:
                false_reject += 1
            details.append({
                "id": q["id"], "question": q["question"],
                "matched": matched, "distance": top_score,
                "false_reject": is_false_reject,
            })
        else:
            noanswer += 1
            ok = top_score > THRESHOLD
            if ok:
                correct_reject += 1
            details.append({
                "id": q["id"], "question": q["question"],
                "matched": None, "distance": top_score,
                "correct_reject": ok,
            })

    return {
        "hit": hit, "should": should,
        "hit_rate": hit / should if should else 0,
        "correct_reject": correct_reject, "noanswer": noanswer,
        "reject_rate": correct_reject / noanswer if noanswer else 0,
        "false_reject": false_reject,
        "details": details,
    }


def sweep_threshold(vs, questions, k):
    """扫描不同阈值下的误拒数与拒答准确率"""
    answer_dists, noanswer_dists = [], []
    for q in questions:
        s = retrieve(vs, q["question"], k)[0][1]
        if q["should_answer"]:
            answer_dists.append(s)
        else:
            noanswer_dists.append(s)

    print(f"\n{'阈值':>6} {'误拒数':>7} {'拒答准确率':>10}")
    print("-" * 28)
    for th in [0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]:
        fr = sum(1 for s in answer_dists if s > th)
        rr = sum(1 for s in noanswer_dists if s > th) / len(noanswer_dists)
        print(f"{th:>6.2f} {fr:>7} {rr:>9.1%}")


# ---------- 主流程 ----------
def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    print(f"共加载 {len(questions)} 条测试问题\n")

    configs = [
        (300, 50, 3),
    ]

    print(f"{'chunk_size':>10} {'k':>3} {'检索命中率':>10} {'误拒数':>7} {'拒答准确率':>10}")
    print("-" * 50)

    result = None
    vs = None
    for cs, co, k in configs:
        vs = build_vectorstore(cs, co)
        result = evaluate(vs, questions, k)
        print(f"{cs:>10} {k:>3} {result['hit_rate']:>9.1%} "
              f"{result['false_reject']:>7} {result['reject_rate']:>9.1%}")

    print("\n" + "=" * 75)
    print("明细（chunk_size=300, k=3）")
    print("=" * 75)
    for d in result["details"]:
        if d["matched"] is None:
            tag = "✅ 正确拒答" if d["correct_reject"] else "❌ 漏放"
        else:
            tag = "✅ 命中" if d["matched"] else "❌ 未命中"
            if d["false_reject"]:
                tag += " / ⚠️ 误拒"
        print(f"  [{d['id']:>2}] {tag:<18} 距离={d['distance']:.4f}  {d['question']}")

    sweep_threshold(vs, questions, 3)


if __name__ == "__main__":
    main()