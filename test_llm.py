import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model = os.getenv("LLM_MODEL")

print("【检查1】Key 读到了吗：", (api_key[:10] + "...") if api_key else "❌ 没读到！")
print("【检查2】base_url =", base_url)
print("【检查3】model =", model)

client = OpenAI(api_key=api_key,base_url=base_url)

try:
    resp=client.chat.completions.create(
        model=model,
        messages=[{'role':'user',"content":'只回复两个字:通了'}],
    )
    print("\n✅ 模型回复：", resp.choices[0].message.content)
except Exception as e:
    print("\n❌ 调用失败：")
    print("   错误类型：", type(e).__name__)
    print("   错误内容：", repr(e))
