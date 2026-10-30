"""调用大模型的公共部分：读 key + 一个 POST

抽出来的原因：answer.py（生成）和 rewrite.py（改写）都要调模型，
但 rewrite 要 import answer 的 ask_llm、answer 又要 import rewrite 的检索，
互相导入会死锁。把「发请求」这件事单独放一层，两边都依赖它，谁也不依赖谁。

这一层不该懂业务——system 提示由调用方传进来。
"""
import json
import os
import urllib.error
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"


def load_key():
    """先读环境变量，再读同目录 .env（两行就够了，不引第三方库）"""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY"):
                    return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def ask_llm(prompt, key, system, temperature=0.1, max_tokens=512):
    """OpenAI 兼容接口，一个 POST 就够，没必要为它装个 SDK

    system 必传：这一层不懂业务，宪法和改写规则都由调用方决定。
    """
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"], data.get("usage", {})
