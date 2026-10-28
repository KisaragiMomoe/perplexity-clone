import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from openai import OpenAI
import trafilatura
from urllib.parse import urlparse
import json

# 连接本地 Ollama
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
MODEL = "qwen2.5:7b"

def fetch_webpage_text(url: str, max_chars: int = 3000):
    """访问网页，提取纯文本，截断到 max_chars"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        # 1. 发送请求
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        # 2. 用 trafilatura 提取核心正文（自动剔除广告、导航栏）
        extracted_text = trafilatura.extract(response.text)
        
        if not extracted_text:
            return ""
            
        # 截断
        return extracted_text[:max_chars].strip()
        
    except Exception as e:
        # 如果 requests 报错，这里不打印长篇大论的报错，直接返回空
        return ""
def perplexity_search_pipeline(query: str):
    print(f"\n🔍 用户提问：{query}")

    print("  [1/3] 正在搜索互联网...")
    with DDGS() as ddgs:
        search_results = list(ddgs.text(query, max_results = 3))

    if not search_results:
        return "未找到相关结果。"

    print("  [2/3] 正在爬取网页正文...")
    context_text = ""
    urls = []

    for i, res in enumerate(search_results):
        url = res['href']
        title = res['title']
        snippet = res.get('body', '')
        urls.append(url)
        print(f"        -> 正在阅读网址 [{i+1}]: {url}")
        page_text = fetch_webpage_text(url)
        if page_text:
            context_text += f"\n--- 网页 [{i+1}] 标题: {title} ---\n{page_text}\n"
        elif snippet:
            context_text += f"\n--- 网页 [{i+1}] 标题: {title} ---\n[只有摘要] {snippet}\n"
        else:
            context_text += f"\n--- 网页 [{i+1}] 标题: {title} ---\n(无法提取内容)\n"

    print("  [3/3] 正在让大模型精读并生成回答...")
    system_prompt = """你是一个专业的 AI 搜索助手。请严格根据我提供的网页内容来回答用户问题。
【核心要求】：
1. 答案必须真实、准确、精炼。
2. 你的回答中，每一个事实陈述后面，必须标注信息来源的编号，格式严格为：[1] 或 [2]。
3. 如果提供的网页内容中找不到答案，请直接回答“根据当前搜索结果，无法确定”，绝对不要编造。
4. 不要罗列你的思考过程，直接给出最终回答。"""

    user_prompt = f"""【用户问题】：{query}

【参考网页内容】：
{context_text}
"""
    response = client.chat.completions.create(
        model = MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1 # 搜索场景要求严谨
    )

    answer = response.choices[0].message.content

    print("\n" + "="*50)
    print(f"🤖 智能搜索回答：\n{answer}")
    print("="*50)
    print("📚 参考来源：")
    for i, url in enumerate(urls):
        print(f"[{i+1}] {url}")
        
    return answer

# ========== 测试运行 ==========
if __name__ == "__main__":
    # 测试一个需要结合实时信息的复杂问题
    perplexity_search_pipeline("2024年诺贝尔物理学奖颁给了谁？他们的主要贡献是什么？")