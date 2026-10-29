import requests
import trafilatura
import json
from ddgs import DDGS
from openai import OpenAI
import time

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
MODEL = "qwen2.5:7b"

def fetch_webpage_text(url: str, max_chars: int = 2000):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        extracted_text = trafilatura.extract(response.text)
        return extracted_text[:max_chars].strip() if extracted_text else ""
    except:
        return ""

def generate_sub_queries(user_query: str):
    """让大模型把复杂问题拆解成多个搜索词"""
    print("  [🧠 大脑] 正在拆解复杂问题...")

    system_prompt = """你是一个专业的搜索引擎查询生成器。
用户会给你一个复杂的提问，请你把它拆解成 3 个不同角度的、简短且精准的搜索关键词（像人类在百度/谷歌里搜的那样）。
请直接输出一个 JSON 格式的字符串数组，不要输出任何其他废话。
例如：["iPhone 16 影像评测", "华为 Mate 70 拍照", "iPhone 16 华为 对比"]"""

    response = client.chat.completions.create(
        model = MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.1
    )

    try:
        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        queries = json.loads(raw_text)
        print(f"         -> 拆解出的搜索词: {queries}")
        return queries
    except Exception as e:
        print(f"         [⚠️ 解析失败] 模型未输出合法 JSON，使用原问题作为搜索词。")
        return [user_query]

def multi_search_and_crawl(sub_queries: list, max_results_per_query=2):
    """遍历多个子查询，获取网址并爬取正文"""
    print(f"  [🌐 检索] 正在并发执行 {len(sub_queries)} 路搜索...")

    context_text = ""
    urls = []
    seen_urls = set() # 用于去重

    for q_idx, q in enumerate(sub_queries):
        try:
            # 加一个 try-except，防止某一个搜索词没结果导致整个程序崩溃
            with DDGS() as ddgs:
                results = list(ddgs.text(q, max_results=max_results_per_query))
        except Exception as e:
            print(f"        [⚠️ 搜索被拦截或失败] 关键词 '{q}' 搜不到结果: {e}")
            # 如果搜不到，等待 2 秒，继续搜下一个词
            time.sleep(2) 
            continue

        for res in results:
            url = res['href']

            if url in seen_urls: # 去重
                continue
            seen_urls.add(url)

            title = res['title']
            snippet = res.get('body', '')
            urls.append(url)

            print(f"        -> 阅读网址 [{len(urls)}]: {url}")
            page_text = fetch_webpage_text(url)

            if page_text:
                context_text += f"\n--- 网页 [{len(urls)}] 来源搜索词: {q} | 标题: {title} ---\n{page_text}\n"
            elif snippet:
                context_text += f"\n--- 网页 [{len(urls)}] 来源搜索词: {q} | 标题: {title} ---\n[摘要] {snippet}\n"
        time.sleep(1.5)
    return context_text, urls

def perplexity_fanout_pipeline(user_query: str):
    print(f"\n🔍 用户提问：{user_query}")
    
    # 第 1 步：拆解
    sub_queries = generate_sub_queries(user_query)
    
    # 第 2 步：多路检索
    context_text, urls = multi_search_and_crawl(sub_queries)
    
    # 第 3 步：综合生成
    print("  [🤖 生成] 正在综合多个网页信息生成回答...")
    system_prompt = """你是一个专业的 AI 搜索助手。请严格根据提供的【参考网页内容】来回答用户问题。
【核心要求】：
1. 答案要全面，尽量对比或综合不同网页的信息。
2. 每一个事实陈述后面，必须标注信息来源的编号，格式严格为：[1] 或 [2]。
3. 如果找不到答案，直接回答“无法确定”，绝对不要编造。"""

    user_prompt = f"""【用户问题】：{user_query}

【参考网页内容】：
{context_text}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    
    answer = response.choices[0].message.content

    print("\n" + "="*50)
    print(f"🤖 智能搜索回答：\n{answer}")
    print("="*50)
    print("📚 参考来源：")
    for i, url in enumerate(urls):
        print(f"[{i+1}] {url}")

if __name__ == "__main__":
    # 测试一个复杂问题
    perplexity_fanout_pipeline("今年最新的国产大模型（比如 Qwen 和 DeepSeek）在开源界有什么重大突破？")
