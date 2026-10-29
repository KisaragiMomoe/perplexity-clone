import requests
import trafilatura
import json
import time
import re
from ddgs import DDGS
from openai import OpenAI
# 👈 引入重排模型
from sentence_transformers import CrossEncoder

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
MODEL = "qwen2.5:7b"

# ========== 0. 加载重排模型（全局加载一次即可） ==========
print("正在加载 BGE-Reranker 模型（首次运行需下载，请稍候）...")
# BAAI/bge-reranker-base 是国内智源开源的神级重排模型
reranker = CrossEncoder('BAAI/bge-reranker-base')
print("重排模型加载完成！\n")

def fetch_webpage_text(url: str):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        extracted_text = trafilatura.extract(response.text)
        return extracted_text if extracted_text else ""
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

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50):
    chunks = []
    paragraphs = text.split('\n')
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) <= chunk_size:
            current_chunk += p + "\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def rerank_chunks(query: str, all_chunks: list, top_k: int = 4):
    print(f"  [🔄 重排] 正在给 {len(all_chunks)} 个文本块打分...")
    pairs = [[query, chunk] for chunk in all_chunks]
    scores = reranker.predict(pairs)
    scored_chunks = list(zip(scores, all_chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]
    print(f"         -> 保留了 {len(top_chunks)} 个最相关的文本块。")
    return top_chunks

def perplexity_rerank_pipeline(user_query: str):
    print(f"\n🔍 用户提问：{user_query}")

    sub_queries = generate_sub_queries(user_query)
    context_text, urls = multi_search_and_crawl(sub_queries)
    
    print("  [✂️ 切块] 正在对网页内容进行分块...")
    all_chunks = chunk_text(context_text, chunk_size=400)
    print(f"         -> 共切出 {len(all_chunks)} 个文本块。")

    top_chunks = rerank_chunks(user_query, all_chunks, top_k=8)

    final_context = ""
    for i, chunk in enumerate(top_chunks):
        final_context += f"\n【资料 {i+1}】: {chunk}\n"

    print("  [🤖 生成] 正在基于精华文本生成回答...")
    system_prompt = """你是一个专业的 AI 搜索助手。请严格根据提供的【参考内容】回答用户问题。
【核心要求】：
1. 每一个事实陈述后面，必须标注信息来源的资料编号，格式严格为：[1] 或 [2]。
2. 语言要高度凝练，不要罗列废话。
3. **【重要】请仔细阅读所有资料。如果用户的问题中包含多个实体（比如比较 Qwen 和 DeepSeek），请确保覆盖所有实体。如果资料中确实只提到了其中一个，请在回答结尾明确指出“资料中未提及另一个”**。"""

    user_prompt = f"""【用户问题】：{user_query}
【参考内容】：
{final_context}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.3
    )
    
    print("\n" + "="*50)
    print(f"🤖 智能搜索回答：\n{response.choices[0].message.content}")
    print("="*50)

if __name__ == "__main__":
    # 测试
    perplexity_rerank_pipeline("今年最新的国产大模型（比如 Qwen 和 DeepSeek）在开源界有什么重大突破？")
