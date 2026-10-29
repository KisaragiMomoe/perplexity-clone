import time
import json
import requests
import trafilatura
from ddgs import DDGS
from openai import OpenAI
from sentence_transformers import CrossEncoder

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
MODEL = "qwen2.5:7b"

print("加载重排模型...")
reranker = CrossEncoder('BAAI/bge-reranker-base')

def fetch_webpage_text(url: str):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        extracted_text = trafilatura.extract(response.text)
        return extracted_text if extracted_text else ""
    except:
        return ""

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50):
    chunks = []

    for p in text.split('\n'):
        if p.strip():
            chunks.append(p.strip())

    final_chunks = []
    current_chunk = ""
    for p in chunks:
        if len(current_chunk) + len(p) <= chunk_size:
            current_chunk += p + "\n"
        else:
            if current_chunk.strip(): 
                final_chunks.append(current_chunk.strip())
            current_chunk = p + "\n"
    if current_chunk.strip():
        final_chunks.append(current_chunk.strip())
    return final_chunks

def rerank_chunks(query: str, all_chunks: list, top_k: int = 4):
    if not all_chunks:  
        return []
    pairs = [[query, chunk] for chunk in all_chunks]
    scores = reranker.predict(pairs)
    scored_chunks = list(zip(scores, all_chunks))
    scored_chunks.sort(key = lambda x: x[0], reverse = True)
    return [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

def search_web(query: str, max_results = 2):
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results = max_results))
    except:
        time.sleep(1)
        return []

def route_query(user_query: str):
    """判断问题属于哪种类型，返回: chat / fast_search / deep_research"""
    print("  [🚦 路由] 正在分析用户意图...")
    system_prompt = """你是一个意图分类器。请将用户的问题分为以下三类之一，只输出对应的英文单词：
1. chat: 日常闲聊、问候、或者纯主观的无需搜索的问题（如：你是谁、写一首诗）。
2. fast_search: 只需要查一个事实或单一信息的问题（如：今天天气、现任联合国秘书长是谁）。
3. deep_research: 需要对比、多实体查询、长篇分析、或者最新行业动态的问题（如：对比苹果和华为、最新大模型突破）。

只输出：chat 或 fast_search 或 deep_research。不要输出任何其他标点符号。"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
        temperature=0.1
    )
    route_type = response.choices[0].message.content.strip().lower()

    if "chat" in route_type: return "chat"
    elif "fast" in route_type: return "fast_search"
    else: return "deep_research"

def check_information_sufficiency(user_query: str, current_context: str):
    """
    让大模型反思当前的资料是否足够。
    返回：(是否充分, 建议补充的搜索词)
    """
    print("  [🧠 反思] 检查资料是否全面...")
    system_prompt = """你是一个信息审计员。请判断【当前资料】是否足够全面回答【用户问题】。
特别是当问题包含多个实体（如A和B）时，必须确保A和B的信息都有。
如果资料充分，只输出：["SUFFICIENT"]
如果资料不全面，请输出一个 JSON 数组，包含1个建议补充搜索的关键词。
不要输出任何其他解释。"""

    user_prompt = f"【用户问题】：{user_query}\n\n【当前资料】：\n{current_context[:2500]}"

    response = client.chat.completions.create(
        model = MODEL,
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.1 
    )

    try:
        raw = response.choices[0].message.content.strip()
        if "```json" in raw: raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        
        if result == ["SUFFICIENT"] or "SUFFICIENT" in result:
            return True, None
        else:
            return False, result[0] # 返回补充搜索词
    except Exception as e:
        return True, None
