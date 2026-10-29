import requests
import trafilatura
import json
import time
import re
from ddgs import DDGS
from openai import OpenAI
from sentence_transformers import CrossEncoder

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
MODEL = "qwen2.5:7b"

print("正在加载 BGE-Reranker 模型...")
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

def check_information_sufficiency(user_query: str, current_context: str):
    """
    让大模型反思当前的资料是否足够。
    返回：(是否充分, 建议补充的搜索词)
    """
    print("  [🧠 反思] 检查资料是否全面...")
    system_prompt = """你是一个信息审计员。请判断【当前资料】是否足够全面回答【用户问题】。
特别是当问题包含多个实体（如A和B）时，必须确保A和B的信息都有。
如果资料充分，只输出：["SUFFICIENT"]
如果资料不全面，请输出一个 JSON 数组，包含1个建议补充搜索的关键词。例如：["DeepSeek 突破"]
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
        print(f"         [⚠️ 反思解析失败] {e}，默认判定充分。")
        return True, None

def search_web(query: str, max_results = 2):
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results = max_results))
    except:
        time.sleep(2)
        return []

def perplexity_multihop_pipeline(user_query: str):
    print(f"\n🔍 用户提问：{user_query}")
    
    all_context_chunks = []
    seen_urls = set()

    print("  [--- 第一跳搜索 ---]")
    sub_queries = [user_query]
    try:
        res = client.chat.completions.create(
            model = MODEL,
            messages=[{"role": "user", "content": f"将问题拆解为2个简短搜索词，只输出JSON数组：{user_query}"}],
        ).choices[0].message.content
        parsed = json.loads(res.replace("```json", "").replace("```", "").strip())
        sub_queries = [str(item) for item in parsed]
        if isinstance(parsed[0], list):
            sub_queries = [str(sub_item) for item in parsed for sub_item in item]
    except: pass

    print(f"         -> 初始搜索词: {sub_queries}")

    for q in sub_queries:
        results = search_web(q)
        for r in results:
            if r['href'] in seen_urls: continue
            seen_urls.add(r['href'])
            text = fetch_webpage_text(r['href'])
            if text: all_context_chunks.extend(chunk_text(text))
            time.sleep(1)

    top_chunks = rerank_chunks(user_query, all_context_chunks, top_k = 4)
    current_context = "\n".join(top_chunks)

    is_sufficient, new_query = check_information_sufficiency(user_query, current_context)

    if not is_sufficient and new_query:
        print(f"  [🚀 触发第二跳] 补充搜索词：{new_query}")
        results = search_web(new_query)
        for r in results:
            if r['href'] in seen_urls: continue
            seen_urls.add(r['href'])
            text = fetch_webpage_text(r['href'])
            if text: all_context_chunks.extend(chunk_text(text))
            time.sleep(1)
            
        # 第二跳重排，扩充上下文
        print("  [🔄 重新重排全部资料]")
        top_chunks = rerank_chunks(user_query, all_context_chunks, top_k=6) # 这次多留一点
        current_context = "\n".join(top_chunks)
    else:
        print("  [✅ 资料已充分，无需补充]")

    print("  [🤖 生成] 正在综合最终资料生成回答...")
    system_prompt = """你是一个专业的 AI 搜索助手。请严格根据提供的【参考内容】回答用户问题。
【核心要求】：
1. 事实必须带有引用编号，例如 [1] 或 [2]。
2. 请仔细阅读所有资料，并全面回答用户关于各个实体的问题。
3. 如果资料中确实**完全缺失**某个实体的信息，请在回答的末尾简短说明（如果资料中已有该实体信息，绝对不要加此说明）。"""
    user_prompt = f"【用户问题】：{user_query}\n\n【参考内容】：\n{current_context}"
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.1
    )
    
    print("\n" + "="*50)
    print(f"🤖 智能搜索回答：\n{response.choices[0].message.content}")
    print("="*50)

if __name__ == "__main__":
    perplexity_multihop_pipeline("今年最新的国产大模型（比如 Qwen 和 DeepSeek）在开源界有什么重大突破？")
