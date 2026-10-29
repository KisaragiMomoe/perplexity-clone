import json
import time
import requests
import trafilatura
from ddgs import DDGS
from openai import OpenAI
from sentence_transformers import CrossEncoder

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
MODEL = "qwen2.5:7b"
reranker = CrossEncoder('BAAI/bge-reranker-base')

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

def handle_chat(user_query: str):
    print("  [💬 通道：直接回答]")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_query}]
    )
    return response.choices[0].message.content

def handle_fast_search(user_query):
    print("  [⚡ 通道：单步快速搜索]")
    results = search_web(user_query)
    context = ""
    for r in results:
        text = fetch_webpage_text(r['href'])
        if text: context += text[:1500] + "\n" # 简单截取
        else: context += r.get('body', '') + "\n"

    system_prompt = "请根据资料回答用户问题，必须带上引用编号如[1]。不要废话。"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"问题：{user_query}\n资料：{context}"}],
        temperature=0.1
    )
    return response.choices[0].message.content

def handle_deep_research(user_query):
    print("  [🕵️ 通道：深度多跳搜索]")
    all_chunks_by_query = {}
    seen_urls = set()

    sub_queries = [user_query]
    try:
        # 👈 必须加上这两行，去请求大模型！
        res_obj = client.chat.completions.create(
            model=MODEL, 
            messages=[{"role": "user", "content": f"将问题拆解为2个简短搜索词，只输出JSON数组：{user_query}"}]
        )
        res = res_obj.choices[0].message.content # 提取文本
        
        # 清理格式并解析
        parsed = json.loads(res.replace("```json", "").replace("```", "").strip())
        sub_queries = [str(sub_item) for item in parsed for sub_item in (item if isinstance(item, list) else [item])]
    except Exception as e: 
        print(f"拆解失败: {e}")
        pass
    
    # 2. 第一跳搜索
    for q in sub_queries:
        all_chunks_by_query[q] = []
        for r in search_web(q):
            if r['href'] in seen_urls: continue
            seen_urls.add(r['href'])
            text = fetch_webpage_text(r['href'])
            if text: all_chunks_by_query[q].extend(chunk_text(text))
            time.sleep(1)

    temp_all_chunks = []
    for q_chunks in all_chunks_by_query.values(): 
        temp_all_chunks.extend(q_chunks)
    temp_top_chunks = rerank_chunks(user_query, temp_all_chunks, top_k = 5)

    is_sufficient, new_query = check_information_sufficiency(user_query, "\n".join(temp_top_chunks))
    
    # 5. 如果不够，触发第二跳
    if not is_sufficient and new_query:
        print(f"  [🚀 触发第二跳] 补充搜索词：{new_query}")
        all_chunks_by_query[new_query] = []
        for r in search_web(new_query):
            if r['href'] in seen_urls: continue
            seen_urls.add(r['href'])
            text = fetch_webpage_text(r['href'])
            if text: all_chunks_by_query[new_query].extend(chunk_text(text))
            time.sleep(1)
    else:
        print("  [✅ 资料已充分，无需补充]")

    # ==========================================
    # 6. 👈 核心：分组重排（绝对公平，每个词保底3块）
    # ==========================================

    final_top_chunks = []

    for q, chunks in all_chunks_by_query.items():
        if chunks:
            best_chunks_for_q = rerank_chunks(user_query, chunks, top_k = 3)
            for c in best_chunks_for_q:
                # 打上来源标签，方便大模型知道这是关于谁的
                final_top_chunks.append(f"【来源方向: {q}】\n{c}")
    current_context = "\n\n".join(final_top_chunks)
    # 7. 最终生成
    system_prompt = """你是一个专业的 AI 搜索助手。请严格根据提供的【参考内容】回答用户问题。
【核心要求】：
1. 每一个事实陈述后面，必须标注信息来源的资料编号，格式严格为：[1] 或 [2]。
2. 请仔细阅读所有资料，确保覆盖用户问题中的所有实体。"""

    user_prompt = f"【用户问题】：{user_query}\n\n【参考内容】：\n{current_context}"
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

def ultimate_agent(user_query: str):
    print(f"\n🔍 用户提问：{user_query}")
    
    # 1. 路由决策
    route = route_query(user_query)
    print(f"  [🎯 决策] 将问题分配给：{route}")

    start_time = time.time()
    if route == "chat":
        answer = handle_chat(user_query)
    elif route == "fast_search":
        answer = handle_fast_search(user_query)
    else:
        answer = handle_deep_research(user_query)
        
    end_time = time.time()

    print("\n" + "="*50)
    print(f"🤖 最终回答（耗时 {end_time - start_time:.2f} 秒）：\n{answer}")
    print("="*50)

if __name__ == "__main__":
    # 测试 1：闲聊
    ultimate_agent("你好呀，请问你叫什么名字？")
    
    # 测试 2：快速事实
    ultimate_agent("今年诺贝尔物理学奖得主是谁？")
    
    # 测试 3：深度搜索
    ultimate_agent("对比一下今年最新的 Qwen 和 DeepSeek 在开源大模型领域的突破。")