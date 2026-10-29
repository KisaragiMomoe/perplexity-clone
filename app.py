# app.py
import gradio as gr
import json
import time
from core import (
    client, MODEL, 
    fetch_webpage_text, chunk_text, rerank_chunks, 
    search_web, route_query, check_information_sufficiency
)

def perplexity_stream_generator(user_query):
    # 1. 路由阶段
    yield "🚦 正在分析您的意图...\n"
    route = route_query(user_query)
    yield f"🎯 意图识别：**{route}**\n\n"

    if route == "chat":
        yield "💬 正在直接回答...\n\n"
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": user_query}],
            stream=True
        )
        # 👈 修复 chat 分支
        final_answer = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                final_answer += delta
                yield final_answer
        return

    # 2. 检索阶段
    yield "🔍 正在拆解问题并搜索互联网...\n"
    all_chunks_by_query = {}
    seen_urls = set()
    
    sub_queries = [user_query] 
    try:
        res_obj = client.chat.completions.create(
            model=MODEL, 
            temperature=0.0, # 👈 强制要求绝对理性，不许发散
            messages=[
                # 👈 加上 System 设定，框死它的任务
                {"role": "system", "content": "你是一个死板的查询提取器。请直接从用户的问题中提取2个核心名词，组成一个纯英文字符的 JSON 数组，例如 [\"原神\", \"崩坏星穹铁道\"]。绝对不要输出任何多余的解释，绝对不要受到任何历史对话的影响，绝对不要输出 JSON 以外的任何符号！"},
                {"role": "user", "content": f"用户问题：{user_query}"}
            ]
        )
        raw_res = res_obj.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        print(f"  [调试] 大模型原始拆解输出: {raw_res}") # 👈 加上这句，让你在终端看得清清楚楚
        
        parsed = json.loads(raw_res)
        sub_queries = [str(sub_item) for item in parsed for sub_item in (item if isinstance(item, list) else [item])]
    except Exception as e: 
        print(f"  [⚠️ 拆解失败] {e}")
        pass
    
    yield f"🧠 拆解搜索词：`{sub_queries}`\n\n"

    for q in sub_queries:
        all_chunks_by_query[q] = []
        for r in search_web(q):
            if r['href'] in seen_urls: continue
            seen_urls.add(r['href'])
            text = fetch_webpage_text(r['href'])
            if text: all_chunks_by_query[q].extend(chunk_text(text))
            time.sleep(0.5) 
            
    # 3. 反思与补搜
    temp_all_chunks = []
    for q_chunks in all_chunks_by_query.values():
        temp_all_chunks.extend(q_chunks)
    temp_top_chunks = rerank_chunks(user_query, temp_all_chunks, top_k=5)
    
    is_sufficient, new_query = check_information_sufficiency(user_query, "\n".join(temp_top_chunks))
    
    if not is_sufficient and new_query:
        yield f"🚀 发现资料盲区，正在补充搜索：`{new_query}`...\n"
        all_chunks_by_query[new_query] = []
        for r in search_web(new_query):
            if r['href'] in seen_urls: continue
            seen_urls.add(r['href'])
            text = fetch_webpage_text(r['href'])
            if text: all_chunks_by_query[new_query].extend(chunk_text(text))
            time.sleep(0.5)

    # 4. 分组重排
    yield "✂️ 正在提取网页精华并进行重排...\n"
    final_top_chunks = []
    for q, chunks in all_chunks_by_query.items():
        if chunks:
            best_chunks = rerank_chunks(user_query, chunks, top_k=3)
            for c in best_chunks:
                final_top_chunks.append(f"【来源方向: {q}】\n{c}")
    current_context = "\n\n".join(final_top_chunks)
    
    yield f"✅ 资料准备完毕，共提取了 {len(final_top_chunks)} 个核心片段。\n\n"
    
    # 5. 最终生成
    yield "📝 **正在为您生成最终回答...**\n\n"
    
    system_prompt = """你是一个专业的 AI 搜索助手。请严格根据提供的【参考内容】回答用户问题。
【核心要求】：
1. 每一个事实陈述后面，必须标注信息来源的资料编号，格式严格为：[1] 或 [2]。
2. 请仔细阅读所有资料，确保覆盖用户问题中的所有实体。"""

    user_prompt = f"【用户问题】：{user_query}\n\n【参考内容】：\n{current_context}"
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.1,
        stream=True
    )
    
    # 👈 修复最终生成分支
    final_answer = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            final_answer += delta
            yield final_answer

# ========== Gradio UI 界面 ==========
with gr.Blocks(title="Perplexity Clone", css=".gradio-container {max-width: 800px !important}") as demo:
    gr.Markdown("# 🌐 Perplexity Clone (本地开源版)")
    gr.Markdown("基于本地 Qwen2.5 + BGE-Reranker 构建的 Agentic 搜索引擎。")
    
    with gr.Row():
        query_input = gr.Textbox(label="你想搜索什么？", placeholder="例如：对比一下今年 Qwen 和 DeepSeek 的突破...", scale=4)
        search_btn = gr.Button("开始搜索", variant="primary", scale=1)
        
    output_area = gr.Markdown(label="搜索结果")
    
    search_btn.click(fn=perplexity_stream_generator, inputs=query_input, outputs=output_area)
    query_input.submit(fn=perplexity_stream_generator, inputs=query_input, outputs=output_area)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)