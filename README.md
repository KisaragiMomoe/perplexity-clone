# 本地版 Perplexity (Agentic RAG Search)

这是一个完全基于本地开源生态（Ollama + Qwen2.5 + BGE-Reranker + ChromaDB）构建的 **Perplexity 风格** AI 搜索引擎。无需翻墙，无需 OpenAI API，完全在本地运行。

## ✨ 核心特性
- **智能意图路由 (Query Router)**：自动判断闲聊、快速查询、深度研究，节省算力。
- **多路检索与扇出 (Fan-out)**：自动拆解复杂问题为多个子查询。
- **精准重排 (Reranker)**：引入 BAAI/bge-reranker-base 对网页抓取内容进行二次精排，过滤广告与无关信息。
- **分组重排配额**：解决多实体查询中的“偏科”问题，保证对比对象的信息都能被召回。
- **自反思与多跳搜索 (Multi-hop & Self-Reflection)**：如果第一次搜索结果不够全面，模型会自主生成补充搜索词，进行第二跳深挖。
- **流式输出 (Streaming)**：基于 Gradio 实现打字机效果，并附带实时搜索状态提示。

## 🛠️ 技术栈
- **本地大模型**：Ollama (Qwen2.5-7B)
- **嵌入与重排**：BAAI/bge-m3 (可选), BAAI/bge-reranker-base
- **搜索引擎**：DuckDuckGo Search API
- **网页正文提取**：Trafilatura
- **前端框架**：Gradio

## 🚀 快速开始
1. 安装 [Ollama](https://ollama.com/) 并拉取模型：
   ```bash
   ollama run qwen2.5:7b
   ```
2. 安装依赖：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. 启动应用：
   ```bash
   python app.py
   ```
4. 在浏览器打开 `http://127.0.0.1:7860`。