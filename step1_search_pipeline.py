import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from openai import OpenAI
import json

# 连接本地 Ollama
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
MODEL = "qwen2.5:7b"

