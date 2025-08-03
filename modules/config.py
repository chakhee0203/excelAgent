import streamlit as st
from typing import Dict
import json
import os

# 预设模型配置
DEFAULT_MODELS = {
    "OpenAI GPT-4": {
        "model_name": "gpt-4",
        "base_url": "https://api.openai.com/v1",
        "description": "OpenAI官方GPT-4模型，功能最强大"
    },
    "OpenAI GPT-3.5-Turbo": {
        "model_name": "gpt-3.5-turbo",
        "base_url": "https://api.openai.com/v1",
        "description": "OpenAI官方GPT-3.5模型，性价比高"
    },
    "DeepSeek Chat": {
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "description": "DeepSeek智能对话模型，中文友好"
    },
    "Moonshot v1": {
        "model_name": "moonshot-v1-8k",
        "base_url": "https://api.moonshot.cn/v1",
        "description": "月之暗面Kimi模型，长文本处理能力强"
    },
    "通义千问": {
        "model_name": "qwen-turbo",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "description": "阿里云通义千问模型，中文理解优秀"
    },
    "智谱GLM": {
        "model_name": "glm-4",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "description": "智谱AI GLM模型，多模态能力强"
    },
    "百川大模型": {
        "model_name": "Baichuan2-Turbo",
        "base_url": "https://api.baichuan-ai.com/v1",
        "description": "百川智能大模型，推理能力强"
    },
    "文心一言": {
        "model_name": "ERNIE-Bot-turbo",
        "base_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
        "description": "百度文心一言，中文语言模型"
    },
    "讯飞星火": {
        "model_name": "generalv3.5",
        "base_url": "https://spark-api-open.xf-yun.com/v1",
        "description": "科大讯飞星火认知大模型"
    },
    "自定义模型": {
        "model_name": "",
        "base_url": "",
        "description": "自定义配置的模型"
    }
}

def setup_page_config():
    """设置页面配置"""
    st.set_page_config(
        page_title="Excel智能助手",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_custom_css():
    """加载自定义CSS样式"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .ai-response {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        font-family: 'Microsoft YaHei', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

def load_config():
    """加载配置"""
    config_file = "config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "api_key": "",
            "model_config": DEFAULT_MODELS["OpenAI GPT-3.5-Turbo"]
        }

def save_config(config):
    """保存配置"""
    with open("config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def get_default_models():
    """获取默认模型配置"""
    return DEFAULT_MODELS