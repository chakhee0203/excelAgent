import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64
from datetime import datetime
import os
from pathlib import Path
import json
from typing import Dict, List, Optional

# 尝试导入LangChain相关模块
try:
    from langchain_openai import ChatOpenAI
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    # 只在session state中记录错误，避免重复显示警告
    if 'langchain_import_error' not in st.session_state:
        st.session_state.langchain_import_error = str(e)
        st.error(f"❌ LangChain导入失败: {str(e)}")
        st.info("💡 请运行以下命令安装LangChain: pip install langchain langchain-openai langchain-community")

# 页面配置
st.set_page_config(
    page_title="Excel智能助手",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
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
        "description": "DeepSeek深度求索，国产优秀模型"
    },
    "DeepSeek Coder": {
        "model_name": "deepseek-coder",
        "base_url": "https://api.deepseek.com/v1",
        "description": "DeepSeek代码专用模型，编程能力强"
    },
    "通义千问 Qwen-Plus": {
        "model_name": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "description": "阿里云通义千问Plus，中文理解优秀"
    },
    "通义千问 Qwen-Turbo": {
        "model_name": "qwen-turbo",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "description": "阿里云通义千问Turbo，响应速度快"
    },
    "通义千问 Qwen-Max": {
        "model_name": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "description": "阿里云通义千问Max，最强性能版本"
    },
    "豆包 Doubao-Pro": {
        "model_name": "doubao-pro-4k",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "description": "字节跳动豆包Pro，多模态能力强"
    },
    "豆包 Doubao-Lite": {
        "model_name": "doubao-lite-4k",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "description": "字节跳动豆包Lite，轻量高效"
    },
    "智谱 GLM-4": {
        "model_name": "glm-4",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "description": "智谱AI GLM-4，国产大模型标杆"
    },
    "智谱 GLM-4-Flash": {
        "model_name": "glm-4-flash",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "description": "智谱AI GLM-4-Flash，超快响应"
    },
    "智谱 GLM-3-Turbo": {
        "model_name": "glm-3-turbo",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "description": "智谱AI GLM-3-Turbo，性价比优选"
    },
    "百川 Baichuan2-Turbo": {
        "model_name": "baichuan2-turbo",
        "base_url": "https://api.baichuan-ai.com/v1",
        "description": "百川智能Baichuan2-Turbo，中文优化"
    },
    "文心一言 ERNIE-4.0": {
        "model_name": "ernie-4.0-8k",
        "base_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
        "description": "百度文心一言4.0，理解能力强"
    },
    "Moonshot AI": {
        "model_name": "moonshot-v1-8k",
        "base_url": "https://api.moonshot.cn/v1",
        "description": "月之暗面Kimi，长文本处理专家"
    },
    "Azure OpenAI": {
        "model_name": "gpt-4",
        "base_url": "https://your-resource.openai.azure.com/",
        "description": "Azure OpenAI服务"
    },
    "国内代理服务": {
        "model_name": "gpt-3.5-turbo",
        "base_url": "https://api.chatanywhere.com.cn/v1",
        "description": "国内代理服务，访问稳定"
    },
    "自定义模型": {
        "model_name": "custom-model",
        "base_url": "https://your-custom-api.com/v1",
        "description": "自定义API服务"
    }
}

class ExcelAgentFull:
    """完整版Excel智能分析助手"""
    
    def __init__(self, api_key: str, model_config: Dict):
        self.api_key = api_key
        self.model_config = model_config
        self.llm = None
        self.agent = None
        
        if LANGCHAIN_AVAILABLE and api_key:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """初始化语言模型"""
        try:
            self.llm = ChatOpenAI(
                model=self.model_config["model_name"],
                openai_api_key=self.api_key,
                openai_api_base=self.model_config["base_url"],
                temperature=0.1,
                max_tokens=2000
            )
            return True
        except Exception as e:
            st.error(f"❌ 模型初始化失败: {str(e)}")
            return False
    
    def create_dataframe_agent(self, df: pd.DataFrame):
        """创建数据框智能代理"""
        if not self.llm:
            return None
        
        try:
            self.agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                verbose=True,
                allow_dangerous_code=True,
                return_intermediate_steps=True
            )
            return self.agent
        except Exception as e:
            st.error(f"❌ 智能代理创建失败: {str(e)}")
            return None
    
    def analyze_data_with_ai(self, df: pd.DataFrame, query: str) -> str:
        """使用AI分析数据"""
        if not self.llm:
            return "❌ AI功能不可用，请检查API配置"
        
        try:
            # 创建数据分析提示模板
            analysis_prompt = PromptTemplate(
                input_variables=["data_info", "query"],
                template="""
                你是一个专业的数据分析师。请根据以下数据信息和用户查询，提供详细的中文分析报告。
                
                数据信息：
                {data_info}
                
                用户查询：{query}
                
                请提供：
                1. 数据概览和关键发现
                2. 具体的分析结果
                3. 数据洞察和建议
                4. 可能的后续分析方向
                
                请用中文回答，格式清晰，重点突出。
                """
            )
            
            # 准备数据信息
            data_info = f"""
            数据形状: {df.shape[0]}行 x {df.shape[1]}列
            列名: {list(df.columns)}
            数据类型: {df.dtypes.to_dict()}
            基本统计: {df.describe().to_string()}
            缺失值: {df.isnull().sum().to_dict()}
            """
            
            # 创建分析链
            analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
            
            # 执行分析
            result = analysis_chain.invoke({
                "data_info": data_info,
                "query": query
            })
            
            return result
            
        except Exception as e:
            return f"❌ AI分析失败: {str(e)}"
    
    def generate_chart_suggestions(self, df: pd.DataFrame) -> List[Dict]:
        """生成图表建议"""
        if not self.llm:
            return self._get_basic_chart_suggestions(df)
        
        try:
            suggestion_prompt = PromptTemplate(
                input_variables=["data_info"],
                template="""
                根据以下数据信息，推荐最适合的3-5种图表类型，并说明原因。
                
                数据信息：
                {data_info}
                
                请以JSON格式返回，包含：
                - chart_type: 图表类型（柱状图/折线图/散点图/饼图/热力图/箱线图）
                - reason: 推荐原因
                - columns: 建议使用的列
                
                示例格式：
                [{"chart_type": "柱状图", "reason": "适合比较分类数据", "columns": ["x列", "y列"]}]
                """
            )
            
            data_info = f"""
            列名和类型: {[(col, str(dtype)) for col, dtype in df.dtypes.items()]}
            数值列: {df.select_dtypes(include=[np.number]).columns.tolist()}
            分类列: {df.select_dtypes(include=['object']).columns.tolist()}
            """
            
            chain = LLMChain(llm=self.llm, prompt=suggestion_prompt)
            result = chain.invoke({"data_info": data_info})
            
            # 尝试解析JSON结果
            try:
                suggestions = json.loads(result)
                return suggestions
            except:
                return self._get_basic_chart_suggestions(df)
                
        except Exception as e:
            return self._get_basic_chart_suggestions(df)
    
    def _get_basic_chart_suggestions(self, df: pd.DataFrame) -> List[Dict]:
        """获取基础图表建议"""
        suggestions = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_cols) >= 1:
            suggestions.append({
                "chart_type": "柱状图",
                "reason": "适合展示数值数据的分布",
                "columns": numeric_cols[:2]
            })
        
        if len(numeric_cols) >= 2:
            suggestions.append({
                "chart_type": "散点图",
                "reason": "适合分析两个数值变量的关系",
                "columns": numeric_cols[:2]
            })
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            suggestions.append({
                "chart_type": "饼图",
                "reason": "适合展示分类数据的比例",
                "columns": [categorical_cols[0], numeric_cols[0]]
            })
        
        return suggestions
    
    def natural_language_query(self, df: pd.DataFrame, query: str) -> str:
        """自然语言查询"""
        if not self.agent:
            self.create_dataframe_agent(df)
        
        if not self.agent:
            return "❌ 智能代理不可用，请检查API配置"
        
        try:
            # 添加中文提示
            enhanced_query = f"""
            请用中文回答以下问题，并提供详细的分析过程：
            
            {query}
            
            注意：
            1. 如果需要计算，请显示计算步骤
            2. 如果需要筛选数据，请说明筛选条件
            3. 提供具体的数值结果
            4. 用中文解释结果的含义
            """
            
            # 使用invoke方法替代run方法，因为代理有多个输出键
            response = self.agent.invoke({"input": enhanced_query})
            
            # 提取输出结果
            if isinstance(response, dict):
                if "output" in response:
                    return response["output"]
                else:
                    return str(response)
            else:
                return str(response)
            
        except Exception as e:
            return f"❌ 查询失败: {str(e)}"

def load_config():
    """加载配置"""
    if 'config' not in st.session_state:
        st.session_state.config = {
            'api_key': '',
            'selected_model': 'OpenAI GPT-3.5-Turbo',
            'custom_model_name': '',
            'custom_base_url': ''
        }
    return st.session_state.config

def save_config(config):
    """保存配置"""
    st.session_state.config = config

def setup_sidebar():
    """设置侧边栏"""
    with st.sidebar:
        st.markdown('<h2 class="sub-header">🔧 模型配置</h2>', unsafe_allow_html=True)
        
        config = load_config()
        
        # API Key输入
        api_key = st.text_input(
            "🔑 API Key",
            value=config['api_key'],
            type="password",
            help="输入您的OpenAI API Key或兼容的API Key"
        )
        
        # 模型选择
        selected_model = st.selectbox(
            "🤖 选择模型",
            list(DEFAULT_MODELS.keys()),
            index=list(DEFAULT_MODELS.keys()).index(config['selected_model'])
        )
        
        # 显示模型信息
        model_info = DEFAULT_MODELS[selected_model]
        st.info(f"📝 {model_info['description']}")
        
        # 自定义配置（当选择自定义模型时）
        if selected_model == "自定义模型":
            custom_model_name = st.text_input(
                "模型名称",
                value=config['custom_model_name'],
                placeholder="例如: gpt-3.5-turbo"
            )
            custom_base_url = st.text_input(
                "API Base URL",
                value=config['custom_base_url'],
                placeholder="例如: https://api.example.com/v1"
            )
            
            # 更新自定义配置
            if custom_model_name and custom_base_url:
                DEFAULT_MODELS["自定义模型"]["model_name"] = custom_model_name
                DEFAULT_MODELS["自定义模型"]["base_url"] = custom_base_url
        
        # 保存配置
        config.update({
            'api_key': api_key,
            'selected_model': selected_model,
            'custom_model_name': config.get('custom_model_name', ''),
            'custom_base_url': config.get('custom_base_url', '')
        })
        save_config(config)
        
        # 连接测试
        if api_key and st.button("🔗 测试连接", type="primary"):
            test_connection(api_key, DEFAULT_MODELS[selected_model])
        
        st.markdown("---")
        
        # 功能选择
        st.markdown('<h2 class="sub-header">🎯 功能选择</h2>', unsafe_allow_html=True)
        
        feature = st.selectbox(
            "选择功能",
            ["AI智能分析", "数据分析", "图表生成", "数据清洗", "统计分析", "自然语言查询", 
             "🤖 机器学习预测", "🔧 高级数据处理", "📊 数据对比分析", "📄 报告生成",
             "📋 数据表格操作", "🔢 公式计算器", "📈 财务分析", "📅 时间序列分析", 
             "🎯 目标跟踪", "📊 仪表板创建", "🔄 数据导入导出", "📝 数据验证",
             "🎨 条件格式化", "📑 工作表管理", "🔍 数据筛选排序", "📐 数学统计函数",
             "💼 商业智能分析", "🏢 企业报表", "📱 移动端适配", "🔐 数据安全"]
        )
        
        # AI状态显示
        if LANGCHAIN_AVAILABLE and api_key:
            st.markdown('<div class="success-box">✅ AI功能已启用</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ AI功能未启用<br>请配置API Key</div>', unsafe_allow_html=True)
        
        # LangChain状态检查
        if st.button("🔍 检查LangChain状态"):
            is_available, status_msg = check_langchain_status()
            if is_available:
                st.success(status_msg)
            else:
                st.error(status_msg)
                st.info("💡 请运行: pip install langchain langchain-openai langchain-community")
        
        return feature, config

def check_langchain_status():
    """检查LangChain安装状态"""
    try:
        import langchain
        import langchain_openai
        import langchain_community
        return True, f"✅ LangChain已正确安装 (版本: {langchain.__version__})"
    except ImportError as e:
        return False, f"❌ LangChain导入失败: {str(e)}"

def test_connection(api_key: str, model_config: Dict):
    """测试API连接"""
    if not LANGCHAIN_AVAILABLE:
        st.error("❌ LangChain模块不可用，无法测试连接")
        if 'langchain_import_error' in st.session_state:
            st.error(f"导入错误详情: {st.session_state.langchain_import_error}")
        
        # 提供诊断信息
        with st.expander("🔧 诊断和解决方案"):
            st.write("**请尝试以下解决方案:**")
            st.code("pip install langchain langchain-openai langchain-community", language="bash")
            st.write("**如果问题仍然存在，请尝试:**")
            st.code("pip uninstall langchain langchain-openai langchain-community\npip install langchain langchain-openai langchain-community", language="bash")
            st.write("**然后重启应用**")
        return
    
    try:
        with st.spinner("正在测试连接..."):
            llm = ChatOpenAI(
                model=model_config["model_name"],
                openai_api_key=api_key,
                openai_api_base=model_config["base_url"],
                temperature=0.1,
                max_tokens=50
            )
            
            # 发送测试消息
            response = llm.invoke([HumanMessage(content="你好，请回复'连接成功'")])
            
            if response and response.content:
                st.success("✅ 连接成功！")
            else:
                st.error("❌ 连接失败：无响应")
                
    except Exception as e:
        st.error(f"❌ 连接失败: {str(e)}")

def main():
    """主函数"""
    # 主标题
    st.markdown('<h1 class="main-header">🤖 Excel智能分析助手 - 完整版</h1>', unsafe_allow_html=True)
    
    # 设置侧边栏并获取配置
    feature, config = setup_sidebar()
    
    # 文件上传
    st.markdown('<h2 class="sub-header">📁 文件上传</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "选择Excel文件",
        type=["xlsx", "xls"],
        help="支持.xlsx和.xls格式的Excel文件，建议文件大小不超过50MB"
    )
    
    if uploaded_file is not None:
        try:
            # 读取Excel文件
            df = pd.read_excel(uploaded_file)
            
            # 显示文件信息
            st.markdown('<div class="success-box">✅ 文件上传成功！</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("数据行数", len(df))
            with col2:
                st.metric("数据列数", len(df.columns))
            with col3:
                st.metric("文件大小", f"{uploaded_file.size / 1024:.1f} KB")
            with col4:
                st.metric("缺失值", df.isnull().sum().sum())
            
            # 数据预览
            st.markdown('<h2 class="sub-header">👀 数据预览</h2>', unsafe_allow_html=True)
            
            # 数据展示设置区域
            with st.container():
                st.markdown("### ⚙️ 数据展示设置")
                
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                
                with col1:
                    rows_per_page = st.selectbox(
                        "📄 每页显示行数",
                        options=[10, 25, 50, 100, 200],
                        index=0,
                        key="rows_per_page",
                        help="选择每页显示的数据行数"
                    )
                
                with col2:
                    show_all = st.checkbox(
                        "📊 显示全部数据", 
                        key="show_all_data",
                        help="勾选后将显示所有数据，取消分页"
                    )
                
                with col3:
                    if not show_all:
                        st.metric(
                            "📈 总页数", 
                            value=(len(df) - 1) // rows_per_page + 1,
                            help="根据当前设置计算的总页数"
                        )
                    else:
                        st.metric(
                            "📊 总行数", 
                            value=len(df),
                            help="数据集的总行数"
                        )
                
                with col4:
                    st.metric(
                        "📋 总列数", 
                        value=len(df.columns),
                        help="数据集的总列数"
                    )
                
                st.divider()
            
            # 计算总页数
            total_rows = len(df)
            total_pages = (total_rows - 1) // rows_per_page + 1 if not show_all else 1
            
            if show_all:
                # 显示全部数据
                st.info(f"📊 显示全部数据：共 {total_rows} 行 × {len(df.columns)} 列")
                st.dataframe(df, use_container_width=True, height=600)
            else:
                  # 计算当前页的数据范围
                  current_page = st.session_state.get('current_page', 1)
                  start_idx = (current_page - 1) * rows_per_page
                  end_idx = min(start_idx + rows_per_page, total_rows)
                  
                  # 数据内容展示区域
                  st.markdown("### 📊 数据内容")
                  
                  # 页面信息和分页导航
                  info_col, nav_col = st.columns([2.5, 1.5])
                  
                  with info_col:
                      st.info(f"📄 第 **{current_page}** 页 | 显示第 **{start_idx + 1}-{end_idx}** 行 | 共 **{total_rows}** 行数据")
                  
                  with nav_col:
                      # 超紧凑水平分页导航
                      nav_cols = st.columns([0.8, 0.8, 1.4, 0.8, 0.8])
                      
                      with nav_cols[0]:
                          if st.button(
                              "⏮️", 
                              disabled=current_page == 1,
                              help="首页",
                              key="first_page_btn"
                          ):
                              st.session_state.current_page = 1
                              st.rerun()
                      
                      with nav_cols[1]:
                          if st.button(
                              "⬅️", 
                              disabled=current_page == 1,
                              help="上一页",
                              key="prev_page_btn"
                          ):
                              st.session_state.current_page = max(1, current_page - 1)
                              st.rerun()
                      
                      with nav_cols[2]:
                          new_page = st.number_input(
                              "",
                              min_value=1,
                              max_value=total_pages,
                              value=current_page,
                              key="compact_page_input",
                              help=f"页码跳转 (共{total_pages}页)",
                              label_visibility="collapsed"
                          )
                          if new_page != current_page:
                              st.session_state.current_page = new_page
                      
                      with nav_cols[3]:
                          if st.button(
                              "➡️", 
                              disabled=current_page == total_pages,
                              help="下一页",
                              key="next_page_btn"
                          ):
                              st.session_state.current_page = min(total_pages, current_page + 1)
                              st.rerun()
                      
                      with nav_cols[4]:
                          if st.button(
                              "⏭️", 
                              disabled=current_page == total_pages,
                              help="末页",
                              key="last_page_btn"
                          ):
                              st.session_state.current_page = total_pages
                              st.rerun()
                  
                  # 显示当前页数据
                  current_data = df.iloc[start_idx:end_idx]
                  st.dataframe(
                      current_data, 
                      use_container_width=True,
                      height=400
                  )
                  
                  # 当前页数据统计信息
                  st.markdown("### 📋 当前页统计")
                  
                  stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                  
                  with stat_col1:
                      st.metric(
                          "📄 当前页行数", 
                          len(current_data),
                          delta=f"共{total_rows}行",
                          help="当前页显示的数据行数"
                      )
                  
                  with stat_col2:
                      numeric_cols = current_data.select_dtypes(include=[np.number]).columns
                      st.metric(
                          "🔢 数值列数", 
                          len(numeric_cols),
                          delta=f"共{len(current_data.columns)}列",
                          help="当前页中包含数值的列数"
                      )
                  
                  with stat_col3:
                      missing_values = current_data.isnull().sum().sum()
                      missing_rate = (missing_values / (len(current_data) * len(current_data.columns)) * 100) if len(current_data) > 0 else 0
                      st.metric(
                          "❌ 缺失值", 
                          missing_values,
                          delta=f"{missing_rate:.1f}%",
                          help="当前页中的缺失值数量和比例"
                      )
                  
                  with stat_col4:
                      unique_values = sum(current_data.nunique())
                      st.metric(
                          "🎯 唯一值", 
                          unique_values,
                          help="当前页所有列的唯一值总数"
                      )
            
            # 初始化AI助手
            agent = None
            if config['api_key'] and LANGCHAIN_AVAILABLE:
                agent = ExcelAgentFull(config['api_key'], DEFAULT_MODELS[config['selected_model']])
            
            # 根据选择的功能执行相应操作
            if feature == "AI智能分析":
                ai_analysis_section(df, agent)
            elif feature == "数据分析":
                data_analysis_section(df)
            elif feature == "图表生成":
                chart_generation_section(df, agent)
            elif feature == "数据清洗":
                data_cleaning_section(df)
            elif feature == "统计分析":
                statistical_analysis_section(df)
            elif feature == "自然语言查询":
                natural_language_section(df, agent)
            elif feature == "🤖 机器学习预测":
                machine_learning_section(df, agent)
            elif feature == "🔧 高级数据处理":
                advanced_data_processing_section(df)
            elif feature == "📊 数据对比分析":
                data_comparison_section(df)
            elif feature == "📄 报告生成":
                report_generation_section(df, agent)
            elif feature == "📋 数据表格操作":
                table_operations_section(df)
            elif feature == "🔢 公式计算器":
                formula_calculator_section(df)
            elif feature == "📈 财务分析":
                financial_analysis_section(df)
            elif feature == "📅 时间序列分析":
                time_series_analysis_section(df)
            elif feature == "🎯 目标跟踪":
                goal_tracking_section(df)
            elif feature == "📊 仪表板创建":
                dashboard_creation_section(df)
            elif feature == "🔄 数据导入导出":
                data_import_export_section(df)
            elif feature == "📝 数据验证":
                data_validation_section(df)
            elif feature == "🎨 条件格式化":
                conditional_formatting_section(df)
            elif feature == "📑 工作表管理":
                worksheet_management_section(df)
            elif feature == "🔍 数据筛选排序":
                data_filtering_sorting_section(df)
            elif feature == "📐 数学统计函数":
                mathematical_functions_section(df)
            elif feature == "💼 商业智能分析":
                business_intelligence_section(df)
            elif feature == "🏢 企业报表":
                enterprise_reports_section(df)
            elif feature == "📱 移动端适配":
                mobile_adaptation_section(df)
            elif feature == "🔐 数据安全":
                data_security_section(df)
                
        except Exception as e:
            st.error(f"❌ 文件读取失败: {str(e)}")
    else:
        # 显示欢迎信息和功能介绍
        show_welcome_page()

def ai_analysis_section(df: pd.DataFrame, agent: Optional[ExcelAgentFull]):
    """AI智能分析功能"""
    st.markdown('<h2 class="sub-header">🤖 AI智能分析</h2>', unsafe_allow_html=True)
    
    if not agent:
        st.markdown('<div class="warning-box">⚠️ AI功能不可用，请在侧边栏配置API Key</div>', unsafe_allow_html=True)
        return
    
    # 快速分析按钮
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 数据概览分析", use_container_width=True):
            with st.spinner("AI正在分析数据概览..."):
                result = agent.analyze_data_with_ai(df, "请对这个数据集进行全面的概览分析，包括数据质量、分布特征、关键指标等")
                st.markdown(f'<div class="ai-response">{result}</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button("🔍 异常值检测", use_container_width=True):
            with st.spinner("AI正在检测异常值..."):
                result = agent.analyze_data_with_ai(df, "请检测数据中的异常值，分析可能的原因，并提供处理建议")
                st.markdown(f'<div class="ai-response">{result}</div>', unsafe_allow_html=True)
    
    with col3:
        if st.button("📈 趋势分析", use_container_width=True):
            with st.spinner("AI正在分析趋势..."):
                result = agent.analyze_data_with_ai(df, "请分析数据中的趋势和模式，识别关键的变化点和规律")
                st.markdown(f'<div class="ai-response">{result}</div>', unsafe_allow_html=True)
    
    # 自定义分析
    st.markdown("### 🎯 自定义AI分析")
    analysis_query = st.text_area(
        "输入您的分析需求",
        placeholder="例如：分析销售数据的季节性特征，找出影响销售的关键因素",
        height=100
    )
    
    if st.button("🚀 开始AI分析", type="primary"):
        if analysis_query:
            with st.spinner("AI正在深度分析中，请稍候..."):
                result = agent.analyze_data_with_ai(df, analysis_query)
                st.markdown(f'<div class="ai-response">{result}</div>', unsafe_allow_html=True)
        else:
            st.warning("请输入分析需求")

def natural_language_section(df: pd.DataFrame, agent: Optional[ExcelAgentFull]):
    """自然语言查询功能"""
    st.markdown('<h2 class="sub-header">💬 自然语言查询</h2>', unsafe_allow_html=True)
    
    if not agent:
        st.markdown('<div class="warning-box">⚠️ 自然语言查询功能不可用，请在侧边栏配置API Key</div>', unsafe_allow_html=True)
        return
    
    # 示例查询
    st.markdown("### 💡 查询示例")
    examples = [
        "数据中哪一列的平均值最高？",
        "找出销售额最高的前5个记录",
        "计算各个类别的总和",
        "数据中有多少个缺失值？",
        "显示数据的基本统计信息"
    ]
    
    selected_example = st.selectbox("选择示例查询", ["自定义查询"] + examples)
    
    # 查询输入
    if selected_example == "自定义查询":
        query = st.text_area(
            "输入您的查询",
            placeholder="例如：找出销售额大于10000的所有记录",
            height=100
        )
    else:
        query = st.text_area(
            "查询内容",
            value=selected_example,
            height=100
        )
    
    if st.button("🔍 执行查询", type="primary"):
        if query:
            with st.spinner("AI正在处理您的查询..."):
                result = agent.natural_language_query(df, query)
                st.markdown("### 📋 查询结果")
                st.markdown(f'<div class="ai-response">{result}</div>', unsafe_allow_html=True)
        else:
            st.warning("请输入查询内容")

def chart_generation_section(df: pd.DataFrame, agent: Optional[ExcelAgentFull]):
    """图表生成功能（增强版）"""
    import numpy as np  # 确保在函数内部可以访问numpy
    st.markdown('<h2 class="sub-header">📊 智能图表生成</h2>', unsafe_allow_html=True)
    
    # AI图表建议
    if agent:
        st.markdown("### 🤖 AI图表建议")
        if st.button("获取AI图表建议"):
            with st.spinner("AI正在分析数据并生成图表建议..."):
                suggestions = agent.generate_chart_suggestions(df)
                
                if suggestions:
                    st.markdown("#### 📈 推荐图表：")
                    for i, suggestion in enumerate(suggestions, 1):
                        st.markdown(f"""
                        **{i}. {suggestion['chart_type']}**
                        - 推荐原因：{suggestion['reason']}
                        - 建议列：{', '.join(suggestion['columns'])}
                        """)
    
    # 手动图表生成
    st.markdown("### 🎨 手动图表生成")
    
    # 选择图表类型
    chart_type = st.selectbox(
        "选择图表类型",
        ["柱状图", "折线图", "散点图", "饼图", "热力图", "箱线图", "直方图", "小提琴图"]
    )
    
    # 选择列
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if chart_type in ["柱状图", "折线图", "散点图"]:
            x_column = st.selectbox("选择X轴", df.columns)
            y_column = st.selectbox("选择Y轴", numeric_columns)
            color_column = st.selectbox("颜色分组（可选）", ["无"] + categorical_columns)
        elif chart_type == "饼图":
            category_column = st.selectbox("选择分类列", categorical_columns)
            value_column = st.selectbox("选择数值列", numeric_columns)
        elif chart_type in ["热力图"]:
            st.info("热力图将显示数值列之间的相关性")
        elif chart_type in ["箱线图", "直方图", "小提琴图"]:
            box_column = st.selectbox("选择数值列", numeric_columns)
            group_column = st.selectbox("分组列（可选）", ["无"] + categorical_columns)
    
    with col2:
        # 图表样式设置
        st.markdown("#### 🎨 样式设置")
        chart_title = st.text_input("图表标题", value=f"{chart_type}分析")
        color_theme = st.selectbox("颜色主题", ["默认", "蓝色", "红色", "绿色", "紫色"])
        
        # 高级设置
        with st.expander("高级设置"):
            show_grid = st.checkbox("显示网格", value=True)
            show_legend = st.checkbox("显示图例", value=True)
            chart_height = st.slider("图表高度", 300, 800, 500)
    
    if st.button("🎨 生成图表", type="primary"):
        try:
            fig = None
            
            # 颜色映射
            color_map = {
                "默认": None,
                "蓝色": "Blues",
                "红色": "Reds", 
                "绿色": "Greens",
                "紫色": "Purples"
            }
            
            if chart_type == "柱状图":
                color = None if color_column == "无" else color_column
                fig = px.bar(df, x=x_column, y=y_column, color=color, 
                           title=chart_title, color_discrete_sequence=px.colors.qualitative.Set3)
            
            elif chart_type == "折线图":
                color = None if color_column == "无" else color_column
                fig = px.line(df, x=x_column, y=y_column, color=color, title=chart_title)
            
            elif chart_type == "散点图":
                color = None if color_column == "无" else color_column
                fig = px.scatter(df, x=x_column, y=y_column, color=color, title=chart_title)
            
            elif chart_type == "饼图":
                pie_data = df.groupby(category_column)[value_column].sum().reset_index()
                fig = px.pie(pie_data, values=value_column, names=category_column, title=chart_title)
            
            elif chart_type == "热力图":
                if len(numeric_columns) >= 2:
                    corr_matrix = df[numeric_columns].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                                  title=chart_title, color_continuous_scale=color_map[color_theme])
                else:
                    st.warning("需要至少2个数值列生成热力图")
            
            elif chart_type == "箱线图":
                if group_column != "无":
                    fig = px.box(df, x=group_column, y=box_column, title=chart_title)
                else:
                    fig = px.box(df, y=box_column, title=chart_title)
            
            elif chart_type == "直方图":
                if group_column != "无":
                    fig = px.histogram(df, x=box_column, color=group_column, title=chart_title)
                else:
                    fig = px.histogram(df, x=box_column, title=chart_title)
            
            elif chart_type == "小提琴图":
                if group_column != "无":
                    fig = px.violin(df, x=group_column, y=box_column, title=chart_title)
                else:
                    fig = px.violin(df, y=box_column, title=chart_title)
            
            if fig:
                # 应用样式设置
                fig.update_layout(
                    showlegend=show_legend,
                    height=chart_height,
                    xaxis_showgrid=show_grid,
                    yaxis_showgrid=show_grid
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 图表解读（如果有AI）
                if agent:
                    if st.button("🤖 AI图表解读"):
                        with st.spinner("AI正在解读图表..."):
                            chart_description = f"这是一个{chart_type}，显示了{chart_title}的相关信息"
                            interpretation = agent.analyze_data_with_ai(df, f"请解读这个{chart_description}，分析其中的关键信息和趋势")
                            st.markdown(f'<div class="ai-response">{interpretation}</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"图表生成失败: {str(e)}")

def data_analysis_section(df: pd.DataFrame):
    """数据分析功能（基础版）"""
    import numpy as np  # 确保在函数内部可以访问numpy
    st.markdown('<h2 class="sub-header">📈 数据分析</h2>', unsafe_allow_html=True)
    
    # 基本统计信息
    st.markdown("### 📊 基本统计信息")
    st.dataframe(df.describe(), use_container_width=True)
    
    # 数据类型信息
    st.markdown("### 🔍 数据类型信息")
    dtype_df = pd.DataFrame({
        '列名': df.columns,
        '数据类型': df.dtypes.values,
        '非空值数量': df.count().values,
        '空值数量': df.isnull().sum().values,
        '空值比例': (df.isnull().sum() / len(df) * 100).round(2).astype(str) + '%'
    })
    st.dataframe(dtype_df, use_container_width=True)
    
    # 数据质量报告
    st.markdown("### 📋 数据质量报告")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_rate = (missing_cells / total_cells) * 100
        
        st.metric(
            "数据完整性",
            f"{100 - missing_rate:.1f}%",
            delta=f"{missing_cells} 个缺失值"
        )
    
    with col2:
        duplicate_rows = df.duplicated().sum()
        duplicate_rate = (duplicate_rows / len(df)) * 100
        
        st.metric(
            "数据唯一性",
            f"{100 - duplicate_rate:.1f}%",
            delta=f"{duplicate_rows} 行重复"
        )
    
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            outlier_count += len(outliers)
        
        st.metric(
            "异常值检测",
            f"{outlier_count} 个",
            delta="基于IQR方法"
        )

def data_cleaning_section(df: pd.DataFrame):
    """数据清洗功能"""
    import numpy as np  # 确保在函数内部可以访问numpy
    st.markdown('<h2 class="sub-header">🧹 数据清洗</h2>', unsafe_allow_html=True)
    
    # 显示数据质量概览
    st.markdown("### 📋 数据质量概览")
    
    quality_info = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_percent = (null_count / len(df)) * 100
        duplicate_count = df[col].duplicated().sum()
        unique_count = df[col].nunique()
        
        quality_info.append({
            '列名': col,
            '数据类型': str(df[col].dtype),
            '空值数量': null_count,
            '空值比例': f"{null_percent:.2f}%",
            '重复值数量': duplicate_count,
            '唯一值数量': unique_count
        })
    
    quality_df = pd.DataFrame(quality_info)
    st.dataframe(quality_df, use_container_width=True)
    
    # 清洗选项
    st.markdown("### 🔧 清洗选项")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 处理空值
        st.markdown("#### 处理空值")
        null_action = st.selectbox(
            "空值处理方式",
            ["不处理", "删除含空值的行", "删除含空值的列", "用均值填充", "用中位数填充", "用众数填充", "向前填充", "向后填充"]
        )
        
        # 处理重复值
        st.markdown("#### 处理重复值")
        duplicate_action = st.selectbox(
            "重复值处理方式",
            ["不处理", "删除重复行", "标记重复行"]
        )
        
        # 异常值处理
        st.markdown("#### 异常值处理")
        outlier_action = st.selectbox(
            "异常值处理方式",
            ["不处理", "删除异常值", "用边界值替换", "标记异常值"]
        )
    
    with col2:
        # 数据类型转换
        st.markdown("#### 数据类型转换")
        convert_column = st.selectbox("选择要转换的列", ["无"] + list(df.columns))
        if convert_column != "无":
            target_type = st.selectbox(
                "目标数据类型",
                ["int", "float", "string", "datetime", "category"]
            )
        
        # 列操作
        st.markdown("#### 列操作")
        column_action = st.selectbox(
            "列操作",
            ["无操作", "删除列", "重命名列"]
        )
        
        if column_action == "删除列":
            columns_to_drop = st.multiselect("选择要删除的列", df.columns)
        elif column_action == "重命名列":
            old_name = st.selectbox("选择要重命名的列", df.columns)
            new_name = st.text_input("新列名")
    
    if st.button("🚀 执行数据清洗", type="primary"):
        cleaned_df = df.copy()
        cleaning_steps = []
        
        try:
            # 处理空值
            if null_action != "不处理":
                if null_action == "删除含空值的行":
                    before_count = len(cleaned_df)
                    cleaned_df = cleaned_df.dropna()
                    cleaning_steps.append(f"删除了 {before_count - len(cleaned_df)} 行含空值的数据")
                elif null_action == "删除含空值的列":
                    before_cols = len(cleaned_df.columns)
                    cleaned_df = cleaned_df.dropna(axis=1)
                    cleaning_steps.append(f"删除了 {before_cols - len(cleaned_df.columns)} 列含空值的列")
                elif null_action == "用均值填充":
                    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
                    cleaning_steps.append("用均值填充了数值列的空值")
                elif null_action == "用中位数填充":
                    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
                    cleaning_steps.append("用中位数填充了数值列的空值")
                elif null_action == "用众数填充":
                    for col in cleaned_df.columns:
                        mode_val = cleaned_df[col].mode()
                        if not mode_val.empty:
                            cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
                    cleaning_steps.append("用众数填充了空值")
                elif null_action == "向前填充":
                    cleaned_df = cleaned_df.fillna(method='ffill')
                    cleaning_steps.append("使用向前填充处理了空值")
                elif null_action == "向后填充":
                    cleaned_df = cleaned_df.fillna(method='bfill')
                    cleaning_steps.append("使用向后填充处理了空值")
            
            # 处理重复值
            if duplicate_action == "删除重复行":
                before_count = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                cleaning_steps.append(f"删除了 {before_count - len(cleaned_df)} 行重复数据")
            elif duplicate_action == "标记重复行":
                cleaned_df['is_duplicate'] = cleaned_df.duplicated()
                cleaning_steps.append("添加了重复行标记列")
            
            # 处理异常值
            if outlier_action != "不处理":
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                outlier_count = 0
                
                for col in numeric_cols:
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    if outlier_action == "删除异常值":
                        before_count = len(cleaned_df)
                        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
                        outlier_count += before_count - len(cleaned_df)
                    elif outlier_action == "用边界值替换":
                        cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
                    elif outlier_action == "标记异常值":
                        cleaned_df[f'{col}_outlier'] = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
                
                if outlier_action == "删除异常值":
                    cleaning_steps.append(f"删除了 {outlier_count} 个异常值")
                elif outlier_action == "用边界值替换":
                    cleaning_steps.append("用边界值替换了异常值")
                elif outlier_action == "标记异常值":
                    cleaning_steps.append("添加了异常值标记列")
            
            # 数据类型转换
            if convert_column != "无":
                try:
                    if target_type == "int":
                        cleaned_df[convert_column] = pd.to_numeric(cleaned_df[convert_column], errors='coerce').astype('Int64')
                    elif target_type == "float":
                        cleaned_df[convert_column] = pd.to_numeric(cleaned_df[convert_column], errors='coerce')
                    elif target_type == "string":
                        cleaned_df[convert_column] = cleaned_df[convert_column].astype(str)
                    elif target_type == "datetime":
                        cleaned_df[convert_column] = pd.to_datetime(cleaned_df[convert_column], errors='coerce')
                    elif target_type == "category":
                        cleaned_df[convert_column] = cleaned_df[convert_column].astype('category')
                    cleaning_steps.append(f"将列 {convert_column} 转换为 {target_type} 类型")
                except Exception as e:
                    st.warning(f"类型转换失败: {str(e)}")
            
            # 列操作
            if column_action == "删除列" and 'columns_to_drop' in locals() and columns_to_drop:
                cleaned_df = cleaned_df.drop(columns=columns_to_drop)
                cleaning_steps.append(f"删除了列: {', '.join(columns_to_drop)}")
            elif column_action == "重命名列" and 'old_name' in locals() and 'new_name' in locals() and new_name:
                cleaned_df = cleaned_df.rename(columns={old_name: new_name})
                cleaning_steps.append(f"将列 {old_name} 重命名为 {new_name}")
            
            # 显示清洗结果
            st.markdown("### ✅ 清洗完成")
            
            if cleaning_steps:
                st.markdown("#### 执行的清洗步骤:")
                for step in cleaning_steps:
                    st.write(f"• {step}")
            
            # 清洗前后对比
            col1, col2 = st.columns(2)
            with col1:
                st.metric("原始数据行数", len(df))
                st.metric("原始数据列数", len(df.columns))
                st.metric("原始缺失值", df.isnull().sum().sum())
            
            with col2:
                st.metric("清洗后行数", len(cleaned_df), delta=int(len(cleaned_df) - len(df)))
                st.metric("清洗后列数", len(cleaned_df.columns), delta=int(len(cleaned_df.columns) - len(df.columns)))
                st.metric("清洗后缺失值", cleaned_df.isnull().sum().sum(), delta=int(cleaned_df.isnull().sum().sum() - df.isnull().sum().sum()))
            
            # 显示清洗后的数据
            st.markdown("#### 清洗后的数据预览:")
            st.dataframe(cleaned_df.head(10), use_container_width=True)
            
            # 提供下载链接
            csv = cleaned_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载清洗后的数据 (CSV)",
                data=csv,
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ 数据清洗失败: {str(e)}")

def statistical_analysis_section(df: pd.DataFrame):
    """统计分析功能"""
    import numpy as np  # 确保在函数内部可以访问numpy
    st.markdown('<h2 class="sub-header">📊 统计分析</h2>', unsafe_allow_html=True)
    
    # 选择分析类型
    analysis_type = st.selectbox(
        "选择分析类型",
        ["描述性统计", "相关性分析", "分组统计", "假设检验", "回归分析"]
    )
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if analysis_type == "描述性统计":
        st.markdown("### 📈 描述性统计")
        selected_columns = st.multiselect(
            "选择要分析的数值列", 
            numeric_columns, 
            default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
        )
        
        if selected_columns:
            # 基础统计
            stats_df = df[selected_columns].describe()
            st.dataframe(stats_df, use_container_width=True)
            
            # 额外统计指标
            extra_stats = []
            for col in selected_columns:
                skewness = df[col].skew()
                kurtosis = df[col].kurtosis()
                extra_stats.append({
                    '列名': col,
                    '偏度': f"{skewness:.3f}",
                    '峰度': f"{kurtosis:.3f}",
                    '变异系数': f"{(df[col].std() / df[col].mean()):.3f}" if df[col].mean() != 0 else "N/A"
                })
            
            st.markdown("#### 📊 额外统计指标")
            st.dataframe(pd.DataFrame(extra_stats), use_container_width=True)
            
            # 分布图
            for col in selected_columns:
                fig = px.histogram(df, x=col, title=f"{col} 分布图", marginal="box")
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "相关性分析":
        st.markdown("### 🔗 相关性分析")
        if len(numeric_columns) >= 2:
            # 选择相关性方法
            corr_method = st.selectbox("相关性方法", ["pearson", "spearman", "kendall"])
            
            corr_matrix = df[numeric_columns].corr(method=corr_method)
            
            # 相关性热力图
            fig = px.imshow(corr_matrix, 
                          text_auto=True, 
                          aspect="auto", 
                          title=f"相关性矩阵热力图 ({corr_method})",
                          color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)
            
            # 强相关性对
            st.markdown("#### 相关性分析结果:")
            
            # 设置阈值
            threshold = st.slider("相关性阈值", 0.0, 1.0, 0.7, 0.1)
            
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        strength = "强正相关" if corr_val > 0 else "强负相关"
                        strong_corr.append({
                            '变量1': corr_matrix.columns[i],
                            '变量2': corr_matrix.columns[j],
                            '相关系数': f"{corr_val:.3f}",
                            '相关强度': strength
                        })
            
            if strong_corr:
                st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
            else:
                st.info(f"未发现相关系数绝对值大于 {threshold} 的变量对")
        else:
            st.warning("需要至少2个数值列进行相关性分析")
    
    elif analysis_type == "分组统计":
        st.markdown("### 📊 分组统计")
        if categorical_columns and numeric_columns:
            group_col = st.selectbox("选择分组列", categorical_columns)
            value_cols = st.multiselect("选择数值列", numeric_columns, default=numeric_columns[:2])
            
            if value_cols and st.button("执行分组统计"):
                for value_col in value_cols:
                    st.markdown(f"#### {value_col} 按 {group_col} 分组统计")
                    
                    grouped_stats = df.groupby(group_col)[value_col].agg([
                        'count', 'mean', 'median', 'std', 'min', 'max'
                    ]).round(3)
                    
                    st.dataframe(grouped_stats, use_container_width=True)
                    
                    # 分组箱线图
                    fig = px.box(df, x=group_col, y=value_col, 
                               title=f"{value_col} 按 {group_col} 分组分布")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("需要至少一个分类列和一个数值列进行分组统计")
    
    elif analysis_type == "假设检验":
        st.markdown("### 🧪 假设检验")
        
        test_type = st.selectbox(
            "选择检验类型",
            ["单样本t检验", "独立样本t检验", "配对样本t检验", "卡方检验"]
        )
        
        if test_type == "单样本t检验":
            if numeric_columns:
                test_column = st.selectbox("选择检验列", numeric_columns)
                test_value = st.number_input("检验值", value=0.0)
                
                if st.button("执行检验"):
                    from scipy import stats
                    
                    sample_data = df[test_column].dropna()
                    t_stat, p_value = stats.ttest_1samp(sample_data, test_value)
                    
                    st.markdown("#### 检验结果:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("t统计量", f"{t_stat:.4f}")
                    with col2:
                        st.metric("p值", f"{p_value:.4f}")
                    with col3:
                        significance = "显著" if p_value < 0.05 else "不显著"
                        st.metric("显著性(α=0.05)", significance)
        
        elif test_type == "独立样本t检验":
            if categorical_columns and numeric_columns:
                group_col = st.selectbox("选择分组列", categorical_columns)
                test_col = st.selectbox("选择检验列", numeric_columns)
                
                groups = df[group_col].unique()
                if len(groups) == 2:
                    if st.button("执行检验"):
                        from scipy import stats
                        
                        group1_data = df[df[group_col] == groups[0]][test_col].dropna()
                        group2_data = df[df[group_col] == groups[1]][test_col].dropna()
                        
                        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                        
                        st.markdown("#### 检验结果:")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("t统计量", f"{t_stat:.4f}")
                        with col2:
                            st.metric("p值", f"{p_value:.4f}")
                        with col3:
                            significance = "显著" if p_value < 0.05 else "不显著"
                            st.metric("显著性(α=0.05)", significance)
                        
                        # 显示组间统计
                        st.markdown("#### 组间统计:")
                        group_stats = pd.DataFrame({
                            '组别': [groups[0], groups[1]],
                            '样本量': [len(group1_data), len(group2_data)],
                            '均值': [group1_data.mean(), group2_data.mean()],
                            '标准差': [group1_data.std(), group2_data.std()]
                        })
                        st.dataframe(group_stats, use_container_width=True)
                else:
                    st.warning("分组列必须恰好包含2个不同的值")
            else:
                st.warning("需要分类列和数值列进行独立样本t检验")
        
        elif test_type == "卡方检验":
            if len(categorical_columns) >= 2:
                col1_name = st.selectbox("选择第一个分类列", categorical_columns)
                col2_name = st.selectbox("选择第二个分类列", [col for col in categorical_columns if col != col1_name])
                
                if st.button("执行检验"):
                    from scipy import stats
                    
                    # 创建交叉表
                    contingency_table = pd.crosstab(df[col1_name], df[col2_name])
                    
                    # 执行卡方检验
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    
                    st.markdown("#### 检验结果:")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("卡方统计量", f"{chi2:.4f}")
                    with col2:
                        st.metric("p值", f"{p_value:.4f}")
                    with col3:
                        st.metric("自由度", dof)
                    with col4:
                        significance = "显著" if p_value < 0.05 else "不显著"
                        st.metric("显著性(α=0.05)", significance)
                    
                    # 显示交叉表
                    st.markdown("#### 交叉表:")
                    st.dataframe(contingency_table, use_container_width=True)
            else:
                st.warning("需要至少2个分类列进行卡方检验")
    
    elif analysis_type == "回归分析":
        st.markdown("### 📈 回归分析")
        
        if len(numeric_columns) >= 2:
            y_column = st.selectbox("选择因变量(Y)", numeric_columns)
            x_columns = st.multiselect(
                "选择自变量(X)", 
                [col for col in numeric_columns if col != y_column],
                default=[col for col in numeric_columns if col != y_column][:2]
            )
            
            if x_columns and st.button("执行回归分析"):
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score, mean_squared_error
                import numpy as np
                
                # 准备数据
                X = df[x_columns].dropna()
                y = df[y_column].dropna()
                
                # 确保X和y的索引对齐
                common_index = X.index.intersection(y.index)
                X = X.loc[common_index]
                y = y.loc[common_index]
                
                if len(X) > 0:
                    # 执行回归
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    
                    # 计算指标
                    r2 = r2_score(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    
                    st.markdown("#### 回归结果:")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R²决定系数", f"{r2:.4f}")
                    with col2:
                        st.metric("RMSE", f"{rmse:.4f}")
                    with col3:
                        st.metric("样本量", len(X))
                    
                    # 回归系数
                    coef_df = pd.DataFrame({
                        '变量': ['截距'] + x_columns,
                        '系数': [model.intercept_] + list(model.coef_),
                        '绝对值': [abs(model.intercept_)] + [abs(coef) for coef in model.coef_]
                    })
                    
                    st.markdown("#### 回归系数:")
                    st.dataframe(coef_df, use_container_width=True)
                    
                    # 残差图
                    residuals = y - y_pred
                    fig = px.scatter(x=y_pred, y=residuals, title="残差图",
                                   labels={'x': '预测值', 'y': '残差'})
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 实际值vs预测值
                    fig2 = px.scatter(x=y, y=y_pred, title="实际值 vs 预测值",
                                    labels={'x': '实际值', 'y': '预测值'})
                    # 添加对角线
                    min_val = min(y.min(), y_pred.min())
                    max_val = max(y.max(), y_pred.max())
                    fig2.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                                 line=dict(color="red", dash="dash"))
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("数据不足，无法进行回归分析")
        else:
            st.warning("需要至少2个数值列进行回归分析")

def show_welcome_page():
    """显示欢迎页面"""
    st.markdown('<div class="info-box">📤 请上传Excel文件开始智能分析</div>', unsafe_allow_html=True)
    
    # 功能介绍
    st.markdown('<h2 class="sub-header">🌟 功能特色</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 AI智能分析
        - 自然语言数据查询
        - 智能数据洞察生成
        - AI驱动的图表建议
        - 自动异常值检测
        """)
    
    with col2:
        st.markdown("""
        ### 📊 数据可视化
        - 8种图表类型
        - 交互式图表操作
        - 自定义样式设置
        - AI图表解读
        """)
    
    with col3:
        st.markdown("""
        ### 🧹 数据处理
        - 智能数据清洗
        - 多种统计分析
        - 假设检验工具
        - 回归分析功能
        """)
    
    # 示例数据生成
    st.markdown('<h2 class="sub-header">🎯 生成示例数据</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📈 生成销售数据", use_container_width=True):
            generate_sample_data("sales")
    
    with col2:
        if st.button("👥 生成员工数据", use_container_width=True):
            generate_sample_data("employee")

def generate_sample_data(data_type: str):
    """生成示例数据"""
    import numpy as np  # 确保在函数内部可以访问numpy
    try:
        np.random.seed(42)
        
        if data_type == "sales":
            # 生成销售数据
            dates = pd.date_range('2023-01-01', periods=365, freq='D')
            n_records = len(dates)
            
            sample_data = {
                '日期': dates,
                '销售额': np.random.normal(15000, 3000, n_records).round(2),
                '产品类别': np.random.choice(['电子产品', '服装', '家居用品', '食品', '图书'], n_records),
                '销售区域': np.random.choice(['北京', '上海', '广州', '深圳', '杭州', '成都'], n_records),
                '销售员': np.random.choice(['张三', '李四', '王五', '赵六', '钱七'], n_records),
                '客户数量': np.random.poisson(25, n_records),
                '折扣率': np.random.uniform(0, 0.3, n_records).round(3),
                '利润率': np.random.normal(0.2, 0.05, n_records).round(3)
            }
            
            # 添加一些季节性和趋势
            for i, date in enumerate(dates):
                # 季节性影响
                month = date.month
                if month in [11, 12]:  # 双十一、双十二
                    sample_data['销售额'][i] *= 1.5
                elif month in [6, 7]:  # 夏季促销
                    sample_data['销售额'][i] *= 1.2
                
                # 周末影响
                if date.weekday() >= 5:  # 周末
                    sample_data['客户数量'][i] = int(sample_data['客户数量'][i] * 1.3)
            
            filename = "销售数据示例.xlsx"
            
        elif data_type == "employee":
            # 生成员工数据
            n_records = 500
            
            departments = ['技术部', '销售部', '市场部', '人事部', '财务部', '运营部']
            positions = ['初级', '中级', '高级', '专家', '经理', '总监']
            cities = ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '西安']
            
            sample_data = {
                '员工ID': [f'EMP{str(i+1).zfill(4)}' for i in range(n_records)],
                '姓名': [f'员工{i+1}' for i in range(n_records)],
                '部门': np.random.choice(departments, n_records),
                '职位': np.random.choice(positions, n_records),
                '工作城市': np.random.choice(cities, n_records),
                '入职日期': pd.date_range('2020-01-01', '2023-12-31', periods=n_records),
                '年龄': np.random.randint(22, 60, n_records),
                '基本工资': np.random.normal(12000, 4000, n_records).round(0),
                '绩效评分': np.random.normal(85, 10, n_records).round(1),
                '工作年限': np.random.randint(0, 15, n_records),
                '学历': np.random.choice(['本科', '硕士', '博士', '专科'], n_records, p=[0.6, 0.25, 0.1, 0.05]),
                '是否在职': np.random.choice(['是', '否'], n_records, p=[0.9, 0.1])
            }
            
            # 调整工资与职位、工作年限的关系
            for i in range(n_records):
                position = sample_data['职位'][i]
                years = sample_data['工作年限'][i]
                
                # 职位系数
                position_multiplier = {
                    '初级': 0.8, '中级': 1.0, '高级': 1.3, 
                    '专家': 1.5, '经理': 1.8, '总监': 2.5
                }[position]
                
                # 工作年限影响
                years_multiplier = 1 + years * 0.05
                
                sample_data['基本工资'][i] = int(sample_data['基本工资'][i] * position_multiplier * years_multiplier)
            
            filename = "员工数据示例.xlsx"
        
        # 创建DataFrame
        sample_df = pd.DataFrame(sample_data)
        
        # 保存为Excel文件
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            sample_df.to_excel(writer, index=False, sheet_name='数据')
        
        st.download_button(
            label=f"📥 下载{filename}",
            data=output.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.success(f"✅ {filename}生成成功！点击上方按钮下载")
        
        # 显示数据预览
        st.markdown("#### 数据预览:")
        st.dataframe(sample_df.head(10), use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ 示例数据生成失败: {str(e)}")

def machine_learning_section(df: pd.DataFrame, agent: Optional[ExcelAgentFull]):
    """机器学习预测功能"""
    st.markdown('<h2 class="sub-header">🤖 机器学习预测</h2>', unsafe_allow_html=True)
    
    # 检查数据是否适合机器学习
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ 数据中数值列不足，无法进行机器学习预测。至少需要2个数值列。")
        return
    
    st.markdown("### 🎯 预测模型配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_col = st.selectbox(
            "选择目标变量（要预测的列）",
            numeric_cols,
            help="选择您想要预测的数值列"
        )
    
    with col2:
        feature_cols = st.multiselect(
            "选择特征变量（用于预测的列）",
            [col for col in numeric_cols if col != target_col],
            default=[col for col in numeric_cols if col != target_col][:3],
            help="选择用于预测的特征列"
        )
    
    if not feature_cols:
        st.warning("请至少选择一个特征变量")
        return
    
    # 模型选择
    model_type = st.selectbox(
        "选择预测模型",
        ["线性回归", "随机森林", "梯度提升"],
        help="不同模型适用于不同类型的数据"
    )
    
    # 数据分割比例
    test_size = st.slider(
        "测试集比例",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="用于测试模型性能的数据比例"
    )
    
    if st.button("🚀 开始训练模型", type="primary"):
        try:
            # 准备数据
            X = df[feature_cols].dropna()
            y = df.loc[X.index, target_col]
            
            if len(X) < 10:
                st.error("数据量太少，无法训练模型。至少需要10行有效数据。")
                return
            
            # 导入机器学习库
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LinearRegression
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                from sklearn.preprocessing import StandardScaler
            except ImportError:
                st.error("❌ 缺少机器学习库。请安装: pip install scikit-learn")
                return
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 选择模型
            if model_type == "线性回归":
                model = LinearRegression()
            elif model_type == "随机森林":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:  # 梯度提升
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            # 训练模型
            with st.spinner("正在训练模型..."):
                if model_type == "线性回归":
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
            
            # 计算评估指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 显示结果
            st.markdown("### 📊 模型性能评估")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² 分数", f"{r2:.4f}", help="决定系数，越接近1越好")
            with col2:
                st.metric("RMSE", f"{rmse:.4f}", help="均方根误差，越小越好")
            with col3:
                st.metric("MAE", f"{mae:.4f}", help="平均绝对误差，越小越好")
            with col4:
                st.metric("MSE", f"{mse:.4f}", help="均方误差，越小越好")
            
            # 预测vs实际值图表
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='预测值',
                marker=dict(color='blue', size=8)
            ))
            
            # 添加理想线
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='理想预测线',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="预测值 vs 实际值",
                xaxis_title="实际值",
                yaxis_title="预测值",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 特征重要性（仅对树模型）
            if model_type in ["随机森林", "梯度提升"]:
                st.markdown("### 🎯 特征重要性")
                
                importance_df = pd.DataFrame({
                    '特征': feature_cols,
                    '重要性': model.feature_importances_
                }).sort_values('重要性', ascending=False)
                
                fig_importance = px.bar(
                    importance_df,
                    x='重要性',
                    y='特征',
                    orientation='h',
                    title="特征重要性排序"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # AI解释
            if agent:
                st.markdown("### 🤖 AI模型解释")
                if st.button("获取AI解释"):
                    with st.spinner("AI正在分析模型结果..."):
                        explanation_query = f"""
                        请分析以下机器学习模型的结果：
                        
                        模型类型：{model_type}
                        目标变量：{target_col}
                        特征变量：{', '.join(feature_cols)}
                        R²分数：{r2:.4f}
                        RMSE：{rmse:.4f}
                        MAE：{mae:.4f}
                        
                        请解释：
                        1. 模型性能如何？
                        2. 哪些特征最重要？
                        3. 模型的可靠性如何？
                        4. 有什么改进建议？
                        """
                        
                        result = agent.analyze_data_with_ai(df, explanation_query)
                        st.markdown(f'<div class="ai-response">{result}</div>', unsafe_allow_html=True)
            
            # 保存模型到session state
            st.session_state['trained_model'] = {
                'model': model,
                'scaler': scaler if model_type == "线性回归" else None,
                'feature_cols': feature_cols,
                'target_col': target_col,
                'model_type': model_type,
                'performance': {'r2': r2, 'rmse': rmse, 'mae': mae, 'mse': mse}
            }
            
            st.success("✅ 模型训练完成！模型已保存到当前会话中。")
            
        except Exception as e:
            st.error(f"❌ 模型训练失败: {str(e)}")
    
    # 使用已训练的模型进行预测
    if 'trained_model' in st.session_state:
        st.markdown("### 🔮 使用模型进行预测")
        
        model_info = st.session_state['trained_model']
        
        st.info(f"当前模型：{model_info['model_type']} | 目标变量：{model_info['target_col']} | R²：{model_info['performance']['r2']:.4f}")
        
        # 输入预测值
        st.markdown("#### 输入特征值进行预测：")
        
        prediction_inputs = {}
        cols = st.columns(len(model_info['feature_cols']))
        
        for i, feature in enumerate(model_info['feature_cols']):
            with cols[i]:
                # 获取该特征的统计信息
                feature_stats = df[feature].describe()
                prediction_inputs[feature] = st.number_input(
                    f"{feature}",
                    value=float(feature_stats['mean']),
                    help=f"范围: {feature_stats['min']:.2f} - {feature_stats['max']:.2f}"
                )
        
        if st.button("🎯 进行预测"):
            try:
                # 准备预测数据
                pred_data = pd.DataFrame([prediction_inputs])
                
                # 应用相同的预处理
                if model_info['scaler']:
                    pred_data_scaled = model_info['scaler'].transform(pred_data)
                    prediction = model_info['model'].predict(pred_data_scaled)[0]
                else:
                    prediction = model_info['model'].predict(pred_data)[0]
                
                st.success(f"🎯 预测结果：{model_info['target_col']} = {prediction:.4f}")
                
                # 显示置信区间（简单估计）
                rmse = model_info['performance']['rmse']
                st.info(f"📊 预测区间（±1个RMSE）：{prediction-rmse:.4f} ~ {prediction+rmse:.4f}")
                
            except Exception as e:
                st.error(f"❌ 预测失败: {str(e)}")

def advanced_data_processing_section(df: pd.DataFrame):
    """高级数据处理功能"""
    st.markdown('<h2 class="sub-header">🔧 高级数据处理</h2>', unsafe_allow_html=True)
    
    # 数据处理选项
    processing_option = st.selectbox(
        "选择数据处理功能",
        ["数据标准化", "异常值处理", "数据分箱", "特征工程", "数据透视表", "数据合并"]
    )
    
    if processing_option == "数据标准化":
        st.markdown("### 📊 数据标准化")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("没有数值列可以标准化")
            return
        
        selected_cols = st.multiselect(
            "选择要标准化的列",
            numeric_cols,
            default=numeric_cols[:3]
        )
        
        if selected_cols:
            method = st.selectbox(
                "标准化方法",
                ["Z-score标准化", "Min-Max标准化", "Robust标准化"]
            )
            
            if st.button("执行标准化"):
                try:
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                    
                    df_processed = df.copy()
                    
                    if method == "Z-score标准化":
                        scaler = StandardScaler()
                    elif method == "Min-Max标准化":
                        scaler = MinMaxScaler()
                    else:
                        scaler = RobustScaler()
                    
                    df_processed[selected_cols] = scaler.fit_transform(df[selected_cols])
                    
                    st.success(f"✅ 使用{method}完成标准化")
                    
                    # 显示前后对比
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 原始数据统计")
                        st.dataframe(df[selected_cols].describe())
                    
                    with col2:
                        st.markdown("#### 标准化后统计")
                        st.dataframe(df_processed[selected_cols].describe())
                    
                    # 可视化对比
                    for col in selected_cols[:2]:  # 最多显示2列的对比
                        fig = go.Figure()
                        
                        fig.add_trace(go.Histogram(
                            x=df[col],
                            name=f"原始 {col}",
                            opacity=0.7,
                            nbinsx=30
                        ))
                        
                        fig.add_trace(go.Histogram(
                            x=df_processed[col],
                            name=f"标准化 {col}",
                            opacity=0.7,
                            nbinsx=30
                        ))
                        
                        fig.update_layout(
                            title=f"{col} 标准化前后对比",
                            barmode='overlay',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 下载处理后的数据
                    csv = df_processed.to_csv(index=False)
                    st.download_button(
                        label="📥 下载标准化后的数据",
                        data=csv,
                        file_name="standardized_data.csv",
                        mime="text/csv"
                    )
                    
                except ImportError:
                    st.error("❌ 缺少scikit-learn库。请安装: pip install scikit-learn")
                except Exception as e:
                    st.error(f"❌ 标准化失败: {str(e)}")
    
    elif processing_option == "异常值处理":
        st.markdown("### 🎯 异常值处理")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("没有数值列可以处理异常值")
            return
        
        selected_col = st.selectbox("选择要处理的列", numeric_cols)
        
        # 异常值检测方法
        method = st.selectbox(
            "异常值检测方法",
            ["IQR方法", "Z-score方法", "百分位数方法"]
        )
        
        if method == "IQR方法":
            Q1 = df[selected_col].quantile(0.25)
            Q3 = df[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
            
        elif method == "Z-score方法":
            z_threshold = st.slider("Z-score阈值", 1.0, 4.0, 3.0, 0.1)
            z_scores = np.abs((df[selected_col] - df[selected_col].mean()) / df[selected_col].std())
            outliers = df[z_scores > z_threshold]
            
        else:  # 百分位数方法
            lower_percentile = st.slider("下百分位数", 0.0, 10.0, 1.0, 0.5)
            upper_percentile = st.slider("上百分位数", 90.0, 100.0, 99.0, 0.5)
            lower_bound = df[selected_col].quantile(lower_percentile/100)
            upper_bound = df[selected_col].quantile(upper_percentile/100)
            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
        
        st.info(f"检测到 {len(outliers)} 个异常值（占总数的 {len(outliers)/len(df)*100:.2f}%）")
        
        if len(outliers) > 0:
            # 显示异常值
            st.markdown("#### 检测到的异常值：")
            st.dataframe(outliers[[selected_col]].head(10))
            
            # 可视化
            fig = go.Figure()
            
            # 正常值
            normal_data = df[~df.index.isin(outliers.index)]
            fig.add_trace(go.Scatter(
                x=normal_data.index,
                y=normal_data[selected_col],
                mode='markers',
                name='正常值',
                marker=dict(color='blue', size=6)
            ))
            
            # 异常值
            fig.add_trace(go.Scatter(
                x=outliers.index,
                y=outliers[selected_col],
                mode='markers',
                name='异常值',
                marker=dict(color='red', size=8, symbol='x')
            ))
            
            fig.update_layout(
                title=f"{selected_col} 异常值检测结果",
                xaxis_title="索引",
                yaxis_title=selected_col,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 处理选项
            treatment = st.selectbox(
                "异常值处理方式",
                ["删除异常值", "用中位数替换", "用均值替换", "用边界值替换"]
            )
            
            if st.button("应用处理"):
                df_processed = df.copy()
                
                if treatment == "删除异常值":
                    df_processed = df_processed[~df_processed.index.isin(outliers.index)]
                    st.success(f"✅ 已删除 {len(outliers)} 个异常值")
                    
                elif treatment == "用中位数替换":
                    median_val = df[selected_col].median()
                    df_processed.loc[outliers.index, selected_col] = median_val
                    st.success(f"✅ 已用中位数 {median_val:.4f} 替换异常值")
                    
                elif treatment == "用均值替换":
                    mean_val = df[selected_col].mean()
                    df_processed.loc[outliers.index, selected_col] = mean_val
                    st.success(f"✅ 已用均值 {mean_val:.4f} 替换异常值")
                    
                else:  # 用边界值替换
                    if method == "IQR方法":
                        df_processed.loc[df_processed[selected_col] < lower_bound, selected_col] = lower_bound
                        df_processed.loc[df_processed[selected_col] > upper_bound, selected_col] = upper_bound
                    elif method == "百分位数方法":
                        df_processed.loc[df_processed[selected_col] < lower_bound, selected_col] = lower_bound
                        df_processed.loc[df_processed[selected_col] > upper_bound, selected_col] = upper_bound
                    st.success("✅ 已用边界值替换异常值")
                
                # 显示处理后的统计信息
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 处理前统计")
                    st.write(df[selected_col].describe())
                
                with col2:
                    st.markdown("#### 处理后统计")
                    st.write(df_processed[selected_col].describe())
                
                # 下载处理后的数据
                csv = df_processed.to_csv(index=False)
                st.download_button(
                    label="📥 下载处理后的数据",
                    data=csv,
                    file_name="outlier_processed_data.csv",
                    mime="text/csv"
                )
    
    elif processing_option == "数据分箱":
        st.markdown("### 📦 数据分箱")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("没有数值列可以分箱")
            return
        
        selected_col = st.selectbox("选择要分箱的列", numeric_cols)
        
        # 分箱方法
        binning_method = st.selectbox(
            "分箱方法",
            ["等宽分箱", "等频分箱", "自定义分箱"]
        )
        
        if binning_method in ["等宽分箱", "等频分箱"]:
            n_bins = st.slider("分箱数量", 2, 20, 5)
        
        if binning_method == "自定义分箱":
            bin_edges_input = st.text_input(
                "输入分箱边界（用逗号分隔）",
                placeholder="例如: 0, 10, 20, 30, 40"
            )
        
        if st.button("执行分箱"):
            try:
                df_processed = df.copy()
                
                if binning_method == "等宽分箱":
                    df_processed[f'{selected_col}_binned'], bin_edges = pd.cut(
                        df[selected_col], 
                        bins=n_bins, 
                        retbins=True,
                        labels=[f'Bin_{i+1}' for i in range(n_bins)]
                    )
                    
                elif binning_method == "等频分箱":
                    df_processed[f'{selected_col}_binned'], bin_edges = pd.qcut(
                        df[selected_col], 
                        q=n_bins, 
                        retbins=True,
                        labels=[f'Bin_{i+1}' for i in range(n_bins)],
                        duplicates='drop'
                    )
                    
                else:  # 自定义分箱
                    if not bin_edges_input:
                        st.error("请输入分箱边界")
                        return
                    
                    try:
                        bin_edges = [float(x.strip()) for x in bin_edges_input.split(',')]
                        bin_edges = sorted(bin_edges)
                        
                        df_processed[f'{selected_col}_binned'] = pd.cut(
                            df[selected_col],
                            bins=bin_edges,
                            labels=[f'Bin_{i+1}' for i in range(len(bin_edges)-1)],
                            include_lowest=True
                        )
                        
                    except ValueError:
                        st.error("分箱边界格式错误，请输入数字并用逗号分隔")
                        return
                
                st.success(f"✅ 分箱完成！创建了新列：{selected_col}_binned")
                
                # 显示分箱结果
                st.markdown("#### 分箱统计")
                bin_stats = df_processed[f'{selected_col}_binned'].value_counts().sort_index()
                st.dataframe(bin_stats)
                
                # 可视化分箱结果
                fig = px.histogram(
                    df_processed,
                    x=f'{selected_col}_binned',
                    title=f"{selected_col} 分箱结果"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示分箱边界
                st.markdown("#### 分箱边界")
                if binning_method != "自定义分箱":
                    bin_info = pd.DataFrame({
                        '分箱': [f'Bin_{i+1}' for i in range(len(bin_edges)-1)],
                        '下界': bin_edges[:-1],
                        '上界': bin_edges[1:]
                    })
                    st.dataframe(bin_info)
                
                # 下载处理后的数据
                csv = df_processed.to_csv(index=False)
                st.download_button(
                    label="📥 下载分箱后的数据",
                    data=csv,
                    file_name="binned_data.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ 分箱失败: {str(e)}")
    
    elif processing_option == "特征工程":
        st.markdown("### ⚙️ 特征工程")
        
        feature_type = st.selectbox(
            "选择特征工程类型",
            ["多项式特征", "交互特征", "对数变换", "平方根变换", "时间特征提取"]
        )
        
        if feature_type == "多项式特征":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.warning("没有数值列可以创建多项式特征")
                return
            
            selected_cols = st.multiselect(
                "选择列创建多项式特征",
                numeric_cols,
                default=numeric_cols[:2]
            )
            
            degree = st.slider("多项式度数", 2, 5, 2)
            
            if selected_cols and st.button("创建多项式特征"):
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    
                    poly = PolynomialFeatures(degree=degree, include_bias=False)
                    poly_features = poly.fit_transform(df[selected_cols])
                    
                    # 获取特征名称
                    feature_names = poly.get_feature_names_out(selected_cols)
                    
                    # 创建新的DataFrame
                    poly_df = pd.DataFrame(poly_features, columns=feature_names)
                    
                    # 合并到原数据
                    df_processed = pd.concat([df, poly_df], axis=1)
                    
                    st.success(f"✅ 创建了 {len(feature_names)} 个多项式特征")
                    
                    # 显示新特征
                    st.markdown("#### 新创建的特征")
                    st.dataframe(poly_df.head())
                    
                    # 下载处理后的数据
                    csv = df_processed.to_csv(index=False)
                    st.download_button(
                        label="📥 下载特征工程后的数据",
                        data=csv,
                        file_name="polynomial_features_data.csv",
                        mime="text/csv"
                    )
                    
                except ImportError:
                    st.error("❌ 缺少scikit-learn库。请安装: pip install scikit-learn")
                except Exception as e:
                    st.error(f"❌ 特征创建失败: {str(e)}")
        
        elif feature_type == "时间特征提取":
            # 检测日期时间列
            datetime_cols = []
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)
                else:
                    # 尝试转换为日期时间
                    try:
                        pd.to_datetime(df[col].dropna().iloc[:5])
                        datetime_cols.append(col)
                    except:
                        pass
            
            if not datetime_cols:
                st.warning("没有检测到日期时间列")
                
                # 让用户选择列进行日期转换
                all_cols = df.columns.tolist()
                selected_col = st.selectbox("选择包含日期的列", all_cols)
                
                if st.button("尝试转换为日期"):
                    try:
                        df[f'{selected_col}_datetime'] = pd.to_datetime(df[selected_col])
                        datetime_cols = [f'{selected_col}_datetime']
                        st.success(f"✅ 成功转换 {selected_col} 为日期时间格式")
                    except Exception as e:
                        st.error(f"❌ 日期转换失败: {str(e)}")
                        return
            
            if datetime_cols:
                selected_datetime_col = st.selectbox("选择日期时间列", datetime_cols)
                
                features_to_extract = st.multiselect(
                    "选择要提取的时间特征",
                    ["年份", "月份", "日期", "星期几", "小时", "分钟", "季度", "是否周末"],
                    default=["年份", "月份", "星期几"]
                )
                
                if features_to_extract and st.button("提取时间特征"):
                    try:
                        df_processed = df.copy()
                        
                        # 确保列是datetime类型
                        if not pd.api.types.is_datetime64_any_dtype(df_processed[selected_datetime_col]):
                            df_processed[selected_datetime_col] = pd.to_datetime(df_processed[selected_datetime_col])
                        
                        dt_col = df_processed[selected_datetime_col]
                        
                        if "年份" in features_to_extract:
                            df_processed[f'{selected_datetime_col}_year'] = dt_col.dt.year
                        if "月份" in features_to_extract:
                            df_processed[f'{selected_datetime_col}_month'] = dt_col.dt.month
                        if "日期" in features_to_extract:
                            df_processed[f'{selected_datetime_col}_day'] = dt_col.dt.day
                        if "星期几" in features_to_extract:
                            df_processed[f'{selected_datetime_col}_weekday'] = dt_col.dt.weekday
                        if "小时" in features_to_extract:
                            df_processed[f'{selected_datetime_col}_hour'] = dt_col.dt.hour
                        if "分钟" in features_to_extract:
                            df_processed[f'{selected_datetime_col}_minute'] = dt_col.dt.minute
                        if "季度" in features_to_extract:
                            df_processed[f'{selected_datetime_col}_quarter'] = dt_col.dt.quarter
                        if "是否周末" in features_to_extract:
                            df_processed[f'{selected_datetime_col}_is_weekend'] = (dt_col.dt.weekday >= 5).astype(int)
                        
                        st.success(f"✅ 成功提取了 {len(features_to_extract)} 个时间特征")
                        
                        # 显示新特征
                        new_cols = [col for col in df_processed.columns if col.startswith(f'{selected_datetime_col}_')]
                        st.markdown("#### 新提取的时间特征")
                        st.dataframe(df_processed[new_cols].head())
                        
                        # 下载处理后的数据
                        csv = df_processed.to_csv(index=False)
                        st.download_button(
                            label="📥 下载时间特征提取后的数据",
                            data=csv,
                            file_name="time_features_data.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ 时间特征提取失败: {str(e)}")

def data_comparison_section(df: pd.DataFrame):
    """数据对比分析功能"""
    st.markdown('<h2 class="sub-header">📊 数据对比分析</h2>', unsafe_allow_html=True)
    
    # 上传第二个文件进行对比
    st.markdown("### 📁 上传对比数据")
    
    comparison_file = st.file_uploader(
        "选择要对比的Excel文件",
        type=["xlsx", "xls"],
        help="上传另一个Excel文件进行数据对比分析",
        key="comparison_file"
    )
    
    if comparison_file is not None:
        try:
            df_compare = pd.read_excel(comparison_file)
            
            st.success(f"✅ 对比文件上传成功！数据形状：{df_compare.shape}")
            
            # 基本信息对比
            st.markdown("### 📋 基本信息对比")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 原始数据")
                st.metric("行数", len(df))
                st.metric("列数", len(df.columns))
                st.metric("缺失值", df.isnull().sum().sum())
                st.metric("数值列数", len(df.select_dtypes(include=[np.number]).columns))
            
            with col2:
                st.markdown("#### 对比数据")
                st.metric("行数", len(df_compare))
                st.metric("列数", len(df_compare.columns))
                st.metric("缺失值", df_compare.isnull().sum().sum())
                st.metric("数值列数", len(df_compare.select_dtypes(include=[np.number]).columns))
            
            # 列名对比
            st.markdown("### 📝 列名对比")
            
            cols_original = set(df.columns)
            cols_compare = set(df_compare.columns)
            
            common_cols = cols_original & cols_compare
            only_original = cols_original - cols_compare
            only_compare = cols_compare - cols_original
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 共同列")
                if common_cols:
                    for col in sorted(common_cols):
                        st.write(f"✅ {col}")
                else:
                    st.write("无共同列")
            
            with col2:
                st.markdown("#### 仅原始数据有")
                if only_original:
                    for col in sorted(only_original):
                        st.write(f"🔵 {col}")
                else:
                    st.write("无独有列")
            
            with col3:
                st.markdown("#### 仅对比数据有")
                if only_compare:
                    for col in sorted(only_compare):
                        st.write(f"🟡 {col}")
                else:
                    st.write("无独有列")
            
            # 如果有共同的数值列，进行统计对比
            numeric_common_cols = []
            if common_cols:
                numeric_common_cols = [col for col in common_cols 
                                     if col in df.select_dtypes(include=[np.number]).columns 
                                     and col in df_compare.select_dtypes(include=[np.number]).columns]
                
                if numeric_common_cols:
                    st.markdown("### 📊 数值列统计对比")
                    
                    selected_col = st.selectbox("选择要对比的数值列", numeric_common_cols)
                    
                    # 统计对比表
                    stats_original = df[selected_col].describe()
                    stats_compare = df_compare[selected_col].describe()
                    
                    comparison_stats = pd.DataFrame({
                        '原始数据': stats_original,
                        '对比数据': stats_compare,
                        '差异': stats_compare - stats_original,
                        '差异百分比': ((stats_compare - stats_original) / stats_original * 100).round(2)
                    })
                    
                    st.dataframe(comparison_stats)
                    
                    # 分布对比图
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=df[selected_col],
                        name='原始数据',
                        opacity=0.7,
                        nbinsx=30
                    ))
                    
                    fig.add_trace(go.Histogram(
                        x=df_compare[selected_col],
                        name='对比数据',
                        opacity=0.7,
                        nbinsx=30
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_col} 分布对比",
                        barmode='overlay',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 箱线图对比
                    fig_box = go.Figure()
                    
                    fig_box.add_trace(go.Box(
                        y=df[selected_col],
                        name='原始数据',
                        boxpoints='outliers'
                    ))
                    
                    fig_box.add_trace(go.Box(
                        y=df_compare[selected_col],
                        name='对比数据',
                        boxpoints='outliers'
                    ))
                    
                    fig_box.update_layout(
                        title=f"{selected_col} 箱线图对比",
                        height=400
                    )
                    
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # 统计检验
                    st.markdown("### 🔬 统计检验")
                    
                    try:
                        from scipy import stats
                        
                        # 去除缺失值
                        data1 = df[selected_col].dropna()
                        data2 = df_compare[selected_col].dropna()
                        
                        # t检验
                        t_stat, t_pvalue = stats.ttest_ind(data1, data2)
                        
                        # Mann-Whitney U检验
                        u_stat, u_pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        
                        # Kolmogorov-Smirnov检验
                        ks_stat, ks_pvalue = stats.ks_2samp(data1, data2)
                        
                        test_results = pd.DataFrame({
                            '检验方法': ['独立样本t检验', 'Mann-Whitney U检验', 'Kolmogorov-Smirnov检验'],
                            '统计量': [t_stat, u_stat, ks_stat],
                            'p值': [t_pvalue, u_pvalue, ks_pvalue],
                            '显著性(α=0.05)': [
                                '显著' if t_pvalue < 0.05 else '不显著',
                                '显著' if u_pvalue < 0.05 else '不显著',
                                '显著' if ks_pvalue < 0.05 else '不显著'
                            ]
                        })
                        
                        st.dataframe(test_results)
                        
                        st.info("""
                        📝 **检验说明：**
                        - **t检验**：检验两组数据均值是否有显著差异（假设数据正态分布）
                        - **Mann-Whitney U检验**：非参数检验，检验两组数据分布是否有显著差异
                        - **Kolmogorov-Smirnov检验**：检验两组数据是否来自同一分布
                        - **p值 < 0.05**：表示差异显著
                        """)
                        
                    except ImportError:
                        st.warning("⚠️ 缺少scipy库，无法进行统计检验。请安装: pip install scipy")
                    except Exception as e:
                        st.error(f"❌ 统计检验失败: {str(e)}")
            
            # 相关性对比
            if len(numeric_common_cols) >= 2:
                st.markdown("### 🔗 相关性对比")
                
                # 计算相关性矩阵
                corr_original = df[numeric_common_cols].corr()
                corr_compare = df_compare[numeric_common_cols].corr()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 原始数据相关性")
                    fig_corr1 = px.imshow(
                        corr_original,
                        title="原始数据相关性热图",
                        color_continuous_scale='RdBu_r',
                        aspect='auto'
                    )
                    st.plotly_chart(fig_corr1, use_container_width=True)
                
                with col2:
                    st.markdown("#### 对比数据相关性")
                    fig_corr2 = px.imshow(
                        corr_compare,
                        title="对比数据相关性热图",
                        color_continuous_scale='RdBu_r',
                        aspect='auto'
                    )
                    st.plotly_chart(fig_corr2, use_container_width=True)
                
                # 相关性差异
                st.markdown("#### 相关性差异")
                corr_diff = corr_compare - corr_original
                
                fig_diff = px.imshow(
                    corr_diff,
                    title="相关性差异热图（对比数据 - 原始数据）",
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                st.plotly_chart(fig_diff, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ 对比文件读取失败: {str(e)}")
    
    else:
        st.info("请上传一个Excel文件进行数据对比分析")

def report_generation_section(df: pd.DataFrame, agent: Optional[ExcelAgentFull]):
    """报告生成功能"""
    st.markdown('<h2 class="sub-header">📄 报告生成</h2>', unsafe_allow_html=True)
    
    # 报告类型选择
    report_type = st.selectbox(
        "选择报告类型",
        ["数据概览报告", "统计分析报告", "AI智能报告", "自定义报告"]
    )
    
    # 报告配置
    st.markdown("### ⚙️ 报告配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_charts = st.checkbox("包含图表", value=True)
        include_statistics = st.checkbox("包含统计信息", value=True)
    
    with col2:
        include_data_quality = st.checkbox("包含数据质量分析", value=True)
        include_recommendations = st.checkbox("包含建议", value=True)
    
    # 报告标题和描述
    report_title = st.text_input(
        "报告标题",
        value=f"数据分析报告 - {datetime.now().strftime('%Y-%m-%d')}"
    )
    
    report_description = st.text_area(
        "报告描述",
        placeholder="请输入报告的背景和目的..."
    )
    
    if st.button("🚀 生成报告", type="primary"):
        try:
            # 创建报告内容
            report_content = []
            
            # 报告标题
            report_content.append(f"# {report_title}\n")
            report_content.append(f"**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if report_description:
                report_content.append(f"**报告描述：** {report_description}\n")
            
            report_content.append("---\n")
            
            # 数据概览
            report_content.append("## 📊 数据概览\n")
            report_content.append(f"- **数据形状：** {df.shape[0]} 行 × {df.shape[1]} 列\n")
            report_content.append(f"- **数值列数：** {len(df.select_dtypes(include=[np.number]).columns)}\n")
            report_content.append(f"- **文本列数：** {len(df.select_dtypes(include=['object']).columns)}\n")
            report_content.append(f"- **缺失值总数：** {df.isnull().sum().sum()}\n")
            report_content.append(f"- **缺失值比例：** {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%\n\n")
            
            # 列信息
            report_content.append("### 📋 列信息\n")
            for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df) * 100)
                report_content.append(f"{i}. **{col}** ({dtype}) - 缺失值: {missing_count} ({missing_pct:.1f}%)\n")
            
            report_content.append("\n")
            
            # 统计信息
            if include_statistics:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    report_content.append("## 📈 统计分析\n")
                    
                    for col in numeric_cols:
                        stats = df[col].describe()
                        report_content.append(f"### {col}\n")
                        report_content.append(f"- **均值：** {stats['mean']:.4f}\n")
                        report_content.append(f"- **中位数：** {stats['50%']:.4f}\n")
                        report_content.append(f"- **标准差：** {stats['std']:.4f}\n")
                        report_content.append(f"- **最小值：** {stats['min']:.4f}\n")
                        report_content.append(f"- **最大值：** {stats['max']:.4f}\n")
                        report_content.append(f"- **偏度：** {df[col].skew():.4f}\n")
                        report_content.append(f"- **峰度：** {df[col].kurtosis():.4f}\n\n")
            
            # 数据质量分析
            if include_data_quality:
                report_content.append("## 🔍 数据质量分析\n")
                
                # 缺失值分析
                missing_analysis = df.isnull().sum()
                missing_cols = missing_analysis[missing_analysis > 0]
                
                if len(missing_cols) > 0:
                    report_content.append("### 缺失值分析\n")
                    for col, count in missing_cols.items():
                        pct = (count / len(df) * 100)
                        report_content.append(f"- **{col}：** {count} 个缺失值 ({pct:.1f}%)\n")
                else:
                    report_content.append("### 缺失值分析\n")
                    report_content.append("✅ 数据中没有缺失值\n")
                
                report_content.append("\n")
                
                # 重复值分析
                duplicate_count = df.duplicated().sum()
                report_content.append("### 重复值分析\n")
                if duplicate_count > 0:
                    report_content.append(f"⚠️ 发现 {duplicate_count} 行重复数据 ({duplicate_count/len(df)*100:.1f}%)\n")
                else:
                    report_content.append("✅ 数据中没有重复行\n")
                
                report_content.append("\n")
            
            # AI智能分析
            if report_type == "AI智能报告" and agent:
                report_content.append("## 🤖 AI智能分析\n")
                
                with st.spinner("AI正在生成智能分析..."):
                    ai_analysis = agent.analyze_data_with_ai(
                        df, 
                        "请对这个数据集进行全面的分析，包括数据特征、潜在问题、关键发现和业务洞察"
                    )
                    report_content.append(ai_analysis)
                    report_content.append("\n\n")
            
            # 建议
            if include_recommendations:
                report_content.append("## 💡 建议和后续步骤\n")
                
                recommendations = []
                
                # 基于数据质量的建议
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
                if missing_pct > 5:
                    recommendations.append("处理缺失值：考虑删除、填充或插值方法")
                
                if df.duplicated().sum() > 0:
                    recommendations.append("处理重复数据：检查并删除重复行")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    recommendations.append("进行相关性分析：探索变量间的关系")
                    recommendations.append("考虑机器学习建模：预测或分类分析")
                
                if len(df) > 1000:
                    recommendations.append("数据采样：对于大数据集，考虑采样分析")
                
                recommendations.append("数据可视化：创建图表以更好地理解数据")
                recommendations.append("定期更新：建立数据更新和监控机制")
                
                for i, rec in enumerate(recommendations, 1):
                    report_content.append(f"{i}. {rec}\n")
                
                report_content.append("\n")
            
            # 附录
            report_content.append("## 📎 附录\n")
            report_content.append("### 数据样本\n")
            report_content.append("前5行数据：\n")
            report_content.append(df.head().to_string())
            report_content.append("\n\n")
            
            # 合并报告内容
            full_report = "".join(report_content)
            
            # 显示报告预览
            st.markdown("### 📋 报告预览")
            st.markdown(full_report)
            
            # 下载选项
            st.markdown("### 📥 下载报告")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Markdown下载
                st.download_button(
                    label="📄 下载Markdown报告",
                    data=full_report,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            with col2:
                # HTML下载
                try:
                    import markdown
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <title>{report_title}</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 40px; }}
                            h1, h2, h3 {{ color: #333; }}
                            table {{ border-collapse: collapse; width: 100%; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                            th {{ background-color: #f2f2f2; }}
                        </style>
                    </head>
                    <body>
                        {markdown.markdown(full_report, extensions=['tables'])}
                    </body>
                    </html>
                    """
                    
                    st.download_button(
                        label="🌐 下载HTML报告",
                        data=html_content,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                except ImportError:
                    st.info("安装markdown库以启用HTML导出: pip install markdown")
            
            with col3:
                # PDF下载（需要额外库）
                st.info("PDF导出需要安装额外库")
                if st.button("📋 复制报告内容"):
                    st.code(full_report)
            
            st.success("✅ 报告生成完成！")
            
        except Exception as e:
            st.error(f"❌ 报告生成失败: {str(e)}")

def table_operations_section(df: pd.DataFrame):
    """数据表格操作功能"""
    st.markdown('<h2 class="sub-header">📋 数据表格操作</h2>', unsafe_allow_html=True)
    
    operation = st.selectbox(
        "选择表格操作",
        ["行列操作", "单元格编辑", "数据插入", "数据删除", "表格合并", "表格拆分"]
    )
    
    if operation == "行列操作":
        st.markdown("### 🔄 行列操作")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 行操作")
            if st.button("📊 显示行统计"):
                st.write(f"总行数: {len(df)}")
                st.write(f"非空行数: {len(df.dropna())}")
                st.write(f"重复行数: {df.duplicated().sum()}")
            
            if st.button("🔄 转置表格"):
                df_transposed = df.T
                st.markdown("#### 转置后的表格:")
                st.dataframe(df_transposed)
                
                csv = df_transposed.to_csv(index=True)
                st.download_button(
                    label="📥 下载转置表格",
                    data=csv,
                    file_name="transposed_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.markdown("#### 列操作")
            if st.button("📊 显示列统计"):
                st.write(f"总列数: {len(df.columns)}")
                st.write(f"数值列数: {len(df.select_dtypes(include=[np.number]).columns)}")
                st.write(f"文本列数: {len(df.select_dtypes(include=['object']).columns)}")
            
            selected_cols = st.multiselect("选择要重新排序的列", df.columns.tolist())
            if selected_cols and st.button("🔄 重新排序列"):
                df_reordered = df[selected_cols]
                st.markdown("#### 重新排序后的表格:")
                st.dataframe(df_reordered)
    
    elif operation == "单元格编辑":
        st.markdown("### ✏️ 单元格编辑")
        
        if len(df) > 0:
            row_idx = st.number_input("选择行索引", 0, len(df)-1, 0)
            col_name = st.selectbox("选择列", df.columns.tolist())
            
            current_value = df.iloc[row_idx][col_name]
            st.info(f"当前值: {current_value}")
            
            new_value = st.text_input("输入新值", str(current_value))
            
            if st.button("✅ 更新单元格"):
                df_edited = df.copy()
                try:
                    # 尝试保持原数据类型
                    if pd.api.types.is_numeric_dtype(df[col_name]):
                        df_edited.iloc[row_idx, df_edited.columns.get_loc(col_name)] = float(new_value)
                    else:
                        df_edited.iloc[row_idx, df_edited.columns.get_loc(col_name)] = new_value
                    
                    st.success(f"✅ 已更新 ({row_idx}, {col_name}) 的值")
                    st.dataframe(df_edited.head(10))
                    
                    csv = df_edited.to_csv(index=False)
                    st.download_button(
                        label="📥 下载编辑后的数据",
                        data=csv,
                        file_name="edited_data.csv",
                        mime="text/csv"
                    )
                except ValueError as e:
                    st.error(f"❌ 数据类型错误: {str(e)}")

def formula_calculator_section(df: pd.DataFrame):
    """公式计算器功能"""
    st.markdown('<h2 class="sub-header">🔢 公式计算器</h2>', unsafe_allow_html=True)
    
    calc_type = st.selectbox(
        "选择计算类型",
        ["基础数学运算", "统计函数", "逻辑函数", "文本函数", "日期函数", "自定义公式"]
    )
    
    if calc_type == "基础数学运算":
        st.markdown("### ➕ 基础数学运算")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1_name = st.selectbox("选择第一列", numeric_cols)
            operation = st.selectbox("选择运算", ["+", "-", "*", "/", "**", "%"])
            col2_name = st.selectbox("选择第二列", numeric_cols)
            
            if st.button("🧮 执行计算"):
                try:
                    col1_data = df[col1_name]
                    col2_data = df[col2_name]
                    
                    if operation == "+":
                        result = col1_data + col2_data
                    elif operation == "-":
                        result = col1_data - col2_data
                    elif operation == "*":
                        result = col1_data * col2_data
                    elif operation == "/":
                        result = col1_data / col2_data
                    elif operation == "**":
                        result = col1_data ** col2_data
                    elif operation == "%":
                        result = col1_data % col2_data
                    
                    result_name = f"{col1_name} {operation} {col2_name}"
                    
                    # 显示结果
                    result_df = pd.DataFrame({
                        col1_name: col1_data,
                        col2_name: col2_data,
                        result_name: result
                    })
                    
                    st.markdown(f"#### 计算结果: {result_name}")
                    st.dataframe(result_df.head(10))
                    
                    # 统计信息
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("结果均值", f"{result.mean():.4f}")
                    with col2:
                        st.metric("结果总和", f"{result.sum():.4f}")
                    with col3:
                        st.metric("结果标准差", f"{result.std():.4f}")
                    
                    # 下载结果
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载计算结果",
                        data=csv,
                        file_name="calculation_result.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ 计算错误: {str(e)}")
        else:
            st.warning("需要至少两个数值列进行运算")
    
    elif calc_type == "统计函数":
        st.markdown("### 📊 统计函数")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("选择列", numeric_cols)
            
            stat_functions = {
                "平均值": lambda x: x.mean(),
                "中位数": lambda x: x.median(),
                "众数": lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
                "标准差": lambda x: x.std(),
                "方差": lambda x: x.var(),
                "最小值": lambda x: x.min(),
                "最大值": lambda x: x.max(),
                "四分位数": lambda x: [x.quantile(0.25), x.quantile(0.5), x.quantile(0.75)],
                "偏度": lambda x: x.skew(),
                "峰度": lambda x: x.kurtosis()
            }
            
            selected_functions = st.multiselect(
                "选择统计函数",
                list(stat_functions.keys()),
                default=["平均值", "中位数", "标准差"]
            )
            
            if st.button("📊 计算统计量"):
                results = {}
                data = df[selected_col].dropna()
                
                for func_name in selected_functions:
                    try:
                        result = stat_functions[func_name](data)
                        if func_name == "四分位数":
                            results["Q1"] = result[0]
                            results["Q2 (中位数)"] = result[1]
                            results["Q3"] = result[2]
                        else:
                            results[func_name] = result
                    except Exception as e:
                        results[func_name] = f"错误: {str(e)}"
                
                # 显示结果
                st.markdown(f"#### {selected_col} 的统计结果:")
                
                for name, value in results.items():
                    if isinstance(value, (int, float)):
                        st.metric(name, f"{value:.4f}")
                    else:
                        st.write(f"**{name}**: {value}")
                
                # 可视化分布
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data,
                    nbinsx=30,
                    name=selected_col
                ))
                fig.update_layout(
                    title=f"{selected_col} 数据分布",
                    xaxis_title=selected_col,
                    yaxis_title="频次"
                )
                st.plotly_chart(fig, use_container_width=True)

def financial_analysis_section(df: pd.DataFrame):
    """财务分析功能"""
    st.markdown('<h2 class="sub-header">📈 财务分析</h2>', unsafe_allow_html=True)
    
    analysis_type = st.selectbox(
        "选择财务分析类型",
        ["盈利能力分析", "现金流分析", "投资回报分析", "成本效益分析", "预算分析", "财务比率计算"]
    )
    
    if analysis_type == "盈利能力分析":
        st.markdown("### 💰 盈利能力分析")
        
        # 检查是否有必要的财务列
        required_cols = ["收入", "成本", "费用"]
        available_cols = [col for col in required_cols if col in df.columns]
        
        if not available_cols:
            st.info("💡 请确保数据包含以下列名: 收入、成本、费用、销售额等")
            
            # 让用户映射列名
            st.markdown("#### 📋 列名映射")
            revenue_col = st.selectbox("选择收入列", [None] + df.columns.tolist())
            cost_col = st.selectbox("选择成本列", [None] + df.columns.tolist())
            expense_col = st.selectbox("选择费用列", [None] + df.columns.tolist())
            
            if revenue_col and cost_col:
                # 计算利润指标
                revenue = df[revenue_col]
                cost = df[cost_col]
                expense = df[expense_col] if expense_col else 0
                
                gross_profit = revenue - cost
                net_profit = gross_profit - expense
                
                profit_margin = (gross_profit / revenue * 100).round(2)
                net_margin = (net_profit / revenue * 100).round(2)
                
                # 显示结果
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("总收入", f"¥{revenue.sum():,.2f}")
                with col2:
                    st.metric("总成本", f"¥{cost.sum():,.2f}")
                with col3:
                    st.metric("毛利润", f"¥{gross_profit.sum():,.2f}")
                with col4:
                    st.metric("净利润", f"¥{net_profit.sum():,.2f}")
                
                # 利润率分析
                st.markdown("#### 📊 利润率分析")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("平均毛利率", f"{profit_margin.mean():.2f}%")
                with col2:
                    st.metric("平均净利率", f"{net_margin.mean():.2f}%")
                
                # 可视化
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=revenue,
                    mode='lines+markers',
                    name='收入',
                    line=dict(color='green')
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=cost,
                    mode='lines+markers',
                    name='成本',
                    line=dict(color='red')
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=gross_profit,
                    mode='lines+markers',
                    name='毛利润',
                    line=dict(color='blue')
                ))
                
                fig.update_layout(
                    title="收入、成本与利润趋势",
                    xaxis_title="时间/序号",
                    yaxis_title="金额",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "投资回报分析":
        st.markdown("### 📈 投资回报分析")
        
        # ROI计算器
        st.markdown("#### 🧮 ROI计算器")
        
        investment_col = st.selectbox("选择投资金额列", [None] + df.columns.tolist())
        return_col = st.selectbox("选择回报金额列", [None] + df.columns.tolist())
        
        if investment_col and return_col:
            investment = df[investment_col]
            returns = df[return_col]
            
            # 计算ROI
            roi = ((returns - investment) / investment * 100).round(2)
            
            # 显示统计
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总投资", f"¥{investment.sum():,.2f}")
            with col2:
                st.metric("总回报", f"¥{returns.sum():,.2f}")
            with col3:
                st.metric("净收益", f"¥{(returns - investment).sum():,.2f}")
            with col4:
                st.metric("平均ROI", f"{roi.mean():.2f}%")
            
            # ROI分布
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=roi,
                nbinsx=20,
                name="ROI分布"
            ))
            fig.update_layout(
                title="投资回报率(ROI)分布",
                xaxis_title="ROI (%)",
                yaxis_title="频次"
            )
            st.plotly_chart(fig, use_container_width=True)

def time_series_analysis_section(df: pd.DataFrame):
    """时间序列分析功能"""
    st.markdown('<h2 class="sub-header">📅 时间序列分析</h2>', unsafe_allow_html=True)
    
    # 检测日期列
    date_cols = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col], errors='raise')
            date_cols.append(col)
        except:
            continue
    
    if not date_cols:
        st.warning("未检测到日期列，请确保数据包含日期时间信息")
        
        # 手动指定日期列
        date_col = st.selectbox("选择日期列", df.columns.tolist())
        if date_col and st.button("🔄 转换为日期格式"):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                date_cols = [date_col]
                st.success(f"✅ 成功转换 {date_col} 为日期格式")
            except Exception as e:
                st.error(f"❌ 日期转换失败: {str(e)}")
                return
    
    if date_cols:
        date_col = st.selectbox("选择日期列", date_cols)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            value_col = st.selectbox("选择数值列", numeric_cols)
            
            analysis_type = st.selectbox(
                "选择分析类型",
                ["趋势分析", "季节性分析", "移动平均", "增长率分析", "预测分析"]
            )
            
            # 准备时间序列数据
            ts_data = df[[date_col, value_col]].copy()
            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
            ts_data = ts_data.sort_values(date_col)
            
            if analysis_type == "趋势分析":
                st.markdown("### 📈 趋势分析")
                
                # 基本趋势图
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts_data[date_col],
                    y=ts_data[value_col],
                    mode='lines+markers',
                    name=value_col,
                    line=dict(color='blue')
                ))
                
                # 添加趋势线
                from scipy import stats
                x_numeric = np.arange(len(ts_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, ts_data[value_col])
                trend_line = slope * x_numeric + intercept
                
                fig.add_trace(go.Scatter(
                    x=ts_data[date_col],
                    y=trend_line,
                    mode='lines',
                    name='趋势线',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{value_col} 时间序列趋势分析",
                    xaxis_title="日期",
                    yaxis_title=value_col,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 趋势统计
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("趋势斜率", f"{slope:.4f}")
                with col2:
                    st.metric("相关系数", f"{r_value:.4f}")
                with col3:
                    st.metric("P值", f"{p_value:.4f}")
                with col4:
                    trend_direction = "上升" if slope > 0 else "下降" if slope < 0 else "平稳"
                    st.metric("趋势方向", trend_direction)
            
            elif analysis_type == "移动平均":
                st.markdown("### 📊 移动平均分析")
                
                window_size = st.slider("移动平均窗口大小", 3, min(30, len(ts_data)//2), 7)
                
                # 计算移动平均
                ts_data['MA'] = ts_data[value_col].rolling(window=window_size).mean()
                
                # 可视化
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=ts_data[date_col],
                    y=ts_data[value_col],
                    mode='lines+markers',
                    name='原始数据',
                    line=dict(color='lightblue'),
                    opacity=0.7
                ))
                
                fig.add_trace(go.Scatter(
                    x=ts_data[date_col],
                    y=ts_data['MA'],
                    mode='lines',
                    name=f'{window_size}期移动平均',
                    line=dict(color='red', width=3)
                ))
                
                fig.update_layout(
                    title=f"{value_col} 移动平均分析",
                    xaxis_title="日期",
                    yaxis_title=value_col,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示平滑后的统计
                smoothed_data = ts_data['MA'].dropna()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("平滑后均值", f"{smoothed_data.mean():.4f}")
                with col2:
                    st.metric("平滑后标准差", f"{smoothed_data.std():.4f}")
                with col3:
                    volatility_reduction = (1 - smoothed_data.std() / ts_data[value_col].std()) * 100
                    st.metric("波动性降低", f"{volatility_reduction:.2f}%")

def goal_tracking_section(df: pd.DataFrame):
    """目标跟踪功能"""
    st.markdown('<h2 class="sub-header">🎯 目标跟踪</h2>', unsafe_allow_html=True)
    
    tracking_type = st.selectbox(
        "选择跟踪类型",
        ["KPI跟踪", "销售目标", "预算执行", "项目进度", "绩效指标"]
    )
    
    if tracking_type == "KPI跟踪":
        st.markdown("### 📊 KPI跟踪分析")
        
        # KPI设置
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            kpi_col = st.selectbox("选择KPI指标列", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                target_value = st.number_input("设置目标值", value=float(df[kpi_col].mean()))
            with col2:
                threshold_type = st.selectbox("阈值类型", ["大于等于", "小于等于", "等于"])
            
            # 计算达成情况
            current_value = df[kpi_col].iloc[-1] if len(df) > 0 else 0
            
            if threshold_type == "大于等于":
                achievement = (current_value / target_value * 100) if target_value != 0 else 0
                is_achieved = current_value >= target_value
            elif threshold_type == "小于等于":
                achievement = (target_value / current_value * 100) if current_value != 0 else 0
                is_achieved = current_value <= target_value
            else:  # 等于
                achievement = 100 - abs((current_value - target_value) / target_value * 100) if target_value != 0 else 0
                is_achieved = abs(current_value - target_value) < (target_value * 0.05)  # 5%容差
            
            # 显示结果
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("当前值", f"{current_value:.2f}")
            with col2:
                st.metric("目标值", f"{target_value:.2f}")
            with col3:
                st.metric("达成率", f"{achievement:.1f}%")
            with col4:
                status = "✅ 已达成" if is_achieved else "❌ 未达成"
                st.metric("状态", status)
            
            # KPI趋势图
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[kpi_col],
                mode='lines+markers',
                name='实际值',
                line=dict(color='blue')
            ))
            
            fig.add_hline(
                y=target_value,
                line_dash="dash",
                line_color="red",
                annotation_text="目标线"
            )
            
            fig.update_layout(
                title=f"{kpi_col} KPI跟踪",
                xaxis_title="时间/序号",
                yaxis_title=kpi_col,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 达成率分析
            achievement_history = []
            for i in range(len(df)):
                val = df[kpi_col].iloc[i]
                if threshold_type == "大于等于":
                    ach = (val / target_value * 100) if target_value != 0 else 0
                elif threshold_type == "小于等于":
                    ach = (target_value / val * 100) if val != 0 else 0
                else:
                    ach = 100 - abs((val - target_value) / target_value * 100) if target_value != 0 else 0
                achievement_history.append(min(ach, 150))  # 限制最大值
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df.index,
                y=achievement_history,
                mode='lines+markers',
                name='达成率',
                line=dict(color='green'),
                fill='tonexty'
            ))
            
            fig2.add_hline(
                y=100,
                line_dash="dash",
                line_color="red",
                annotation_text="100%达成线"
            )
            
            fig2.update_layout(
                title="KPI达成率趋势",
                xaxis_title="时间/序号",
                yaxis_title="达成率 (%)",
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)

def dashboard_creation_section(df: pd.DataFrame):
    """仪表板创建功能"""
    st.markdown('<h2 class="sub-header">📊 仪表板创建</h2>', unsafe_allow_html=True)
    
    dashboard_type = st.selectbox(
        "选择仪表板类型",
        ["执行仪表板", "运营仪表板", "分析仪表板", "自定义仪表板"]
    )
    
    if dashboard_type == "执行仪表板":
        st.markdown("### 📈 执行仪表板")
        
        # 关键指标卡片
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 4:
            selected_metrics = st.multiselect(
                "选择关键指标 (最多6个)",
                numeric_cols,
                default=numeric_cols[:4],
                max_selections=6
            )
            
            if selected_metrics:
                # 指标卡片布局
                cols = st.columns(min(len(selected_metrics), 3))
                
                for i, metric in enumerate(selected_metrics):
                    with cols[i % 3]:
                        current_value = df[metric].iloc[-1] if len(df) > 0 else 0
                        previous_value = df[metric].iloc[-2] if len(df) > 1 else current_value
                        
                        delta = current_value - previous_value
                        delta_percent = (delta / previous_value * 100) if previous_value != 0 else 0
                        
                        st.metric(
                            label=metric,
                            value=f"{current_value:.2f}",
                            delta=f"{delta:.2f} ({delta_percent:+.1f}%)"
                        )
                
                # 趋势图表
                st.markdown("#### 📈 趋势概览")
                
                fig = go.Figure()
                
                for metric in selected_metrics[:3]:  # 最多显示3个趋势
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[metric],
                        mode='lines+markers',
                        name=metric
                    ))
                
                fig.update_layout(
                    title="关键指标趋势",
                    xaxis_title="时间/序号",
                    yaxis_title="数值",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 数据表格
                st.markdown("#### 📋 详细数据")
                st.dataframe(df[selected_metrics].tail(10), use_container_width=True)
    
    elif dashboard_type == "自定义仪表板":
        st.markdown("### 🎨 自定义仪表板")
        
        # 布局选择
        layout_type = st.selectbox(
            "选择布局",
            ["2x2网格", "3x2网格", "单列布局", "双列布局"]
        )
        
        # 图表类型选择
        chart_types = ["折线图", "柱状图", "饼图", "散点图", "热力图", "指标卡片"]
        
        if layout_type == "2x2网格":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 图表1")
                chart1_type = st.selectbox("图表类型", chart_types, key="chart1")
                if chart1_type != "指标卡片":
                    chart1_col = st.selectbox("选择数据列", df.columns.tolist(), key="col1")
                    
                    if chart1_type == "折线图":
                        fig1 = px.line(df, y=chart1_col, title=f"{chart1_col} 趋势")
                    elif chart1_type == "柱状图":
                        fig1 = px.bar(df.head(20), y=chart1_col, title=f"{chart1_col} 分布")
                    elif chart1_type == "散点图":
                        if len(df.select_dtypes(include=[np.number]).columns) >= 2:
                            x_col = st.selectbox("X轴", df.select_dtypes(include=[np.number]).columns.tolist(), key="x1")
                            fig1 = px.scatter(df, x=x_col, y=chart1_col, title=f"{x_col} vs {chart1_col}")
                        else:
                            fig1 = px.scatter(df, y=chart1_col, title=f"{chart1_col} 散点图")
                    
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    metric_col = st.selectbox("选择指标列", df.select_dtypes(include=[np.number]).columns.tolist(), key="metric1")
                    if metric_col:
                        value = df[metric_col].iloc[-1] if len(df) > 0 else 0
                        st.metric(metric_col, f"{value:.2f}")
            
            with col2:
                st.markdown("#### 图表2")
                chart2_type = st.selectbox("图表类型", chart_types, key="chart2")
                if chart2_type != "指标卡片":
                    chart2_col = st.selectbox("选择数据列", df.columns.tolist(), key="col2")
                    
                    if chart2_type == "饼图":
                        # 对于饼图，需要分组数据
                        if df[chart2_col].dtype == 'object':
                            pie_data = df[chart2_col].value_counts().head(10)
                            fig2 = px.pie(values=pie_data.values, names=pie_data.index, title=f"{chart2_col} 分布")
                        else:
                            # 数值列分箱
                            bins = pd.cut(df[chart2_col], bins=5)
                            pie_data = bins.value_counts()
                            fig2 = px.pie(values=pie_data.values, names=[str(x) for x in pie_data.index], title=f"{chart2_col} 分布")
                    elif chart2_type == "柱状图":
                        fig2 = px.bar(df.head(20), y=chart2_col, title=f"{chart2_col} 分布")
                    else:
                        fig2 = px.line(df, y=chart2_col, title=f"{chart2_col} 趋势")
                    
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    metric_col = st.selectbox("选择指标列", df.select_dtypes(include=[np.number]).columns.tolist(), key="metric2")
                    if metric_col:
                        value = df[metric_col].iloc[-1] if len(df) > 0 else 0
                        st.metric(metric_col, f"{value:.2f}")

def data_import_export_section(df: pd.DataFrame):
    """数据导入导出功能"""
    st.markdown('<h2 class="sub-header">🔄 数据导入导出</h2>', unsafe_allow_html=True)
    
    operation_type = st.selectbox(
        "选择操作类型",
        ["数据导出", "格式转换", "数据合并", "数据拆分", "批量处理"]
    )
    
    if operation_type == "数据导出":
        st.markdown("### 📤 数据导出")
        
        export_format = st.selectbox(
            "选择导出格式",
            ["CSV", "Excel (XLSX)", "JSON", "HTML", "Parquet"]
        )
        
        # 导出选项
        col1, col2 = st.columns(2)
        
        with col1:
            include_index = st.checkbox("包含索引", value=False)
            selected_columns = st.multiselect(
                "选择要导出的列 (留空表示全部)",
                df.columns.tolist()
            )
        
        with col2:
            if len(df) > 1000:
                export_rows = st.number_input(
                    "导出行数 (0表示全部)",
                    min_value=0,
                    max_value=len(df),
                    value=1000
                )
            else:
                export_rows = len(df)
        
        # 准备导出数据
        export_df = df.copy()
        
        if selected_columns:
            export_df = export_df[selected_columns]
        
        if export_rows > 0 and export_rows < len(export_df):
            export_df = export_df.head(export_rows)
        
        # 显示预览
        st.markdown("#### 📋 导出预览")
        st.info(f"将导出 {len(export_df)} 行 × {len(export_df.columns)} 列数据")
        st.dataframe(export_df.head(5))
        
        # 生成下载按钮
        if export_format == "CSV":
            data = export_df.to_csv(index=include_index, encoding='utf-8-sig')
            filename = "exported_data.csv"
            mime_type = "text/csv"
        
        elif export_format == "Excel (XLSX)":
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=include_index, sheet_name='Data')
            data = buffer.getvalue()
            filename = "exported_data.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        elif export_format == "JSON":
            data = export_df.to_json(orient='records', force_ascii=False, indent=2)
            filename = "exported_data.json"
            mime_type = "application/json"
        
        elif export_format == "HTML":
            data = export_df.to_html(index=include_index, escape=False)
            filename = "exported_data.html"
            mime_type = "text/html"
        
        elif export_format == "Parquet":
            buffer = BytesIO()
            export_df.to_parquet(buffer, index=include_index)
            data = buffer.getvalue()
            filename = "exported_data.parquet"
            mime_type = "application/octet-stream"
        
        st.download_button(
            label=f"📥 下载 {export_format} 文件",
            data=data,
            file_name=filename,
            mime=mime_type
        )
    
    elif operation_type == "格式转换":
        st.markdown("### 🔄 格式转换")
        
        st.info("💡 支持在不同数据格式之间转换")
        
        # 数据类型转换
        st.markdown("#### 📊 数据类型转换")
        
        conversion_options = {
            "数值转文本": "将数值列转换为文本格式",
            "文本转数值": "将文本列转换为数值格式",
            "日期格式化": "标准化日期格式",
            "布尔转换": "转换为布尔值"
        }
        
        conversion_type = st.selectbox(
            "选择转换类型",
            list(conversion_options.keys())
        )
        
        st.info(conversion_options[conversion_type])
        
        if conversion_type == "数值转文本":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("选择数值列", numeric_cols)
                
                if st.button("🔄 执行转换"):
                    df_converted = df.copy()
                    df_converted[selected_col] = df_converted[selected_col].astype(str)
                    
                    st.success(f"✅ 已将 {selected_col} 转换为文本格式")
                    
                    # 显示转换结果
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 转换前")
                        st.write(f"数据类型: {df[selected_col].dtype}")
                        st.dataframe(df[[selected_col]].head())
                    
                    with col2:
                        st.markdown("#### 转换后")
                        st.write(f"数据类型: {df_converted[selected_col].dtype}")
                        st.dataframe(df_converted[[selected_col]].head())
                    
                    # 下载转换后的数据
                    csv = df_converted.to_csv(index=False)
                    st.download_button(
                        label="📥 下载转换后的数据",
                        data=csv,
                        file_name="converted_data.csv",
                        mime="text/csv"
                    )
        
        elif conversion_type == "文本转数值":
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            if text_cols:
                selected_col = st.selectbox("选择文本列", text_cols)
                
                # 检查是否可以转换为数值
                sample_data = df[selected_col].dropna().head(10)
                st.markdown("#### 📋 数据预览")
                st.dataframe(sample_data)
                
                if st.button("🔄 执行转换"):
                    try:
                        df_converted = df.copy()
                        df_converted[selected_col] = pd.to_numeric(df_converted[selected_col], errors='coerce')
                        
                        # 统计转换结果
                        null_count = df_converted[selected_col].isnull().sum()
                        success_rate = (len(df_converted) - null_count) / len(df_converted) * 100
                        
                        st.success(f"✅ 转换完成，成功率: {success_rate:.1f}%")
                        
                        if null_count > 0:
                            st.warning(f"⚠️ 有 {null_count} 个值无法转换，已设为 NaN")
                        
                        # 显示转换结果
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### 转换前")
                            st.write(f"数据类型: {df[selected_col].dtype}")
                            st.dataframe(df[[selected_col]].head())
                        
                        with col2:
                            st.markdown("#### 转换后")
                            st.write(f"数据类型: {df_converted[selected_col].dtype}")
                            st.dataframe(df_converted[[selected_col]].head())
                        
                        # 下载转换后的数据
                        csv = df_converted.to_csv(index=False)
                        st.download_button(
                            label="📥 下载转换后的数据",
                            data=csv,
                            file_name="converted_data.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ 转换失败: {str(e)}")

def data_validation_section(df: pd.DataFrame):
    """数据验证功能"""
    st.markdown('<h2 class="sub-header">📝 数据验证</h2>', unsafe_allow_html=True)
    
    validation_type = st.selectbox(
        "选择验证类型",
        ["数据完整性检查", "数据格式验证", "业务规则验证", "重复值检测", "异常值检测"]
    )
    
    if validation_type == "数据完整性检查":
        st.markdown("### 🔍 数据完整性检查")
        
        # 缺失值分析
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df) * 100).round(2)
        
        # 创建缺失值报告
        missing_report = pd.DataFrame({
            '列名': missing_data.index,
            '缺失值数量': missing_data.values,
            '缺失值比例(%)': missing_percent.values,
            '数据类型': [str(df[col].dtype) for col in missing_data.index]
        })
        
        missing_report = missing_report[missing_report['缺失值数量'] > 0].sort_values('缺失值数量', ascending=False)
        
        if len(missing_report) > 0:
            st.markdown("#### ❌ 发现缺失值")
            st.dataframe(missing_report, use_container_width=True)
            
            # 可视化缺失值
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=missing_report['列名'],
                y=missing_report['缺失值比例(%)'],
                name='缺失值比例',
                marker_color='red'
            ))
            
            fig.update_layout(
                title="各列缺失值比例",
                xaxis_title="列名",
                yaxis_title="缺失值比例 (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 处理建议
            st.markdown("#### 💡 处理建议")
            for _, row in missing_report.iterrows():
                col_name = row['列名']
                missing_pct = row['缺失值比例(%)']
                
                if missing_pct > 50:
                    suggestion = "考虑删除该列或寻找替代数据源"
                    color = "🔴"
                elif missing_pct > 20:
                    suggestion = "考虑使用插值或均值填充"
                    color = "🟡"
                else:
                    suggestion = "可以删除缺失行或简单填充"
                    color = "🟢"
                
                st.write(f"{color} **{col_name}** ({missing_pct}%): {suggestion}")
        
        else:
            st.success("✅ 数据完整性良好，未发现缺失值")
        
        # 数据类型一致性检查
        st.markdown("#### 📊 数据类型分析")
        
        dtype_summary = pd.DataFrame({
            '列名': df.columns,
            '数据类型': [str(dtype) for dtype in df.dtypes],
            '唯一值数量': [df[col].nunique() for col in df.columns],
            '样本值': [str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else 'N/A' for col in df.columns]
        })
        
        st.dataframe(dtype_summary, use_container_width=True)
    
    elif validation_type == "重复值检测":
        st.markdown("### 🔍 重复值检测")
        
        # 完全重复行检测
        duplicate_rows = df.duplicated()
        duplicate_count = duplicate_rows.sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总行数", len(df))
        with col2:
            st.metric("重复行数", duplicate_count)
        with col3:
            duplicate_rate = (duplicate_count / len(df) * 100) if len(df) > 0 else 0
            st.metric("重复率", f"{duplicate_rate:.2f}%")
        
        if duplicate_count > 0:
            st.warning(f"⚠️ 发现 {duplicate_count} 行完全重复的数据")
            
            # 显示重复行
            st.markdown("#### 📋 重复行预览")
            duplicate_data = df[duplicate_rows]
            st.dataframe(duplicate_data.head(10))
            
            # 处理选项
            if st.button("🗑️ 删除重复行"):
                df_cleaned = df.drop_duplicates()
                removed_count = len(df) - len(df_cleaned)
                
                st.success(f"✅ 已删除 {removed_count} 行重复数据")
                
                # 显示清理后的统计
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("清理前行数", len(df))
                with col2:
                    st.metric("清理后行数", len(df_cleaned))
                
                # 下载清理后的数据
                csv = df_cleaned.to_csv(index=False)
                st.download_button(
                    label="📥 下载去重后的数据",
                    data=csv,
                    file_name="deduplicated_data.csv",
                    mime="text/csv"
                )
        else:
            st.success("✅ 未发现完全重复的行")
        
        # 按特定列检测重复
        st.markdown("#### 🎯 按列检测重复")
        
        selected_columns = st.multiselect(
            "选择要检查重复的列",
            df.columns.tolist(),
            help="选择一个或多个列来检测基于这些列的重复值"
        )
        
        if selected_columns:
            column_duplicates = df.duplicated(subset=selected_columns)
            column_duplicate_count = column_duplicates.sum()
            
            st.info(f"基于选定列的重复行数: {column_duplicate_count}")
            
            if column_duplicate_count > 0:
                st.dataframe(df[column_duplicates][selected_columns].head(10))

def conditional_formatting_section(df: pd.DataFrame):
    """条件格式化功能"""
    st.markdown('<h2 class="sub-header">🎨 条件格式化</h2>', unsafe_allow_html=True)
    
    st.info("💡 通过颜色和样式突出显示符合特定条件的数据")
    
    formatting_type = st.selectbox(
        "选择格式化类型",
        ["数值条件格式", "文本条件格式", "热力图着色", "数据条", "图标集"]
    )
    
    if formatting_type == "数值条件格式":
        st.markdown("### 🔢 数值条件格式")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("选择数值列", numeric_cols)
            
            # 条件设置
            col1, col2, col3 = st.columns(3)
            
            with col1:
                condition = st.selectbox(
                    "选择条件",
                    ["大于", "小于", "等于", "介于", "前N%", "后N%"]
                )
            
            with col2:
                if condition in ["大于", "小于", "等于"]:
                    threshold = st.number_input("阈值", value=float(df[selected_col].mean()))
                elif condition == "介于":
                    min_val = st.number_input("最小值", value=float(df[selected_col].min()))
                elif condition in ["前N%", "后N%"]:
                    percentage = st.slider("百分比", 1, 50, 10)
            
            with col3:
                if condition == "介于":
                    max_val = st.number_input("最大值", value=float(df[selected_col].max()))
            
            # 颜色选择
            highlight_color = st.selectbox(
                "选择高亮颜色",
                ["红色", "绿色", "蓝色", "黄色", "橙色", "紫色"]
            )
            
            color_map = {
                "红色": "background-color: #ffcccc",
                "绿色": "background-color: #ccffcc",
                "蓝色": "background-color: #ccccff",
                "黄色": "background-color: #ffffcc",
                "橙色": "background-color: #ffddcc",
                "紫色": "background-color: #ffccff"
            }
            
            if st.button("🎨 应用格式化"):
                # 创建条件函数
                def highlight_condition(val):
                    if condition == "大于":
                        return color_map[highlight_color] if val > threshold else ""
                    elif condition == "小于":
                        return color_map[highlight_color] if val < threshold else ""
                    elif condition == "等于":
                        return color_map[highlight_color] if abs(val - threshold) < 0.001 else ""
                    elif condition == "介于":
                        return color_map[highlight_color] if min_val <= val <= max_val else ""
                    elif condition == "前N%":
                        threshold_val = df[selected_col].quantile(1 - percentage/100)
                        return color_map[highlight_color] if val >= threshold_val else ""
                    elif condition == "后N%":
                        threshold_val = df[selected_col].quantile(percentage/100)
                        return color_map[highlight_color] if val <= threshold_val else ""
                    return ""
                
                # 应用样式
                styled_df = df.style.applymap(highlight_condition, subset=[selected_col])
                
                st.markdown("#### 🎨 格式化结果")
                st.dataframe(styled_df, use_container_width=True)
                
                # 统计符合条件的数据
                if condition == "大于":
                    matching_count = (df[selected_col] > threshold).sum()
                elif condition == "小于":
                    matching_count = (df[selected_col] < threshold).sum()
                elif condition == "等于":
                    matching_count = (abs(df[selected_col] - threshold) < 0.001).sum()
                elif condition == "介于":
                    matching_count = ((df[selected_col] >= min_val) & (df[selected_col] <= max_val)).sum()
                elif condition == "前N%":
                    threshold_val = df[selected_col].quantile(1 - percentage/100)
                    matching_count = (df[selected_col] >= threshold_val).sum()
                elif condition == "后N%":
                    threshold_val = df[selected_col].quantile(percentage/100)
                    matching_count = (df[selected_col] <= threshold_val).sum()
                
                matching_rate = (matching_count / len(df) * 100) if len(df) > 0 else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("符合条件的行数", matching_count)
                with col2:
                    st.metric("符合条件的比例", f"{matching_rate:.1f}%")
    
    elif formatting_type == "热力图着色":
        st.markdown("### 🌡️ 热力图着色")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect(
                "选择要生成热力图的列",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if selected_cols:
                color_scheme = st.selectbox(
                    "选择颜色方案",
                    ["红蓝渐变", "绿红渐变", "蓝白红", "黄橙红"]
                )
                
                if st.button("🎨 生成热力图"):
                    # 创建热力图
                    correlation_matrix = df[selected_cols].corr()
                    
                    color_scales = {
                        "红蓝渐变": "RdBu_r",
                        "绿红渐变": "RdYlGn_r",
                        "蓝白红": "coolwarm",
                        "黄橙红": "YlOrRd"
                    }
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns,
                        colorscale=color_scales[color_scheme],
                        text=correlation_matrix.round(3).values,
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        hoverongaps=False
                    ))
                    
                    fig.update_layout(
                        title="数据相关性热力图",
                        width=600,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 数值热力图
                    st.markdown("#### 📊 数值分布热力图")
                    
                    # 标准化数据用于热力图显示
                    normalized_data = (df[selected_cols] - df[selected_cols].min()) / (df[selected_cols].max() - df[selected_cols].min())
                    
                    # 应用背景渐变
                    styled_df = df[selected_cols].style.background_gradient(
                        cmap=color_scales[color_scheme],
                        subset=selected_cols
                    )
                    
                    st.dataframe(styled_df, use_container_width=True)

def worksheet_management_section(df: pd.DataFrame):
    """工作表管理功能"""
    st.markdown('<h2 class="sub-header">📋 工作表管理</h2>', unsafe_allow_html=True)
    
    management_type = st.selectbox(
        "选择管理操作",
        ["工作表信息", "数据分割", "数据合并", "工作表比较", "数据备份"]
    )
    
    if management_type == "工作表信息":
        st.markdown("### 📊 工作表信息")
        
        # 基本信息
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总行数", len(df))
        with col2:
            st.metric("总列数", len(df.columns))
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("内存占用", f"{memory_usage:.2f} MB")
        with col4:
            st.metric("数据类型数", len(df.dtypes.unique()))
        
        # 详细信息
        st.markdown("#### 📋 列详细信息")
        
        column_info = []
        for col in df.columns:
            col_data = df[col]
            info = {
                '列名': col,
                '数据类型': str(col_data.dtype),
                '非空值数': col_data.count(),
                '缺失值数': col_data.isnull().sum(),
                '唯一值数': col_data.nunique(),
                '内存使用(KB)': col_data.memory_usage(deep=True) / 1024
            }
            
            if col_data.dtype in ['int64', 'float64']:
                info.update({
                    '最小值': col_data.min(),
                    '最大值': col_data.max(),
                    '平均值': col_data.mean()
                })
            
            column_info.append(info)
        
        info_df = pd.DataFrame(column_info)
        st.dataframe(info_df, use_container_width=True)
        
        # 数据质量评分
        st.markdown("#### 🎯 数据质量评分")
        
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness_score = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # 一致性评分（基于数据类型的一致性）
        consistency_score = 85  # 简化评分
        
        # 准确性评分（基于异常值检测）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            outlier_count += len(outliers)
        
        accuracy_score = max(0, 100 - (outlier_count / len(df) * 100)) if len(df) > 0 else 100
        
        overall_score = (completeness_score + consistency_score + accuracy_score) / 3
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("完整性", f"{completeness_score:.1f}%")
        with col2:
            st.metric("一致性", f"{consistency_score:.1f}%")
        with col3:
            st.metric("准确性", f"{accuracy_score:.1f}%")
        with col4:
            st.metric("总体评分", f"{overall_score:.1f}%")
    
    elif management_type == "数据分割":
        st.markdown("### ✂️ 数据分割")
        
        split_method = st.selectbox(
            "选择分割方式",
            ["按行数分割", "按比例分割", "按列值分割", "随机分割"]
        )
        
        if split_method == "按行数分割":
            rows_per_split = st.number_input(
                "每个分割的行数",
                min_value=1,
                max_value=len(df),
                value=min(1000, len(df)//2)
            )
            
            if st.button("🔪 执行分割"):
                splits = []
                for i in range(0, len(df), rows_per_split):
                    split_df = df.iloc[i:i+rows_per_split]
                    splits.append(split_df)
                
                st.success(f"✅ 数据已分割为 {len(splits)} 个部分")
                
                for i, split_df in enumerate(splits):
                    st.markdown(f"#### 📄 分割 {i+1} ({len(split_df)} 行)")
                    st.dataframe(split_df.head(3))
                    
                    # 下载按钮
                    csv = split_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 下载分割 {i+1}",
                        data=csv,
                        file_name=f"split_{i+1}.csv",
                        mime="text/csv",
                        key=f"download_split_{i}"
                    )
        
        elif split_method == "按比例分割":
            train_ratio = st.slider("训练集比例", 0.1, 0.9, 0.7, 0.1)
            test_ratio = 1 - train_ratio
            
            st.info(f"训练集: {train_ratio*100:.0f}%, 测试集: {test_ratio*100:.0f}%")
            
            if st.button("🔪 执行分割"):
                train_size = int(len(df) * train_ratio)
                
                train_df = df.iloc[:train_size]
                test_df = df.iloc[train_size:]
                
                st.success("✅ 数据已按比例分割")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### 📊 训练集 ({len(train_df)} 行)")
                    st.dataframe(train_df.head(3))
                    
                    csv_train = train_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载训练集",
                        data=csv_train,
                        file_name="train_data.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.markdown(f"#### 📊 测试集 ({len(test_df)} 行)")
                    st.dataframe(test_df.head(3))
                    
                    csv_test = test_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载测试集",
                        data=csv_test,
                        file_name="test_data.csv",
                        mime="text/csv"
                    )

def data_filtering_sorting_section(df: pd.DataFrame):
    """数据筛选排序功能"""
    st.markdown('<h2 class="sub-header">🔍 数据筛选排序</h2>', unsafe_allow_html=True)
    
    operation_type = st.selectbox(
        "选择操作类型",
        ["数据筛选", "数据排序", "高级筛选", "条件筛选", "组合操作"]
    )
    
    if operation_type == "数据筛选":
        st.markdown("### 🔍 数据筛选")
        
        # 选择筛选列
        filter_column = st.selectbox("选择筛选列", df.columns.tolist())
        
        col_data = df[filter_column]
        
        if col_data.dtype in ['int64', 'float64']:
            # 数值筛选
            st.markdown("#### 🔢 数值筛选")
            
            filter_type = st.selectbox(
                "筛选类型",
                ["范围筛选", "条件筛选", "百分位筛选"]
            )
            
            if filter_type == "范围筛选":
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                
                selected_range = st.slider(
                    f"选择 {filter_column} 的范围",
                    min_val,
                    max_val,
                    (min_val, max_val)
                )
                
                filtered_df = df[(df[filter_column] >= selected_range[0]) & 
                               (df[filter_column] <= selected_range[1])]
            
            elif filter_type == "条件筛选":
                condition = st.selectbox("选择条件", ["大于", "小于", "等于", "不等于"])
                threshold = st.number_input("阈值", value=float(col_data.mean()))
                
                if condition == "大于":
                    filtered_df = df[df[filter_column] > threshold]
                elif condition == "小于":
                    filtered_df = df[df[filter_column] < threshold]
                elif condition == "等于":
                    filtered_df = df[df[filter_column] == threshold]
                else:  # 不等于
                    filtered_df = df[df[filter_column] != threshold]
            
            elif filter_type == "百分位筛选":
                percentile = st.slider("选择百分位", 1, 99, 50)
                threshold_val = col_data.quantile(percentile/100)
                
                direction = st.selectbox("筛选方向", ["高于百分位", "低于百分位"])
                
                if direction == "高于百分位":
                    filtered_df = df[df[filter_column] >= threshold_val]
                else:
                    filtered_df = df[df[filter_column] <= threshold_val]
                
                st.info(f"第{percentile}百分位值: {threshold_val:.2f}")
        
        else:
            # 文本筛选
            st.markdown("#### 📝 文本筛选")
            
            unique_values = col_data.unique()
            
            if len(unique_values) <= 50:
                # 多选筛选
                selected_values = st.multiselect(
                    f"选择 {filter_column} 的值",
                    unique_values,
                    default=unique_values[:min(5, len(unique_values))]
                )
                
                filtered_df = df[df[filter_column].isin(selected_values)]
            
            else:
                # 文本搜索
                search_text = st.text_input(f"搜索 {filter_column} 中包含的文本")
                
                if search_text:
                    filtered_df = df[df[filter_column].astype(str).str.contains(search_text, case=False, na=False)]
                else:
                    filtered_df = df
        
        # 显示筛选结果
        st.markdown("#### 📊 筛选结果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("原始行数", len(df))
        with col2:
            st.metric("筛选后行数", len(filtered_df))
        with col3:
            retention_rate = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
            st.metric("保留率", f"{retention_rate:.1f}%")
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # 下载筛选结果
        if len(filtered_df) > 0:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 下载筛选结果",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
    
    elif operation_type == "数据排序":
        st.markdown("### 📊 数据排序")
        
        # 选择排序列
        sort_columns = st.multiselect(
            "选择排序列 (按优先级顺序)",
            df.columns.tolist(),
            default=[df.columns[0]]
        )
        
        if sort_columns:
            # 排序方向
            sort_orders = []
            
            for col in sort_columns:
                order = st.selectbox(
                    f"{col} 的排序方向",
                    ["升序", "降序"],
                    key=f"sort_order_{col}"
                )
                sort_orders.append(order == "升序")
            
            if st.button("📊 执行排序"):
                sorted_df = df.sort_values(
                    by=sort_columns,
                    ascending=sort_orders
                )
                
                st.success("✅ 数据排序完成")
                
                # 显示排序信息
                sort_info = ", ".join([
                    f"{col}({'升序' if asc else '降序'})"
                    for col, asc in zip(sort_columns, sort_orders)
                ])
                
                st.info(f"排序规则: {sort_info}")
                
                # 显示排序结果
                st.dataframe(sorted_df, use_container_width=True)
                
                # 下载排序结果
                csv = sorted_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载排序结果",
                    data=csv,
                    file_name="sorted_data.csv",
                    mime="text/csv"
                )

def mathematical_functions_section(df: pd.DataFrame):
    """数学统计函数功能"""
    st.markdown('<h2 class="sub-header">🧮 数学统计函数</h2>', unsafe_allow_html=True)
    
    function_category = st.selectbox(
        "选择函数类别",
        ["基础数学函数", "统计函数", "三角函数", "对数函数", "自定义公式"]
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("⚠️ 没有找到数值列，无法执行数学函数操作")
        return
    
    if function_category == "基础数学函数":
        st.markdown("### ➕ 基础数学函数")
        
        selected_col = st.selectbox("选择数值列", numeric_cols)
        
        math_function = st.selectbox(
            "选择数学函数",
            ["平方", "平方根", "立方", "绝对值", "四舍五入", "向上取整", "向下取整"]
        )
        
        if st.button("🧮 计算"):
            result_col_name = f"{selected_col}_{math_function}"
            
            if math_function == "平方":
                df[result_col_name] = df[selected_col] ** 2
            elif math_function == "平方根":
                df[result_col_name] = np.sqrt(np.abs(df[selected_col]))
            elif math_function == "立方":
                df[result_col_name] = df[selected_col] ** 3
            elif math_function == "绝对值":
                df[result_col_name] = np.abs(df[selected_col])
            elif math_function == "四舍五入":
                decimal_places = st.number_input("小数位数", min_value=0, max_value=10, value=2)
                df[result_col_name] = np.round(df[selected_col], decimal_places)
            elif math_function == "向上取整":
                df[result_col_name] = np.ceil(df[selected_col])
            elif math_function == "向下取整":
                df[result_col_name] = np.floor(df[selected_col])
            
            st.success(f"✅ 已计算 {math_function}，结果保存在 {result_col_name} 列")
            
            # 显示结果对比
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 原始数据")
                st.dataframe(df[[selected_col]].head(10))
            
            with col2:
                st.markdown("#### 计算结果")
                st.dataframe(df[[result_col_name]].head(10))
            
            # 可视化对比
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(df[:100]))),
                y=df[selected_col][:100],
                mode='lines',
                name=f'原始 {selected_col}',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(len(df[:100]))),
                y=df[result_col_name][:100],
                mode='lines',
                name=f'{math_function} 结果',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f"{selected_col} vs {math_function}结果",
                xaxis_title="数据点",
                yaxis_title="值",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif function_category == "统计函数":
        st.markdown("### 📊 统计函数")
        
        selected_cols = st.multiselect(
            "选择数值列",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        if selected_cols:
            stat_functions = [
                "均值", "中位数", "众数", "标准差", "方差", 
                "偏度", "峰度", "最小值", "最大值", "范围",
                "四分位距", "变异系数", "相关系数"
            ]
            
            selected_functions = st.multiselect(
                "选择统计函数",
                stat_functions,
                default=["均值", "标准差", "中位数"]
            )
            
            if st.button("📊 计算统计量"):
                results = {}
                
                for col in selected_cols:
                    col_results = {}
                    col_data = df[col].dropna()
                    
                    if "均值" in selected_functions:
                        col_results["均值"] = col_data.mean()
                    if "中位数" in selected_functions:
                        col_results["中位数"] = col_data.median()
                    if "众数" in selected_functions:
                        mode_val = col_data.mode()
                        col_results["众数"] = mode_val.iloc[0] if len(mode_val) > 0 else np.nan
                    if "标准差" in selected_functions:
                        col_results["标准差"] = col_data.std()
                    if "方差" in selected_functions:
                        col_results["方差"] = col_data.var()
                    if "偏度" in selected_functions:
                        col_results["偏度"] = col_data.skew()
                    if "峰度" in selected_functions:
                        col_results["峰度"] = col_data.kurtosis()
                    if "最小值" in selected_functions:
                        col_results["最小值"] = col_data.min()
                    if "最大值" in selected_functions:
                        col_results["最大值"] = col_data.max()
                    if "范围" in selected_functions:
                        col_results["范围"] = col_data.max() - col_data.min()
                    if "四分位距" in selected_functions:
                        col_results["四分位距"] = col_data.quantile(0.75) - col_data.quantile(0.25)
                    if "变异系数" in selected_functions:
                        col_results["变异系数"] = col_data.std() / col_data.mean() if col_data.mean() != 0 else np.nan
                    
                    results[col] = col_results
                
                # 显示结果表格
                results_df = pd.DataFrame(results).T
                st.markdown("#### 📊 统计结果")
                st.dataframe(results_df.round(4), use_container_width=True)
                
                # 相关系数矩阵
                if "相关系数" in selected_functions and len(selected_cols) > 1:
                    st.markdown("#### 🔗 相关系数矩阵")
                    corr_matrix = df[selected_cols].corr()
                    
                    # 热力图
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu_r',
                        text=corr_matrix.round(3).values,
                        texttemplate="%{text}",
                        textfont={"size": 10}
                    ))
                    
                    fig.update_layout(
                        title="相关系数热力图",
                        width=500,
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(corr_matrix.round(4), use_container_width=True)
                
                # 下载统计结果
                csv = results_df.to_csv()
                st.download_button(
                    label="📥 下载统计结果",
                    data=csv,
                    file_name="statistical_results.csv",
                    mime="text/csv"
                 )

def business_intelligence_section(df: pd.DataFrame):
    """商业智能分析功能"""
    st.markdown('<h2 class="sub-header">📈 商业智能分析</h2>', unsafe_allow_html=True)
    
    bi_type = st.selectbox(
        "选择分析类型",
        ["销售分析", "客户分析", "产品分析", "趋势预测", "KPI仪表板"]
    )
    
    if bi_type == "销售分析":
        st.markdown("### 💰 销售分析")
        
        # 检查是否有销售相关列
        potential_sales_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sales', '销售', 'revenue', '收入', 'amount', '金额'])]
        potential_date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', '日期', 'time', '时间'])]
        
        if potential_sales_cols:
            sales_col = st.selectbox("选择销售金额列", potential_sales_cols)
            
            if potential_date_cols:
                date_col = st.selectbox("选择日期列", potential_date_cols)
                
                # 销售趋势分析
                st.markdown("#### 📊 销售趋势分析")
                
                try:
                    # 转换日期列
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    
                    # 按月汇总销售数据
                    monthly_sales = df.groupby(df[date_col].dt.to_period('M'))[sales_col].sum().reset_index()
                    monthly_sales[date_col] = monthly_sales[date_col].astype(str)
                    
                    # 绘制销售趋势图
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=monthly_sales[date_col],
                        y=monthly_sales[sales_col],
                        mode='lines+markers',
                        name='月销售额',
                        line=dict(color='blue', width=3)
                    ))
                    
                    fig.update_layout(
                        title="月度销售趋势",
                        xaxis_title="月份",
                        yaxis_title="销售额",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 销售统计
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_sales = df[sales_col].sum()
                        st.metric("总销售额", f"{total_sales:,.2f}")
                    
                    with col2:
                        avg_sales = df[sales_col].mean()
                        st.metric("平均销售额", f"{avg_sales:,.2f}")
                    
                    with col3:
                        max_sales = df[sales_col].max()
                        st.metric("最高销售额", f"{max_sales:,.2f}")
                    
                    with col4:
                        sales_growth = ((monthly_sales[sales_col].iloc[-1] - monthly_sales[sales_col].iloc[0]) / monthly_sales[sales_col].iloc[0] * 100) if len(monthly_sales) > 1 else 0
                        st.metric("增长率", f"{sales_growth:.1f}%")
                    
                except Exception as e:
                    st.error(f"日期处理错误: {str(e)}")
            
            # 销售分布分析
            st.markdown("#### 📊 销售分布分析")
            
            fig = go.Figure(data=[go.Histogram(x=df[sales_col], nbinsx=20)])
            fig.update_layout(
                title="销售额分布",
                xaxis_title="销售额",
                yaxis_title="频次",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ 未找到销售相关列，请确保数据包含销售金额信息")
    
    elif bi_type == "KPI仪表板":
        st.markdown("### 📊 KPI仪表板")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # 选择KPI指标
            selected_kpis = st.multiselect(
                "选择KPI指标",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
            
            if selected_kpis:
                # 创建KPI卡片
                cols = st.columns(len(selected_kpis))
                
                for i, kpi in enumerate(selected_kpis):
                    with cols[i]:
                        current_value = df[kpi].iloc[-1] if len(df) > 0 else 0
                        avg_value = df[kpi].mean()
                        
                        # 计算变化率
                        if len(df) > 1:
                            previous_value = df[kpi].iloc[-2]
                            change_rate = ((current_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
                        else:
                            change_rate = 0
                        
                        st.metric(
                            label=kpi,
                            value=f"{current_value:.2f}",
                            delta=f"{change_rate:.1f}%"
                        )
                
                # KPI趋势图
                st.markdown("#### 📈 KPI趋势")
                
                fig = go.Figure()
                
                for kpi in selected_kpis:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(df))),
                        y=df[kpi],
                        mode='lines',
                        name=kpi
                    ))
                
                fig.update_layout(
                    title="KPI趋势图",
                    xaxis_title="时间点",
                    yaxis_title="值",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

def enterprise_reports_section(df: pd.DataFrame):
    """企业报表功能"""
    st.markdown('<h2 class="sub-header">📋 企业报表</h2>', unsafe_allow_html=True)
    
    report_type = st.selectbox(
        "选择报表类型",
        ["数据摘要报表", "财务报表", "运营报表", "自定义报表", "定期报表"]
    )
    
    if report_type == "数据摘要报表":
        st.markdown("### 📊 数据摘要报表")
        
        # 报表标题
        report_title = st.text_input("报表标题", value="数据分析摘要报表")
        
        # 生成报表
        if st.button("📋 生成报表"):
            st.markdown(f"# {report_title}")
            st.markdown(f"**生成时间:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown("---")
            
            # 基本信息
            st.markdown("## 📊 基本信息")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总行数", len(df))
            with col2:
                st.metric("总列数", len(df.columns))
            with col3:
                missing_rate = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 0
                st.metric("缺失率", f"{missing_rate:.1f}%")
            with col4:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("内存占用", f"{memory_mb:.1f}MB")
            
            # 数值列统计
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                st.markdown("## 🔢 数值列统计")
                
                numeric_summary = df[numeric_cols].describe().round(2)
                st.dataframe(numeric_summary, use_container_width=True)
                
                # 相关性分析
                if len(numeric_cols) > 1:
                    st.markdown("## 🔗 相关性分析")
                    
                    corr_matrix = df[numeric_cols].corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu_r',
                        text=corr_matrix.round(2).values,
                        texttemplate="%{text}",
                        textfont={"size": 10}
                    ))
                    
                    fig.update_layout(
                        title="变量相关性热力图",
                        width=600,
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # 文本列分析
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if text_cols:
                st.markdown("## 📝 文本列分析")
                
                text_summary = []
                for col in text_cols:
                    summary = {
                        '列名': col,
                        '唯一值数量': df[col].nunique(),
                        '最频繁值': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A',
                        '缺失值数量': df[col].isnull().sum()
                    }
                    text_summary.append(summary)
                
                text_summary_df = pd.DataFrame(text_summary)
                st.dataframe(text_summary_df, use_container_width=True)
            
            # 数据质量评估
            st.markdown("## ✅ 数据质量评估")
            
            quality_metrics = {
                '完整性': f"{100 - missing_rate:.1f}%",
                '一致性': "85.0%",  # 简化评分
                '准确性': "90.0%",  # 简化评分
                '及时性': "95.0%"   # 简化评分
            }
            
            quality_df = pd.DataFrame(list(quality_metrics.items()), columns=['指标', '评分'])
            st.dataframe(quality_df, use_container_width=True)
            
            # 建议和结论
            st.markdown("## 💡 建议和结论")
            
            recommendations = []
            
            if missing_rate > 10:
                recommendations.append("• 数据缺失率较高，建议进行数据清洗和补全")
            
            if len(numeric_cols) > 0:
                high_corr_pairs = []
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.8:
                                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                
                if high_corr_pairs:
                    recommendations.append("• 发现高相关性变量，建议考虑特征选择")
            
            if len(df) > 10000:
                recommendations.append("• 数据量较大，建议考虑采样或分批处理")
            
            if not recommendations:
                recommendations.append("• 数据质量良好，可以进行进一步分析")
            
            for rec in recommendations:
                st.markdown(rec)
            
            # 导出报表
            st.markdown("---")
            
            # 生成HTML报表
            html_content = f"""
            <html>
            <head>
                <title>{report_title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2E86AB; }}
                    h2 {{ color: #A23B72; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>{report_title}</h1>
                <p><strong>生成时间:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                
                <h2>基本信息</h2>
                <ul>
                    <li>总行数: {len(df)}</li>
                    <li>总列数: {len(df.columns)}</li>
                    <li>缺失率: {missing_rate:.1f}%</li>
                    <li>内存占用: {memory_mb:.1f}MB</li>
                </ul>
                
                <h2>数据质量评估</h2>
                {quality_df.to_html(index=False)}
                
                <h2>建议和结论</h2>
                <ul>
                    {''.join([f'<li>{rec[2:]}</li>' for rec in recommendations])}
                </ul>
            </body>
            </html>
            """
            
            st.download_button(
                label="📥 下载HTML报表",
                data=html_content,
                file_name=f"{report_title}_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                mime="text/html"
            )

def mobile_adaptation_section(df: pd.DataFrame):
    """移动端适配功能"""
    st.markdown('<h2 class="sub-header">📱 移动端适配</h2>', unsafe_allow_html=True)
    
    st.info("💡 优化数据展示以适配移动设备")
    
    adaptation_type = st.selectbox(
        "选择适配类型",
        ["响应式表格", "移动端图表", "简化视图", "触摸优化", "离线功能"]
    )
    
    if adaptation_type == "响应式表格":
        st.markdown("### 📱 响应式表格")
        
        # 列选择（移动端显示较少列）
        max_cols_mobile = st.slider("移动端最大显示列数", 1, min(5, len(df.columns)), 3)
        
        selected_cols = st.multiselect(
            "选择要在移动端显示的列",
            df.columns.tolist(),
            default=df.columns.tolist()[:max_cols_mobile]
        )
        
        if selected_cols:
            # 移动端优化的表格显示
            st.markdown("#### 📱 移动端预览")
            
            mobile_df = df[selected_cols].head(10)
            
            # 使用更紧凑的显示方式
            st.dataframe(
                mobile_df,
                use_container_width=True,
                height=300
            )
            
            # 分页控制
            st.markdown("#### 📄 分页控制")
            
            page_size = st.selectbox("每页显示行数", [5, 10, 20, 50], index=1)
            total_pages = (len(df) - 1) // page_size + 1
            
            page_num = st.number_input(
                "页码",
                min_value=1,
                max_value=total_pages,
                value=1
            )
            
            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, len(df))
            
            page_df = df[selected_cols].iloc[start_idx:end_idx]
            
            st.info(f"显示第 {start_idx + 1}-{end_idx} 行，共 {len(df)} 行")
            st.dataframe(page_df, use_container_width=True)
    
    elif adaptation_type == "移动端图表":
        st.markdown("### 📊 移动端图表")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            chart_col = st.selectbox("选择图表数据列", numeric_cols)
            
            chart_type = st.selectbox(
                "选择图表类型",
                ["简化柱状图", "迷你折线图", "饼图", "仪表盘"]
            )
            
            if chart_type == "简化柱状图":
                # 移动端优化的柱状图
                top_n = st.slider("显示前N个数据点", 5, 20, 10)
                
                chart_data = df.nlargest(top_n, chart_col)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=chart_data.index[:top_n],
                        y=chart_data[chart_col][:top_n],
                        marker_color='lightblue'
                    )
                ])
                
                fig.update_layout(
                    title=f"Top {top_n} - {chart_col}",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(size=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "迷你折线图":
                # 移动端优化的折线图
                sample_size = min(50, len(df))
                sample_data = df.sample(n=sample_size) if len(df) > sample_size else df
                
                fig = go.Figure(data=[
                    go.Scatter(
                        x=list(range(len(sample_data))),
                        y=sample_data[chart_col],
                        mode='lines',
                        line=dict(width=2, color='blue')
                    )
                ])
                
                fig.update_layout(
                    title=f"{chart_col} 趋势",
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False,
                    font=dict(size=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)

def data_security_section(df: pd.DataFrame):
    """数据安全功能"""
    st.markdown('<h2 class="sub-header">🔒 数据安全</h2>', unsafe_allow_html=True)
    
    security_type = st.selectbox(
        "选择安全功能",
        ["数据脱敏", "访问控制", "数据加密", "审计日志", "隐私保护"]
    )
    
    if security_type == "数据脱敏":
        st.markdown("### 🎭 数据脱敏")
        
        st.info("💡 对敏感数据进行脱敏处理，保护隐私信息")
        
        # 选择需要脱敏的列
        sensitive_cols = st.multiselect(
            "选择需要脱敏的列",
            df.columns.tolist(),
            help="选择包含敏感信息的列（如姓名、电话、邮箱等）"
        )
        
        if sensitive_cols:
            masking_method = st.selectbox(
                "选择脱敏方法",
                ["部分遮蔽", "完全替换", "哈希处理", "随机化"]
            )
            
            if st.button("🎭 执行脱敏"):
                masked_df = df.copy()
                
                for col in sensitive_cols:
                    if masking_method == "部分遮蔽":
                        # 保留前2位和后2位，中间用*替换
                        masked_df[col] = masked_df[col].astype(str).apply(
                            lambda x: x[:2] + '*' * max(0, len(x) - 4) + x[-2:] if len(x) > 4 else '*' * len(x)
                        )
                    
                    elif masking_method == "完全替换":
                        # 完全替换为固定字符
                        masked_df[col] = '***MASKED***'
                    
                    elif masking_method == "哈希处理":
                        # 使用哈希函数
                        import hashlib
                        masked_df[col] = masked_df[col].astype(str).apply(
                            lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[:8]
                        )
                    
                    elif masking_method == "随机化":
                        # 随机打乱数据
                        masked_df[col] = masked_df[col].sample(frac=1).reset_index(drop=True)
                
                st.success("✅ 数据脱敏完成")
                
                # 显示脱敏前后对比
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 原始数据")
                    st.dataframe(df[sensitive_cols].head(5))
                
                with col2:
                    st.markdown("#### 脱敏后数据")
                    st.dataframe(masked_df[sensitive_cols].head(5))
                
                # 下载脱敏后的数据
                csv = masked_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载脱敏数据",
                    data=csv,
                    file_name="masked_data.csv",
                    mime="text/csv"
                )
    
    elif security_type == "隐私保护":
        st.markdown("### 🛡️ 隐私保护")
        
        st.info("💡 检测和保护数据中的个人隐私信息")
        
        # 隐私信息检测
        privacy_patterns = {
            '邮箱': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '电话': r'\b(?:\+?86)?1[3-9]\d{9}\b',
            '身份证': r'\b\d{17}[\dXx]\b',
            '银行卡': r'\b\d{16,19}\b'
        }
        
        detected_privacy = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                col_text = ' '.join(df[col].astype(str).tolist())
                
                for privacy_type, pattern in privacy_patterns.items():
                    import re
                    matches = re.findall(pattern, col_text)
                    
                    if matches:
                        if col not in detected_privacy:
                            detected_privacy[col] = []
                        detected_privacy[col].append({
                            'type': privacy_type,
                            'count': len(matches)
                        })
        
        if detected_privacy:
            st.warning("⚠️ 检测到可能的隐私信息")
            
            for col, privacy_info in detected_privacy.items():
                st.markdown(f"**列 '{col}' 中发现:**")
                for info in privacy_info:
                    st.write(f"- {info['type']}: {info['count']} 个")
            
            # 提供保护建议
            st.markdown("#### 💡 保护建议")
            st.markdown("""
            - 对包含邮箱的列进行部分遮蔽处理
            - 对电话号码进行中间位数遮蔽
            - 对身份证号进行脱敏处理
            - 对银行卡号进行加密存储
            - 建议实施访问控制和审计机制
            """)
        
        else:
            st.success("✅ 未检测到明显的隐私信息")
        
        # 数据安全评分
        st.markdown("#### 🎯 数据安全评分")
        
        security_score = 100
        
        # 根据检测到的隐私信息扣分
        if detected_privacy:
            security_score -= len(detected_privacy) * 10
        
        # 根据数据规模调整评分
        if len(df) > 10000:
            security_score -= 5  # 大数据集风险更高
        
        security_score = max(0, security_score)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("安全评分", f"{security_score}/100")
        with col2:
            risk_level = "低" if security_score >= 80 else "中" if security_score >= 60 else "高"
            st.metric("风险等级", risk_level)
        with col3:
            st.metric("隐私列数", len(detected_privacy))

if __name__ == "__main__":
    main()