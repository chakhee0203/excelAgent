import pandas as pd
import streamlit as st
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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

class ExcelAgentFull:
    """Excel智能分析代理"""
    
    def __init__(self, api_key: str, model_config: Dict):
        self.api_key = api_key
        self.model_config = model_config
        self.llm = None
        self.agent = None
        
        if LANGCHAIN_AVAILABLE and api_key:
            try:
                self.llm = ChatOpenAI(
                    api_key=api_key,
                    model=model_config.get('model_name', 'gpt-3.5-turbo'),
                    base_url=model_config.get('base_url'),
                    temperature=0.1
                )
            except Exception as e:
                st.error(f"❌ 模型初始化失败: {str(e)}")
    
    def create_agent(self, df: pd.DataFrame):
        """创建数据分析代理"""
        if not self.llm:
            return None
        
        try:
            self.agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                verbose=True,
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
            return self.agent
        except Exception as e:
            st.error(f"❌ 代理创建失败: {str(e)}")
            return None
    
    def analyze_data(self, df: pd.DataFrame, analysis_type: str = "comprehensive") -> str:
        """智能数据分析"""
        if not self.llm:
            return "❌ AI模型未正确配置，无法进行智能分析"
        
        try:
            # 构建分析提示
            data_info = f"""
            数据基本信息：
            - 行数：{len(df)}
            - 列数：{len(df.columns)}
            - 列名：{', '.join(df.columns.tolist())}
            - 数据类型：{df.dtypes.to_dict()}
            - 缺失值：{df.isnull().sum().to_dict()}
            
            数据预览：
            {df.head().to_string()}
            
            数值列统计：
            {df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else '无数值列'}
            """
            
            if analysis_type == "comprehensive":
                prompt = f"""
                请对以下Excel数据进行全面的智能分析：
                
                {data_info}
                
                请提供：
                1. 数据质量评估
                2. 关键发现和洞察
                3. 数据分布特征
                4. 异常值检测
                5. 相关性分析
                6. 业务建议
                
                请用中文回答，并提供具体的数据支撑。
                """
            elif analysis_type == "quality":
                prompt = f"""
                请对以下Excel数据进行数据质量分析：
                
                {data_info}
                
                请重点分析：
                1. 数据完整性（缺失值情况）
                2. 数据一致性（格式统一性）
                3. 数据准确性（异常值检测）
                4. 数据及时性（时间相关分析）
                5. 改进建议
                
                请用中文回答。
                """
            elif analysis_type == "business":
                prompt = f"""
                请从商业角度分析以下Excel数据：
                
                {data_info}
                
                请提供：
                1. 关键业务指标识别
                2. 趋势分析
                3. 业务机会发现
                4. 风险点识别
                5. 决策建议
                
                请用中文回答，并提供可执行的建议。
                """
            
            # 调用LLM进行分析
            messages = [SystemMessage(content="你是一个专业的数据分析师，擅长Excel数据分析和商业洞察。"),
                       HumanMessage(content=prompt)]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"❌ 分析过程中出现错误: {str(e)}"
    
    def generate_chart_suggestions(self, df: pd.DataFrame) -> List[Dict]:
        """生成图表建议"""
        suggestions = []
        
        # 获取数值列和分类列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 基础图表建议
        suggestions.extend(self._get_basic_chart_suggestions(df))
        
        # 如果有AI模型，生成智能建议
        if self.llm:
            try:
                ai_suggestions = self._get_ai_chart_suggestions(df, numeric_cols, categorical_cols)
                suggestions.extend(ai_suggestions)
            except Exception as e:
                st.warning(f"AI图表建议生成失败: {str(e)}")
        
        return suggestions
    
    def _get_basic_chart_suggestions(self, df: pd.DataFrame) -> List[Dict]:
        """获取基础图表建议"""
        suggestions = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 单变量分析建议
        for col in numeric_cols[:3]:  # 限制数量
            suggestions.append({
                'type': 'histogram',
                'title': f'{col} 分布直方图',
                'description': f'显示 {col} 的数据分布情况',
                'columns': [col],
                'priority': 'medium'
            })
        
        # 双变量分析建议
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'scatter',
                'title': f'{numeric_cols[0]} vs {numeric_cols[1]} 散点图',
                'description': f'分析 {numeric_cols[0]} 和 {numeric_cols[1]} 的相关性',
                'columns': numeric_cols[:2],
                'priority': 'high'
            })
        
        return suggestions
    
    def _get_ai_chart_suggestions(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> List[Dict]:
        """获取AI图表建议"""
        # 这里可以实现更复杂的AI图表建议逻辑
        return []
    
    def natural_language_query(self, df: pd.DataFrame, query: str) -> str:
        """自然语言查询"""
        if not self.agent:
            self.create_agent(df)
        
        if not self.agent:
            return "❌ AI代理未正确配置，无法处理自然语言查询"
        
        try:
            # 构建查询提示
            enhanced_query = f"""
            请分析数据并回答以下问题：{query}
            
            请注意：
            1. 如果需要计算，请提供具体的数值结果
            2. 如果需要筛选数据，请说明筛选条件和结果
            3. 请用中文回答
            4. 如果无法回答，请说明原因
            5. 请确保回答格式正确，避免输出解析错误
            """
            
            # 尝试多种调用方式来处理解析错误
            try:
                response = self.agent.invoke({"input": enhanced_query})
                if isinstance(response, dict):
                    return response.get('output', response.get('result', str(response)))
                else:
                    return str(response)
            except Exception as invoke_error:
                # 如果invoke失败，尝试使用run方法
                try:
                    response = self.agent.run(enhanced_query)
                    return str(response)
                except Exception as run_error:
                    # 如果都失败了，返回更详细的错误信息
                    error_msg = str(invoke_error)
                    if "output parsing error" in error_msg.lower():
                        return f"❌ AI回答格式解析失败，请尝试重新提问或简化问题。原始错误: {error_msg}"
                    else:
                        return f"❌ 查询处理失败: {error_msg}"
            
        except Exception as e:
            return f"❌ 查询处理失败: {str(e)}"

def check_langchain_status():
    """检查LangChain状态"""
    if not LANGCHAIN_AVAILABLE:
        st.error("❌ LangChain未安装或导入失败")
        st.info("💡 请运行: pip install langchain langchain-openai langchain-community")
        return False
    return True

def test_connection(api_key: str, model_config: Dict):
    """测试连接"""
    if not check_langchain_status():
        return False, "LangChain未正确安装"
    
    if not api_key:
        return False, "请输入API密钥"
    
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_config.get('model_name', 'gpt-3.5-turbo'),
            base_url=model_config.get('base_url'),
            temperature=0.1
        )
        
        # 发送测试消息
        test_message = [HumanMessage(content="Hello, this is a test message. Please respond with 'Connection successful'.")]
        response = llm.invoke(test_message)
        
        if response and response.content:
            return True, "连接成功"
        else:
            return False, "连接失败：未收到有效响应"
            
    except Exception as e:
        return False, f"连接失败: {str(e)}"