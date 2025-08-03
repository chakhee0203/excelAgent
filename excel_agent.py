import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional
import json
import os
from datetime import datetime

class ExcelAgent:
    """Excel智能分析代理，基于LangChain 1.0.0"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """
        初始化Excel Agent
        
        Args:
            model_name: 使用的模型名称
            temperature: 模型温度参数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.df = None
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 初始化各种分析链
        self._setup_analysis_chains()
        
    def _setup_analysis_chains(self):
        """设置各种分析链"""
        
        # 数据分析链
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""你是一个专业的数据分析师。你需要分析用户提供的Excel数据，并给出专业的分析结果。
            
            分析时请注意：
            1. 提供清晰的数据洞察
            2. 识别数据中的模式和趋势
            3. 指出异常值或有趣的发现
            4. 给出可行的建议
            5. 使用中文回答
            
            数据信息将在用户消息中提供。"""),
            HumanMessage(content="{input}")
        ])
        
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=self.analysis_prompt,
            memory=self.memory
        )
        
        # 图表解读链
        self.chart_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""你是一个专业的数据可视化专家。你需要解读用户生成的图表，并提供专业的见解。
            
            解读时请注意：
            1. 描述图表显示的主要趋势
            2. 识别关键数据点
            3. 解释数据背后的含义
            4. 提供基于图表的建议
            5. 使用中文回答"""),
            HumanMessage(content="图表类型：{chart_type}\n用户需求：{input}")
        ])
        
        self.chart_chain = LLMChain(
            llm=self.llm,
            prompt=self.chart_prompt
        )
        
        # 数据清洗链
        self.cleaning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""你是一个数据清洗专家。你需要根据用户的需求，提供数据清洗的建议和方法。
            
            请注意：
            1. 分析数据质量问题
            2. 提供具体的清洗步骤
            3. 解释清洗的原因
            4. 建议最佳实践
            5. 使用中文回答"""),
            HumanMessage(content="{input}")
        ])
        
        self.cleaning_chain = LLMChain(
            llm=self.llm,
            prompt=self.cleaning_prompt
        )
        
        # 统计分析链
        self.stats_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""你是一个统计分析专家。你需要进行专业的统计分析，并解释结果的意义。
            
            分析时请注意：
            1. 选择合适的统计方法
            2. 解释统计结果的含义
            3. 评估结果的可靠性
            4. 提供实际应用建议
            5. 使用中文回答"""),
            HumanMessage(content="{input}")
        ])
        
        self.stats_chain = LLMChain(
            llm=self.llm,
            prompt=self.stats_prompt
        )
    
    def load_data(self, df: pd.DataFrame):
        """加载数据"""
        self.df = df.copy()
        
        # 创建pandas agent
        self.pandas_agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            allow_dangerous_code=True
        )
        
        return f"数据加载成功！数据形状：{self.df.shape}"
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        if self.df is None:
            return {"error": "未加载数据"}
        
        summary = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "null_counts": self.df.isnull().sum().to_dict(),
            "memory_usage": self.df.memory_usage(deep=True).sum(),
            "numeric_columns": list(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.df.select_dtypes(include=['object']).columns)
        }
        
        return summary
    
    def analyze_data(self, query: str) -> str:
        """分析数据"""
        if self.df is None:
            return "错误：未加载数据"
        
        try:
            # 获取数据基本信息
            data_info = self._get_data_context()
            
            # 构建分析输入
            analysis_input = f"""
            数据基本信息：
            {data_info}
            
            用户分析需求：{query}
            
            请基于以上数据信息进行专业分析。
            """
            
            # 执行分析
            result = self.analysis_chain.invoke({"input": analysis_input})
            
            return result
            
        except Exception as e:
            return f"分析失败：{str(e)}"
    
    def interpret_chart(self, user_request: str, chart_type: str) -> str:
        """解读图表"""
        try:
            result = self.chart_chain.invoke({
                "input": user_request,
                "chart_type": chart_type
            })
            return result
        except Exception as e:
            return f"图表解读失败：{str(e)}"
    
    def clean_data(self, cleaning_request: str) -> str:
        """数据清洗建议"""
        if self.df is None:
            return "错误：未加载数据"
        
        try:
            # 获取数据质量信息
            quality_info = self._get_data_quality_info()
            
            cleaning_input = f"""
            数据质量信息：
            {quality_info}
            
            用户清洗需求：{cleaning_request}
            
            请提供具体的数据清洗建议。
            """
            
            result = self.cleaning_chain.invoke({"input": cleaning_input})
            return result
            
        except Exception as e:
            return f"数据清洗建议生成失败：{str(e)}"
    
    def statistical_analysis(self, stats_request: str) -> str:
        """统计分析"""
        if self.df is None:
            return "错误：未加载数据"
        
        try:
            # 获取统计信息
            stats_info = self._get_statistical_context()
            
            stats_input = f"""
            数据统计信息：
            {stats_info}
            
            用户统计分析需求：{stats_request}
            
            请进行专业的统计分析。
            """
            
            result = self.stats_chain.invoke({"input": stats_input})
            return result
            
        except Exception as e:
            return f"统计分析失败：{str(e)}"
    
    def natural_language_query(self, query: str) -> Any:
        """自然语言查询"""
        if self.df is None:
            return "错误：未加载数据"
        
        try:
            # 使用pandas agent执行查询
            # 首先尝试使用run方法，如果失败则使用invoke方法
            try:
                result = self.pandas_agent.run(query)
                return result
            except Exception as run_error:
                # 如果run方法失败，尝试使用invoke方法
                if "not exactly one output key" in str(run_error):
                    response = self.pandas_agent.invoke({"input": query})
                    if isinstance(response, dict) and "output" in response:
                        return response["output"]
                    else:
                        return str(response)
                else:
                    raise run_error
            
        except Exception as e:
            return f"查询执行失败：{str(e)}"
    
    def generate_insights(self) -> List[str]:
        """生成数据洞察"""
        if self.df is None:
            return ["错误：未加载数据"]
        
        insights = []
        
        try:
            # 基本统计洞察
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if self.df[col].std() > 0:  # 避免除零错误
                    cv = self.df[col].std() / self.df[col].mean()
                    if cv > 1:
                        insights.append(f"列 '{col}' 的变异系数较高 ({cv:.2f})，数据分散度大")
                
                # 检查异常值
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = self.df[(self.df[col] < Q1 - 1.5 * IQR) | 
                                 (self.df[col] > Q3 + 1.5 * IQR)]
                
                if len(outliers) > 0:
                    outlier_pct = len(outliers) / len(self.df) * 100
                    insights.append(f"列 '{col}' 存在 {len(outliers)} 个异常值 ({outlier_pct:.1f}%)")
            
            # 缺失值洞察
            missing_cols = self.df.columns[self.df.isnull().any()]
            for col in missing_cols:
                missing_pct = self.df[col].isnull().sum() / len(self.df) * 100
                if missing_pct > 10:
                    insights.append(f"列 '{col}' 缺失值较多 ({missing_pct:.1f}%)，需要注意")
            
            # 相关性洞察
            if len(numeric_cols) >= 2:
                corr_matrix = self.df[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            high_corr_pairs.append(
                                (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                            )
                
                for col1, col2, corr in high_corr_pairs:
                    insights.append(f"'{col1}' 和 '{col2}' 高度相关 (r={corr:.3f})")
            
            return insights if insights else ["数据质量良好，未发现明显问题"]
            
        except Exception as e:
            return [f"洞察生成失败：{str(e)}"]
    
    def _get_data_context(self) -> str:
        """获取数据上下文信息"""
        if self.df is None:
            return "未加载数据"
        
        context = f"""
        数据形状：{self.df.shape[0]} 行，{self.df.shape[1]} 列
        
        列信息：
        {self.df.dtypes.to_string()}
        
        数值列统计：
        {self.df.describe().to_string()}
        
        缺失值统计：
        {self.df.isnull().sum().to_string()}
        
        前5行数据：
        {self.df.head().to_string()}
        """
        
        return context
    
    def _get_data_quality_info(self) -> str:
        """获取数据质量信息"""
        if self.df is None:
            return "未加载数据"
        
        quality_info = f"""
        数据形状：{self.df.shape}
        
        各列缺失值情况：
        {self.df.isnull().sum().to_string()}
        
        各列重复值情况：
        {self.df.duplicated().sum()} 行重复数据
        
        数据类型：
        {self.df.dtypes.to_string()}
        
        内存使用：{self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
        """
        
        return quality_info
    
    def _get_statistical_context(self) -> str:
        """获取统计上下文"""
        if self.df is None:
            return "未加载数据"
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        stats_context = f"""
        数据形状：{self.df.shape}
        
        数值列：{list(numeric_cols)}
        
        描述性统计：
        {self.df[numeric_cols].describe().to_string()}
        """
        
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()
            stats_context += f"""
            
        相关性矩阵：
        {corr_matrix.to_string()}
        """
        
        return stats_context
    
    def export_analysis_report(self, filename: str = None) -> str:
        """导出分析报告"""
        if self.df is None:
            return "错误：未加载数据"
        
        if filename is None:
            filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            report = f"""
# Excel数据分析报告
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据概览
{self._get_data_context()}

## 数据质量评估
{self._get_data_quality_info()}

## 自动洞察
{chr(10).join(self.generate_insights())}

## 统计信息
{self._get_statistical_context()}
            """
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            return f"分析报告已保存到：{filename}"
            
        except Exception as e:
            return f"报告导出失败：{str(e)}"
    
    def get_column_suggestions(self, operation: str) -> List[str]:
        """获取列建议"""
        if self.df is None:
            return []
        
        suggestions = []
        
        if operation == "numeric_analysis":
            suggestions = list(self.df.select_dtypes(include=[np.number]).columns)
        elif operation == "categorical_analysis":
            suggestions = list(self.df.select_dtypes(include=['object']).columns)
        elif operation == "datetime_analysis":
            suggestions = list(self.df.select_dtypes(include=['datetime64']).columns)
        else:
            suggestions = list(self.df.columns)
        
        return suggestions
    
    def validate_operation(self, operation: str, columns: List[str]) -> Dict[str, Any]:
        """验证操作的有效性"""
        if self.df is None:
            return {"valid": False, "message": "未加载数据"}
        
        # 检查列是否存在
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            return {
                "valid": False, 
                "message": f"列不存在：{missing_cols}"
            }
        
        # 根据操作类型验证
        if operation in ["correlation", "regression"]:
            numeric_cols = [col for col in columns 
                          if col in self.df.select_dtypes(include=[np.number]).columns]
            if len(numeric_cols) < 2:
                return {
                    "valid": False,
                    "message": "此操作需要至少2个数值列"
                }
        
        return {"valid": True, "message": "操作有效"}