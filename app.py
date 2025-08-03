import streamlit as st
import pandas as pd
import numpy as np
from modules import (
    setup_page_config,
    load_custom_css,
    ExcelAgentFull,
    FUNCTION_MODULES,
    FUNCTION_CATEGORIES,
    get_function_by_name
)

def main():
    """主应用程序入口"""
    # 设置页面配置
    setup_page_config()
    
    # 加载自定义CSS
    load_custom_css()
    
    # 应用标题
    st.title("Excel智能助手")
    st.markdown("---")
    
    # 初始化Excel代理
    if 'excel_agent' not in st.session_state:
        st.session_state.excel_agent = None
    
    # 侧边栏配置
    with st.sidebar:
        st.title("模型配置")
        
        # API配置
        with st.expander("API配置", expanded=False):
            api_key = st.text_input("API密钥", type="password", help="输入您的AI模型API密钥")
            
            model_options = [
                "OpenAI GPT-4", "OpenAI GPT-3.5-Turbo", "DeepSeek Chat", 
                "Moonshot v1", "通义千问", "智谱GLM", "百川大模型", 
                "文心一言", "讯飞星火", "自定义模型"
            ]
            selected_model = st.selectbox("选择模型", model_options)
            
            if st.button("保存配置"):
                if api_key:
                    try:
                        # 将选择的模型转换为配置字典
                        model_config = {
                            "model_name": "gpt-3.5-turbo",
                            "base_url": None
                        }
                        
                        # 根据选择的模型设置配置
                        if "GPT-4" in selected_model:
                            model_config["model_name"] = "gpt-4"
                        elif "GPT-3.5" in selected_model:
                            model_config["model_name"] = "gpt-3.5-turbo"
                        elif "DeepSeek" in selected_model:
                            model_config["model_name"] = "deepseek-chat"
                            model_config["base_url"] = "https://api.deepseek.com"
                        elif "Moonshot" in selected_model:
                            model_config["model_name"] = "moonshot-v1-8k"
                            model_config["base_url"] = "https://api.moonshot.cn/v1"
                        
                        st.session_state.excel_agent = ExcelAgentFull(api_key, model_config)
                        st.success("✅ 配置保存成功！")
                    except Exception as e:
                        st.error(f"❌ 模型初始化失败: {str(e)}")
                else:
                    st.warning("⚠️ 请输入API密钥")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "选择Excel文件", 
        type=['xlsx', 'xls', 'csv'],
        help="支持Excel和CSV格式文件"
    )
    
    if uploaded_file is not None:
        try:
            # 读取文件
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # 显示数据基本信息
            st.success(f"✅ 文件上传成功！数据形状: {df.shape}")
            
            # 侧边栏功能选择
            st.sidebar.title("功能选择")
            
            # 初始化选中的功能
            if 'selected_function' not in st.session_state:
                st.session_state.selected_function = None
            
            # 按分类显示功能
            for category, functions in FUNCTION_CATEGORIES.items():
                with st.sidebar.expander(f"📁 {category}", expanded=False):
                    for func_name in functions:
                        if st.button(func_name, key=f"btn_{func_name}"):
                            st.session_state.selected_function = func_name
                            st.rerun()  # 重新运行以更新界面
            
            # 添加清除选择按钮
            if st.session_state.selected_function is not None:
                if st.sidebar.button("返回数据预览", key="clear_selection"):
                    st.session_state.selected_function = None
                    st.rerun()
            
            # 如果没有选择功能，显示数据预览
            if st.session_state.selected_function is None:
                st.subheader("数据预览")
                st.dataframe(df.head(10), use_container_width=True)
                
                # 显示数据基本统计信息
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("行数", df.shape[0])
                    st.metric("列数", df.shape[1])
                with col2:
                    st.metric("数值列数", len(df.select_dtypes(include=[np.number]).columns))
                    st.metric("文本列数", len(df.select_dtypes(include=['object']).columns))
                
                # 显示列信息
                st.subheader("列信息")
                col_info = pd.DataFrame({
                    '列名': df.columns.tolist(),
                    '数据类型': [str(dtype) for dtype in df.dtypes],
                    '非空值数量': df.count().tolist(),
                    '缺失值数量': df.isnull().sum().tolist(),
                    '缺失值比例': (df.isnull().sum() / len(df) * 100).round(2).tolist()
                })
                st.dataframe(col_info, use_container_width=True)
                
                # 使用说明
                st.info("💡 请从左侧边栏选择要使用的功能模块")
            
            else:
                # 执行选中的功能
                # st.subheader(f"{st.session_state.selected_function}")
                function = get_function_by_name(st.session_state.selected_function)
                if function:
                    try:
                        # 检查函数是否需要agent参数
                        import inspect
                        sig = inspect.signature(function)
                        if 'agent' in sig.parameters:
                            function(df, st.session_state.excel_agent)
                        else:
                            function(df)
                    except Exception as e:
                        st.error(f"执行功能时出错: {str(e)}")
                        st.exception(e)
                else:
                    st.error(f"未找到功能: {st.session_state.selected_function}")
                    
        except Exception as e:
            st.error(f"文件读取失败: {str(e)}")
            st.exception(e)
    
    else:
        # 未上传文件时的欢迎界面
        st.markdown("""
        ## 功能特色
        
        ### 核心分析功能
        - **AI智能分析**: 基于人工智能的数据洞察
        - **自然语言查询**: 用自然语言查询数据
        - **图表生成**: 智能推荐和生成可视化图表
        
        ### 数据处理功能
        - **数据清洗**: 处理缺失值、重复值、异常值
        - **统计分析**: 描述性统计、相关性分析、假设检验
        - **高级数据处理**: 标准化、特征工程、聚类分析
        - **数据对比**: 多维度数据对比分析
        
        ### 业务应用功能
        - **表格操作**: 数据预览、列行操作、透视表
        - **公式计算器**: 数学、统计、财务、日期函数
        - **财务分析**: 盈利能力、偿债能力、运营效率分析
        - **时间序列分析**: 趋势分析、季节性分析、预测
        - **目标跟踪**: 销售、KPI、预算跟踪
        - **仪表板创建**: 销售、财务、运营仪表板
        
        ### 可视化和界面功能
        - **条件格式化**: 数据条、色阶、突出显示规则
        - **移动端适配**: 响应式表格、卡片视图
        - **工作表管理**: 工作表信息、拆分、合并
        
        ### 安全和隐私功能
        - **数据安全**: 数据脱敏、隐私保护
        - **数学统计函数**: 高级数学和统计计算
        
        ---
        
        ### 开始使用
        1. 上传您的Excel或CSV文件
        2. 从左侧边栏选择需要的功能
        3. 根据提示进行操作和分析
        
        ### 支持的文件格式
        - Excel文件 (.xlsx, .xls)
        - CSV文件 (.csv)
        """)
        
        # 示例数据
        st.markdown("### 或者使用示例数据")
        if st.button("生成示例销售数据"):
            # 生成示例数据
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            sample_data = pd.DataFrame({
                '日期': dates,
                '销售额': np.random.normal(10000, 2000, 100).round(2),
                '产品类别': np.random.choice(['电子产品', '服装', '食品', '家居'], 100),
                '销售人员': np.random.choice(['张三', '李四', '王五', '赵六'], 100),
                '客户满意度': np.random.uniform(3.0, 5.0, 100).round(1)
            })
            
            st.session_state['sample_data'] = sample_data
            st.rerun()  # 重新运行以显示数据分析界面
    
    # 检查是否有示例数据
    if 'sample_data' in st.session_state and st.session_state['sample_data'] is not None:
        df = st.session_state['sample_data']
        
        # 显示数据基本信息
        st.success(f"✅ 示例数据已加载！数据形状: {df.shape}")
        
        # 侧边栏功能选择
        st.sidebar.title("功能选择")
        
        # 初始化选中的功能
        if 'selected_function' not in st.session_state:
            st.session_state.selected_function = None
        
        # 按分类显示功能
        for category, functions in FUNCTION_CATEGORIES.items():
            with st.sidebar.expander(f"📁 {category}", expanded=False):
                for func_name in functions:
                    if st.button(func_name, key=f"sample_btn_{func_name}"):
                        st.session_state.selected_function = func_name
                        st.rerun()  # 重新运行以更新界面
        
        # 添加清除选择按钮
        if st.session_state.selected_function is not None:
            if st.sidebar.button("返回数据预览", key="sample_clear_selection"):
                st.session_state.selected_function = None
                st.rerun()
        
        # 添加清除示例数据按钮
        if st.sidebar.button("清除示例数据", key="clear_sample_data"):
            st.session_state['sample_data'] = None
            st.session_state.selected_function = None
            st.rerun()
        
        # 如果没有选择功能，显示数据预览
        if st.session_state.selected_function is None:
            st.subheader("数据预览")
            st.dataframe(df.head(10), use_container_width=True)
            
            # 显示数据基本统计信息
            col1, col2 = st.columns(2)
            with col1:
                st.metric("行数", df.shape[0])
                st.metric("列数", df.shape[1])
            with col2:
                st.metric("数值列数", len(df.select_dtypes(include=[np.number]).columns))
                st.metric("文本列数", len(df.select_dtypes(include=['object']).columns))
            
            # 显示列信息
            st.subheader("列信息")
            col_info = pd.DataFrame({
                '列名': df.columns.tolist(),
                '数据类型': [str(dtype) for dtype in df.dtypes],
                '非空值数量': df.count().tolist(),
                '缺失值数量': df.isnull().sum().tolist(),
                '缺失值比例': (df.isnull().sum() / len(df) * 100).round(2).tolist()
            })
            st.dataframe(col_info, use_container_width=True)
            
            # 使用说明
            st.info("💡 请从左侧边栏选择要使用的功能模块")
        
        else:
            # 执行选中的功能
            st.subheader(f" {st.session_state.selected_function}")
            function = get_function_by_name(st.session_state.selected_function)
            if function:
                try:
                    # 检查函数是否需要agent参数
                    import inspect
                    sig = inspect.signature(function)
                    if 'agent' in sig.parameters:
                        function(df, st.session_state.excel_agent)
                    else:
                        function(df)
                except Exception as e:
                    st.error(f"执行功能时出错: {str(e)}")
                    st.exception(e)
            else:
                st.error(f"未找到功能: {st.session_state.selected_function}")

if __name__ == "__main__":
    main()