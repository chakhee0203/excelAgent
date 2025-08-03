import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
from .excel_agent import ExcelAgentFull

# 尝试导入Plotly相关模块
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

def ai_analysis_section(df: pd.DataFrame, agent: Optional[ExcelAgentFull]):
    """AI智能分析功能"""
    st.markdown('<h2 class="sub-header">🤖 AI智能分析</h2>', unsafe_allow_html=True)
    
    if not agent or not agent.llm:
        st.warning("⚠️ AI功能需要配置API密钥才能使用")
        st.info("💡 请在侧边栏配置您的API密钥和模型设置")
        return
    
    st.info("💡 AI将自动分析您的数据并提供专业洞察")
    
    # 分析类型选择
    analysis_type = st.selectbox(
        "选择分析类型",
        ["comprehensive", "quality", "business"],
        format_func=lambda x: {
            "comprehensive": "全面分析",
            "quality": "数据质量分析", 
            "business": "商业分析"
        }[x]
    )
    
    if st.button("开始AI分析", type="primary"):
        with st.spinner("AI正在分析您的数据..."):
            try:
                analysis_result = agent.analyze_data(df, analysis_type)
                
                st.markdown('<div class="ai-response">', unsafe_allow_html=True)
                st.markdown("### AI分析结果")
                st.markdown(analysis_result)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 保存分析结果
                if 'analysis_history' not in st.session_state:
                    st.session_state.analysis_history = []
                
                st.session_state.analysis_history.append({
                    'timestamp': pd.Timestamp.now(),
                    'type': analysis_type,
                    'result': analysis_result
                })
                
                st.success("✅ 分析完成！")
                
            except Exception as e:
                st.error(f"❌ 分析失败: {str(e)}")
    
    # 显示历史分析记录
    if 'analysis_history' in st.session_state and st.session_state.analysis_history:
        with st.expander("历史分析记录"):
            for i, record in enumerate(reversed(st.session_state.analysis_history[-5:])):
                st.markdown(f"**{record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {record['type']}**")
                st.text(record['result'][:200] + "..." if len(record['result']) > 200 else record['result'])
                st.divider()

def natural_language_section(df: pd.DataFrame, agent: Optional[ExcelAgentFull]):
    """自然语言查询功能"""
    st.markdown('<h2 class="sub-header">自然语言查询</h2>', unsafe_allow_html=True)
    
    if not agent or not agent.llm:
        st.warning("⚠️ 自然语言查询需要配置API密钥才能使用")
        st.info("💡 请在侧边栏配置您的API密钥和模型设置")
        return
    
    st.info("💡 用自然语言描述您想了解的数据问题，AI将为您分析并回答")
    
    # 预设查询示例
    example_queries = [
        "数据中有多少行和多少列？",
        "哪一列的缺失值最多？",
        "数值列的平均值是多少？",
        "找出最大值和最小值",
        "数据中有哪些异常值？",
        "各列之间的相关性如何？"
    ]
    
    st.markdown("### 💡 查询示例")
    cols = st.columns(3)
    for i, example in enumerate(example_queries):
        with cols[i % 3]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.query_input = example
    
    # 查询输入
    query = st.text_area(
        "请输入您的问题",
        value=st.session_state.get('query_input', ''),
        height=100,
        placeholder="例如：这个数据集中销售额最高的是哪个月份？"
    )
    
    if st.button("🔍 查询", type="primary") and query:
        with st.spinner("AI正在分析您的问题..."):
            try:
                response = agent.natural_language_query(df, query)
                
                st.markdown('<div class="ai-response">', unsafe_allow_html=True)
                st.markdown("### 查询结果")
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 保存查询历史
                if 'query_history' not in st.session_state:
                    st.session_state.query_history = []
                
                st.session_state.query_history.append({
                    'timestamp': pd.Timestamp.now(),
                    'query': query,
                    'response': response
                })
                
            except Exception as e:
                st.error(f"❌ 查询失败: {str(e)}")
    
    # 显示查询历史
    if 'query_history' in st.session_state and st.session_state.query_history:
        with st.expander("查询历史"):
            for i, record in enumerate(reversed(st.session_state.query_history[-5:])):
                st.markdown(f"**Q: {record['query']}**")
                st.markdown(f"A: {record['response'][:200]}..." if len(record['response']) > 200 else f"A: {record['response']}")
                st.caption(f"时间: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.divider()

def chart_generation_section(df: pd.DataFrame, agent: Optional[ExcelAgentFull]):
    """图表生成功能"""
    st.markdown('<h2 class="sub-header">智能图表生成</h2>', unsafe_allow_html=True)
    
    # 获取数值列和分类列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if not numeric_cols and not categorical_cols:
        st.warning("⚠️ 数据中没有可用于生成图表的列")
        return
    
    # 图表类型选择
    chart_type = st.selectbox(
        "选择图表类型",
        ["柱状图", "折线图", "散点图", "饼图", "直方图", "箱线图", "热力图", "相关性矩阵"]
    )
    
    if chart_type == "自动推荐":
        st.markdown("### AI图表推荐")
        
        if agent and agent.llm:
            if st.button("获取AI推荐"):
                with st.spinner("🤖 AI正在分析最适合的图表类型..."):
                    suggestions = agent.generate_chart_suggestions(df)
                    
                    if suggestions:
                        for i, suggestion in enumerate(suggestions[:3]):
                            with st.expander(f"推荐 {i+1}: {suggestion['title']}"):
                                st.write(suggestion['description'])
                                if st.button(f"生成图表 {i+1}", key=f"gen_{i}"):
                                    # 这里可以根据建议生成对应的图表
                                    st.info("图表生成功能开发中...")
                    else:
                        st.info("暂无图表推荐")
        else:
            # 基础推荐逻辑
            st.info("💡 基于数据特征的基础推荐：")
            
            recommendations = []
            
            if len(numeric_cols) >= 2:
                recommendations.append("散点图 - 适合分析两个数值变量的关系")
                recommendations.append("热力图 - 适合显示变量间的相关性")
            
            if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                recommendations.append("柱状图 - 适合比较不同类别的数值")
                recommendations.append("箱线图 - 适合显示数值分布")
            
            if len(categorical_cols) >= 1:
                recommendations.append("饼图 - 适合显示类别占比")
            
            for rec in recommendations:
                st.write(f"• {rec}")
    
    else:
        # 手动图表生成
        st.markdown(f"### {chart_type}生成")
        
        if chart_type == "柱状图":
            if categorical_cols and numeric_cols:
                x_col = st.selectbox("选择X轴（分类）", categorical_cols)
                y_col = st.selectbox("选择Y轴（数值）", numeric_cols)
                
                if st.button("生成柱状图"):
                    if not PLOTLY_AVAILABLE:
                        st.error("❌ Plotly未安装，无法生成交互式图表")
                        st.info("💡 请运行以下命令安装Plotly: pip install plotly")
                    else:
                        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "折线图":
            if numeric_cols:
                if datetime_cols:
                    x_col = st.selectbox("选择X轴", datetime_cols + numeric_cols)
                else:
                    x_col = st.selectbox("选择X轴", numeric_cols)
                y_col = st.selectbox("选择Y轴", numeric_cols)
                
                if st.button("生成折线图"):
                    if not PLOTLY_AVAILABLE:
                        st.error("❌ Plotly未安装，无法生成交互式图表")
                        st.info("💡 请运行以下命令安装Plotly: pip install plotly")
                    else:
                        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "散点图":
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("选择X轴", numeric_cols)
                y_col = st.selectbox("选择Y轴", [col for col in numeric_cols if col != x_col])
                
                color_col = None
                if categorical_cols:
                    use_color = st.checkbox("使用颜色分组")
                    if use_color:
                        color_col = st.selectbox("选择颜色分组列", categorical_cols)
                
                if st.button("生成散点图"):
                    if not PLOTLY_AVAILABLE:
                        st.error("❌ Plotly未安装，无法生成交互式图表")
                        st.info("💡 请运行以下命令安装Plotly: pip install plotly")
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                                       title=f"{y_col} vs {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "饼图":
            if categorical_cols:
                cat_col = st.selectbox("选择分类列", categorical_cols)
                
                if st.button("生成饼图"):
                    if not PLOTLY_AVAILABLE:
                        st.error("❌ Plotly未安装，无法生成交互式图表")
                        st.info("💡 请运行以下命令安装Plotly: pip install plotly")
                    else:
                        value_counts = df[cat_col].value_counts()
                        fig = px.pie(values=value_counts.values, names=value_counts.index,
                                   title=f"{cat_col} 分布")
                        st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "直方图":
            if numeric_cols:
                col = st.selectbox("选择数值列", numeric_cols)
                bins = st.slider("直方图分组数", 10, 100, 30)
                
                if st.button("生成直方图"):
                    if not PLOTLY_AVAILABLE:
                        st.error("❌ Plotly未安装，无法生成交互式图表")
                        st.info("💡 请运行以下命令安装Plotly: pip install plotly")
                    else:
                        fig = px.histogram(df, x=col, nbins=bins, title=f"{col} 分布")
                        st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "箱线图":
            if numeric_cols:
                y_col = st.selectbox("选择数值列", numeric_cols)
                
                x_col = None
                if categorical_cols:
                    use_group = st.checkbox("按类别分组")
                    if use_group:
                        x_col = st.selectbox("选择分组列", categorical_cols)
                
                if st.button("生成箱线图"):
                    if not PLOTLY_AVAILABLE:
                        st.error("❌ Plotly未安装，无法生成交互式图表")
                        st.info("💡 请运行以下命令安装Plotly: pip install plotly")
                    else:
                        fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} 箱线图")
                        st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "热力图":
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect(
                    "选择要显示的数值列",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if selected_cols and st.button("生成热力图"):
                    if not PLOTLY_AVAILABLE:
                        st.error("❌ Plotly未安装，无法生成交互式图表")
                        st.info("💡 请运行以下命令安装Plotly: pip install plotly")
                    else:
                        corr_matrix = df[selected_cols].corr()
                        fig = px.imshow(corr_matrix, 
                                      title="相关性热力图",
                                      color_continuous_scale="RdBu_r")
                        st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "相关性矩阵":
            if len(numeric_cols) >= 2:
                if st.button("生成相关性矩阵"):
                    if not PLOTLY_AVAILABLE:
                        st.error("❌ Plotly未安装，无法生成交互式图表")
                        st.info("💡 请运行以下命令安装Plotly: pip install plotly")
                    else:
                        corr_matrix = df[numeric_cols].corr()
                        
                        # 使用plotly创建交互式相关性矩阵
                        fig = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale='RdBu_r',
                            zmid=0
                        ))
                        
                        fig.update_layout(
                            title="数值列相关性矩阵",
                            xaxis_title="变量",
                            yaxis_title="变量"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示强相关性对
                    st.markdown("### 强相关性分析")
                    
                    strong_corr = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.7:  # 强相关性阈值
                                strong_corr.append({
                                    '变量1': corr_matrix.columns[i],
                                    '变量2': corr_matrix.columns[j],
                                    '相关系数': round(corr_val, 3),
                                    '相关性': '强正相关' if corr_val > 0.7 else '强负相关'
                                })
                    
                    if strong_corr:
                        st.dataframe(pd.DataFrame(strong_corr))
                    else:
                        st.info("未发现强相关性（|r| > 0.7）的变量对")

def machine_learning_section(df: pd.DataFrame, agent: Optional[ExcelAgentFull]):
    """机器学习预测功能"""
    st.markdown('<h2 class="sub-header">机器学习预测</h2>', unsafe_allow_html=True)
    
    # 检查数据是否适合机器学习
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ 数据中数值列不足，无法进行机器学习预测。至少需要2个数值列。")
        return
    
    st.markdown("### 预测模型配置")
    
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
    
    if st.button("开始训练模型", type="primary"):
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
            st.markdown("### 模型性能评估")
            
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
            if PLOTLY_AVAILABLE:
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
            else:
                st.warning("⚠️ Plotly未安装，无法显示预测vs实际值图表")
            
            # 特征重要性（仅对树模型）
            if model_type in ["随机森林", "梯度提升"]:
                st.markdown("### 特征重要性")
                
                importance_df = pd.DataFrame({
                    '特征': feature_cols,
                    '重要性': model.feature_importances_
                }).sort_values('重要性', ascending=False)
                
                if PLOTLY_AVAILABLE:
                    fig_importance = px.bar(
                        importance_df,
                        x='重要性',
                        y='特征',
                        orientation='h',
                        title="特征重要性排序"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.warning("⚠️ Plotly未安装，无法显示特征重要性图表")
                    st.dataframe(importance_df)
            
            # AI解释
            if agent:
                st.markdown("### AI模型解释")
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
        st.markdown("### 使用模型进行预测")
        
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
        
        if st.button("进行预测"):
            try:
                # 准备预测数据
                pred_data = pd.DataFrame([prediction_inputs])
                
                # 应用相同的预处理
                if model_info['scaler']:
                    pred_data_scaled = model_info['scaler'].transform(pred_data)
                    prediction = model_info['model'].predict(pred_data_scaled)[0]
                else:
                    prediction = model_info['model'].predict(pred_data)[0]
                
                st.success(f"预测结果：{model_info['target_col']} = {prediction:.4f}")
                
                # 显示置信区间（简单估计）
                rmse = model_info['performance']['rmse']
                st.info(f"预测区间（±1个RMSE）：{prediction-rmse:.4f} ~ {prediction+rmse:.4f}")
                
            except Exception as e:
                st.error(f"❌ 预测失败: {str(e)}")