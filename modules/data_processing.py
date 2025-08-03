import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

# 尝试导入Plotly相关模块
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

def data_analysis_section(df: pd.DataFrame):
    """数据分析功能（基础版）"""
    import numpy as np  # 确保在函数内部可以访问numpy
    st.markdown('<h2 class="sub-header">数据分析</h2>', unsafe_allow_html=True)
    
    # 基本统计信息
    st.markdown("### 基本统计信息")
    st.dataframe(df.describe(), use_container_width=True)
    
    # 数据类型信息
    st.markdown("### 数据类型信息")
    dtype_df = pd.DataFrame({
        '列名': df.columns,
        '数据类型': df.dtypes.values,
        '非空值数量': df.count().values,
        '空值数量': df.isnull().sum().values,
        '空值比例': (df.isnull().sum() / len(df) * 100).round(2).astype(str) + '%'
    })
    st.dataframe(dtype_df, use_container_width=True)
    
    # 数据质量报告
    st.markdown("### 数据质量报告")
    
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
    st.markdown('<h2 class="sub-header">🧹 数据清洗</h2>', unsafe_allow_html=True)
    
    cleaning_option = st.selectbox(
        "选择清洗操作",
        ["缺失值处理", "重复值处理", "异常值检测", "数据类型转换", "文本清洗"]
    )
    
    if cleaning_option == "缺失值处理":
        st.markdown("### 缺失值分析")
        
        # 显示缺失值统计
        missing_stats = df.isnull().sum()
        missing_percent = (missing_stats / len(df)) * 100
        
        missing_df = pd.DataFrame({
            '列名': missing_stats.index,
            '缺失数量': missing_stats.values,
            '缺失比例(%)': missing_percent.values
        })
        missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df)
            
            # 缺失值可视化
            if PLOTLY_AVAILABLE:
                fig = px.bar(missing_df, x='列名', y='缺失比例(%)', 
                            title='各列缺失值比例')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Plotly未安装，无法显示缺失值可视化图表")
                st.bar_chart(missing_df.set_index('列名')['缺失比例(%)'])
            
            # 处理选项
            st.markdown("### 缺失值处理")
            
            cols_with_missing = missing_df['列名'].tolist()
            selected_cols = st.multiselect("选择要处理的列", cols_with_missing)
            
            if selected_cols:
                method = st.selectbox(
                    "选择处理方法",
                    ["删除含缺失值的行", "用均值填充", "用中位数填充", "用众数填充", "前向填充", "后向填充", "自定义值填充"]
                )
                
                if method == "自定义值填充":
                    fill_value = st.text_input("输入填充值")
                
                if st.button("执行处理"):
                    cleaned_df = df.copy()
                    
                    for col in selected_cols:
                        if method == "删除含缺失值的行":
                            cleaned_df = cleaned_df.dropna(subset=[col])
                        elif method == "用均值填充" and pd.api.types.is_numeric_dtype(df[col]):
                            cleaned_df[col].fillna(df[col].mean(), inplace=True)
                        elif method == "用中位数填充" and pd.api.types.is_numeric_dtype(df[col]):
                            cleaned_df[col].fillna(df[col].median(), inplace=True)
                        elif method == "用众数填充":
                            mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                            cleaned_df[col].fillna(mode_val, inplace=True)
                        elif method == "前向填充":
                            cleaned_df[col].fillna(method='ffill', inplace=True)
                        elif method == "后向填充":
                            cleaned_df[col].fillna(method='bfill', inplace=True)
                        elif method == "自定义值填充":
                            cleaned_df[col].fillna(fill_value, inplace=True)
                    
                    st.success(f"✅ 处理完成！原数据 {len(df)} 行，处理后 {len(cleaned_df)} 行")
                    
                    # 显示处理结果
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 处理前")
                        st.write(f"缺失值总数: {df[selected_cols].isnull().sum().sum()}")
                    with col2:
                        st.markdown("#### 处理后")
                        st.write(f"缺失值总数: {cleaned_df[selected_cols].isnull().sum().sum()}")
                    
                    # 保存到session state
                    st.session_state.cleaned_df = cleaned_df
                    
                    # 下载清洗后的数据
                    csv = cleaned_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载清洗后的数据",
                        data=csv,
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )
        else:
            st.success("✅ 数据中没有缺失值")
    
    elif cleaning_option == "重复值处理":
        st.markdown("### 重复值分析")
        
        # 检测重复值
        duplicates = df.duplicated()
        duplicate_count = duplicates.sum()
        
        st.metric("重复行数", duplicate_count)
        st.metric("重复比例", f"{(duplicate_count/len(df)*100):.2f}%")
        
        if duplicate_count > 0:
            st.warning(f"⚠️ 发现 {duplicate_count} 行重复数据")
            
            # 显示重复数据
            if st.checkbox("显示重复数据"):
                duplicate_rows = df[df.duplicated(keep=False)].sort_values(df.columns.tolist())
                st.dataframe(duplicate_rows)
            
            # 处理选项
            st.markdown("### 重复值处理")
            
            keep_option = st.selectbox(
                "保留策略",
                ["保留第一个", "保留最后一个", "全部删除"]
            )
            
            subset_cols = st.multiselect(
                "基于特定列检测重复（留空则基于所有列）",
                df.columns.tolist()
            )
            
            if st.button("删除重复值"):
                if keep_option == "保留第一个":
                    keep = 'first'
                elif keep_option == "保留最后一个":
                    keep = 'last'
                else:
                    keep = False
                
                subset = subset_cols if subset_cols else None
                cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
                
                st.success(f"✅ 处理完成！原数据 {len(df)} 行，处理后 {len(cleaned_df)} 行")
                
                # 保存到session state
                st.session_state.cleaned_df = cleaned_df
                
                # 下载清洗后的数据
                csv = cleaned_df.to_csv(index=False)
                st.download_button(
                    label="下载去重后的数据",
                    data=csv,
                    file_name="deduplicated_data.csv",
                    mime="text/csv"
                )
        else:
            st.success("✅ 数据中没有重复值")
    
    elif cleaning_option == "异常值检测":
        st.markdown("### 异常值检测")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ 没有数值列可以进行异常值检测")
            return
        
        selected_col = st.selectbox("选择要检测的列", numeric_cols)
        
        method = st.selectbox(
            "选择检测方法",
            ["IQR方法", "Z-Score方法", "Isolation Forest", "箱线图可视化"]
        )
        
        if method == "IQR方法":
            Q1 = df[selected_col].quantile(0.25)
            Q3 = df[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
            
            st.write(f"检测到 {len(outliers)} 个异常值")
            st.write(f"正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            if len(outliers) > 0:
                st.dataframe(outliers)
        
        elif method == "Z-Score方法":
            threshold = st.slider("Z-Score阈值", 1.0, 4.0, 3.0, 0.1)
            
            z_scores = np.abs(stats.zscore(df[selected_col].dropna()))
            outliers = df[z_scores > threshold]
            
            st.write(f"检测到 {len(outliers)} 个异常值（Z-Score > {threshold}）")
            
            if len(outliers) > 0:
                st.dataframe(outliers)
        
        elif method == "Isolation Forest":
            contamination = st.slider("异常值比例", 0.01, 0.5, 0.1, 0.01)
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(df[[selected_col]].dropna())
            
            outliers = df[outlier_labels == -1]
            
            st.write(f"检测到 {len(outliers)} 个异常值")
            
            if len(outliers) > 0:
                st.dataframe(outliers)
        
        elif method == "箱线图可视化":
            if PLOTLY_AVAILABLE:
                fig = px.box(df, y=selected_col, title=f"{selected_col} 箱线图")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Plotly未安装，无法显示箱线图")
                # 使用matplotlib作为备选
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.boxplot(df[selected_col].dropna())
                ax.set_title(f"{selected_col} 箱线图")
                st.pyplot(fig)
            
            # 显示统计信息
            st.markdown("### 统计信息")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最小值", f"{df[selected_col].min():.2f}")
            with col2:
                st.metric("第一四分位数", f"{df[selected_col].quantile(0.25):.2f}")
            with col3:
                st.metric("中位数", f"{df[selected_col].median():.2f}")
            with col4:
                st.metric("第三四分位数", f"{df[selected_col].quantile(0.75):.2f}")

def statistical_analysis_section(df: pd.DataFrame):
    """统计分析功能"""
    st.markdown('<h2 class="sub-header">统计分析</h2>', unsafe_allow_html=True)
    
    analysis_type = st.selectbox(
        "选择分析类型",
        ["描述性统计", "相关性分析", "分布分析", "假设检验", "回归分析"]
    )
    
    if analysis_type == "描述性统计":
        st.markdown("### 描述性统计")
        
        # 数值列统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.markdown("#### 数值列统计")
            desc_stats = df[numeric_cols].describe()
            st.dataframe(desc_stats)
            
            # 添加更多统计指标
            additional_stats = pd.DataFrame({
                '偏度': df[numeric_cols].skew(),
                '峰度': df[numeric_cols].kurtosis(),
                '变异系数': df[numeric_cols].std() / df[numeric_cols].mean()
            })
            
            st.markdown("#### 额外统计指标")
            st.dataframe(additional_stats)
        
        # 分类列统计
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            st.markdown("#### 分类列统计")
            
            selected_cat_col = st.selectbox("选择分类列", categorical_cols)
            
            value_counts = df[selected_cat_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(value_counts)
            
            with col2:
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"{selected_cat_col} 分布")
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "相关性分析":
        st.markdown("### 相关性分析")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ 需要至少2个数值列进行相关性分析")
            return
        
        # 相关性矩阵
        corr_matrix = df[numeric_cols].corr()
        
        # 热力图
        fig = px.imshow(corr_matrix, 
                       title="相关性热力图",
                       color_continuous_scale="RdBu_r",
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # 强相关性对
        st.markdown("#### 强相关性分析")
        
        threshold = st.slider("相关性阈值", 0.5, 1.0, 0.7, 0.05)
        
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    strong_corr.append({
                        '变量1': corr_matrix.columns[i],
                        '变量2': corr_matrix.columns[j],
                        '相关系数': round(corr_val, 3),
                        '相关性强度': '强' if abs(corr_val) >= 0.8 else '中等',
                        '相关性方向': '正相关' if corr_val > 0 else '负相关'
                    })
        
        if strong_corr:
            st.dataframe(pd.DataFrame(strong_corr))
        else:
            st.info(f"未发现相关系数绝对值 >= {threshold} 的变量对")
    
    elif analysis_type == "分布分析":
        st.markdown("### 分布分析")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ 没有数值列可以进行分布分析")
            return
        
        selected_col = st.selectbox("选择要分析的列", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 直方图
            fig_hist = px.histogram(df, x=selected_col, 
                                  title=f"{selected_col} 分布直方图",
                                  marginal="box")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Q-Q图
            fig_qq = go.Figure()
            
            # 计算理论分位数和样本分位数
            sorted_data = np.sort(df[selected_col].dropna())
            n = len(sorted_data)
            theoretical_quantiles = stats.norm.ppf(np.arange(1, n+1) / (n+1))
            
            fig_qq.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sorted_data,
                mode='markers',
                name='数据点'
            ))
            
            # 添加理论直线
            fig_qq.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=np.mean(sorted_data) + np.std(sorted_data) * theoretical_quantiles,
                mode='lines',
                name='理论直线',
                line=dict(color='red')
            ))
            
            fig_qq.update_layout(title=f"{selected_col} Q-Q图",
                               xaxis_title="理论分位数",
                               yaxis_title="样本分位数")
            
            st.plotly_chart(fig_qq, use_container_width=True)
        
        # 正态性检验
        st.markdown("#### 正态性检验")
        
        # Shapiro-Wilk检验（适用于小样本）
        if len(df[selected_col].dropna()) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(df[selected_col].dropna())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Shapiro-Wilk统计量", f"{shapiro_stat:.4f}")
            with col2:
                st.metric("p值", f"{shapiro_p:.4f}")
            
            if shapiro_p > 0.05:
                st.success("✅ 数据可能服从正态分布（p > 0.05）")
            else:
                st.warning("⚠️ 数据可能不服从正态分布（p <= 0.05）")
        
        # Kolmogorov-Smirnov检验
        ks_stat, ks_p = stats.kstest(df[selected_col].dropna(), 'norm')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("K-S统计量", f"{ks_stat:.4f}")
        with col2:
            st.metric("p值", f"{ks_p:.4f}")

def advanced_data_processing_section(df: pd.DataFrame):
    """高级数据处理功能"""
    st.markdown('<h2 class="sub-header">高级数据处理</h2>', unsafe_allow_html=True)
    
    processing_type = st.selectbox(
        "选择处理类型",
        ["数据标准化", "特征工程", "聚类分析", "主成分分析", "数据采样"]
    )
    
    if processing_type == "数据标准化":
        st.markdown("### 数据标准化")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ 没有数值列可以进行标准化")
            return
        
        selected_cols = st.multiselect("选择要标准化的列", numeric_cols, default=numeric_cols)
        
        if selected_cols:
            method = st.selectbox(
                "选择标准化方法",
                ["Z-Score标准化", "Min-Max标准化", "Robust标准化"]
            )
            
            if st.button("执行标准化"):
                scaled_df = df.copy()
                
                if method == "Z-Score标准化":
                    scaler = StandardScaler()
                    scaled_df[selected_cols] = scaler.fit_transform(df[selected_cols])
                    st.info("使用Z-Score标准化：(x - mean) / std")
                
                elif method == "Min-Max标准化":
                    scaler = MinMaxScaler()
                    scaled_df[selected_cols] = scaler.fit_transform(df[selected_cols])
                    st.info("使用Min-Max标准化：(x - min) / (max - min)")
                
                elif method == "Robust标准化":
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    scaled_df[selected_cols] = scaler.fit_transform(df[selected_cols])
                    st.info("使用Robust标准化：(x - median) / IQR")
                
                # 显示标准化前后对比
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 标准化前")
                    st.dataframe(df[selected_cols].describe())
                
                with col2:
                    st.markdown("#### 标准化后")
                    st.dataframe(scaled_df[selected_cols].describe())
                
                # 保存结果
                st.session_state.scaled_df = scaled_df
                
                # 下载标准化后的数据
                csv = scaled_df.to_csv(index=False)
                st.download_button(
                    label="下载标准化数据",
                    data=csv,
                    file_name="scaled_data.csv",
                    mime="text/csv"
                )
    
    elif processing_type == "聚类分析":
        st.markdown("### 聚类分析")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ 需要至少2个数值列进行聚类分析")
            return
        
        selected_cols = st.multiselect(
            "选择用于聚类的列",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        if len(selected_cols) >= 2:
            n_clusters = st.slider("聚类数量", 2, 10, 3)
            
            if st.button("执行K-Means聚类"):
                # 数据预处理
                data_for_clustering = df[selected_cols].dropna()
                
                # 标准化
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_for_clustering)
                
                # K-Means聚类
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_data)
                
                # 添加聚类标签到原数据
                clustered_df = data_for_clustering.copy()
                clustered_df['聚类标签'] = cluster_labels
                
                # 显示聚类结果
                st.markdown("#### 聚类结果")
                
                # 聚类统计
                cluster_stats = clustered_df.groupby('聚类标签').agg({
                    col: ['mean', 'count'] for col in selected_cols
                }).round(2)
                
                st.dataframe(cluster_stats)
                
                # 可视化（如果有2-3个特征）
                if len(selected_cols) == 2:
                    fig = px.scatter(
                        clustered_df,
                        x=selected_cols[0],
                        y=selected_cols[1],
                        color='聚类标签',
                        title="聚类结果可视化"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif len(selected_cols) >= 3:
                    fig = px.scatter_3d(
                        clustered_df,
                        x=selected_cols[0],
                        y=selected_cols[1],
                        z=selected_cols[2],
                        color='聚类标签',
                        title="聚类结果3D可视化"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 聚类评估
                from sklearn.metrics import silhouette_score
                silhouette_avg = silhouette_score(scaled_data, cluster_labels)
                st.metric("轮廓系数", f"{silhouette_avg:.3f}")
                
                if silhouette_avg > 0.5:
                    st.success("✅ 聚类效果良好")
                elif silhouette_avg > 0.25:
                    st.warning("⚠️ 聚类效果一般")
                else:
                    st.error("❌ 聚类效果较差")
    
    elif processing_type == "主成分分析":
        st.markdown("### 🔍 主成分分析 (PCA)")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ 需要至少2个数值列进行PCA分析")
            return
        
        selected_cols = st.multiselect(
            "选择用于PCA的列",
            numeric_cols,
            default=numeric_cols
        )
        
        if len(selected_cols) >= 2:
            n_components = st.slider(
                "主成分数量",
                1,
                min(len(selected_cols), len(df)),
                min(3, len(selected_cols))
            )
            
            if st.button("执行PCA分析"):
                # 数据预处理
                data_for_pca = df[selected_cols].dropna()
                
                # 标准化
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_for_pca)
                
                # PCA
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(scaled_data)
                
                # 创建PCA结果DataFrame
                pca_df = pd.DataFrame(
                    pca_result,
                    columns=[f'PC{i+1}' for i in range(n_components)]
                )
                
                # 显示解释方差比
                st.markdown("#### 主成分解释方差")
                
                variance_df = pd.DataFrame({
                    '主成分': [f'PC{i+1}' for i in range(n_components)],
                    '解释方差比': pca.explained_variance_ratio_,
                    '累积解释方差比': np.cumsum(pca.explained_variance_ratio_)
                })
                
                st.dataframe(variance_df)
                
                # 可视化解释方差
                fig = px.bar(
                    variance_df,
                    x='主成分',
                    y='解释方差比',
                    title="各主成分解释方差比"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 如果有至少2个主成分，显示散点图
                if n_components >= 2:
                    fig_scatter = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        title="前两个主成分散点图"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # 显示主成分载荷
                st.markdown("#### 主成分载荷")
                
                loadings = pd.DataFrame(
                    pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(n_components)],
                    index=selected_cols
                )
                
                st.dataframe(loadings)

def data_comparison_section(df: pd.DataFrame):
    """数据对比功能"""
    st.markdown('<h2 class="sub-header">数据对比</h2>', unsafe_allow_html=True)
    
    comparison_type = st.selectbox(
        "选择对比类型",
        ["列间对比", "分组对比", "时间序列对比", "统计对比"]
    )
    
    if comparison_type == "列间对比":
        st.markdown("### 列间数据对比")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ 需要至少2个数值列进行对比")
            return
        
        col1_name = st.selectbox("选择第一列", numeric_cols)
        col2_name = st.selectbox("选择第二列", [col for col in numeric_cols if col != col1_name])
        
        if st.button("开始对比"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {col1_name} 统计")
                stats1 = df[col1_name].describe()
                st.dataframe(stats1)
            
            with col2:
                st.markdown(f"#### {col2_name} 统计")
                stats2 = df[col2_name].describe()
                st.dataframe(stats2)
            
            # 对比可视化
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=df[col1_name],
                name=col1_name,
                boxpoints='outliers'
            ))
            
            fig.add_trace(go.Box(
                y=df[col2_name],
                name=col2_name,
                boxpoints='outliers'
            ))
            
            fig.update_layout(title="列间数据分布对比")
            st.plotly_chart(fig, use_container_width=True)
            
            # 相关性分析
            correlation = df[col1_name].corr(df[col2_name])
            st.metric("相关系数", f"{correlation:.3f}")
            
            # 散点图
            fig_scatter = px.scatter(
                df,
                x=col1_name,
                y=col2_name,
                title=f"{col1_name} vs {col2_name}",
                trendline="ols"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    elif comparison_type == "分组对比":
        st.markdown("### 👥 分组数据对比")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not categorical_cols or not numeric_cols:
            st.warning("⚠️ 需要至少1个分类列和1个数值列进行分组对比")
            return
        
        group_col = st.selectbox("选择分组列", categorical_cols)
        value_col = st.selectbox("选择数值列", numeric_cols)
        
        if st.button("开始分组对比"):
            # 分组统计
            group_stats = df.groupby(group_col)[value_col].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
            
            st.markdown("#### 分组统计")
            st.dataframe(group_stats)
            
            # 箱线图对比
            fig_box = px.box(
                df,
                x=group_col,
                y=value_col,
                title=f"{value_col} 按 {group_col} 分组对比"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # 小提琴图
            fig_violin = px.violin(
                df,
                x=group_col,
                y=value_col,
                title=f"{value_col} 分布对比（小提琴图）"
            )
            st.plotly_chart(fig_violin, use_container_width=True)
            
            # 方差分析（如果分组数量合适）
            groups = [group[value_col].dropna() for name, group in df.groupby(group_col)]
            
            if len(groups) >= 2:
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("F统计量", f"{f_stat:.3f}")
                    with col2:
                        st.metric("p值", f"{p_value:.3f}")
                    
                    if p_value < 0.05:
                        st.success("✅ 各组间存在显著差异（p < 0.05）")
                    else:
                        st.info("ℹ️ 各组间无显著差异（p >= 0.05）")
                        
                except Exception as e:
                    st.warning(f"方差分析失败: {str(e)}")


def data_import_export_section(df: pd.DataFrame):
    """数据导入导出功能"""
    st.markdown('<h2 class="sub-header">数据导入导出</h2>', unsafe_allow_html=True)
    
    operation_type = st.selectbox(
        "选择操作类型",
        ["数据导出", "格式转换", "数据合并", "数据拆分", "批量处理"]
    )
    
    if operation_type == "数据导出":
        st.markdown("### 数据导出")
        
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
        st.markdown("#### 导出预览")
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
            label=f"下载 {export_format} 文件",
            data=data,
            file_name=filename,
            mime=mime_type
        )
    
    elif operation_type == "格式转换":
        st.markdown("### 格式转换")
        
        st.info("💡 支持在不同数据格式之间转换")
        
        # 数据类型转换
        st.markdown("#### 数据类型转换")
        
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
                
                if st.button("执行转换"):
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
                        label="下载转换后的数据",
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
                st.markdown("#### 数据预览")
                st.dataframe(sample_data)
                
                if st.button("执行转换"):
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
                            label="下载转换后的数据",
                            data=csv,
                            file_name="converted_data.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ 转换失败: {str(e)}")


def data_validation_section(df: pd.DataFrame):
    """数据验证功能"""
    st.markdown('<h2 class="sub-header">数据验证</h2>', unsafe_allow_html=True)
    
    validation_type = st.selectbox(
        "选择验证类型",
        ["数据完整性检查", "数据格式验证", "业务规则验证", "重复值检测", "异常值检测"]
    )
    
    if validation_type == "数据完整性检查":
        st.markdown("### 数据完整性检查")
        
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
        st.markdown("#### 数据类型分析")
        
        dtype_summary = pd.DataFrame({
            '列名': df.columns,
            '数据类型': [str(dtype) for dtype in df.dtypes],
            '唯一值数量': [df[col].nunique() for col in df.columns],
            '样本值': [str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else 'N/A' for col in df.columns]
        })
        
        st.dataframe(dtype_summary, use_container_width=True)
    
    elif validation_type == "重复值检测":
        st.markdown("### 重复值检测")
        
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
            st.markdown("#### 重复行预览")
            duplicate_data = df[duplicate_rows]
            st.dataframe(duplicate_data.head(10))
            
            # 处理选项
            if st.button("删除重复行"):
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
                    label="下载去重后的数据",
                    data=csv,
                    file_name="deduplicated_data.csv",
                    mime="text/csv"
                )
        else:
            st.success("✅ 未发现完全重复的行")
        
        # 按特定列检测重复
        st.markdown("#### 按列检测重复")
        
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