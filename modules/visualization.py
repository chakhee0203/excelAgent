import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


def conditional_formatting_section(df: pd.DataFrame):
    """条件格式化功能"""
    st.markdown('<h2 class="sub-header">条件格式化</h2>', unsafe_allow_html=True)
    
    st.info("💡 通过颜色和样式突出显示符合特定条件的数据")
    
    formatting_type = st.selectbox(
        "选择格式化类型",
        ["数值条件格式", "文本条件格式", "热力图着色", "数据条", "图标集"]
    )
    
    if formatting_type == "数值条件格式":
        st.markdown("### 数值条件格式")
        
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
            
            if st.button("应用格式化"):
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
                
                st.markdown("#### 格式化结果")
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
        st.markdown("### 热力图着色")
        
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
                
                if st.button("生成热力图"):
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
                    st.markdown("#### 数值分布热力图")
                    
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
    st.markdown('<h2 class="sub-header">工作表管理</h2>', unsafe_allow_html=True)
    
    management_type = st.selectbox(
        "选择管理功能",
        ["工作表信息", "数据分割", "数据合并", "工作表比较", "批量操作"]
    )
    
    if management_type == "工作表信息":
        st.markdown("### ℹ工作表信息")
        
        # 基本信息
        st.markdown("#### 基本统计")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总行数", len(df))
        with col2:
            st.metric("总列数", len(df.columns))
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("内存使用", f"{memory_usage:.2f} MB")
        with col4:
            null_count = df.isnull().sum().sum()
            st.metric("缺失值总数", null_count)
        
        # 列信息详情
        st.markdown("#### 列信息详情")
        
        column_info = []
        for col in df.columns:
            col_data = df[col]
            info = {
                "列名": col,
                "数据类型": str(col_data.dtype),
                "非空值数量": col_data.count(),
                "缺失值数量": col_data.isnull().sum(),
                "缺失率": f"{(col_data.isnull().sum() / len(df)) * 100:.1f}%",
                "唯一值数量": col_data.nunique()
            }
            
            if col_data.dtype in ['int64', 'float64']:
                info.update({
                    "最小值": col_data.min(),
                    "最大值": col_data.max(),
                    "平均值": f"{col_data.mean():.2f}"
                })
            
            column_info.append(info)
        
        info_df = pd.DataFrame(column_info)
        st.dataframe(info_df, use_container_width=True)
        
        # 数据质量评估
        st.markdown("#### 数据质量评估")
        
        quality_score = 100
        issues = []
        
        # 检查缺失值
        missing_rate = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_rate > 10:
            quality_score -= 20
            issues.append(f"缺失值比例较高 ({missing_rate:.1f}%)")
        elif missing_rate > 5:
            quality_score -= 10
            issues.append(f"存在一定缺失值 ({missing_rate:.1f}%)")
        
        # 检查重复行
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_rate = (duplicate_count / len(df)) * 100
            if duplicate_rate > 5:
                quality_score -= 15
                issues.append(f"重复行较多 ({duplicate_count} 行, {duplicate_rate:.1f}%)")
            else:
                quality_score -= 5
                issues.append(f"存在少量重复行 ({duplicate_count} 行)")
        
        # 显示质量评分
        col1, col2 = st.columns(2)
        
        with col1:
            if quality_score >= 90:
                st.success(f"✅ 数据质量评分: {quality_score}/100 (优秀)")
            elif quality_score >= 70:
                st.warning(f"⚠️ 数据质量评分: {quality_score}/100 (良好)")
            else:
                st.error(f"❌ 数据质量评分: {quality_score}/100 (需要改进)")
        
        with col2:
            if issues:
                st.markdown("**发现的问题:**")
                for issue in issues:
                    st.write(f"• {issue}")
            else:
                st.success("✅ 未发现明显的数据质量问题")
    
    elif management_type == "数据分割":
        st.markdown("### 数据分割")
        
        split_method = st.selectbox(
            "选择分割方法",
            ["按行数分割", "按列分割", "按条件分割", "随机分割"]
        )
        
        if split_method == "按行数分割":
            rows_per_split = st.number_input(
                "每个分割的行数",
                value=min(1000, len(df) // 2),
                min_value=1,
                max_value=len(df)
            )
            
            if st.button("执行分割"):
                splits = []
                for i in range(0, len(df), rows_per_split):
                    split_df = df.iloc[i:i+rows_per_split]
                    splits.append(split_df)
                
                st.success(f"✅ 数据已分割为 {len(splits)} 个部分")
                
                # 显示分割结果
                for i, split_df in enumerate(splits):
                    with st.expander(f"分割 {i+1} ({len(split_df)} 行)"):
                        st.dataframe(split_df.head())
                        
                        # 下载按钮
                        csv = split_df.to_csv(index=False)
                        st.download_button(
                            label=f"下载分割 {i+1}",
                            data=csv,
                            file_name=f"split_{i+1}.csv",
                            mime="text/csv"
                        )
        
        elif split_method == "按列分割":
            # 选择列进行分组
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not categorical_cols:
                st.warning("⚠️ 没有分类列可用于分割")
                return
            
            split_col = st.selectbox("选择分割列", categorical_cols)
            
            if st.button("执行分割"):
                unique_values = df[split_col].unique()
                
                st.success(f"✅ 按 '{split_col}' 分割为 {len(unique_values)} 个部分")
                
                for value in unique_values:
                    subset = df[df[split_col] == value]
                    
                    with st.expander(f"{split_col} = {value} ({len(subset)} 行)"):
                        st.dataframe(subset.head())
                        
                        # 下载按钮
                        csv = subset.to_csv(index=False)
                        st.download_button(
                            label=f"📥 下载 {value}",
                            data=csv,
                            file_name=f"{split_col}_{value}.csv",
                            mime="text/csv"
                        )
        
        elif split_method == "随机分割":
            train_ratio = st.slider("训练集比例", 0.1, 0.9, 0.7, 0.1)
            
            if st.button("执行随机分割"):
                # 随机分割
                train_df = df.sample(frac=train_ratio, random_state=42)
                test_df = df.drop(train_df.index)
                
                st.success(f"✅ 数据已随机分割")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### 训练集 ({len(train_df)} 行)")
                    st.dataframe(train_df.head())
                    
                    csv_train = train_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载训练集",
                        data=csv_train,
                        file_name="train_data.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.markdown(f"#### 测试集 ({len(test_df)} 行)")
                    st.dataframe(test_df.head())
                    
                    csv_test = test_df.to_csv(index=False)
                    st.download_button(
                        label="下载测试集",
                        data=csv_test,
                        file_name="test_data.csv",
                        mime="text/csv"
                    )
    
    elif management_type == "工作表比较":
        st.markdown("### 工作表比较")
        
        st.info("💡 上传另一个文件进行比较")
        
        uploaded_file2 = st.file_uploader(
            "选择第二个Excel文件",
            type=['xlsx', 'xls', 'csv'],
            key="compare_file"
        )
        
        if uploaded_file2 is not None:
            try:
                if uploaded_file2.name.endswith('.csv'):
                    df2 = pd.read_csv(uploaded_file2)
                else:
                    df2 = pd.read_excel(uploaded_file2)
                
                st.success("✅ 第二个文件加载成功")
                
                # 基本比较
                st.markdown("#### 基本信息比较")
                
                comparison_data = {
                    "指标": ["行数", "列数", "内存使用(MB)", "缺失值总数"],
                    "文件1": [
                        len(df),
                        len(df.columns),
                        f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}",
                        df.isnull().sum().sum()
                    ],
                    "文件2": [
                        len(df2),
                        len(df2.columns),
                        f"{df2.memory_usage(deep=True).sum() / 1024 / 1024:.2f}",
                        df2.isnull().sum().sum()
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # 列比较
                st.markdown("#### 列结构比较")
                
                cols1 = set(df.columns)
                cols2 = set(df2.columns)
                
                common_cols = cols1.intersection(cols2)
                only_in_1 = cols1 - cols2
                only_in_2 = cols2 - cols1
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("共同列", len(common_cols))
                    if common_cols:
                        st.write("共同列:")
                        for col in sorted(common_cols):
                            st.write(f"• {col}")
                
                with col2:
                    st.metric("仅在文件1", len(only_in_1))
                    if only_in_1:
                        st.write("仅在文件1:")
                        for col in sorted(only_in_1):
                            st.write(f"• {col}")
                
                with col3:
                    st.metric("仅在文件2", len(only_in_2))
                    if only_in_2:
                        st.write("仅在文件2:")
                        for col in sorted(only_in_2):
                            st.write(f"• {col}")
                
                # 数据类型比较
                if common_cols:
                    st.markdown("#### 🔍 数据类型比较")
                    
                    dtype_comparison = []
                    for col in sorted(common_cols):
                        dtype_comparison.append({
                            "列名": col,
                            "文件1类型": str(df[col].dtype),
                            "文件2类型": str(df2[col].dtype),
                            "类型匹配": "✅" if df[col].dtype == df2[col].dtype else "❌"
                        })
                    
                    dtype_df = pd.DataFrame(dtype_comparison)
                    st.dataframe(dtype_df, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ 文件加载失败: {str(e)}")