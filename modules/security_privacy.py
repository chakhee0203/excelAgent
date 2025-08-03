import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import random
import string
import re
from datetime import datetime, timedelta
import math
from scipy import stats

def data_security_section(df: pd.DataFrame):
    """数据安全功能"""
    st.markdown('<h2 class="sub-header">🔒 数据安全</h2>', unsafe_allow_html=True)
    
    security_type = st.selectbox(
        "选择安全功能",
        ["数据脱敏", "隐私保护", "数据加密", "访问控制", "审计日志"]
    )
    
    if security_type == "数据脱敏":
        st.markdown("### 数据脱敏")
        
        st.info("💡 对敏感数据进行脱敏处理，保护隐私信息")
        
        masking_method = st.selectbox(
            "选择脱敏方法",
            ["字符替换", "部分隐藏", "哈希脱敏", "随机化", "泛化处理"]
        )
        
        # 选择要脱敏的列
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if masking_method == "字符替换":
            if not text_cols:
                st.warning("⚠️ 没有文本列可用于字符替换")
                return
            
            target_col = st.selectbox("选择要脱敏的列", text_cols)
            replacement_char = st.text_input("替换字符", value="*")
            
            replacement_type = st.selectbox(
                "替换类型",
                ["全部替换", "保留首尾", "保留首字符", "保留末字符"]
            )
            
            if st.button("执行字符替换脱敏"):
                masked_df = df.copy()
                
                def mask_text(text):
                    if pd.isna(text):
                        return text
                    
                    text = str(text)
                    if len(text) <= 1:
                        return replacement_char
                    
                    if replacement_type == "全部替换":
                        return replacement_char * len(text)
                    elif replacement_type == "保留首尾":
                        if len(text) <= 2:
                            return text
                        return text[0] + replacement_char * (len(text) - 2) + text[-1]
                    elif replacement_type == "保留首字符":
                        return text[0] + replacement_char * (len(text) - 1)
                    elif replacement_type == "保留末字符":
                        return replacement_char * (len(text) - 1) + text[-1]
                
                masked_df[target_col] = df[target_col].apply(mask_text)
                
                st.success("✅ 字符替换脱敏完成")
                
                # 显示对比
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 原始数据")
                    st.dataframe(df[[target_col]].head(10))
                
                with col2:
                    st.markdown("#### 脱敏后数据")
                    st.dataframe(masked_df[[target_col]].head(10))
                
                # 保存脱敏数据
                st.session_state.masked_df = masked_df
        
        elif masking_method == "哈希脱敏":
            if not text_cols:
                st.warning("⚠️ 没有文本列可用于哈希脱敏")
                return
            
            target_col = st.selectbox("选择要脱敏的列", text_cols)
            
            hash_method = st.selectbox(
                "哈希算法",
                ["MD5", "SHA256", "SHA1"]
            )
            
            preserve_length = st.checkbox("保持原始长度")
            
            if st.button("执行哈希脱敏"):
                masked_df = df.copy()
                
                def hash_text(text):
                    if pd.isna(text):
                        return text
                    
                    text = str(text)
                    
                    if hash_method == "MD5":
                        hash_obj = hashlib.md5(text.encode('utf-8'))
                    elif hash_method == "SHA256":
                        hash_obj = hashlib.sha256(text.encode('utf-8'))
                    else:  # SHA1
                        hash_obj = hashlib.sha1(text.encode('utf-8'))
                    
                    hash_value = hash_obj.hexdigest()
                    
                    if preserve_length and len(text) < len(hash_value):
                        return hash_value[:len(text)]
                    
                    return hash_value
                
                masked_df[target_col] = df[target_col].apply(hash_text)
                
                st.success("✅ 哈希脱敏完成")
                
                # 显示对比
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 原始数据")
                    st.dataframe(df[[target_col]].head(10))
                
                with col2:
                    st.markdown("#### 脱敏后数据")
                    st.dataframe(masked_df[[target_col]].head(10))
                
                # 保存脱敏数据
                st.session_state.masked_df = masked_df
        
        elif masking_method == "随机化":
            if not numeric_cols:
                st.warning("⚠️ 没有数值列可用于随机化")
                return
            
            target_col = st.selectbox("选择要脱敏的列", numeric_cols)
            
            randomization_type = st.selectbox(
                "随机化类型",
                ["加噪声", "范围随机", "排序随机"]
            )
            
            if randomization_type == "加噪声":
                noise_level = st.slider("噪声水平 (%)", 1, 50, 10)
            elif randomization_type == "范围随机":
                min_val = st.number_input("最小值", value=float(df[target_col].min()))
                max_val = st.number_input("最大值", value=float(df[target_col].max()))
            
            if st.button("执行随机化脱敏"):
                masked_df = df.copy()
                
                if randomization_type == "加噪声":
                    # 添加高斯噪声
                    std_dev = df[target_col].std() * (noise_level / 100)
                    noise = np.random.normal(0, std_dev, len(df))
                    masked_df[target_col] = df[target_col] + noise
                
                elif randomization_type == "范围随机":
                    # 在指定范围内随机生成
                    masked_df[target_col] = np.random.uniform(min_val, max_val, len(df))
                
                elif randomization_type == "排序随机":
                    # 保持分布但打乱顺序
                    masked_df[target_col] = np.random.permutation(df[target_col].values)
                
                st.success("✅ 随机化脱敏完成")
                
                # 显示统计对比
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 原始数据统计")
                    st.write(df[target_col].describe())
                
                with col2:
                    st.markdown("#### 脱敏后数据统计")
                    st.write(masked_df[target_col].describe())
                
                # 保存脱敏数据
                st.session_state.masked_df = masked_df
    
    elif security_type == "隐私保护":
        st.markdown("### 隐私保护")
        
        st.info("💡 识别和保护个人隐私信息")
        
        protection_method = st.selectbox(
            "选择保护方法",
            ["PII检测", "数据匿名化", "差分隐私", "K-匿名"]
        )
        
        if protection_method == "PII检测":
            st.markdown("#### 个人身份信息(PII)检测")
            
            # 定义PII模式
            pii_patterns = {
                "邮箱": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "电话号码": r'\b(?:\+?86)?1[3-9]\d{9}\b',
                "身份证号": r'\b[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]\b',
                "银行卡号": r'\b\d{16,19}\b',
                "IP地址": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            }
            
            if st.button("检测PII信息"):
                pii_results = {}
                
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                for col in text_cols:
                    col_pii = []
                    
                    for idx, value in df[col].items():
                        if pd.isna(value):
                            continue
                        
                        value_str = str(value)
                        
                        for pii_type, pattern in pii_patterns.items():
                            matches = re.findall(pattern, value_str)
                            if matches:
                                col_pii.append({
                                    "行号": idx,
                                    "PII类型": pii_type,
                                    "匹配内容": matches[0][:10] + "..." if len(matches[0]) > 10 else matches[0]
                                })
                    
                    if col_pii:
                        pii_results[col] = col_pii
                
                if pii_results:
                    st.warning(f"⚠️ 检测到 {len(pii_results)} 列包含PII信息")
                    
                    for col, pii_list in pii_results.items():
                        with st.expander(f"列 '{col}' - 发现 {len(pii_list)} 个PII"):
                            pii_df = pd.DataFrame(pii_list)
                            st.dataframe(pii_df)
                else:
                    st.success("✅ 未检测到明显的PII信息")
        
        elif protection_method == "数据匿名化":
            st.markdown("#### 数据匿名化")
            
            # 选择准标识符
            quasi_identifiers = st.multiselect(
                "选择准标识符列",
                df.columns.tolist(),
                help="准标识符是可能用于重新识别个人的属性组合"
            )
            
            if not quasi_identifiers:
                st.warning("⚠️ 请选择至少一个准标识符")
                return
            
            anonymization_method = st.selectbox(
                "匿名化方法",
                ["泛化", "抑制", "置换"]
            )
            
            if st.button("执行匿名化"):
                anonymized_df = df.copy()
                
                for col in quasi_identifiers:
                    if anonymization_method == "泛化":
                        if df[col].dtype in ['int64', 'float64']:
                            # 数值泛化：分组到范围
                            bins = st.slider(f"{col} 分组数", 2, 10, 5, key=f"bins_{col}")
                            anonymized_df[col] = pd.cut(df[col], bins=bins, labels=False)
                        else:
                            # 文本泛化：取前缀
                            prefix_length = st.slider(f"{col} 前缀长度", 1, 5, 2, key=f"prefix_{col}")
                            anonymized_df[col] = df[col].astype(str).str[:prefix_length] + "*"
                    
                    elif anonymization_method == "抑制":
                        # 随机抑制部分值
                        suppression_rate = st.slider(f"{col} 抑制比例", 0.1, 0.5, 0.2, key=f"suppress_{col}")
                        mask = np.random.random(len(df)) < suppression_rate
                        anonymized_df.loc[mask, col] = "*"
                    
                    elif anonymization_method == "置换":
                        # 随机置换值
                        anonymized_df[col] = np.random.permutation(df[col].values)
                
                st.success("✅ 数据匿名化完成")
                
                # 显示对比
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 原始数据")
                    st.dataframe(df[quasi_identifiers].head(10))
                
                with col2:
                    st.markdown("#### 匿名化后数据")
                    st.dataframe(anonymized_df[quasi_identifiers].head(10))
                
                # 保存匿名化数据
                st.session_state.anonymized_df = anonymized_df

def mathematical_functions_section(df: pd.DataFrame):
    """数学统计函数功能"""
    st.markdown('<h2 class="sub-header">数学统计函数</h2>', unsafe_allow_html=True)
    
    function_category = st.selectbox(
        "选择函数类别",
        ["基础数学函数", "统计函数", "概率分布", "假设检验", "回归分析"]
    )
    
    if function_category == "基础数学函数":
        st.markdown("### 基础数学函数")
        
        math_function = st.selectbox(
            "选择数学函数",
            ["三角函数", "对数函数", "指数函数", "幂函数", "取整函数"]
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ 没有数值列可用于数学计算")
            return
        
        selected_col = st.selectbox("选择数据列", numeric_cols)
        
        if math_function == "三角函数":
            trig_func = st.selectbox(
                "选择三角函数",
                ["sin", "cos", "tan", "arcsin", "arccos", "arctan"]
            )
            
            angle_unit = st.selectbox("角度单位", ["弧度", "度"])
            
            if st.button("计算三角函数"):
                data = df[selected_col].dropna()
                
                # 角度转换
                if angle_unit == "度" and trig_func in ["sin", "cos", "tan"]:
                    data_rad = np.radians(data)
                else:
                    data_rad = data
                
                # 计算三角函数
                if trig_func == "sin":
                    result = np.sin(data_rad)
                elif trig_func == "cos":
                    result = np.cos(data_rad)
                elif trig_func == "tan":
                    result = np.tan(data_rad)
                elif trig_func == "arcsin":
                    result = np.arcsin(np.clip(data, -1, 1))
                elif trig_func == "arccos":
                    result = np.arccos(np.clip(data, -1, 1))
                elif trig_func == "arctan":
                    result = np.arctan(data)
                
                # 反三角函数结果转换
                if angle_unit == "度" and trig_func.startswith("arc"):
                    result = np.degrees(result)
                
                # 显示结果
                result_df = pd.DataFrame({
                    f"原始值({selected_col})": data,
                    f"{trig_func}结果": result
                })
                
                st.markdown(f"#### {trig_func} 计算结果")
                st.dataframe(result_df.head(20))
                
                # 统计信息
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("最大值", f"{result.max():.4f}")
                with col2:
                    st.metric("最小值", f"{result.min():.4f}")
                with col3:
                    st.metric("平均值", f"{result.mean():.4f}")
        
        elif math_function == "对数函数":
            log_base = st.selectbox(
                "对数底数",
                ["自然对数(e)", "常用对数(10)", "二进制对数(2)", "自定义"]
            )
            
            if log_base == "自定义":
                custom_base = st.number_input("自定义底数", value=2.0, min_value=0.001)
            
            if st.button("计算对数"):
                data = df[selected_col].dropna()
                
                # 过滤正数
                positive_data = data[data > 0]
                
                if len(positive_data) == 0:
                    st.error("❌ 没有正数可用于对数计算")
                    return
                
                # 计算对数
                if log_base == "自然对数(e)":
                    result = np.log(positive_data)
                    base_name = "ln"
                elif log_base == "常用对数(10)":
                    result = np.log10(positive_data)
                    base_name = "log10"
                elif log_base == "二进制对数(2)":
                    result = np.log2(positive_data)
                    base_name = "log2"
                else:
                    result = np.log(positive_data) / np.log(custom_base)
                    base_name = f"log{custom_base}"
                
                # 显示结果
                result_df = pd.DataFrame({
                    f"原始值({selected_col})": positive_data,
                    f"{base_name}结果": result
                })
                
                st.markdown(f"#### {base_name} 计算结果")
                st.dataframe(result_df.head(20))
                
                if len(positive_data) < len(data):
                    st.warning(f"⚠️ 已过滤 {len(data) - len(positive_data)} 个非正数")
    
    elif function_category == "统计函数":
        st.markdown("### 统计函数")
        
        stat_function = st.selectbox(
            "选择统计函数",
            ["描述性统计", "分位数分析", "偏度和峰度", "变异系数", "置信区间"]
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ 没有数值列可用于统计分析")
            return
        
        if stat_function == "描述性统计":
            selected_cols = st.multiselect(
                "选择分析列",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if st.button("计算描述性统计"):
                if not selected_cols:
                    st.warning("⚠️ 请选择至少一列")
                    return
                
                # 计算描述性统计
                desc_stats = df[selected_cols].describe()
                
                # 添加额外统计量
                additional_stats = pd.DataFrame(index=['变异系数', '偏度', '峰度'])
                
                for col in selected_cols:
                    data = df[col].dropna()
                    cv = data.std() / data.mean() if data.mean() != 0 else np.nan
                    skewness = stats.skew(data)
                    kurtosis = stats.kurtosis(data)
                    
                    additional_stats[col] = [cv, skewness, kurtosis]
                
                # 合并统计结果
                full_stats = pd.concat([desc_stats, additional_stats])
                
                st.markdown("#### 描述性统计结果")
                st.dataframe(full_stats.round(4))
                
                # 可视化
                if len(selected_cols) == 1:
                    col = selected_cols[0]
                    data = df[col].dropna()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 直方图
                        import plotly.express as px
                        fig_hist = px.histogram(
                            x=data,
                            title=f"{col} 分布直方图",
                            nbins=30
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # 箱线图
                        fig_box = px.box(
                            y=data,
                            title=f"{col} 箱线图"
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
        
        elif stat_function == "置信区间":
            selected_col = st.selectbox("选择分析列", numeric_cols)
            
            confidence_level = st.slider(
                "置信水平",
                0.80, 0.99, 0.95, 0.01,
                format="%.2f"
            )
            
            if st.button("计算置信区间"):
                data = df[selected_col].dropna()
                
                if len(data) < 2:
                    st.error("❌ 数据点太少，无法计算置信区间")
                    return
                
                # 计算置信区间
                mean = data.mean()
                std_err = stats.sem(data)  # 标准误
                
                # t分布临界值
                alpha = 1 - confidence_level
                t_critical = stats.t.ppf(1 - alpha/2, len(data) - 1)
                
                # 置信区间
                margin_error = t_critical * std_err
                ci_lower = mean - margin_error
                ci_upper = mean + margin_error
                
                # 显示结果
                st.markdown(f"#### {selected_col} 的 {confidence_level:.0%} 置信区间")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("样本均值", f"{mean:.4f}")
                with col2:
                    st.metric("标准误", f"{std_err:.4f}")
                with col3:
                    st.metric("下限", f"{ci_lower:.4f}")
                with col4:
                    st.metric("上限", f"{ci_upper:.4f}")
                
                # 解释
                st.info(
                    f"💡 我们有 {confidence_level:.0%} 的信心认为总体均值在 "
                    f"[{ci_lower:.4f}, {ci_upper:.4f}] 区间内。"
                )
                
                # 可视化置信区间
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # 添加数据点
                fig.add_trace(go.Scatter(
                    x=list(range(len(data))),
                    y=data,
                    mode='markers',
                    name='数据点',
                    marker=dict(color='lightblue')
                ))
                
                # 添加均值线
                fig.add_hline(
                    y=mean,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"均值: {mean:.4f}"
                )
                
                # 添加置信区间
                fig.add_hrect(
                    y0=ci_lower,
                    y1=ci_upper,
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    layer="below",
                    line_width=0,
                    annotation_text=f"{confidence_level:.0%} 置信区间"
                )
                
                fig.update_layout(
                    title=f"{selected_col} 置信区间可视化",
                    xaxis_title="数据点索引",
                    yaxis_title="数值"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif function_category == "概率分布":
        st.markdown("### 概率分布")
        
        distribution_type = st.selectbox(
            "选择分布类型",
            ["正态分布", "泊松分布", "指数分布", "均匀分布", "二项分布"]
        )
        
        if distribution_type == "正态分布":
            st.markdown("#### 📊 正态分布分析")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.warning("⚠️ 没有数值列可用于分布分析")
                return
            
            selected_col = st.selectbox("选择分析列", numeric_cols)
            
            if st.button("分析正态分布"):
                data = df[selected_col].dropna()
                
                if len(data) < 3:
                    st.error("❌ 数据点太少，无法进行分布分析")
                    return
                
                # 正态性检验
                shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Shapiro-Wilk检验限制样本量
                
                # 计算分布参数
                mean = data.mean()
                std = data.std()
                
                # 显示结果
                st.markdown("#### 正态分布参数")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("均值 (μ)", f"{mean:.4f}")
                with col2:
                    st.metric("标准差 (σ)", f"{std:.4f}")
                with col3:
                    st.metric("Shapiro统计量", f"{shapiro_stat:.4f}")
                with col4:
                    st.metric("p值", f"{shapiro_p:.4f}")
                
                # 正态性判断
                if shapiro_p > 0.05:
                    st.success("✅ 数据可能服从正态分布 (p > 0.05)")
                else:
                    st.warning("⚠️ 数据可能不服从正态分布 (p ≤ 0.05)")
                
                # 可视化
                import plotly.figure_factory as ff
                
                # Q-Q图
                fig_qq = go.Figure()
                
                # 理论分位数
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
                sample_quantiles = np.sort(data)
                
                fig_qq.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    name='数据点'
                ))
                
                # 理想直线
                fig_qq.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=theoretical_quantiles * std + mean,
                    mode='lines',
                    name='理想正态分布',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_qq.update_layout(
                    title="Q-Q图 (正态性检验)",
                    xaxis_title="理论分位数",
                    yaxis_title="样本分位数"
                )
                
                st.plotly_chart(fig_qq, use_container_width=True)
                
                # 概率密度函数对比
                x_range = np.linspace(data.min(), data.max(), 100)
                theoretical_pdf = stats.norm.pdf(x_range, mean, std)
                
                fig_pdf = go.Figure()
                
                # 实际数据直方图
                fig_pdf.add_trace(go.Histogram(
                    x=data,
                    histnorm='probability density',
                    name='实际数据',
                    opacity=0.7
                ))
                
                # 理论正态分布
                fig_pdf.add_trace(go.Scatter(
                    x=x_range,
                    y=theoretical_pdf,
                    mode='lines',
                    name='理论正态分布',
                    line=dict(color='red', width=2)
                ))
                
                fig_pdf.update_layout(
                    title="概率密度函数对比",
                    xaxis_title="数值",
                    yaxis_title="概率密度"
                )
                
                st.plotly_chart(fig_pdf, use_container_width=True)