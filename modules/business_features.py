import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

def table_operations_section(df: pd.DataFrame):
    """数据表格操作功能"""
    st.markdown('<h2 class="sub-header">数据表格操作</h2>', unsafe_allow_html=True)
    
    operation = st.selectbox(
        "选择操作类型",
        ["数据预览", "列操作", "行操作", "数据透视表", "数据合并"]
    )
    
    if operation == "数据预览":
        st.markdown("### 数据预览")
        
        # 基本信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总行数", len(df))
        with col2:
            st.metric("总列数", len(df.columns))
        with col3:
            st.metric("内存使用", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")
        with col4:
            st.metric("缺失值", df.isnull().sum().sum())
        
        # 数据类型
        st.markdown("#### 数据类型分布")
        dtype_counts = df.dtypes.value_counts()
        fig = px.pie(values=dtype_counts.values, names=[str(dtype) for dtype in dtype_counts.index], title="数据类型分布")
        st.plotly_chart(fig, use_container_width=True)
        
        # 数据预览
        st.markdown("#### 🔍 数据预览")
        
        preview_rows = st.slider("预览行数", 5, min(100, len(df)), 10)
        preview_type = st.radio("预览类型", ["前N行", "后N行", "随机N行"])
        
        if preview_type == "前N行":
            st.dataframe(df.head(preview_rows))
        elif preview_type == "后N行":
            st.dataframe(df.tail(preview_rows))
        else:
            st.dataframe(df.sample(n=min(preview_rows, len(df))))
    
    elif operation == "列操作":
        st.markdown("### 列操作")
        
        col_operation = st.selectbox(
            "选择列操作",
            ["重命名列", "删除列", "添加计算列", "列排序", "列筛选"]
        )
        
        if col_operation == "重命名列":
            st.markdown("#### 重命名列")
            
            col_to_rename = st.selectbox("选择要重命名的列", df.columns.tolist())
            new_name = st.text_input("输入新列名")
            
            if st.button("重命名") and new_name:
                renamed_df = df.rename(columns={col_to_rename: new_name})
                st.success(f"✅ 列 '{col_to_rename}' 已重命名为 '{new_name}'")
                st.dataframe(renamed_df.head())
                
                # 保存到session state
                st.session_state.modified_df = renamed_df
        
        elif col_operation == "删除列":
            st.markdown("#### 删除列")
            
            cols_to_delete = st.multiselect("选择要删除的列", df.columns.tolist())
            
            if st.button("删除选中列") and cols_to_delete:
                modified_df = df.drop(columns=cols_to_delete)
                st.success(f"✅ 已删除 {len(cols_to_delete)} 列")
                st.write(f"剩余列数: {len(modified_df.columns)}")
                st.dataframe(modified_df.head())
                
                # 保存到session state
                st.session_state.modified_df = modified_df
        
        elif col_operation == "添加计算列":
            st.markdown("#### 添加计算列")
            
            new_col_name = st.text_input("新列名")
            
            calc_type = st.selectbox(
                "计算类型",
                ["两列相加", "两列相减", "两列相乘", "两列相除", "常数运算", "自定义公式"]
            )
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if calc_type in ["两列相加", "两列相减", "两列相乘", "两列相除"]:
                if len(numeric_cols) >= 2:
                    col1 = st.selectbox("选择第一列", numeric_cols)
                    col2 = st.selectbox("选择第二列", [c for c in numeric_cols if c != col1])
                    
                    if st.button("添加计算列") and new_col_name:
                        modified_df = df.copy()
                        
                        if calc_type == "两列相加":
                            modified_df[new_col_name] = df[col1] + df[col2]
                        elif calc_type == "两列相减":
                            modified_df[new_col_name] = df[col1] - df[col2]
                        elif calc_type == "两列相乘":
                            modified_df[new_col_name] = df[col1] * df[col2]
                        elif calc_type == "两列相除":
                            modified_df[new_col_name] = df[col1] / df[col2]
                        
                        st.success(f"✅ 已添加计算列 '{new_col_name}'")
                        st.dataframe(modified_df[[col1, col2, new_col_name]].head())
                        
                        # 保存到session state
                        st.session_state.modified_df = modified_df
                else:
                    st.warning("⚠️ 需要至少2个数值列进行计算")
    
    elif operation == "数据透视表":
        st.markdown("### 数据透视表")
        
        if len(df.columns) < 2:
            st.warning("⚠️ 数据列数不足，无法创建透视表")
            return
        
        # 透视表配置
        index_col = st.selectbox("选择行索引", df.columns.tolist())
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            values_col = st.selectbox("选择数值列", numeric_cols)
            
            agg_func = st.selectbox(
                "聚合函数",
                ["sum", "mean", "count", "min", "max", "std"]
            )
            
            # 可选的列索引
            remaining_cols = [col for col in df.columns if col not in [index_col, values_col]]
            if remaining_cols:
                use_columns = st.checkbox("使用列索引")
                if use_columns:
                    columns_col = st.selectbox("选择列索引", remaining_cols)
                else:
                    columns_col = None
            else:
                columns_col = None
            
            if st.button("生成透视表"):
                try:
                    pivot_table = pd.pivot_table(
                        df,
                        index=index_col,
                        columns=columns_col,
                        values=values_col,
                        aggfunc=agg_func,
                        fill_value=0
                    )
                    
                    st.markdown("#### 透视表结果")
                    st.dataframe(pivot_table)
                    
                    # 透视表可视化
                    if columns_col is None:
                        # 简单柱状图
                        fig = px.bar(
                            x=pivot_table.index,
                            y=pivot_table.values.flatten(),
                            title=f"{values_col} 按 {index_col} 分组的 {agg_func}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # 热力图
                        fig = px.imshow(
                            pivot_table.values,
                            x=pivot_table.columns,
                            y=pivot_table.index,
                            title="透视表热力图"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 保存透视表
                    st.session_state.pivot_table = pivot_table
                    
                    # 下载透视表
                    csv = pivot_table.to_csv()
                    st.download_button(
                        label="下载透视表",
                        data=csv,
                        file_name="pivot_table.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ 透视表生成失败: {str(e)}")
        else:
            st.warning("⚠️ 没有数值列可用于透视表")

def formula_calculator_section(df: pd.DataFrame):
    """公式计算器功能"""
    st.markdown('<h2 class="sub-header">公式计算器</h2>', unsafe_allow_html=True)
    
    calc_type = st.selectbox(
        "选择计算类型",
        ["基础数学运算", "统计函数", "财务函数", "日期函数", "文本函数"]
    )
    
    if calc_type == "基础数学运算":
        st.markdown("### 基础数学运算")
        
        operation = st.selectbox(
            "选择运算",
            ["加法", "减法", "乘法", "除法", "幂运算", "开方", "对数", "三角函数"]
        )
        
        if operation in ["加法", "减法", "乘法", "除法", "幂运算"]:
            col1, col2 = st.columns(2)
            with col1:
                num1 = st.number_input("第一个数", value=0.0)
            with col2:
                num2 = st.number_input("第二个数", value=0.0)
            
            if st.button("计算"):
                if operation == "加法":
                    result = num1 + num2
                elif operation == "减法":
                    result = num1 - num2
                elif operation == "乘法":
                    result = num1 * num2
                elif operation == "除法":
                    result = num1 / num2 if num2 != 0 else "除数不能为0"
                elif operation == "幂运算":
                    result = num1 ** num2
                
                st.success(f"计算结果: {result}")
        
        elif operation == "开方":
            num = st.number_input("输入数值", value=0.0, min_value=0.0)
            root = st.number_input("开几次方", value=2.0, min_value=1.0)
            
            if st.button("计算"):
                result = num ** (1/root)
                st.success(f"{num} 的 {root} 次方根 = {result}")
        
        elif operation == "对数":
            num = st.number_input("输入数值", value=1.0, min_value=0.001)
            base = st.selectbox("对数底数", ["自然对数(e)", "常用对数(10)", "二进制对数(2)", "自定义"])
            
            if base == "自定义":
                custom_base = st.number_input("自定义底数", value=2.0, min_value=0.001)
            
            if st.button("计算"):
                if base == "自然对数(e)":
                    result = math.log(num)
                elif base == "常用对数(10)":
                    result = math.log10(num)
                elif base == "二进制对数(2)":
                    result = math.log2(num)
                else:
                    result = math.log(num, custom_base)
                
                st.success(f"结果: {result}")
    
    elif calc_type == "统计函数":
        st.markdown("### 统计函数")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ 没有数值列可用于统计计算")
            return
        
        selected_col = st.selectbox("选择数据列", numeric_cols)
        
        stat_func = st.selectbox(
            "选择统计函数",
            ["平均值", "中位数", "众数", "标准差", "方差", "最大值", "最小值", "求和", "计数", "分位数"]
        )
        
        if stat_func == "分位数":
            quantile = st.slider("分位数", 0.0, 1.0, 0.5, 0.01)
        
        if st.button("计算统计值"):
            data = df[selected_col].dropna()
            
            if stat_func == "平均值":
                result = data.mean()
            elif stat_func == "中位数":
                result = data.median()
            elif stat_func == "众数":
                mode_result = data.mode()
                result = mode_result.iloc[0] if not mode_result.empty else "无众数"
            elif stat_func == "标准差":
                result = data.std()
            elif stat_func == "方差":
                result = data.var()
            elif stat_func == "最大值":
                result = data.max()
            elif stat_func == "最小值":
                result = data.min()
            elif stat_func == "求和":
                result = data.sum()
            elif stat_func == "计数":
                result = data.count()
            elif stat_func == "分位数":
                result = data.quantile(quantile)
            
            st.success(f"{selected_col} 的 {stat_func}: {result}")
            
            # 显示数据分布
            fig = px.histogram(df, x=selected_col, title=f"{selected_col} 分布")
            st.plotly_chart(fig, use_container_width=True)
    
    elif calc_type == "财务函数":
        st.markdown("### 💰 财务函数")
        
        fin_func = st.selectbox(
            "选择财务函数",
            ["现值计算(PV)", "终值计算(FV)", "年金现值(PVA)", "年金终值(FVA)", "内部收益率(IRR)", "净现值(NPV)"]
        )
        
        if fin_func == "现值计算(PV)":
            st.markdown("#### 现值计算")
            st.info("计算未来现金流的现值")
            
            fv = st.number_input("终值 (FV)", value=1000.0)
            rate = st.number_input("利率 (%)", value=5.0) / 100
            periods = st.number_input("期数", value=1, min_value=1)
            
            if st.button("计算现值"):
                pv = fv / ((1 + rate) ** periods)
                st.success(f"现值 (PV) = {pv:.2f}")
                
                # 显示计算过程
                st.write(f"计算公式: PV = FV / (1 + r)^n")
                st.write(f"PV = {fv} / (1 + {rate:.3f})^{periods} = {pv:.2f}")
        
        elif fin_func == "终值计算(FV)":
            st.markdown("#### 终值计算")
            st.info("计算现在投资在未来的价值")
            
            pv = st.number_input("现值 (PV)", value=1000.0)
            rate = st.number_input("利率 (%)", value=5.0) / 100
            periods = st.number_input("期数", value=1, min_value=1)
            
            if st.button("计算终值"):
                fv = pv * ((1 + rate) ** periods)
                st.success(f"终值 (FV) = {fv:.2f}")
                
                # 显示计算过程
                st.write(f"计算公式: FV = PV × (1 + r)^n")
                st.write(f"FV = {pv} × (1 + {rate:.3f})^{periods} = {fv:.2f}")
        
        elif fin_func == "年金现值(PVA)":
            st.markdown("#### 年金现值")
            st.info("计算一系列等额支付的现值")
            
            pmt = st.number_input("每期支付金额 (PMT)", value=100.0)
            rate = st.number_input("利率 (%)", value=5.0) / 100
            periods = st.number_input("期数", value=10, min_value=1)
            
            if st.button("计算年金现值"):
                if rate == 0:
                    pva = pmt * periods
                else:
                    pva = pmt * ((1 - (1 + rate) ** (-periods)) / rate)
                
                st.success(f"年金现值 (PVA) = {pva:.2f}")
                
                # 显示计算过程
                st.write(f"计算公式: PVA = PMT × [(1 - (1 + r)^(-n)) / r]")
                st.write(f"PVA = {pmt} × [(1 - (1 + {rate:.3f})^(-{periods})) / {rate:.3f}] = {pva:.2f}")

def financial_analysis_section(df: pd.DataFrame):
    """财务分析功能"""
    st.markdown('<h2 class="sub-header">💼 财务分析</h2>', unsafe_allow_html=True)
    
    analysis_type = st.selectbox(
        "选择分析类型",
        ["盈利能力分析", "偿债能力分析", "营运能力分析", "成长能力分析", "投资分析"]
    )
    
    if analysis_type == "盈利能力分析":
        st.markdown("### 盈利能力分析")
        
        st.info("💡 分析企业的盈利能力和效率")
        
        # 输入财务数据
        col1, col2 = st.columns(2)
        
        with col1:
            revenue = st.number_input("营业收入", value=0.0, min_value=0.0)
            cost_of_sales = st.number_input("营业成本", value=0.0, min_value=0.0)
            operating_expenses = st.number_input("营业费用", value=0.0, min_value=0.0)
        
        with col2:
            total_assets = st.number_input("总资产", value=0.0, min_value=0.0)
            shareholders_equity = st.number_input("股东权益", value=0.0, min_value=0.0)
            net_income = st.number_input("净利润", value=0.0)
        
        if st.button("计算盈利能力指标"):
            # 计算各项指标
            gross_profit = revenue - cost_of_sales
            operating_profit = gross_profit - operating_expenses
            
            # 盈利能力比率
            ratios = {}
            
            if revenue > 0:
                ratios['毛利率'] = (gross_profit / revenue) * 100
                ratios['营业利润率'] = (operating_profit / revenue) * 100
                ratios['净利润率'] = (net_income / revenue) * 100
            
            if total_assets > 0:
                ratios['总资产收益率(ROA)'] = (net_income / total_assets) * 100
            
            if shareholders_equity > 0:
                ratios['净资产收益率(ROE)'] = (net_income / shareholders_equity) * 100
            
            # 显示结果
            st.markdown("#### 盈利能力指标")
            
            for ratio_name, ratio_value in ratios.items():
                st.metric(ratio_name, f"{ratio_value:.2f}%")
            
            # 可视化
            if ratios:
                fig = px.bar(
                    x=list(ratios.keys()),
                    y=list(ratios.values()),
                    title="盈利能力指标对比",
                    labels={'x': '指标', 'y': '比率(%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 分析建议
            st.markdown("#### 💡 分析建议")
            
            if '毛利率' in ratios:
                if ratios['毛利率'] > 30:
                    st.success("✅ 毛利率较高，产品竞争力强")
                elif ratios['毛利率'] > 15:
                    st.warning("⚠️ 毛利率中等，需关注成本控制")
                else:
                    st.error("❌ 毛利率较低，建议优化产品结构")
            
            if '净资产收益率(ROE)' in ratios:
                if ratios['净资产收益率(ROE)'] > 15:
                    st.success("✅ ROE较高，股东回报良好")
                elif ratios['净资产收益率(ROE)'] > 8:
                    st.warning("⚠️ ROE中等，有提升空间")
                else:
                    st.error("❌ ROE较低，需改善经营效率")

def time_series_analysis_section(df: pd.DataFrame):
    """时间序列分析功能"""
    st.markdown('<h2 class="sub-header">时间序列分析</h2>', unsafe_allow_html=True)
    
    # 检测日期列
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # 尝试识别可能的日期列
    potential_date_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(10))
                potential_date_cols.append(col)
            except:
                pass
    
    all_date_cols = date_cols + potential_date_cols
    
    if not all_date_cols:
        st.warning("⚠️ 未检测到日期列，请确保数据中包含日期信息")
        
        # 提供手动指定选项
        manual_date_col = st.selectbox("手动选择日期列", df.columns.tolist())
        
        if st.button("尝试转换为日期格式"):
            try:
                df[manual_date_col] = pd.to_datetime(df[manual_date_col])
                st.success(f"✅ 成功将 '{manual_date_col}' 转换为日期格式")
                all_date_cols = [manual_date_col]
            except Exception as e:
                st.error(f"❌ 日期转换失败: {str(e)}")
                return
    
    if all_date_cols:
        date_col = st.selectbox("选择日期列", all_date_cols)
        
        # 确保日期列是datetime类型
        if df[date_col].dtype != 'datetime64[ns]':
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception as e:
                st.error(f"❌ 日期列转换失败: {str(e)}")
                return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ 没有数值列可用于时间序列分析")
            return
        
        value_col = st.selectbox("选择数值列", numeric_cols)
        
        analysis_type = st.selectbox(
            "选择分析类型",
            ["趋势分析", "季节性分析", "移动平均", "增长率分析", "预测分析"]
        )
        
        # 准备时间序列数据
        ts_df = df[[date_col, value_col]].copy()
        ts_df = ts_df.sort_values(date_col)
        ts_df = ts_df.dropna()
        
        if len(ts_df) < 2:
            st.warning("⚠️ 数据点太少，无法进行时间序列分析")
            return
        
        if analysis_type == "趋势分析":
            st.markdown("### 趋势分析")
            
            # 基本时间序列图
            fig = px.line(
                ts_df,
                x=date_col,
                y=value_col,
                title=f"{value_col} 时间序列趋势"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 趋势统计
            st.markdown("#### 趋势统计")
            
            # 计算总体趋势
            first_value = ts_df[value_col].iloc[0]
            last_value = ts_df[value_col].iloc[-1]
            total_change = last_value - first_value
            total_change_pct = (total_change / first_value) * 100 if first_value != 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("起始值", f"{first_value:.2f}")
            with col2:
                st.metric("结束值", f"{last_value:.2f}")
            with col3:
                st.metric("总变化", f"{total_change:.2f}")
            with col4:
                st.metric("总变化率", f"{total_change_pct:.2f}%")
            
            # 趋势方向判断
            if total_change > 0:
                st.success("整体呈上升趋势")
            elif total_change < 0:
                st.error("整体呈下降趋势")
            else:
                st.info("整体趋势平稳")
        
        elif analysis_type == "移动平均":
            st.markdown("### 移动平均分析")
            
            window_size = st.slider("移动平均窗口大小", 2, min(30, len(ts_df)//2), 7)
            
            # 计算移动平均
            ts_df['移动平均'] = ts_df[value_col].rolling(window=window_size).mean()
            
            # 绘制图表
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ts_df[date_col],
                y=ts_df[value_col],
                mode='lines',
                name='原始数据',
                line=dict(color='lightblue')
            ))
            
            fig.add_trace(go.Scatter(
                x=ts_df[date_col],
                y=ts_df['移动平均'],
                mode='lines',
                name=f'{window_size}期移动平均',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f"{value_col} 移动平均分析",
                xaxis_title="日期",
                yaxis_title="数值"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 移动平均统计
            st.markdown("#### 移动平均统计")
            
            ma_data = ts_df['移动平均'].dropna()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("移动平均最大值", f"{ma_data.max():.2f}")
            with col2:
                st.metric("移动平均最小值", f"{ma_data.min():.2f}")
            with col3:
                st.metric("移动平均标准差", f"{ma_data.std():.2f}")

def goal_tracking_section(df: pd.DataFrame):
    """目标跟踪功能"""
    st.markdown('<h2 class="sub-header">目标跟踪</h2>', unsafe_allow_html=True)
    
    tracking_type = st.selectbox(
        "选择跟踪类型",
        ["销售目标", "KPI跟踪", "预算执行", "项目进度", "自定义目标"]
    )
    
    if tracking_type == "销售目标":
        st.markdown("### 💰 销售目标跟踪")
        
        # 目标设置
        col1, col2 = st.columns(2)
        
        with col1:
            sales_target = st.number_input("销售目标", value=100000.0, min_value=0.0)
            time_period = st.selectbox("时间周期", ["月度", "季度", "年度"])
        
        with col2:
            current_sales = st.number_input("当前销售额", value=0.0, min_value=0.0)
            days_passed = st.number_input("已过天数", value=1, min_value=1)
        
        # 计算进度
        if sales_target > 0:
            completion_rate = (current_sales / sales_target) * 100
            
            # 根据时间周期计算预期进度
            if time_period == "月度":
                total_days = 30
            elif time_period == "季度":
                total_days = 90
            else:  # 年度
                total_days = 365
            
            expected_progress = (days_passed / total_days) * 100
            
            # 显示进度
            st.markdown("#### 目标完成情况")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("完成率", f"{completion_rate:.1f}%")
            with col2:
                st.metric("预期进度", f"{expected_progress:.1f}%")
            with col3:
                remaining = sales_target - current_sales
                st.metric("剩余目标", f"{remaining:.0f}")
            with col4:
                if expected_progress > 0:
                    performance = completion_rate / expected_progress
                    st.metric("绩效指数", f"{performance:.2f}")
            
            # 进度条可视化
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['目标进度'],
                y=[completion_rate],
                name='实际完成',
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                x=['目标进度'],
                y=[expected_progress],
                name='预期进度',
                marker_color='orange',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="目标完成进度对比",
                yaxis_title="完成率(%)",
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 分析和建议
            st.markdown("#### 💡 分析建议")
            
            if completion_rate >= expected_progress:
                st.success("✅ 目标完成情况良好，超出预期进度")
            elif completion_rate >= expected_progress * 0.8:
                st.warning("⚠️ 目标完成情况基本符合预期，需继续努力")
            else:
                st.error("❌ 目标完成情况落后于预期，需要加强措施")
            
            # 预测分析
            if days_passed > 0:
                daily_avg = current_sales / days_passed
                remaining_days = total_days - days_passed
                projected_total = current_sales + (daily_avg * remaining_days)
                
                st.markdown("#### 预测分析")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("日均销售", f"{daily_avg:.0f}")
                with col2:
                    st.metric("预测总销售", f"{projected_total:.0f}")
                
                if projected_total >= sales_target:
                    st.success(f"✅ 按当前速度，预计能够完成目标")
                else:
                    shortfall = sales_target - projected_total
                    required_daily = (sales_target - current_sales) / remaining_days if remaining_days > 0 else 0
                    st.warning(f"⚠️ 按当前速度，预计缺口 {shortfall:.0f}")
                    st.info(f"💡 需要日均销售达到 {required_daily:.0f} 才能完成目标")

def dashboard_creation_section(df: pd.DataFrame):
    """仪表板创建功能"""
    st.markdown('<h2 class="sub-header">仪表板创建</h2>', unsafe_allow_html=True)
    
    dashboard_type = st.selectbox(
        "选择仪表板类型",
        ["销售仪表板", "财务仪表板", "运营仪表板", "自定义仪表板"]
    )
    
    if dashboard_type == "销售仪表板":
        st.markdown("### 销售仪表板")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ 需要数值列来创建销售仪表板")
            return
        
        # 配置仪表板
        col1, col2 = st.columns(2)
        
        with col1:
            sales_col = st.selectbox("选择销售额列", numeric_cols)
            
        with col2:
            if categorical_cols:
                category_col = st.selectbox("选择分类列（可选）", ["无"] + categorical_cols)
                if category_col == "无":
                    category_col = None
            else:
                category_col = None
        
        if st.button("生成销售仪表板"):
            # 关键指标
            st.markdown("#### 关键销售指标")
            
            total_sales = df[sales_col].sum()
            avg_sales = df[sales_col].mean()
            max_sales = df[sales_col].max()
            min_sales = df[sales_col].min()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总销售额", f"{total_sales:,.0f}")
            with col2:
                st.metric("平均销售额", f"{avg_sales:,.0f}")
            with col3:
                st.metric("最高销售额", f"{max_sales:,.0f}")
            with col4:
                st.metric("最低销售额", f"{min_sales:,.0f}")
            
            # 销售分布图
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    df,
                    x=sales_col,
                    title="销售额分布",
                    nbins=20
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                if category_col:
                    # 按类别分组的销售
                    category_sales = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)
                    
                    fig_bar = px.bar(
                        x=category_sales.index,
                        y=category_sales.values,
                        title=f"按{category_col}分组的销售额"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    # 销售趋势（如果有足够数据点）
                    if len(df) > 10:
                        df_indexed = df.reset_index()
                        fig_trend = px.line(
                            df_indexed,
                            x='index',
                            y=sales_col,
                            title="销售趋势"
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
            
            # 销售排名（如果有分类列）
            if category_col:
                st.markdown("#### 销售排名")
                
                ranking = df.groupby(category_col)[sales_col].agg([
                    'sum', 'mean', 'count'
                ]).round(2)
                ranking.columns = ['总销售额', '平均销售额', '销售次数']
                ranking = ranking.sort_values('总销售额', ascending=False)
                
                st.dataframe(ranking)
            
            # 销售分析
            st.markdown("#### 销售分析")
            
            # 计算一些统计指标
            sales_std = df[sales_col].std()
            sales_cv = sales_std / avg_sales if avg_sales != 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("销售标准差", f"{sales_std:,.0f}")
                st.metric("变异系数", f"{sales_cv:.2f}")
            
            with col2:
                # 销售集中度分析
                if category_col and len(df[category_col].unique()) > 1:
                    top_category_sales = category_sales.iloc[0]
                    concentration = (top_category_sales / total_sales) * 100
                    st.metric("最大类别占比", f"{concentration:.1f}%")
                    
                    if concentration > 50:
                        st.warning("⚠️ 销售过于集中在单一类别")
                    else:
                        st.success("✅ 销售分布相对均衡")

def report_generation_section(df: pd.DataFrame):
    """报告生成功能"""
    st.markdown('<h2 class="sub-header">报告生成</h2>', unsafe_allow_html=True)
    
    report_type = st.selectbox(
        "选择报告类型",
        ["数据摘要报告", "分析报告", "财务报告", "自定义报告"]
    )
    
    if report_type == "数据摘要报告":
        st.markdown("### 数据摘要报告")
        
        if st.button("生成数据摘要报告"):
            # 基本信息
            st.markdown("#### 数据基本信息")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总行数", len(df))
            with col2:
                st.metric("总列数", len(df.columns))
            with col3:
                st.metric("内存使用", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")
            with col4:
                st.metric("缺失值", df.isnull().sum().sum())
            
            # 数据类型分析
            st.markdown("#### 数据类型分析")
            dtype_counts = df.dtypes.value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(dtype_counts.to_frame('数量'))
            with col2:
                fig = px.pie(values=dtype_counts.values, names=dtype_counts.index, title="数据类型分布")
                st.plotly_chart(fig, use_container_width=True)
            
            # 数值列统计
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.markdown("#### 数值列统计摘要")
                st.dataframe(df[numeric_cols].describe())
            
            # 缺失值分析
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            
            if not missing_data.empty:
                st.markdown("#### ⚠️ 缺失值分析")
                missing_df = pd.DataFrame({
                    '列名': missing_data.index,
                    '缺失数量': missing_data.values,
                    '缺失比例(%)': (missing_data.values / len(df)) * 100
                })
                st.dataframe(missing_df)
            
            # 生成报告总结
            st.markdown("#### 报告总结")
            
            summary_text = f"""
            **数据摘要报告**
            
            - 数据集包含 {len(df)} 行和 {len(df.columns)} 列
            - 数据类型分布：{dict(dtype_counts)}
            - 总缺失值：{df.isnull().sum().sum()} 个
            - 内存使用：{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB
            
            **数据质量评估：**
            """
            
            missing_rate = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_rate < 5:
                summary_text += "\n- ✅ 数据质量良好，缺失值比例较低"
            elif missing_rate < 15:
                summary_text += "\n- ⚠️ 数据质量中等，建议处理缺失值"
            else:
                summary_text += "\n- ❌ 数据质量较差，需要重点处理缺失值"
            
            st.markdown(summary_text)
    
    elif report_type == "分析报告":
        st.markdown("### 分析报告")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ 没有数值列可用于分析")
            return
        
        analysis_col = st.selectbox("选择分析列", numeric_cols)
        
        if st.button("生成分析报告"):
            data = df[analysis_col].dropna()
            
            st.markdown(f"#### {analysis_col} 分析报告")
            
            # 描述性统计
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("平均值", f"{data.mean():.2f}")
            with col2:
                st.metric("中位数", f"{data.median():.2f}")
            with col3:
                st.metric("标准差", f"{data.std():.2f}")
            with col4:
                st.metric("变异系数", f"{data.std()/data.mean():.2f}")
            
            # 分布图
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(df, x=analysis_col, title=f"{analysis_col} 分布")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_box = px.box(df, y=analysis_col, title=f"{analysis_col} 箱线图")
                st.plotly_chart(fig_box, use_container_width=True)
            
            # 异常值检测
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            st.markdown("#### 异常值分析")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("异常值数量", len(outliers))
            with col2:
                st.metric("异常值比例", f"{len(outliers)/len(data)*100:.1f}%")
            with col3:
                st.metric("异常值范围", f"<{lower_bound:.1f} 或 >{upper_bound:.1f}")
            
            # 分析结论
            st.markdown("#### 分析结论")
            
            conclusion = f"""
            **{analysis_col} 数据分析结论：**
            
            1. **中心趋势**：平均值为 {data.mean():.2f}，中位数为 {data.median():.2f}
            2. **离散程度**：标准差为 {data.std():.2f}，变异系数为 {data.std()/data.mean():.2f}
            3. **数据分布**：最小值 {data.min():.2f}，最大值 {data.max():.2f}
            4. **异常值情况**：检测到 {len(outliers)} 个异常值，占比 {len(outliers)/len(data)*100:.1f}%
            """
            
            if data.std()/data.mean() < 0.1:
                conclusion += "\n5. **稳定性**：✅ 数据变异较小，相对稳定"
            elif data.std()/data.mean() < 0.3:
                conclusion += "\n5. **稳定性**：⚠️ 数据变异中等"
            else:
                conclusion += "\n5. **稳定性**：❌ 数据变异较大，波动明显"
            
            st.markdown(conclusion)
    
    elif report_type == "自定义报告":
        st.markdown("### 自定义报告")
        
        st.info("💡 创建您自己的数据报告")
        
        # 报告配置
        report_title = st.text_input("报告标题", "数据分析报告")
        
        # 选择要包含的部分
        st.markdown("#### 选择报告内容")
        
        include_summary = st.checkbox("包含数据摘要", True)
        include_charts = st.checkbox("包含图表分析", True)
        include_statistics = st.checkbox("包含统计分析", True)
        
        if include_charts:
            chart_cols = st.multiselect("选择图表分析的列", df.columns.tolist())
        
        if st.button("生成自定义报告"):
            st.markdown(f"# {report_title}")
            st.markdown(f"**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if include_summary:
                st.markdown("## 数据摘要")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总行数", len(df))
                with col2:
                    st.metric("总列数", len(df.columns))
                with col3:
                    st.metric("缺失值", df.isnull().sum().sum())
            
            if include_statistics:
                st.markdown("## 统计分析")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.dataframe(df[numeric_cols].describe())
                else:
                    st.warning("没有数值列可用于统计分析")
            
            if include_charts and chart_cols:
                st.markdown("## 图表分析")
                
                for col in chart_cols:
                    if df[col].dtype in ['object', 'category']:
                        # 分类数据 - 柱状图
                        value_counts = df[col].value_counts().head(10)
                        fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"{col} 分布")
                        st.plotly_chart(fig, use_container_width=True)
                    elif np.issubdtype(df[col].dtype, np.number):
                        # 数值数据 - 直方图
                        fig = px.histogram(df, x=col, title=f"{col} 分布")
                        st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ 自定义报告生成完成！")