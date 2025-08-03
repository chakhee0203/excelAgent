#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例数据生成器

生成各种类型的示例Excel文件，用于测试Excel Agent的功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

def create_sales_data():
    """创建销售数据示例"""
    np.random.seed(42)
    random.seed(42)
    
    # 生成日期范围
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 产品列表
    products = ['笔记本电脑', '台式机', '显示器', '键盘', '鼠标', '耳机', '音响', '摄像头']
    regions = ['北京', '上海', '广州', '深圳', '杭州', '南京', '成都', '武汉']
    sales_reps = ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十']
    
    # 生成数据
    data = []
    for date in date_range:
        # 每天生成5-15条销售记录
        daily_records = random.randint(5, 15)
        
        for _ in range(daily_records):
            product = random.choice(products)
            region = random.choice(regions)
            sales_rep = random.choice(sales_reps)
            
            # 根据产品类型设置价格范围
            if product == '笔记本电脑':
                unit_price = random.randint(3000, 8000)
            elif product == '台式机':
                unit_price = random.randint(2000, 6000)
            elif product == '显示器':
                unit_price = random.randint(800, 3000)
            else:
                unit_price = random.randint(50, 500)
            
            quantity = random.randint(1, 10)
            total_amount = unit_price * quantity
            
            # 添加一些季节性趋势
            if date.month in [11, 12]:  # 年末促销
                total_amount *= random.uniform(1.1, 1.3)
            elif date.month in [6, 7]:  # 年中促销
                total_amount *= random.uniform(1.05, 1.2)
            
            data.append({
                '日期': date,
                '产品名称': product,
                '销售区域': region,
                '销售代表': sales_rep,
                '单价': round(unit_price, 2),
                '数量': quantity,
                '销售金额': round(total_amount, 2),
                '成本': round(total_amount * random.uniform(0.6, 0.8), 2),
                '利润': round(total_amount * random.uniform(0.2, 0.4), 2)
            })
    
    df = pd.DataFrame(data)
    return df

def create_employee_data():
    """创建员工数据示例"""
    np.random.seed(42)
    random.seed(42)
    
    departments = ['技术部', '销售部', '市场部', '人事部', '财务部', '运营部']
    positions = {
        '技术部': ['软件工程师', '高级工程师', '技术经理', '架构师'],
        '销售部': ['销售专员', '销售经理', '大客户经理', '销售总监'],
        '市场部': ['市场专员', '市场经理', '品牌经理', '市场总监'],
        '人事部': ['人事专员', '招聘经理', '培训经理', '人事总监'],
        '财务部': ['会计', '财务分析师', '财务经理', '财务总监'],
        '运营部': ['运营专员', '运营经理', '产品经理', '运营总监']
    }
    
    cities = ['北京', '上海', '广州', '深圳', '杭州', '南京', '成都', '武汉']
    education_levels = ['本科', '硕士', '博士', '专科']
    
    data = []
    employee_id = 1001
    
    for dept in departments:
        # 每个部门15-25人
        dept_size = random.randint(15, 25)
        
        for _ in range(dept_size):
            position = random.choice(positions[dept])
            city = random.choice(cities)
            education = random.choice(education_levels)
            
            # 根据职位设置薪资范围
            if '总监' in position:
                salary = random.randint(25000, 40000)
            elif '经理' in position:
                salary = random.randint(15000, 30000)
            elif '高级' in position or '架构师' in position:
                salary = random.randint(12000, 25000)
            else:
                salary = random.randint(6000, 15000)
            
            # 工作年限影响薪资
            work_years = random.randint(1, 15)
            salary += work_years * random.randint(500, 1500)
            
            # 年龄
            age = random.randint(22, 55)
            
            # 绩效评分
            performance = round(random.uniform(3.0, 5.0), 1)
            
            data.append({
                '员工编号': f'EMP{employee_id:04d}',
                '姓名': f'员工{employee_id-1000}',
                '部门': dept,
                '职位': position,
                '年龄': age,
                '工作城市': city,
                '学历': education,
                '工作年限': work_years,
                '月薪': salary,
                '绩效评分': performance,
                '入职日期': datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))
            })
            
            employee_id += 1
    
    df = pd.DataFrame(data)
    return df

def create_financial_data():
    """创建财务数据示例"""
    np.random.seed(42)
    
    # 生成月度财务数据
    months = pd.date_range(start='2022-01-01', end='2023-12-31', freq='M')
    
    data = []
    for month in months:
        # 基础收入，带有增长趋势
        base_revenue = 1000000 + (month.year - 2022) * 100000 + month.month * 50000
        revenue = base_revenue * random.uniform(0.8, 1.2)
        
        # 成本约为收入的60-70%
        cost = revenue * random.uniform(0.6, 0.7)
        
        # 运营费用
        operating_expense = revenue * random.uniform(0.15, 0.25)
        
        # 税费
        tax = (revenue - cost - operating_expense) * random.uniform(0.15, 0.25)
        
        # 净利润
        net_profit = revenue - cost - operating_expense - tax
        
        data.append({
            '年月': month.strftime('%Y-%m'),
            '营业收入': round(revenue, 2),
            '营业成本': round(cost, 2),
            '运营费用': round(operating_expense, 2),
            '税费': round(tax, 2),
            '净利润': round(net_profit, 2),
            '毛利率': round((revenue - cost) / revenue * 100, 2),
            '净利率': round(net_profit / revenue * 100, 2)
        })
    
    df = pd.DataFrame(data)
    return df

def create_inventory_data():
    """创建库存数据示例"""
    np.random.seed(42)
    random.seed(42)
    
    categories = ['电子产品', '服装', '食品', '家居用品', '图书', '运动用品']
    suppliers = ['供应商A', '供应商B', '供应商C', '供应商D', '供应商E']
    warehouses = ['北京仓库', '上海仓库', '广州仓库', '成都仓库']
    
    data = []
    product_id = 1
    
    for category in categories:
        # 每个类别20-30个产品
        products_count = random.randint(20, 30)
        
        for _ in range(products_count):
            supplier = random.choice(suppliers)
            warehouse = random.choice(warehouses)
            
            # 库存数量
            current_stock = random.randint(0, 1000)
            min_stock = random.randint(10, 100)
            max_stock = random.randint(500, 1500)
            
            # 价格
            unit_cost = random.uniform(10, 500)
            selling_price = unit_cost * random.uniform(1.2, 2.5)
            
            # 最后进货日期
            last_restock = datetime.now() - timedelta(days=random.randint(1, 90))
            
            # 库存状态
            if current_stock <= min_stock:
                status = '库存不足'
            elif current_stock >= max_stock:
                status = '库存过多'
            else:
                status = '正常'
            
            data.append({
                '产品编号': f'P{product_id:04d}',
                '产品名称': f'{category}产品{product_id}',
                '产品类别': category,
                '供应商': supplier,
                '仓库位置': warehouse,
                '当前库存': current_stock,
                '最小库存': min_stock,
                '最大库存': max_stock,
                '单位成本': round(unit_cost, 2),
                '销售价格': round(selling_price, 2),
                '库存价值': round(current_stock * unit_cost, 2),
                '最后进货日期': last_restock,
                '库存状态': status
            })
            
            product_id += 1
    
    df = pd.DataFrame(data)
    return df

def create_customer_data():
    """创建客户数据示例"""
    np.random.seed(42)
    random.seed(42)
    
    customer_types = ['个人客户', '企业客户', 'VIP客户']
    cities = ['北京', '上海', '广州', '深圳', '杭州', '南京', '成都', '武汉', '西安', '重庆']
    industries = ['科技', '金融', '制造', '零售', '教育', '医疗', '房地产', '物流']
    
    data = []
    customer_id = 10001
    
    for _ in range(500):  # 生成500个客户
        customer_type = random.choice(customer_types)
        city = random.choice(cities)
        
        if customer_type == '企业客户':
            industry = random.choice(industries)
            company_size = random.choice(['小型', '中型', '大型'])
        else:
            industry = None
            company_size = None
        
        # 注册日期
        register_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))
        
        # 最后购买日期
        last_purchase = register_date + timedelta(days=random.randint(0, 365))
        
        # 购买次数和金额
        purchase_count = random.randint(1, 50)
        total_amount = random.uniform(1000, 100000)
        
        if customer_type == 'VIP客户':
            total_amount *= random.uniform(2, 5)
            purchase_count *= random.randint(2, 4)
        
        # 客户状态
        days_since_last_purchase = (datetime.now() - last_purchase).days
        if days_since_last_purchase > 180:
            status = '流失客户'
        elif days_since_last_purchase > 90:
            status = '风险客户'
        else:
            status = '活跃客户'
        
        data.append({
            '客户编号': f'C{customer_id:05d}',
            '客户名称': f'客户{customer_id-10000}',
            '客户类型': customer_type,
            '所在城市': city,
            '行业': industry,
            '公司规模': company_size,
            '注册日期': register_date,
            '最后购买日期': last_purchase,
            '购买次数': purchase_count,
            '累计消费金额': round(total_amount, 2),
            '平均订单金额': round(total_amount / purchase_count, 2),
            '客户状态': status
        })
        
        customer_id += 1
    
    df = pd.DataFrame(data)
    return df

def main():
    """主函数 - 生成所有示例数据"""
    print("🎯 Excel示例数据生成器")
    print("=" * 30)
    
    # 创建示例数据目录
    sample_dir = Path('sample_data')
    sample_dir.mkdir(exist_ok=True)
    
    datasets = {
        '销售数据.xlsx': create_sales_data,
        '员工数据.xlsx': create_employee_data,
        '财务数据.xlsx': create_financial_data,
        '库存数据.xlsx': create_inventory_data,
        '客户数据.xlsx': create_customer_data
    }
    
    for filename, create_func in datasets.items():
        try:
            print(f"📊 正在生成 {filename}...")
            df = create_func()
            filepath = sample_dir / filename
            df.to_excel(filepath, index=False)
            print(f"✅ {filename} 生成完成 ({len(df)} 行数据)")
        except Exception as e:
            print(f"❌ {filename} 生成失败: {e}")
    
    print("\n🎉 所有示例数据生成完成！")
    print(f"📁 文件保存在: {sample_dir.absolute()}")
    print("\n💡 使用说明:")
    print("1. 启动Excel Agent应用")
    print("2. 上传sample_data目录中的任意Excel文件")
    print("3. 体验各种分析功能")
    
    # 显示数据集信息
    print("\n📋 数据集说明:")
    descriptions = {
        '销售数据.xlsx': '包含产品销售记录，适合练习销售分析、趋势预测',
        '员工数据.xlsx': '包含员工信息，适合练习人力资源分析、薪资分析',
        '财务数据.xlsx': '包含财务报表数据，适合练习财务分析、盈利分析',
        '库存数据.xlsx': '包含库存管理数据，适合练习库存分析、供应链分析',
        '客户数据.xlsx': '包含客户信息，适合练习客户分析、客户细分'
    }
    
    for filename, description in descriptions.items():
        print(f"• {filename}: {description}")

if __name__ == "__main__":
    main()