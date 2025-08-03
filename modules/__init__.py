# Excel智能分析助手模块包

# 导入配置模块
from .config import (
    setup_page_config,
    load_custom_css,
    get_default_models,
    load_config,
    save_config
)

# 导入Excel代理模块
from .excel_agent import ExcelAgentFull

# 导入核心分析模块
from .core_analysis import (
    ai_analysis_section,
    natural_language_section,
    chart_generation_section,
    machine_learning_section
)

# 导入数据处理模块
from .data_processing import (
    data_analysis_section,
    data_cleaning_section,
    statistical_analysis_section,
    advanced_data_processing_section,
    data_comparison_section,
    data_import_export_section,
    data_validation_section
)

# 导入业务功能模块
from .business_features import (
    table_operations_section,
    formula_calculator_section,
    financial_analysis_section,
    time_series_analysis_section,
    goal_tracking_section,
    dashboard_creation_section,
    report_generation_section
)

# 导入可视化模块
from .visualization import (
    conditional_formatting_section,
    worksheet_management_section
)

# 导入安全隐私模块
from .security_privacy import (
    data_security_section,
    mathematical_functions_section
)

# 模块版本信息
__version__ = "1.0.0"
__author__ = "Excel智能助手"

# 导出所有功能函数
__all__ = [
    # 配置相关
    'setup_page_config',
    'load_custom_css', 
    'get_default_models',
    'load_config',
    'save_config',
    
    # Excel代理
    'ExcelAgentFull',
    
    # 核心分析功能
    'ai_analysis_section',
    'natural_language_section',
    'chart_generation_section',
    'machine_learning_section',
    
    # 数据处理功能
    'data_analysis_section',
    'data_cleaning_section',
    'statistical_analysis_section', 
    'advanced_data_processing_section',
    'data_comparison_section',
    'data_import_export_section',
    'data_validation_section',
    
    # 业务功能
    'table_operations_section',
    'formula_calculator_section',
    'financial_analysis_section',
    'time_series_analysis_section',
    'goal_tracking_section',
    'dashboard_creation_section',
    'report_generation_section',
    
    # 可视化功能
    'conditional_formatting_section',
    'worksheet_management_section',
    
    # 安全隐私功能
    'data_security_section',
    'mathematical_functions_section'
]

# 功能模块映射
FUNCTION_MODULES = {
    # 核心分析功能
    "AI智能分析": ai_analysis_section,
    "自然语言查询": natural_language_section,
    "图表生成": chart_generation_section,
    "机器学习预测": machine_learning_section,
    
    # 数据处理功能
    "数据分析": data_analysis_section,
    "数据清洗": data_cleaning_section,
    "统计分析": statistical_analysis_section,
    "高级数据处理": advanced_data_processing_section,
    "数据对比": data_comparison_section,
    "数据导入导出": data_import_export_section,
    "数据验证": data_validation_section,
    
    # 业务应用功能
    "表格操作": table_operations_section,
    "公式计算器": formula_calculator_section,
    "财务分析": financial_analysis_section,
    "时间序列分析": time_series_analysis_section,
    "目标跟踪": goal_tracking_section,
    "仪表板创建": dashboard_creation_section,
    "报告生成": report_generation_section,
    
    # 可视化和界面功能
    "条件格式化": conditional_formatting_section,
    "工作表管理": worksheet_management_section,
    
    # 安全和隐私功能
    "数据安全": data_security_section,
    "数学统计函数": mathematical_functions_section
}

# 功能分类
FUNCTION_CATEGORIES = {
    "核心分析功能": [
        "AI智能分析",
        "自然语言查询", 
        "图表生成",
        "机器学习预测"
    ],
    "数据处理功能": [
        "数据分析",
        "数据清洗",
        "统计分析",
        "高级数据处理",
        "数据对比",
        "数据导入导出",
        "数据验证"
    ],
    "业务应用功能": [
        "表格操作",
        "公式计算器",
        "财务分析",
        "时间序列分析",
        "目标跟踪",
        "仪表板创建",
        "报告生成"
    ],
    "可视化和界面功能": [
        "条件格式化",
        "工作表管理"
    ],
    "安全和隐私功能": [
        "数据安全",
        "数学统计函数"
    ]
}

def get_function_by_name(function_name: str):
    """根据功能名称获取对应的函数"""
    return FUNCTION_MODULES.get(function_name)

def get_functions_by_category(category: str):
    """根据分类获取功能列表"""
    return FUNCTION_CATEGORIES.get(category, [])

def get_all_functions():
    """获取所有功能名称列表"""
    return list(FUNCTION_MODULES.keys())

def get_all_categories():
    """获取所有分类名称列表"""
    return list(FUNCTION_CATEGORIES.keys())