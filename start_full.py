#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel智能分析工具 - 完整版启动脚本
作者: AI助手
版本: 2.0.0
描述: 带有AI功能的Excel数据分析工具启动器
"""

import subprocess
import sys
import os
from pathlib import Path

def check_and_install_dependencies():
    """检查并安装依赖项"""
    print("🔍 检查依赖项...")
    
    required_packages = [
        'streamlit', 'pandas', 'plotly', 'openpyxl', 
        'scikit-learn', 'scipy', 'langchain', 'openai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\n📦 正在安装缺失的依赖项: {', '.join(missing_packages)}")
        try:
            # 尝试从requirements_full.txt安装
            requirements_file = Path(__file__).parent / "requirements_full.txt"
            if requirements_file.exists():
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ])
            else:
                # 单独安装核心包
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install"
                ] + missing_packages)
            print("✅ 依赖项安装完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖项安装失败: {e}")
            print("请手动运行: pip install -r requirements_full.txt")
            return False
    
    return True

def start_app():
    """启动Streamlit应用"""
    print("\n🚀 启动Excel智能分析工具 (完整版)...")
    
    # 获取当前脚本目录
    current_dir = Path(__file__).parent
    app_file = current_dir / "app_full.py"
    
    if not app_file.exists():
        print(f"❌ 找不到应用文件: {app_file}")
        return False
    
    try:
        # 启动Streamlit应用
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(app_file),
            "--server.headless", "true",
            "--server.port", "8502",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=current_dir)
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("="*60)
    print("📊 Excel智能分析工具 - 完整版")
    print("="*60)
    print("功能特色:")
    print("🤖 AI智能分析 - 自然语言数据查询")
    print("📈 数据可视化 - 8种交互式图表")
    print("🧹 数据处理 - 智能清洗和统计分析")
    print("🔬 高级分析 - 假设检验和回归分析")
    print("🎯 示例数据 - 内置销售和员工数据生成器")
    print("="*60)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return
    
    # 检查并安装依赖
    if not check_and_install_dependencies():
        print("\n❌ 依赖项检查失败，请手动安装依赖项")
        return
    
    # 启动应用
    if start_app():
        print("\n✅ 应用启动成功")
        print("📱 访问地址: http://localhost:8502")
        print("💡 提示: 按Ctrl+C停止应用")
    else:
        print("\n❌ 应用启动失败")

if __name__ == "__main__":
    main()