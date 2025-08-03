#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excel智能分析助手启动脚本

这个脚本提供了一个简单的方式来启动Excel Agent应用
包含环境检查、依赖验证和应用启动功能
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误：需要Python 3.8或更高版本")
        print(f"当前版本：{sys.version}")
        return False
    print(f"✅ Python版本检查通过：{sys.version.split()[0]}")
    return True

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'streamlit',
        'pandas', 
        'langchain',
        'openai',
        'plotly',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
        else:
            print(f"✅ {package} 已安装")
    
    if missing_packages:
        print(f"❌ 缺少以下依赖包：{', '.join(missing_packages)}")
        print("请运行以下命令安装依赖：")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包检查通过")
    return True

def check_env_file():
    """检查环境配置文件"""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists():
        if env_example.exists():
            print("⚠️  未找到.env文件，但找到了.env.example")
            print("请复制.env.example为.env并配置你的API密钥")
            print("命令：cp .env.example .env")
        else:
            print("❌ 未找到环境配置文件")
        return False
    
    # 检查是否配置了API密钥
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'your_openai_api_key_here' in content or 'OPENAI_API_KEY=' not in content:
                print("⚠️  请在.env文件中配置你的OpenAI API密钥")
                return False
    except Exception as e:
        print(f"❌ 读取.env文件失败：{e}")
        return False
    
    print("✅ 环境配置文件检查通过")
    return True

def start_streamlit_app():
    """启动Streamlit应用"""
    try:
        print("🚀 正在启动Excel智能分析助手...")
        print("应用将在浏览器中自动打开")
        print("如果没有自动打开，请访问：http://localhost:8501")
        print("按 Ctrl+C 停止应用")
        print("-" * 50)
        
        # 启动streamlit应用
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败：{e}")
    except Exception as e:
        print(f"❌ 未知错误：{e}")

def show_help():
    """显示帮助信息"""
    help_text = """
🔧 Excel智能分析助手 - 启动脚本

用法：
  python run.py [选项]

选项：
  --help, -h     显示此帮助信息
  --check, -c    仅检查环境，不启动应用
  --install, -i  安装依赖包

示例：
  python run.py          # 启动应用
  python run.py --check  # 检查环境
  python run.py --install # 安装依赖

首次使用：
1. 确保已安装Python 3.8+
2. 运行：python run.py --install
3. 复制.env.example为.env并配置API密钥
4. 运行：python run.py

需要帮助？请查看README.md文件
    """
    print(help_text)

def install_dependencies():
    """安装依赖包"""
    try:
        print("📦 正在安装依赖包...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✅ 依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败：{e}")
        return False
    except Exception as e:
        print(f"❌ 安装过程中出现错误：{e}")
        return False

def main():
    """主函数"""
    print("🎯 Excel智能分析助手")
    print("=" * 30)
    
    # 解析命令行参数
    args = sys.argv[1:]
    
    if '--help' in args or '-h' in args:
        show_help()
        return
    
    if '--install' in args or '-i' in args:
        if not check_python_version():
            return
        install_dependencies()
        return
    
    # 环境检查
    print("🔍 正在检查运行环境...")
    
    if not check_python_version():
        return
    
    if not check_dependencies():
        print("\n💡 提示：运行 'python run.py --install' 来安装依赖")
        return
    
    if not check_env_file():
        print("\n💡 提示：请配置.env文件后再启动应用")
        return
    
    if '--check' in args or '-c' in args:
        print("\n✅ 环境检查完成，一切正常！")
        return
    
    print("\n✅ 环境检查完成")
    print("=" * 30)
    
    # 启动应用
    start_streamlit_app()

if __name__ == "__main__":
    main()