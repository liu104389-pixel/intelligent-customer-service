# 🤖 智能家居客服助手 - RAG 增强版

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/langchain-0.2+-green.svg)](https://langchain.com/)
[![Gradio](https://img.shields.io/badge/gradio-4.0+-orange.svg)](https://gradio.app/)

## 📌 项目简介
本项目是一个基于**通义千问大模型**和 **RAG（检索增强生成）** 技术的智能客服助手。它能够根据产品手册等私有知识库，为用户提供准确、友好的产品咨询解答。项目从低代码平台验证起步，最终使用 Python + LangChain 完整实现，并提供了 Web 界面。

## ✨ 核心功能
- **多轮对话**：支持上下文记忆，对话连贯自然。
- **私有知识问答**：基于产品手册构建向量库，回答准确率高于纯模型生成。
- **RAG 检索增强**：用户问题先检索相关文档片段，再交由大模型生成答案，有效降低幻觉。
- **Web 界面**：使用 Gradio 构建，交互友好，开箱即用。

## 🛠️ 技术栈
| 组件 | 技术 |
|------|------|
| 大模型 | 通义千问 (qwen-max) |
| 框架 | LangChain, LangChain-OpenAI |
| 向量库 | Chroma (with text-embedding-v2) |
| 后端 | Gradio |


## 📁 项目结构
.
├── main.py # 主程序（包含 RAG 逻辑和 Gradio 界面）
├── build_vector_store.py # 构建向量库脚本（只需运行一次）
├── requirements.txt # 依赖列表
├── .env.example # 环境变量示例
├── .gitignore
└── README.md


## 🚀 快速开始

### 环境要求
- Python 3.10+
- 阿里云 DashScope API Key（需自行申请）

### 安装步骤
1. **克隆仓库**
   ```bash
   git clone https://github.com/liu104389-pixel/intelligent-customer-service.git
   cd intelligent-customer-service
安装依赖
pip install -r requirements.txt
配置环境变量
DASHSCOPE_API_KEY=sk-你的真实密钥
准备知识库
python build_vector_store.py
启动客服助手
python main.py

 界面预览

 <img width="1920" height="1039" alt="image" src="https://github.com/user-attachments/assets/7bd498fd-4eaf-4b6e-b801-22040c7a2fc7" />

 待优化
接入更多文档格式（Word、HTML）

增加意图识别模块

提高检索精度

 许可证
MIT
