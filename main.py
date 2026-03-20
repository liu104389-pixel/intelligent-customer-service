import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
print("API Key from env:", os.getenv("DASHSCOPE_API_KEY"))
# ========== 1. 初始化千问大模型 ==========
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
llm = ChatOpenAI(
    model="qwen-max",
    api_key="sk-71de30a8def44b68a50e2f9236deb674",  # 直接粘贴
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.5,
    max_tokens=2000,
)

# ========== 2. 初始化 Embedding 模型并加载向量库 ==========
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

vector_store = Chroma(
    persist_directory="chroma_product_index",  # 确保路径正确
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # 返回前3个最相关片段

# ========== 3. 定义提示词模板（带 context 占位符） ==========
system_prompt = """你是一家智能家居公司的客服助手。请根据以下【参考资料】来回答客户的问题。
如果参考资料中有相关信息，请基于它们给出准确、友好的回答。
如果参考资料中没有相关信息，请礼貌地告诉用户“我暂时无法从手册中找到这个问题的答案，建议咨询人工客服”。

【回复规范】
1. 语气友好专业，使用"您好"开头
2. 回答简洁准确
3. 千万不要编造参考资料中没有的信息
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt + "\n\n【参考资料】\n{context}"),  # 将 context 合并到 system 消息中
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])
# ========== 4. 构建带检索的 RAG Chain ==========
def retrieve_docs(question):
    """根据用户问题检索相关文档片段，返回拼接后的文本"""
    docs = retriever.invoke(question)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# 使用 LCEL 创建 chain：先检索 context，再调用 prompt 和 llm
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

rag_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | RunnableLambda(retrieve_docs)
    )
    | prompt
    | llm
)

# ========== 5. 添加对话记忆 ==========
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

# ========== 6. 对话函数 ==========
def chat_with_agent(question, session_id="default"):
    response = chain_with_history.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content

# ========== 7. 启动交互式测试 ==========
import gradio as gr


def respond(message, chat_history):
    response = chat_with_agent(message, session_id="gradio_user")
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    return "", chat_history


with gr.Blocks(title="智享家智能客服") as demo:  # 注意这里没有 theme
    gr.Markdown("# 🤖 智享家智能客服")
    gr.Markdown("我是您的智能家居助手，可以回答关于产品功能、价格、售后等问题。")

    chatbot = gr.Chatbot(label="对话历史")  # 去掉 type 参数
    msg = gr.Textbox(label="输入消息", placeholder="例如：智能音箱有什么功能？", lines=2)

    with gr.Row():
        submit = gr.Button("发送", variant="primary")
        clear = gr.ClearButton([msg, chatbot], value="清除对话")

    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())