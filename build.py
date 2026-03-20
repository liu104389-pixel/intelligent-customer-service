import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # 改用 PDF 加载器
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

# 1. 加载 PDF 文件
loader = PyPDFLoader(r"D:\下载\智能家居产品手册.pdf")  # 使用原始字符串
documents = loader.load()
print(f"已加载 {len(documents)} 页文档")

# 2. 文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]
)
chunks = text_splitter.split_documents(documents)
chunks = [chunk for chunk in chunks if chunk.page_content and isinstance(chunk.page_content, str)]
print(f"切分并过滤后得到 {len(chunks)} 个文本块")

# 3. 初始化 DashScope Embeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 测试 embedding 服务
print("测试 embedding 服务...")
test_embedding = embeddings.embed_query("测试")
print(f"测试成功，embedding 维度: {len(test_embedding)}")

# 4. 创建并保存 Chroma 向量库
print("开始生成向量库...")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_product_index"
)
print("向量库已保存到 chroma_product_index 文件夹")