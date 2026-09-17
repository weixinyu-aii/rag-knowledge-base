# RAG Knowledge Base

面向真实项目实践的混合检索知识库。支持文档导入、中文分块、向量检索、BM25、RRF 融合、可选重排、带引用问答、会话持久化、REST API、Streamlit UI 和离线评测。

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![RAG](https://img.shields.io/badge/RAG-Hybrid-success)
![API](https://img.shields.io/badge/API-FastAPI-009688)
![License](https://img.shields.io/badge/License-MIT-yellow)

> 这是一个面向 GitHub 展示和面试讲解的完整实现，不是简单的“上传文档后调用大模型”Demo。项目重点在检索链路、可追溯引用、评测和工程化。

## 功能特性

- PDF、DOCX、Markdown、TXT、CSV 文档解析
- 中文友好的递归长度分块和重叠窗口
- BGE 向量检索
- 无第三方分词依赖的 BM25 关键词检索
- Reciprocal Rank Fusion 混合排名
- 可选 BGE CrossEncoder 重排
- 元数据过滤，可用于来源、部门、租户等过滤场景
- 多轮问题重写和 SQLite 会话持久化
- 答案引用来源、页码和片段
- 大模型异常时自动降级返回检索原文
- FastAPI 文档、上传、删除、重建、检索、问答和会话接口
- Streamlit 可交互演示界面
- 检索评测指标和可复现评测数据格式
- Docker、Docker Compose、GitHub Actions 和单元测试

## 架构

~~~mermaid
flowchart LR
    A[PDF / DOCX / MD / TXT / CSV] --> B[Document Parser]
    B --> C[Chunker + Metadata]
    C --> D[BGE Embedding]
    D --> E[FAISS Vector Index]
    C --> F[BM25 Index]
    Q[User Question] --> V[Vector Search]
    Q --> L[BM25 Search]
    E --> V
    F --> L
    V --> R[RRF Fusion]
    L --> R
    R --> X[Optional Reranker]
    X --> P[Prompt Builder]
    P --> LLM[Qwen via DashScope]
    LLM --> ANS[Answer + Citations]
    ANS --> S[(SQLite Sessions)]
~~~

详细设计见 [docs/架构说明.md](docs/架构说明.md)。

## 项目结构

~~~text
.
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── docs/
├── examples/
│   ├── documents/
│   └── evaluation/
├── scripts/
├── src/rag_knowledge_base/
│   ├── api.py
│   ├── chunking.py
│   ├── cli.py
│   ├── config.py
│   ├── documents.py
│   ├── evaluation.py
│   ├── index.py
│   ├── manifest.py
│   ├── providers.py
│   ├── retrieval.py
│   ├── service.py
│   └── sessions.py
├── tests/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
~~~

## 快速开始

### 1. 创建环境

~~~bash
git clone https://github.com/OWNER/REPOSITORY.git
cd REPOSITORY
python -m venv .venv
~~~

Windows：

~~~powershell
.venv\Scripts\activate
~~~

Linux / macOS：

~~~bash
source .venv/bin/activate
~~~

安装完整依赖：

~~~bash
python -m pip install --upgrade pip
python -m pip install -e ".[rag,api,ui]"
~~~

### 2. 配置环境变量

~~~bash
copy .env.example .env
~~~

Linux / macOS 使用：

~~~bash
cp .env.example .env
~~~

至少填写：

~~~env
DASHSCOPE_API_KEY=sk-your-dashscope-key
~~~

如果暂时没有 API Key，仍然可以导入文档并执行向量、BM25 和混合检索；问答接口会返回检索降级结果。

### 3. 导入示例文档

~~~bash
python -m rag_knowledge_base ingest examples/documents/sample-handbook.md
~~~

### 4. 命令行问答

~~~bash
python -m rag_knowledge_base ask "差旅报销需要哪些材料？"
~~~

输出示例：

~~~text
差旅报销需要提交经审批的出差申请单、完整行程单和合法有效发票，超过 2000 元还需要费用明细。[1]

Sources:
[1] sample-handbook.md
~~~

### 5. 启动 API

~~~bash
python -m rag_knowledge_base serve --reload
~~~

打开 http://localhost:8000/docs 查看交互式 API 文档。

### 6. 启动界面

~~~bash
python -m rag_knowledge_base ui
~~~

也可以直接运行规范入口：

~~~bash
streamlit run streamlit_app.py
~~~

打开 http://localhost:8501。

## API 示例

上传文档：

~~~bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@examples/documents/sample-handbook.md"
~~~

提问：

~~~bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"差旅报销需要哪些材料？","session_id":"demo"}'
~~~

仅执行检索：

~~~bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"question":"年假多少天？","top_k":5}'
~~~

主要接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | /health | 服务、索引和模型配置状态 |
| GET | /api/v1/documents | 文档列表 |
| POST | /api/v1/documents | 上传并建立索引 |
| DELETE | /api/v1/documents/{doc_id} | 删除文档并重建索引 |
| POST | /api/v1/index/rebuild | 从原始文档重建全部索引 |
| POST | /api/v1/search | 只执行检索，不调用大模型 |
| POST | /api/v1/query | 多轮问答并返回引用 |
| GET | /api/v1/sessions/{session_id} | 获取会话历史 |
| DELETE | /api/v1/sessions/{session_id} | 清空会话历史 |

## 配置项

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| DASHSCOPE_API_KEY | 无 | 通义千问 API Key |
| RKB_LLM_MODEL | qwen-turbo | 对话模型 |
| RKB_EMBEDDING_MODEL | D:/langchain01/models/bge-small-zh-v1.5 | 嵌入模型本地路径 |
| RKB_DEVICE | cpu | cpu、cuda、mps |
| RKB_ALLOW_MODEL_DOWNLOAD | false | 是否允许联网下载模型 |
| RKB_DATA_DIR | ./data | 运行数据目录 |
| RKB_CHUNK_SIZE | 800 | 文本块长度 |
| RKB_CHUNK_OVERLAP | 150 | 文本块重叠长度 |
| RKB_RETRIEVER_K | 5 | 默认返回数量 |
| RKB_CANDIDATE_K | 20 | 融合和重排候选数量 |
| RKB_RETRIEVAL_MODE | hybrid | vector、bm25 或 hybrid |
| RKB_RRF_K | 60 | RRF 平滑参数 |
| RKB_RERANKER_ENABLED | false | 是否启用 CrossEncoder |
| RKB_RERANKER_MODEL | D:/langchain01/models/bge-reranker-base | 重排模型本地路径 |
| RKB_MAX_UPLOAD_MB | 20 | 单文件大小限制 |
| RKB_API_KEY | 空 | 可选服务鉴权 Key |
| RKB_API_URL | 空 | Streamlit 远程 API 地址；设置后不再直接访问本地索引 |

项目默认只使用本地模型，禁止联网下载：

~~~env
RKB_EMBEDDING_MODEL=D:/langchain01/models/bge-small-zh-v1.5
RKB_ALLOW_MODEL_DOWNLOAD=false
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
~~~

如果本地模型目录不存在，导入会立即给出明确错误，不会等待网络超时。

## 检索评测

导入示例文档后运行：

~~~bash
python -m rag_knowledge_base evaluate \
  --dataset examples/evaluation/questions.jsonl \
  --k 3
~~~

输出包括：

- Hit Rate@K
- Recall@K
- MRR@K
- Precision@K
- 平均检索耗时

比较不同检索模式：

~~~bash
RKB_RETRIEVAL_MODE=vector python -m rag_knowledge_base evaluate --dataset examples/evaluation/questions.jsonl --k 5
RKB_RETRIEVAL_MODE=bm25 python -m rag_knowledge_base evaluate --dataset examples/evaluation/questions.jsonl --k 5
RKB_RETRIEVAL_MODE=hybrid python -m rag_knowledge_base evaluate --dataset examples/evaluation/questions.jsonl --k 5
~~~

更完整的评测方法见 [docs/评测说明.md](docs/评测说明.md)。

## Docker

准备 .env 后运行：

~~~bash
docker compose up --build
~~~

- API：http://localhost:8000
- API 文档：http://localhost:8000/docs
- Streamlit：http://localhost:8501

数据保存在 Docker Named Volume 中。

## 质量检查

~~~bash
python scripts/run_tests.py
python -m compileall -q src app
ruff check src tests app
~~~

当前测试覆盖分块、BM25、混合检索、元数据过滤、评测指标、清单持久化、会话存储和 API 鉴权。

## 关键设计决策

### 为什么没有只使用 LangChain

项目直接编排解析、分块、索引、检索和生成链路。这样可以明确控制元数据、引用、降级、评测和持久化，也避免核心业务被框架版本升级绑定。面试时可以逐层解释每个环节，而不是只描述“调用了某个 Chain”。

### 为什么同时使用向量和 BM25

向量检索擅长语义相似，BM25 擅长精确术语、金额、编号和专有名词。混合检索对真实企业文档更稳健。

### 为什么使用 RRF

向量相似度和 BM25 分数量纲不同，直接加权需要一个不稳定的归一化过程。RRF 只依赖排名，稳定且易于解释。

### 为什么返回结构化引用

答案文本可能被模型改写。服务层单独返回文档名、页码、文本块 ID 和分数，前端可以稳定展示来源，评测也可以验证引用。

## 当前限制

- 扫描版 PDF 暂未集成 OCR。
- SQLite 和本地 FAISS 适合单机和小规模知识库，不适合大规模并发写入。
- 删除文档后需要重建全部索引。
- CrossEncoder 默认关闭，避免首次启动下载模型。
- 当前回答生成为同步调用，尚未实现 SSE Token 流式输出。
- 没有用户、角色和租户级权限体系。

## Roadmap

- OCR 和表格结构化解析
- 增量索引与后台任务队列
- SSE 流式问答
- OpenTelemetry、Token、成本与延迟监控
- Qdrant、Milvus 或 PGVector 适配器
- 引用忠实度与幻觉评测
- 多租户和行级权限
- 人工反馈与回归数据集

## 适合写进简历的点

**知识库 RAG 问答系统｜个人项目**

- 实现向量、BM25 与 RRF 混合检索，并加入可选 CrossEncoder 重排，提升精确术语和语义问题的综合召回能力。
- 设计文档、文本块、来源、页码的元数据链路，使回答返回结构化引用并支持来源过滤。
- 实现多轮问题重写、SQLite 会话持久化和 LLM 异常降级，提升多轮问答与系统可用性。
- 建立可复现检索评测集，统计 Hit Rate、Recall、MRR、Precision 和检索延迟。
- 使用 FastAPI、Streamlit、Docker 和 GitHub Actions 完成接口、交互、部署与持续集成。

发布前请运行 [docs/发布检查清单.md](docs/发布检查清单.md)。

## 常见问题

### ImportError: attempted relative import with no known parent package

不要使用 python 直接执行包内的 Streamlit 文件。请从仓库根目录运行：

~~~bash
streamlit run streamlit_app.py
~~~

或者使用项目命令：

~~~bash
python -m rag_knowledge_base ui
~~~

### 首次启动加载嵌入模型较慢

程序只读取本地 BGE 模型。第一次加载到内存通常需要几秒到十几秒；同一个进程后续处理小文件会明显更快。

### 没有 DASHSCOPE_API_KEY

文档导入、向量检索、BM25 和混合检索仍然可以使用；问答会返回带来源的检索降级结果。

## License

MIT
