# RAG Agent Service

面向扫地机器人 / 扫拖一体机器人场景的 RAG + Agent 智能客服服务。基于 LangChain 1.x `create_agent`、LangGraph、Chroma 与阿里云 DashScope，具备检索增强问答、ReAct 工具调用、**向量 + BM25 混合检索**、**答案出处标注**、场景化提示词切换与 SSE 流式输出能力。

**技术栈**：Python 3.13 · FastAPI · LangChain 1.5 · LangGraph · Chroma · jieba + rank_bm25 · DashScope（qwen + text-embedding-v4）· Uvicorn

**效果**：自建 80 题评估集（68 正 / 12 负），混合检索 + 精排 hit@3 **68/68**（MRR 0.985；仅向量 76.5%），精排前的纯混合检索为 91.2%。详见 [`eval/report.md`](eval/report.md)。



***

## 架构

### 请求流程



```
用户输入
   │
   ├── CLI（main.py --mode cli）──────────────┐
   └── HTTP（api/server.py，--mode api）──────┤
                                              ▼
                              Agent（create_agent + 中间件）
                                 │  ① 带系统提示词与工具清单调用模型
                                 │  ② 模型决定调哪个工具
                                 │  ③ 工具结果回填 → 再次调用模型
                                 │  ④ 不再请求工具 → 输出回答
                                 ▼
                     最终回答（一次性 / SSE 流式）
```

### 建库流程（启动时执行一次）



```
data/*.pdf|txt
   │
   ├─ DocumentLoader   读取 → 13 个 Document（PDF 按页拆）
   ├─ Chunker          按编号条目切分 + 打包 → 194 个 chunk（每条规则不被腰斩）
   ├─ ChromaStore      按 chunk MD5 去重（已入库的跳过）
   └─ Embedding        DashScope 向量化 → chroma_db/，MD5 记入 md5.text
```

### 分层依赖（单向）



```
utils / config  ←  document_loader / chunker / embedding / vector_store / rag_summarize / agent / api  ←  main.py
```

`main.py` 是唯一读配置、建实例、接线的地方（组装点）；下层模块只接收传进来的对象，不自己读配置、不自己建实例。



***

## 目录结构



```
rag_service/
├── agent/
│   ├── graph.py            # create_agent 组装 + run_agent / run_agent_stream
│   ├── middleware.py       # monitor_tool / log_before_model / build_prompt_switch
│   └── tools.py            # 7 个工具定义
├── api/
│   └── server.py           # FastAPI：/health、/chat、/chat/stream、/feedback、/stats
├── storage/                # SQLite 调用留痕与反馈（qa_log / feedback）
├── chunker/
│   ├── text_splitter.py       # build_splitter / split_documents（字符硬切，fixed 模式）
│   └── structural_splitter.py # 条目对齐切分 + 打包（structural 模式，默认）
├── config/
│   ├── __init__.py         # BASE_DIR
│   ├── rag.yml             # 模型相关
│   ├── chroma.yml          # 知识库与切分相关
│   ├── prompts.yml         # 提示词路径
│   └── agent.yml           # 外部数据路径
├── data/                   # 原始知识库（pdf / txt）+ external/records.csv
├── document_loader/
│   └── loader.py           # DocumentLoader：扫描目录、按类型加载、打 source 标签
├── embedding/
│   └── embeddings.py       # build_embedding_model（DashScope）
├── eval/                   # 检索效果评估（离线，不依赖服务）
│   ├── questions.json      # 80 条评估样本（68 正 + 12 负）
│   ├── check_questions.py  # 题库自检（关键词存在性 + 区分度），输出核对表
│   ├── questions_review.md # 自检生成的核对表
│   ├── run_eval.py         # hit@k / MRR / 来源命中 / 耗时，支持 --mode vector|hybrid
│   └── report.md           # 评估报告（对比两种检索策略）
├── prompts/                # main_prompt.txt / rag_summarize.txt / rag_summarize_cited.txt / report_prompt.txt
├── rag_summarize/
│   └── rag_summarize.py    # RagSummarizeService：检索 + 拼 context + LLM 总结（含出处）
├── retrieval/
│   └── hybrid_retriever.py # HybridRetriever：向量 + BM25 双路召回，RRF 融合
├── utils/                  # config_handler / logger_handler / path_tool / prompt_loader
├── vector_store/
│   └── chroma_store.py     # ChromaStore：初始化、MD5 去重入库、获取 retriever
├── chroma_db/              # 向量库持久化目录（自动生成）
├── logs/app.log            # 运行日志（自动生成）
├── md5.text                # 已入库 chunk 的 MD5 清单（自动生成）
├── main.py                 # 入口：CLI / API
└── requirements.txt
```



***

## 快速开始

### 方式一：Docker（推荐，不用装 Python 环境）

```bash
git clone https://github.com/zhong-suyu/rag-service.git
cd rag-service
cp .env.example .env      # 填入两个 API Key（值相同）
docker compose up -d --build
```

打开 `http://localhost:8000/docs` 验证。

- `.env` 不进镜像，由 compose 在运行时注入
- `chroma_db` 随镜像打包，容器启动即有 194 块（若你 clone 到的仓库里没有 `chroma_db`，首次启动会自动建库，3~5 分钟 + 一次 embedding token）
- 日志写在宿主机 `logs/`（已挂卷）

常用命令：

```bash
docker compose ps                  # 看状态（healthy 才算起好）
docker compose logs -f api         # 看日志
docker compose down                # 停掉并删容器
```

### 方式二：本地 Python 环境

#### 1. 环境



```
cd D:\Python-Project\rag_service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 2. 配置 API Key

对话模型与向量化模型都走阿里云百炼（DashScope），用的是**同一个 key**，但两个客户端读的环境变量名不同，需要都配置：

```
# 向量化（DashScopeEmbeddings）读这个
$env:DASHSCOPE_API_KEY = "sk-xxxx"

# 对话模型（ChatOpenAI）读这个，值和上面相同
$env:OPENAI_API_KEY = "sk-xxxx"
```

也可以在项目根目录建 `.env` 文件（`main.py` 会自动加载），写上面两行即可。

说明：`ChatOpenAI` 是 OpenAI SDK 的封装，默认读 `OPENAI_API_KEY`；本项目未显式传 `api_key`，所以要配它，否则对话模型会 401。

#### 3. 启动



```
# CLI 交互模式（流式打印）
python main.py

# API 服务模式
python main.py --mode api --host 0.0.0.0 --port 8000
```

启动成功会依次打印：`RAG Agent 服务启动` → 文档加载与切分 → `No new documents to add`（首次会是 `Adding 194 new chunks`）→ `Agent 图构建完成` → `Uvicorn running on ...`。

首次启动会向量化 194 个 chunk（耗时、消耗 token）；之后启动命中 MD5 去重，秒级完成。



***

## 接口说明

服务启动后，Swagger 文档：`http://127.0.0.1:8000/docs`

### `GET /health`



```
{"status": "ok"}
```

### `POST /chat` —— 一次性返回

请求：



```
{"message": "拖布发臭怎么处理？"}
```

响应（等待全部工具调用完成后返回，通常 30\~170 秒，取决于答案长度）：



```
{"reply": "拖布发臭主要来自……", "trace_id": "293a51e866ef", "latency_ms": 169904}
```

> `trace_id` 是本次请求的短 ID，用于上报反馈；不是用户标识，不含个人信息。

### `POST /chat/stream` —— SSE 流式

请求体同上。响应为 `text/event-stream`，**首帧就给 `trace_id`**，方便前端在流结束后上报反馈：



```
data: {"trace_id": "293a51e866ef"}
data: {"delta": "拖布"}
data: {"delta": "发臭"}
...
data: {"done": true, "latency_ms": 51234}
data: [DONE]
```

出错时会推送 `data: {"error": "..."}` 后以 `[DONE]` 结束。

### `POST /feedback` —— 点赞 / 点踩

```
{"trace_id": "293a51e866ef", "rating": -1, "comment": "没给具体机型"}
```

`rating` 只接受 `1`（好评）或 `-1`（点踩）；`trace_id` 不存在返回 404。返回 `{"ok": true, ...}`。

### `GET /stats` —— 反馈统计

```
{
  "total": 12,
  "feedback_count": 5,
  "good_rate": 0.8,
  "reject_rate": 0.167,
  "avg_latency_ms": 45320.5,
  "p95_latency_ms": 120341.0,
  "recent_bad_cases": [{"trace_id": "...", "question": "...", "comment": "..."}]
}
```

`good_rate` 的分母是**收到反馈的请求数**（不是总请求数），否则没人点赞时会显示 0%，没有意义。

### 反馈闭环怎么跑

```powershell
# 1. 问一个问题，拿到 trace_id
curl.exe -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" `
  -d '{\"message\":\"地毯需要多大吸力\"}'

# 2. 对这次回答点踩（rating=-1）
curl.exe -X POST http://127.0.0.1:8000/feedback -H "Content-Type: application/json" `
  -d '{\"trace_id\":\"<上一步的 trace_id>\",\"rating\":-1,\"comment\":\"没给具体机型\"}'

# 3. 看统计
curl.exe http://127.0.0.1:8000/stats
```

**没有真实用户怎么办**：`eval/replay_feedback.py` 把评估集回放成线上请求，按规则自动点赞/点踩，跑通 `trace_id → qa_log → feedback → /stats → 归因` 整条链路。

```powershell
.venv\Scripts\python.exe main.py --mode api --port 8000     # 另开一个窗口
.venv\Scripts\python.exe eval\replay_feedback.py --limit 4  # 先小批量验证
```

实测输出（4 题回放 + 1 次手动点踩）：

```
  [1/4] q01 👍 家用扫地机器人建议选多少 Pa 以上吸力？
  [2/4] q02 👍 地毯场景需要多大吸力？
  [3/4] q03 👍 带宠物的家庭优先选哪种避障方式？
  [4/4] n01 👎 这款扫地机器人的售价是多少？  ← 库外问题却没说明未提及
```

> 注意：回放产生的反馈属于**模拟数据**，用于验证链路，不能当作用户反馈的统计结论；真实流量接入后只需换数据源，代码不用改。



```
# 命令行验证（-N 关闭缓冲才能看到逐条推送）
curl.exe -N -X POST http://127.0.0.1:8000/chat/stream `
  -H "Content-Type: application/json" `
  -d '{\"message\":\"拖布发臭怎么处理？\"}'
```



***

## 效果验证

自建 80 题评估集（选购 / 维护 / 故障 / 使用 / 技术五类，含 12 条库里没有答案的负样本），
对 `chroma_db` 里的真实索引跑 top-3 检索，命中标准 = 返回的块里至少含答案关键词。
题库本身先过一遍自检（关键词必须真实存在于库中，且命中块数 ≤ 8 以防用泛词凑分）。

```
.venv\Scripts\python.exe eval\check_questions.py        # 题库自检，生成核对表
.venv\Scripts\python.exe eval\run_eval.py --mode vector  # 基线
.venv\Scripts\python.exe eval\run_eval.py --mode hybrid  # 混合检索
.venv\Scripts\python.exe eval\run_eval.py --mode hybrid_rerank  # 混合 + 精排
```

| 指标 | 仅向量 | 混合检索 | 混合 + 精排 |
| --- | --- | --- | --- |
| hit@3 | 76.5% (52/68) | 91.2% (62/68) | **100% (68/68)** |
| MRR | 0.600 | 0.797 | **0.985** |
| 来源命中率 | 92.6% | 97.1% | **100%** |
| 平均检索耗时 | 229 ms | 237 ms | 1757 ms |

> 100% 是 68 条正样本上的结果，属小样本结论，不能外推成"任何问题都能答对"。

另做过一组切分对照：字符硬切（326 块 / 161 字）→ 条目对齐打包（194 块 / 265 字），
混合检索 hit@3 由 86.8% 提升到 91.2%，MRR 由 0.725 到 0.797 —— 因为语料是
`1. 现象：…；检测：…；修复：…` 的结构化规则，硬切会把一条规则腰斩成"现象块"和"修复块"。

68 道正样本下 95% 置信区间约 ±7pp，所以对外说"约 91%"而非精确值。
提升主要来自参数类问题：`地毯场景需≥4000Pa`、`集尘袋容量≥2.5L`、`充电座前方3米`、`电机功率≥300W`
这类数字+单位的答案，纯向量召不回，BM25 词面命中能抓到。完整分析、剩余漏召与未采用方案见 [`eval/report.md`](eval/report.md)。

***

## 配置说明

配置只在 `main.py`（以及 `agent/tools.py` 读取 `agent.yml`）处读取，**改完必须重启进程**（`load_config` 有进程内缓存）。

### `config/rag.yml`



| 键                 | 说明                       |
| ----------------- | ------------------------ |
| `chat_model`      | 对话模型                     |
| `model_base_url`  | OpenAI 兼容网关地址（DashScope） |
| `embedding_model` | 向量化模型                    |

### `config/chroma.yml`



| 键                                             | 说明                        |
| --------------------------------------------- | ------------------------- |
| `collection_name`                             | Chroma 集合名                |
| `persist_directory`                           | 持久化目录（相对项目根）              |
| `md5_hex_store`                               | 去重清单文件路径；删除该键则关闭去重、每次全量入库 |
| `data_path`                                   | 知识库目录                     |
| `allow_knowledge_file_types`                  | 参与建库的文件后缀                 |
| `chunk_strategy` | `structural` 条目对齐打包（默认）；`fixed` 字符硬切 |
| `chunk_target_size` | structural 模式下每块目标长度（默认 300） |
| `chunk_size` / `chunk_overlap` / `separators` | fixed 模式的切分参数（唯一来源，代码内不再有默认值） |
| `k`                                           | 检索召回条数                    |
| `retrieval_mode`                              | `vector` 仅向量；`hybrid` 向量 + BM25 融合（默认） |
| `hybrid_candidate_k`                          | 每路各召回多少条候选再融合             |
| `rrf_k`                                       | RRF 平滑常数，默认 60              |
| `rerank_enabled` | `true` 开启两阶段检索（召回多捞 → 精排取 top-k） |
| `rerank_model` | `qwen3.7-text-rerank`（默认，实测优于 `qwen3-rerank`） |
| `rerank_candidate_k` | 交给精排的候选条数（默认 10，实测优于 20） |
| `rerank_reject_threshold` | 精排分低于该值就拒答；**默认 0 = 关闭**（分数是请求内相对分，做不了绝对阈值） |

### `config/prompts.yml`

`main_prompt_path`（默认系统提示词）、`report_prompt_path`（报告场景）、`rag_summarize_prompt_path`（检索总结模板，含 `{input}` / `{context}` 占位符）、`rag_summarize_cited_prompt_path`（带 `[编号]` 出处标注的总结模板）。

### `config/agent.yml`

`external_data_path`：用户使用记录 CSV 的路径。



***

## 可用工具



| 工具                                    | 说明                                         |
| ------------------------------------- | ------------------------------------------ |
| `rag_summarize(query)`                | 从向量库检索专业资料并 LLM 总结                         |
| `get_user_location()`                 | 获取用户所在城市（演示用随机值）                           |
| `get_weather(city)`                   | 获取城市天气（演示实现）                               |
| `get_user_id()`                       | 获取用户 ID                                    |
| `get_current_month()`                 | 获取月份（YYYY-MM）                              |
| `fetch_external_data(user_id, month)` | 查询该用户该月使用记录（读 `data/external/records.csv`） |
| `fill_context_for_report()`           | 触发报告场景提示词切换                                |

中间件：`monitor_tool`（工具调用日志 + 报告标记）、`log_before_model`（模型调用前日志）、`build_prompt_switch`（按 `context["report"]` 切换提示词）。



***

## 设计要点



* **配置集中**：`main.py` 读 yml → 传参给下层；模块内不 import 配置，便于单独测试与替换实现。

* **chunk 级幂等入库**：以 chunk 内容 MD5 作为 Chroma 主键，重复启动不重复向量化；同批次内也去重。

* **MD5 清单增量写**：`_append_md5s` 只追加新记录，配合 `compact_md5s()` 整理，避免每次重写整份文件。

* **Agent 编排**：使用 `create_agent`，ReAct 循环、工具次数上限、终止条件交给框架；中间件承担日志与提示词切换。

* **双形态输出**：`run_agent`（一次性）与 `run_agent_stream`（`stream_mode="messages"` 流式）共用同一套工具与配置。

* **可观测**：控制台与 `logs/app.log` 同时记录，日志名带模块路径（如 `rag_service.agent.middleware`），能直接定位来源。

* **混合检索**：向量一路负责语义相近，BM25 一路负责数字/术语字面命中，用 RRF（`Σ 1/(60+rank)`）融合，不依赖两路分数归一化。

* **两阶段检索（精排）**：混合检索先取 10 条候选，再用 `qwen3.7-text-rerank` 重新打分取 top-3。召回负责"别漏"、精排负责"排得准"，MRR 由 0.797 提到 0.985。候选深度实测 10 优于 20（20 条反而干扰精排且 token 多 68%），延迟代价约 +1.5s。

* **不做分数阈值拒答**：距离阈值（正负样本区间重叠）与精排分阈值（分数是请求内相对分，随候选池漂移，同题可差 0.13）都验证过不可行，`rerank_reject_threshold` 保留配置项但默认 0，仍由提示词约束「资料中没有就说明未提及」兜底。

* **反馈闭环**：`trace_id` + SQLite 两张表（`qa_log` / `feedback`）+ `/feedback` + `/stats`。只存 `sources` 不存精排分数——只有会被使用的字段才落库；流式拿不到 token 用量就留空，不编数字。落库失败只记 warning，不影响回答返回。

* **答案可溯源**：`invoke_with_sources` 在参考资料前加 `[编号]` 与来源文件名，提示词要求每条结论标注编号，返回体同时给出 `sources` 列表（文件名 + 片段）。

* **不做距离阈值拒答**：实测负样本 top1 距离（0.52\~0.82）与命中题（最高 1.15）区间重叠，绝对距离分不清相关与否，改由提示词约束「资料中没有就说明未提及」。

* **条目对齐切分**：语料是编号规则（`1. 现象…；检测…；修复…`），字符硬切会把一条规则拆成"现象块"和"修复块"，问"怎么处理"时召回现象块、丢掉答案块。改为按条目边界切、相邻条目打包到 300 字，块数 326→194，hit@3 +4.4pp。



***

## 常见问题

**启动报 DASHSCOPE_API_KEY 相关错误 / 对话模型返回 401**

环境变量没被读到。对话链路需要 `OPENAI_API_KEY`，向量化需要 `DASHSCOPE_API_KEY`，两者填同一个百炼 key；先确认这两个变量都有值，或在项目根放 `.env`。`.env` 不会覆盖系统里已有的同名变量。

`[Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)`

端口被占用。换端口 `--port 8001`；或查占用进程后结束：



```
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object OwningProcess
Stop-Process -Id <PID>
```

**改了 yml 但不生效**

配置有进程内缓存，需重启服务。

**每次启动都显示 No new documents to add 正常吗**

正常。chunk MD5 已在 `md5.text` 中，说明库已建好，没有重复向量化。

**回答要等很久**

Agent 可能多轮调用工具，每次 `rag_summarize` 要检索 + LLM 总结，单轮约 20\~35 秒。想更快可调小 `k` 或收紧系统提示词。流式接口（`/chat/stream`）可先看到首字（约 3 秒）。

**如何重建知识库**

删掉 `chroma_db/` 目录与 `md5.text`，重启即可全量重建（会重新向量化，消耗 token）。

**怎么换模型 / 换向量库**

模型改 `config/rag.yml`；换向量库实现只改 `vector_store/` 一层，上层调用 `as_retriever(k)` 不变。

**某个 PDF 损坏导致启动失败**

`DocumentLoader.load_all()` 目前没有逐文件容错，坏文件会中断建库。建议先从 `data/` 移出该文件。



***

## 改动记录

相对课程原项目 RAG-Agent 的重构与修正：



* **模块化拆分**：按 document\_loader /chunker/embedding /vector\_store/rag\_summarize /agent/api 分层，配置与路径统一收进 utils 与 config。

* **配置集中与传参**：`main.py` 统一读 yml 并显式传参；`build_splitter` 去掉内置默认值，切分参数唯一来源为 `config/chroma.yml`。

* **Agent 编排**：改用 `create_agent` + 中间件（日志、报告提示词切换）。此前手写 StateGraph 时修复过 `messages` 未用 `Annotated[list, add_messages]` 导致历史被覆盖、工具调用死循环的问题。

* **流式输出**：新增 `run_agent_stream` 与 `POST /chat/stream`（SSE）。

* **日志**：子 logger 挂到 `rag_service.*` 命名空间，模块日志可正常写入 `logs/app.log`。

* **入库去重**：判断改用滚动集合 `seen`，覆盖「库内已有 + 本批内重复」。

* **MD5 清单**：全量重写改为增量追加，新增 `compact_md5s()`。

* **文档元数据**：`source` 统一为纯文件名，不写入本机绝对路径。

* **外部数据路径**：`fetch_external_data` 读 `config/agent.yml`。

* **清理**：移除 `if True` 调试残留、重复的 `setup_logger()` 调用、未使用的 `PromptLoader.format()` 与 `add_documents` 的 `upsert` 形参。

* **检索评估**：新增 `eval/`（评估集 + 脚本 + 报告），把「检索好不好」从主观感觉变成可复现的数字。

* **混合检索**：新增 `retrieval/hybrid_retriever.py`，向量 + BM25 双路召回归一化融合，`config/chroma.yml` 用 `retrieval_mode` 切换；hit@3 由 75% 提升到 95%。

* **出处标注**：新增 `prompts/rag_summarize_cited.txt` 与 `RagSummarizeService.invoke_with_sources`，答案带 `[编号]`，同时返回来源文件名与片段。

* **条目对齐切分**：新增 `chunker/structural_splitter.py`，`chunk_strategy: structural|fixed` 可切换；默认 structural（194 块），旧 fixed（326 块）保留作对照。

* **精排（rerank）**：新增 `retrieval/reranker.py`（百炼文本排序 API），`config/chroma.yml` 增 `rerank_enabled / rerank_model / rerank_candidate_k / rerank_reject_threshold`；`RagSummarizeService` 变成"召回 10 → 精排取 3"。80 题 hit@3 由 91.2% 升至 100%（68/68），MRR 0.797 → 0.985；对照了候选深度 10 vs 20 与两个模型，最终选 `qwen3.7-text-rerank` + 候选 10。阈值拒答实测后仍不启用（分数随候选池漂移）。

* **反馈闭环**：新增 `storage/db.py`（SQLite：`qa_log` / `feedback`，WAL + 写锁）、`/feedback`、`/stats`，`/chat` 与 `/chat/stream` 返回 `trace_id`；`rag_summarize` 工具改用 `invoke_with_sources`，把 `sources` 写进 `runtime.context` 随请求落库（只存 sources，不存精排分）。新增 `eval/replay_feedback.py` 用评估集回放模拟线上流量。