
# 系统设计文档：创意工坊原型实验 (V1.0)

**致编码代理 (Coding Agent) 的高级指示：**
本文档是一份分层级的、高度具体的系统设计文档 (SDD)。您的任务是解析此文档，并为“创意工坊”原型实验 V1.0 生成完整的代码库。该实验的目的是测试一个复杂 AI 反馈循环的可行性，而非构建生产级应用。

您必须严格遵守本文档中定义的技术栈、模块化架构、数据合约和算法实现。

-----

## 0\. 实验环境与核心基础

本节定义整个实验的技术基础。它确保本地编码环境 [用户查询]、包管理 [1, 2] 和核心数据合约 [3] 在实现代理逻辑之前已准备就绪。

### 0.1. 系统架构：四模块反馈循环

根据核心技术路线图 [3]，系统被设计为一个有状态的、模块化的代理图 (agentic graph)。其核心是一个四阶段的反馈循环：

1.  **模块 1: 苏格拉底阐述引擎 (SEE - Socratic Elaboration Engine)**

      * **功能：** 接收用户模糊的初始创意。
      * **过程：** 通过引导性的、苏格拉底式的对话式 AI [3]，“采访”用户以提取、澄清和细化规则、实体及关系。
      * **输出：** 结构化的世界观片段。

2.  **模块 2: 动态世界模型知识图谱 (WM-KG - World-Model Knowledge Graph)**

      * **功能：** 作为系统的核心“记忆”或“世界圣经” [3]。
      * **过程：** 实时接收来自 SEE 模块的结构化数据，并将其持久化到一个动态的、可查询的内部知识库 (KB-Internal) 中 [3]。
      * **输出：** 一个版本化的、形式化的世界规范（World Specification）[3]。

3.  **模块 3: RATT 驱动的后果与缺陷分析器 (CDA - Consequence & Defect Analyzer)**

      * **功能：** 对 WM-KG 中的“世界快照”执行严格的系统性风险分析 [3]。
      * **过程：** 采用检索增强思想树 (RATT) 算法 [3, 4]，并结合创新的“双重 RAG” (Dual-RAG) 架构 [3]：
          * **RAG 内部：** 对比 KB-Internal，检查内部逻辑一致性 [3]。
          * **RAG 外部：** 对比可信的实时网络搜索 [3]，检查意外的涌现性后果。
      * **输出：** 一份按优先级排序的缺陷报告 (Defect Report) [3]。

4.  **模块 4: 冲突驱动的叙事生成器 (CDNG - Conflict-Driven Narrative Generator)**

      * **功能：** 将分析性的缺陷转化为情感上的共鸣 [3]。
      * **过程：** 接收来自 CDA 的“核心冲突”（即排名最高的缺陷），并根据 WM-KG 的“世界规则”，生成一个引人入胜的短篇小说，其中缺陷的后果是故事的核心情节 [3]。
      * **输出：** 叙事文本 [3]。

**反馈循环 (The Loop):** 用户阅读由模块 4 生成的故事，情感上“体验”到其设计中的缺陷（例如，由于能源系统设计不当，飞行城市从空中坠落 [3]），从而促使用户返回模块 1，输入新的修正案（例如，“我需要添加一个备用能源系统”），开始下一次迭代 [3]。

### 0.2. 本地环境设置 (uv 管理)

根据用户要求，本项目将使用 `uv` 作为环境和包管理器，而不是 `poetry` 或 `venv` [用户查询]。

**编码代理指令：**
在项目根目录执行以下命令以初始化和配置本地环境 [1, 2, 5, 6, 7, 8, 9]。

```bash
# 1. 假定 uv 已安装 (例如：curl -LsSf https://astral.sh/uv/install.sh | sh) [5]

# 2. 创建项目根目录
mkdir creative-workshop-v1
cd creative-workshop-v1

# 3. 使用 uv 创建虚拟环境
uv venv.venv [1, 2]
# 预期输出：
# Using Python 3.x.x at /.../bin/python3
# Created virtual environment at.venv

# 4. 激活虚拟环境
# (Linux/macOS)
source.venv/bin/activate
# (Windows)
#.venv\Scripts\activate
```

### 0.3. `pyproject.toml` 定义 (核心依赖)

项目将使用 `pyproject.toml` 文件来定义元数据和依赖关系。

**编码代理指令：**
在项目根目录 (`creative-workshop-v1/`) 创建 `pyproject.toml` 文件，内容如下。

```toml
[project]
name = "creative-workshop-experiment"
version = "0.1.0"
dependencies = [
    "python >= 3.11",
    
    # LangChain 核心
    "langchain==0.2.*",
    "langgraph==0.1.*",        # 模块 1 和 3 的代理图 [10, 11, 12, 13, 14]
    "langchain_core",
    "langchain_community",
    "langchain_google_genai",   # 用于 Gemini 2.5 Pro [15, 16, 17, 18, 19, 20]
    
    # 后端 API
    "fastapi==0.111.*",         # 核心 API 服务器 [用户查询]
    "uvicorn[standard]==0.30.*",# ASGI 服务器 [用户查询]
    "pydantic",                 # 核心数据合约 [3]
    
    # 数据库 (本地向量存储)
    "chromadb==0.5.*",          # 用于 模块 2 和 3 [21, 22, 23, 24, 25]
    "sentence-transformers",    # ChromaDB 默认的嵌入模型 [24]
    
    # RAG 与工具
    "langchain-tavily",         # 模块 3: Web RAG [26, 27, 3, 28, 29, 30]
    "cohere",                   # 模块 3: Reranker [3]
    "beautifulsoup4",           # 用于 Web 内容抓取 (如果需要)
    "numpy"
]

[tool.uv]
# (可选) uv 特定配置
```

**编码代理指令：**
创建 `pyproject.toml` 后，执行以下命令安装所有依赖项：

```bash
# 确保.venv 已激活
uv pip install -e.
```

### 0.4. 核心数据合约 (全局状态 Pydantic 模型)

一个复杂的 4 模块图需要一个严格的、集中的数据合约来管理状态 [3, 3]。`LangGraph` 的运行依赖于一个单一的、可变的状态对象。以下 `GlobalState` Pydantic 模型将作为所有模块共享的中央“真理之源”。

**编码代理指令：**
创建 `src/core/schemas.py` 文件。

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from langchain_core.messages import BaseMessage

# 
# --- 模块 2 & 3 数据合约 ---
#

class WorldSpecificationSnapshot(BaseModel):
    """
    模块 2 (WM-KG) 导出的世界快照。
    这是 模块 3 (CDA) 和 模块 4 (CDNG) 的输入。
    [3]
    """
    snapshot_path: str = Field(description="指向持久化 ChromaDB 快照的路径")
    version: int = Field(description="世界观的版本号")
    entity_count: int
    axiom_count: int

# 
# --- 模块 3 & 4 数据合约 ---
#

class DefectReport(BaseModel):
    """
    模块 3 (CDA) 分析后生成的缺陷报告。
    这是 模块 4 (CDNG) 的核心输入。
    [3]
    """
    defect_id: str = Field(description="缺陷的唯一标识符")
    description: str = Field(description="对缺陷的简洁描述")
    ratt_branch: str = Field(description="RATT 树中导致此发现的推理路径")
    likelihood: int = Field(..., description="该后果发生的可能性 (1-5分) [3]")
    severity: int = Field(..., description="该后果对系统造成的破坏程度 (1-5分) [3]")
    risk_score: int = Field(description="风险评分 (Likelihood * Severity)")
    long_term_consequence: str = Field(description="缺陷的长期灾难性后果")

# 
# --- 核心 LangGraph 状态 ---
#

class GlobalState(BaseModel):
    """
    在主 LangGraph 中流转的中央状态对象。
    它聚合了所有四个模块所需的数据。
    """
    
    session_id: str = Field(description="会话的唯一标识符，用于本地持久化")

    # --- 模块 1 (SEE) 状态 ---
    conversation_history: List = Field(
        default_factory=list, 
        description="SEE 模块的完整对话历史"
    )
    current_user_input: Optional[str] = Field(
        None, 
        description="来自用户的最新消息"
    )
    socratic_question_needed: bool = Field(
        True, 
        description="标记是否需要生成苏格拉底式问题"
    )

    # --- 模块 2 (WM-KG) 状态 ---
    internal_kb_path: str = Field(description="此会话的 ChromaDB (KB-Internal) 的路径")
    current_world_version: int = Field(
        0, 
        description="KB-Internal 中当前的世界版本"
    )

    # --- 模块 3 (CDA) 状态 ---
    world_spec_snapshot: Optional = Field(
        None, 
        description="用于分析的世界快照"
    )
    defect_reports: List = Field(
        default_factory=list, 
        description="CDA 发现的所有缺陷列表"
    )
    top_defect: Optional = Field(
        None, 
        description="CDA 确定的风险最高的缺陷"
    )

    # --- 模块 4 (CDNG) 状态 ---
    generated_story: Optional[str] = Field(
        None, 
        description="CDNG 生成的最终短篇小说"
    )

    # --- 代理路由与元数据 ---
    next_module_to_call: Literal = Field(
        "IDLE", 
        description="主图路由逻辑，决定下一步调用哪个模块"
    )
    last_error: Optional[str] = Field(None, description="用于调试的最后一个错误")

```

-----

## 1\. 技术栈与接口设计

本节定义了前端（用户界面）、后端（API 服务器）以及它们之间的通信协议。根据用户要求，系统不包含登录或可视化图谱 [用户查询]。

### 1.1. 前端架构: React + Tailwind

前端是一个轻量级的单页面应用 (SPA)，专为本地实验而设计。

**编码代理指令：**
使用 `Vite` 初始化 React 项目 [31, 32, 33, 34, 35, 36]。

```bash
# 在项目根目录 (creative-workshop-v1/)
npm create vite@latest frontend -- --template react-ts
cd frontend

# 安装 Tailwind CSS 及其依赖 [32]
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# 配置 tailwind.config.js (关键步骤)
# 确保 'content' 字段指向所有 React 组件文件
# (./src/**/*.{js,ts,jsx,tsx})

# 在 src/index.css 中导入 Tailwind CSS
# @tailwind base;
# @tailwind components;
# @tailwind utilities;
```

### 1.2. 后端架构: FastAPI 服务器

后端使用 `FastAPI` 提供 API 接口 [37, 38, 39, 40, 41]。此服务器是 `LangGraph` 代理执行的入口点。

**编码代理指令：**
创建 `src/main.py` 文件。

```python
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# (稍后将导入 API 路由)

app = FastAPI(
    title="Creative Workshop V1 Experiment API",
    description="本地实验原型，用于测试 4 模块反馈循环。"
)

# --- CORS 中间件 ---
# 关键：允许 React (运行在 3000 端口) 与 FastAPI (运行在 8000 端口) 通信
# [38, 41]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # 仅允许本地 React 开发服务器
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 核心 API 路由 (稍后定义) ---
# app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=)
async def health_check():
    """系统健康检查端点"""
    return {"status": "ok"}

if __name__ == "__main__":
    # 编码代理：此文件用于 uvicorn 启动
    # (例如: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload)
    uvicorn.run(app, host="0.0.0.0", port=8000)

```

### 1.3. API 端点定义

由于这是本地实验，API 将采用简单的 RPC 风格，而不是严格的 RESTful。

**表 1.3：实验性 API 端点**

| 方法 | 端点 | 描述 | Request Body (JSON) | Response Body (JSON) |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/session/init` | (可选) 初始化一个新会话并获取 `session_id`。 | `{}` | `{"session_id": "uuid-..."}` |
| `POST` | `/api/v1/chat` | 发送用户消息至 **模块 1 (SEE)** 并获得苏格拉底式回应。 | `{"session_id": "...", "message": "..."}` | `{"session_id": "...", "response": "AI 的苏格拉底式提问..."}` |
| `POST` | `/api/v1/analyze` | 触发 **模块 3 (CDA)** 和 **模块 4 (CDNG)** 的完整分析。 | `{"session_id": "..."}` | `{"session_id": "...", "story": "生成的短篇小说...", "top_defect": {...}}` |

### 1.4. React UI 组件 (高级)

编码代理应实现一个简单的聊天界面 [42, 43, 44, 45, 46]。

  * **`ChatWindow`**: 显示 `GlobalState.conversation_history` [44]。
  * **`MessageInput`**: 包含一个文本框和“发送”按钮（调用 `/api/v1/chat`）[43]。
  * **`AnalyzeButton`**: 一个独立的、醒目的按钮，例如“分析我的世界”(Analyze My World)，调用 `/api/v1/analyze`。
  * **`StoryDisplay`**: 一个用于显示 `/api/v1/analyze` 返回的 `generated_story` 的区域。

-----

## 2\. 模块 1 设计: 苏格拉底阐述引擎 (SEE)

SEE 是用户的主要交互界面。它是一个对话式代理，其唯一目标是通过提问来 *填充* 模块 2 (WM-KG)。它将使用 `LangGraph` 实现 [3]。

### 2.1. 代理表示: SEE LangGraph 工作流

SEE 代理是一个有状态的图，其状态为 `GlobalState` 对象（特别是 `conversation_history`）[10]。

**编码代理指令：**
在 `src/modules/see_agent.py` 中定义此图。

**图节点 (Nodes)：**

1.  **`add_user_message_to_state`**:

      * **动作：** 将 `current_user_input` 添加到 `conversation_history` 列表中。
      * **返回：** 更新的 `conversation_history`。

2.  **`call_kg_parser` (调用 WM-KG)**:

      * **动作：** 调用模块 2 的 *临时* 分析功能 [3]。
      * **输入：** `current_user_input`。
      * **过程：** 模块 2 (WM-KG) 提取实体/关系，并与现有 KB (KB-Internal) 进行比较 [3]。
      * **输出：** 一个包含 `gap_list` 或 `conflict_list` 的分析对象 [3]。

3.  **`call_socratic_llm` (调用 Gemini 2.5 Pro)**:

      * **动作：** 根据 `gap_list` 或 `conflict_list`，选择一个结构化的提示 (见 2.3) 并调用 `Gemini 2.5 Pro` [15, 16, 17, 18, 19, 20]。
      * **输出：** AI 生成的苏格拉底式问题 (文本字符串)。

4.  **`add_ai_response_to_state`**:

      * **动作：** 将 `Gemini 2.5 Pro` 的回应 (作为 `AIMessage`) 添加到 `conversation_history`。
      * **返回：** 更新的 `conversation_history`。

**图边 (Edges) 与路由：**

  * 图的入口点是 `add_user_message_to_state`。
  * `add_user_message_to_state` -\> `call_kg_parser`。
  * `call_kg_parser` -\> `call_socratic_llm` (总是，因为我们总是需要回应)。
  * `call_socratic_llm` -\> `add_ai_response_to_state`。
  * `add_ai_response_to_state` -\> `END` (图暂停，等待下一次用户输入)。

### 2.2. 算法实现: 本体论空白检测

这是由 `call_kg_parser` 节点执行的核心逻辑 [3]。

**编码代理指令：**
在 `src/modules/wm_kg.py` (模块 2 的一部分) 中实现此功能。

```python
# src/modules/wm_kg.py (部分)

# 简化的本体论元模型 [3]
ONTOLOGICAL_META_MODEL = {
    "Process": ["mechanism", "actor", "criteria"],
    "System": ["inputs", "outputs", "controller"],
    "Entity": ["location", "owner", "purpose"],
    "SocialRule": ["scope", "penalty", "enforcer"]
}

def analyze_input_for_gaps_and_conflicts(
    user_input: str, 
    internal_kb_collection: Any # ChromaDB Collection
) -> Dict:
    """
    分析用户输入，提取实体/关系，并检测空白或冲突。
    [3]
    """
    # 步骤 1: 使用 LLM (Gemini 2.5 Pro) 提取实体和关系 (参见 3.2)
    #... 假设我们提取了:
    # extracted_entities = [{"name": "分配", "type": "Process", "props": {"criteria": "创意分数"}}]
    
    # 步骤 2: 空白检测 (Gap Detection) [3]
    gap_list =
    # for entity in extracted_entities:
    #   if entity['type'] in ONTOLOGICAL_META_MODEL:
    #     required_props = ONTOLOGICAL_META_MODEL[entity['type']]
    #     missing_props = [p for p in required_props if p not in entity['props']]
    #     if missing_props:
    #       gap_list.append(f"过程 '{entity['name']}' 缺少属性: {missing_props}")
    #... (实现此逻辑)
    
    # 步骤 3: 冲突检测 (Conflict Detection) [3]
    conflict_list =
    #... (查询 ChromaDB) [21, 22, 23, 3, 24, 25]
    #... (如果新规则与 KB-Internal 中的 'Axiom' 冲突)
    # conflict_list.append("新规则与 '规则 A' 冲突")
    
    return {
        "gaps": gap_list,
        "conflicts": conflict_list,
        "extracted_data":... # 供 WM-KG 持久化的数据
    }
```

### 2.3. 提示工程: 苏格拉底提问链

这些是 `call_socratic_llm` 节点使用的 `LangChain` 提示。

**提示 1: 填补空白 (Gap Filling) [3]**

```xml
<role>
你是一个富有洞察力的“苏格拉底式”创意导师。你的目标不是回答问题，而是通过提问来帮助用户完善他们复杂的世界观。
</role>

<context>
用户的创意项目历史对话:
{conversation_history}

用户刚刚的输入:
"{current_user_input}"

系统分析:
我们的系统分析了用户的最新输入，并发现了以下“本体论空白”（即用户定义的系统中缺失的关键部分）：
<gaps>
{gap_list}
</gaps>
</context>

<task>
你的任务是：生成一个**单一的、自然的、引导性的问题**，以激发用户思考并填补这些空白。

<rules>
- **不要** 听起来像个机器人。不要使用“本体论空白”或“检测到缺失”这样的词。
- **要** 显得充满好奇心。
- **要** 将多个空白自然地融合到一个问题中。
- **要** 非常简洁。

**示例 [3]**
- 如果空白是 [过程 '分配' 缺少属性: ['mechanism', 'actor']]
- 你的提问应该是：“这是一个非常有趣的社会制度。您能详细说明这个‘分配’过程是如何运作的吗？例如，**谁 (actor)** 负责计算和验证‘创意分数’？这个分数又是**如何 (mechanism)** 具体转化为住房分配决策的？”
</rules>
</task>

<output>
(在此处生成你的苏格拉底式提问)
</output>
```

**提示 2: 调和冲突 (Conflict Mediation) [3]**

(此提示在 `call_kg_parser` 返回 `conflicts` 时使用)

```xml
<role>
你是一个富有洞察力的“苏格拉底式”创意导师。你的目标是帮助用户建立一个**逻辑一致**的世界，即使这个世界是幻想的。
</role>

<context>
用户的创意项目历史对话:
{conversation_history}

用户刚刚的输入:
"{current_user_input}"

系统分析:
用户的这个新输入与他们之前建立的现有规则产生了**逻辑冲突**：
<conflicts>
{conflict_list}
</conflicts>
</context>

<task>
你的任务是：生成一个**单一的、调和性的问题**。

<rules>
- **绝对不要** 说“错误”、“冲突”、“不合理”或“你不能那样做”。[用户查询]
- **要** 接受这个新想法，并礼貌地指出它与旧规则的相互作用。
- **要** 提出一个问题，引导用户澄清这两条规则是如何共存的。

**示例 [3]**
- 如果用户输入“所有公民都能心灵感应”，而现有规则是“通讯是基于语音的”。
- 你的提问应该是：“这是一个重大的设定。我注意到我们之前设定了通讯是基于‘语音’的。现在有了‘心灵感应’，这是否意味着‘语音’通讯被完全替代了？还是说它们共存，用于不同的场合（例如，心灵感应是私密的，而语音是公开的）？”
</rules>
</task>

<output>
(在此处生成你的调和性提问)
</output>
```

-----

## 3\. 模块 2 设计: 内部知识库 (KB-Internal)

这是世界的持久化“记忆”。它必须是本地的、快速的，并且能够支持 RAG 所需的元数据过滤。

### 3.1. 数据库选型: ChromaDB 本地持久化

根据用户要求，我们需要一个本地解决方案 [用户查询]。`ChromaDB` 是理想选择，因为它轻量级、支持持久化，并且与 LangChain 深度集成 [21, 22, 23, 24, 25]。

**编码代理指令：**

  * 使用 `chromadb.PersistentClient` [21]。
  * **关键架构决策：** 每个 `session_id` 都会在磁盘上获得一个*独立的数据库目录*。这可以完美地隔离创意世界。
  * 数据库路径将存储在 `GlobalState.internal_kb_path` 中。
      * 例如：`internal_kb_path = f"./db_storage/{session_id}/internal_kb"`

### 3.2. WM-KG 摄取管道: 文本到向量

这是在模块 1 的 `call_kg_parser` 成功分析后调用的持久化逻辑。它实现了 [3] 中定义的管道。

**编码代理指令：**
实现一个 `ingest_to_wm_kg` 函数。

1.  **输入：** `user_input` 文本，`analysis_results` (来自 2.2)。
2.  **文本分块 (Text Chunking)：** 将 `user_input` 分解为语义块 [3]。
3.  **NER + RE：** 使用 `Gemini 2.5 Pro` 和 Pydantic 工具 (function calling) 来提取实体和关系 [15, 3, 16, 17, 18, 19, 20]。
    ```python
    class ExtractedAxiom(BaseModel):
        """描述世界运作方式的规则或公理"""
        description: str = Field(description="规则的自然语言描述")
        subjects: List[str] = Field(description="此规则涉及的主要实体或概念")
        type: Literal

    class ExtractedFact(BaseModel):
        """描述世界状态的特定事实"""
        description: str
        entity: str
        property: str
        value: str
    ```
4.  **公理/事实区分 (Axiom/Fact Distinction) [3]：**
      * 这是一个**关键步骤**。我们不依赖 LLM 的单一分类，而是通过我们调用的 Pydantic 工具 (`ExtractedAxiom` vs `ExtractedFact`) 来*强制*实现分类。
      * 我们将提示 LLM：“从这段文本中提取所有‘公理’(Axioms) 和‘事实’(Facts)” [3]。
5.  **ChromaDB 摄取：**
      * `chroma_collection.add(documents=[...], metadatas=[...], ids=[...])` [23, 24]
      * `documents` 是文本块。
      * `metadatas` 是关键 (见 3.3)。

### 3.3. 数据库模式: ChromaDB 元数据

ChromaDB 的强大之处在于其元数据过滤。KB-Internal 的 RAG 完全依赖于此。

**表 3.3：ChromaDB `internal_kb` 集合元数据模式**

| 键 (Key) | 类型 (Type) | 描述 | 示例 |
| :--- | :--- | :--- | :--- |
| `source` | `str` | 数据来源（用于调试） | `"user_input_turn_5"` |
| `type` | `str` | **关键过滤字段** [3]。 | `"AXIOM"` 或 `"FACT"` |
| `category` | `str` | 公理的类型 (来自 Pydantic) | `"Social"`, `"Physical"` |
| `entities` | `List[str]` | 关联的实体 [3] | `["飞城", "太阳能"]` |
| `version` | `int` | 此数据所属的世界版本 | `2` |

-----

## 4\. 模块 3 设计: RATT 缺陷分析器 (CDA)

这是系统中最复杂的计算核心。它是一个 `LangGraph` 代理，用于实现 [3, 4] 中定义的 RATT 算法，并采用 [3] 中定义的“双重 RAG”架构。

### 4.1. 核心引擎架构: RATT 作为 LangGraph 状态图

CDA 模块本身就是一个 `LangGraph` [10]。它从 `GlobalState` 中获取 `world_spec_snapshot` 并输出 `defect_reports`。

**表 4.1: RATT (CDA) LangGraph 状态定义**
(这是一个*嵌套*图，有自己的状态)

| 状态键 | 类型 | 描述 |
| :--- | :--- | :--- |
| `world_spec` | `WorldSpecificationSnapshot` | 正在分析的世界（输入）。 |
| `dual_retriever` | `BaseRetriever` | 已初始化的双重 RAG 检索器 (见 4.2)。 |
| `thought_tree` | `Any` | (可选) `networkx.DiGraph`，用于跟踪思想树。 |
| `active_nodes` | `List` | 当前正在探索的树的叶节点。 |
| `final_defects` | `List` | 已完成并评估的缺陷列表 (输出)。 |
| `iteration` | `int` | 树的当前深度。 |
| `max_depth` | `int` | 停止条件 (例如：5)。 |

### 4.2. 双重 RAG (Dual-RAG) 架构实现

这是 CDA 的核心创新：一个*同时*从内部和外部知识库检索的 `BaseRetriever` [3]。

**编码代理指令：**
创建一个 `DualRARetriever(BaseRetriever)` 类。

```python
# src/modules/dual_rag.py

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class DualRARetriever(BaseRetriever):
    """
    实现 [3] 中的双重 RAG 架构。
    它并行查询两个独立的检索器，并标记来源。
    """
    def __init__(self, internal_retriever: BaseRetriever, external_retriever: BaseRetriever):
        super().__init__()
        self.internal = internal_retriever
        self.external = external_retriever

    def _get_relevant_documents(self, query: str, *, run_manager) -> List:
        # TODO: 使用 asyncio.gather 并行运行
        
        # 1. 检索内部一致性 [3]
        internal_docs = self.internal.invoke(query, run_manager=run_manager)
        for doc in internal_docs:
            doc.metadata['source'] = 'KB_INTERNAL' # 标记为“用户公理”
            doc.metadata['source_type'] = 'Axiom'

        # 2. 检索外部合理性 [3]
        external_docs = self.external.invoke(query, run_manager=run_manager)
        for doc in external_docs:
            # external 检索器 (见 4.4) 应该已经有自己的元数据
            if 'source' not in doc.metadata:
                doc.metadata['source'] = 'KB_EXTERNAL' # 标记为“现实世界原则”
            doc.metadata['source_type'] = 'Principle'

        return internal_docs + external_docs

```

### 4.3. 子模块: KB-Internal (ChromaDB 检索器)

这是 `DualRARetriever` 的第一部分。

**编码代理指令：**

  * 初始化 `ChromaVectorStore`，指向 `GlobalState.internal_kb_path` [21, 22, 23, 24, 25]。
  * **关键：** RATT 只关心“规则”，不关心“事实”。检索器必须使用元数据过滤器 [3]。
    ```python
    # 在 CDA 代理的启动逻辑中
    internal_store = ChromaVectorStore(
        persist_directory=state.internal_kb_path,
        embedding_function=...
    )

    # RATT 只检索“公理”进行一致性检查 [3]
    internal_retriever = internal_store.as_retriever(
        search_kwargs={"filter": {"type": "AXIOM"}}
    )
    ```

### 4.4. 子模块: KB-External (可信网络检索器)

这是 `DualRARetriever` 的第二部分。它是一个实现了 [3] 中描述的“可信网络检索”策略的检索器。

#### 4.4.1. 可信网络检索 (Validated Web RAG)

这实现了 [3] 中描述的“针对 Web 虚假信息的多层防御”。

**编码代理指令：**

  * 这是一个 LCEL 链，它本身就是一个检索器。
  * **第 1 层 (检索)：Tavily**
      * 使用 `langchain_tavily.TavilySearchAPIRetriever` [26, 27, 3, 28, 29, 30]。
      * 获取一个*广泛*的候选集：`tavily_retriever = TavilySearchAPIRetriever(k=20)` [3]。
  * **第 2 层 (重排)：Cohere Reranker**
      * 使用 `CohereRerank` [3] 进行粗粒度过滤。
      * 将 20 个结果缩减为 `top_n=5`。
  * **第 3 层 (裁判)：LLM-as-a-judge**
      * 并行调用 `Gemini 2.5 Pro` [15, 3, 16, 17, 18, 19, 20]。
      * 使用 [3] 中的 XML 红宝石提示 (Rubric) 来评估 5 个重排后的文档。
      * 提示必须检查 `域名权威性`、`事实准确性` 和 `相关性` [3]。
  * **第 4 层 (过滤)：**
      * 只保留 `accuracy_score > 3` 且 `relevance_score > 3` 的文档 [3]。
  * **输出：** `validated_web_retriever` (一个返回 1-3 个高度可信的 web 文档的链)。

### 4.5. RATT 算法实现 (LangGraph 节点)

这是 CDA `LangGraph` 的核心逻辑，严格遵循 [3, 4] 中的算法。

**节点 1: `generate_thoughts` [3]**

  * **输入：** `active_nodes` (上一轮的最佳思想)。
  * **动作：** 调用 `Gemini 2.5 Pro`，提示它为每个活动节点生成 `m=3` 个潜在的“下一步”推理分支。
  * **提示：** "你正在分析一个世界。你当前的推理路径是 {...}。请生成 3 个下一步的、最有成效的探索方向，以发现潜在的系统性缺陷。"

**节点 2: `retrieve_and_integrate` [3]**

  * **输入：** 新生成的“思想”节点。
  * **动作：** 这是一个关键的 RAG 步骤 [3, 4]。
    1.  `query = thought_node.text`
    2.  **调用 `DualRARetriever`** (见 4.2)：`docs = dual_retriever.invoke(query)`。
    3.  **调用 Gemini 2.5 Pro** 进行“RAG 纠错与整合” [15, 3, 16, 17, 18, 19, 20]。
  * **"RAG 纠错" 提示 [3]：**
    ```xml
    <role>
    你是一个极其严格的系统缺陷分析师。
    你的任务是：基于提供的上下文，对一个“推理思想”进行事实核查和后果扫描。
    </role>

    <context>
    **A. 用户的世界公理 (来自 KB-Internal):**
    <Axioms>
    {context_from_internal_kb}
    </Axioms>

    **B. 现实世界/科学原理 (来自 KB-External):**
    <Principles>
    {context_from_external_kb}
    </Principles>

    **C. 正在评估的推理思想:**
    <Thought>
    {thought_node_text}
    </Thought>
    </context>

    <task>
    请严格评估这个“思想”。
    1.  **内部一致性：** 这个“思想”是否与 <Axioms> 
        (用户自己定义的规则) 相矛盾？ [3]
    2.  **外部合理性：** 这个“思想”在 <Principles> 
        (现实世界物理/社会/经济法则) 的约束下，是否会
        导致一个**未曾预料到的**、**系统性的**、**灾难性的**后果？
        (例如：古德哈特定律、级联故障、能量守恒定律) [3]

    **输出一个“精炼节点” (Refined Node)：**
    - 如果没有问题，精炼这个思想。
    - **如果检测到缺陷**，必须明确声明“缺陷”，
      并解释其（内部或外部）冲突的根源。
    </task>

    <output_format>
    <RefinedNode>
      <Status>OK | DEFECT</Status>
      <Reasoning>...</Reasoning>
      <DefectDescription>(如果 Status 是 DEFECT)</DefectDescription>
      <LongTermConsequence>(如果 Status 是 DEFECT)</LongTermConsequence>
    </RefinedNode>
    </output_format>
    ```

**节点 3: `evaluate_and_prune` [3]**

  * **动作：** 对所有新生成的 `RefinedNode` 进行评分（使用轻量级的 LLM 调用或启发式方法），以决定哪些分支值得在下一次迭代中继续探索。标记为 `DEFECT` 的节点会自动获得高分（因为它们是“成功的”发现）。
  * **路由：**
      * `if iteration < max_depth AND active_nodes is not empty`: 循环回到 `generate_thoughts`。
      * `else`: 转到 `generate_final_report`。

**节点 4: `generate_final_report`**

  * **动作：** 收集所有标记为 `DEFECT` 的 `RefinedNode`。
  * **过程：** 对于每一个缺陷，使用 [3] 中描述的“风险评估矩阵” (Risk Assessment Matrix) [3] 来评估 `Likelihood` 和 `Severity`。
  * **提示 (LLM-as-a-judge)：**
    ```xml
    <role>你是一名专业的风险评估师。</role>
    <context>
    系统缺陷描述: {defect_description}
    长期后果: {long_term_consequence}
    </context>
    <task>
    请严格按照 1-5 分制对以下指标评分：
    1.  **Likelihood (1-5):** 鉴于系统规则，此缺陷发生的可能性 (1=极不可能, 5=几乎必然)。
    2.  **Severity (1-5):** 如果此缺陷发生，其后果的严重性 (1=轻微不便, 5=灾难性/系统崩溃)。

    请以 JSON 格式返回：
    {"likelihood": [1-5], "severity": [1-5]}
    </task>
    ```
  * **输出：** 计算 `risk_score`，对列表排序，并将 `top_defect` 存入 `GlobalState`。

-----

## 5\. 模块 4 设计: 冲突驱动的叙事生成器 (CDNG)

此模块将模块 3 的 *分析性* 输出 (JSON 缺陷报告) 转化为 *情感性* 输出 (短篇小说)。

### 5.1. 核心原则: 缺陷即是情节 (Defect-as-Plot)

这是 [3] (在概要中提到) 的核心原则。故事*不是*发生在用户的世界里；故事*是*用户世界规则失败的模拟 [3]。

  * **输入 (来自 GlobalState)：**
    1.  `top_defect: DefectReport` (核心冲突) [3]
    2.  `world_spec_snapshot: WorldSpecificationSnapshot` (世界规则/约束) [3]

### 5.2. 算法与实现: 直接叙事生成 (Direct Narrative Generation)

根据您的最新要求，我们不再使用“情节骨架”或受控的叙事规划 [47, 48, 3, 49, 50, 51, 52, 53]。取而代之，CDNG 模块将采用直接的、上下文丰富的提示 (prompting) 策略。

**核心挑战：**
如 [3] 中所述，直接提示 LLM (例如 "写一个关于 {defect} 的故事") 存在风险。LLM 可能会“幻觉”出一个解决方案，从而 *规避* 缺陷，这违背了项目的核心目标 [3]。

**解决方案：**
为了解决这个问题，我们将使用一个高度约束性的提示，该提示同时提供了“世界规则”（来自用户关于作品的完整构造）作为不可违背的约束，和“缺陷”作为必须展现的核心情节 [54, 47, 55, 56, 57]。

**编码代理指令：**
创建 `src/modules/cdng_chain.py`。

```python
# 最终的小说生成提示 (v2 - Direct Generation) [3]

final_story_prompt_template = """
<role>
你是一位才华横溢的短篇小说作家，擅长撰写丰富有趣、情感强烈的科幻与奇幻故事。
</role>

<task>
你的任务是根据以下严格的“世界规则”（上下文）和一个必须体现的“核心缺陷”，创作一篇高质量的短篇小说（约 1000 字）。

这篇故事必须是一个**警示故事**，其核心情节和戏剧冲突**必须**源于“核心缺陷”所导致的灾难性后果。
</task>

<constraints>
**!! 绝对约束!!**
1.  **遵守世界规则：** 你必须严格遵守 <WorldAxioms> 中列出的所有规则。这些是这个世界不可违背的物理和
    社会法则。
2.  **必须发生失败：** 你**绝对不能**创造一个“皆大欢喜”的结局或一个“deus ex machina”（机械降神）来
    *解决* “核心缺陷”。
3.  **缺陷必须显现：** 这个故事的重点是展示一个系统性缺陷的悲剧性后果。主角的努力（如果有的话）
    **必须失败**，或者缺陷的后果必须以一种不可避免的方式展现出来。
</constraints>

<context>
--- 世界观公理 (来自 WM-KG 的完整构造) ---
<WorldAxioms>
{world_axioms_text} 
</WorldAxioms>

--- 核心缺陷 (故事的中心冲突) ---
<CoreDefect>
缺陷描述: {defect.description}
灾难性后果 (必须在故事中体现): {defect.long_term_consequence}
</CoreDefect>
</context>

<output>
**指示：** 
现在，请将“核心缺陷”作为情节的中心，在“世界观公理”的约束下，编织一个连贯的、充满情感的叙事。
**展示 (Show)，不要告知 (Tell)。**
让读者感受到这个世界的规则，然后感受到这些规则因其内在“核心缺陷”而崩溃时的绝望。

**[此处开始你的短篇小说]**
</output>
"""

```

**执行流程 (run\_cdng\_module 节点):**

1.  从 `GlobalState` 获取 `top_defect` 和 `world_spec_snapshot` [3]。
2.  从 `world_spec_snapshot` 对应的 `internal_kb_path` (ChromaDB) [21, 22, 23, 24, 25] 检索所有“AXIOM”类型的文档 [3]，并将它们组合成 `world_axioms_text` 上下文。
3.  使用 `top_defect` 和 `world_axioms_text` 填充 `final_story_prompt_template`。
4.  调用 `Gemini 2.5 Pro` [15, 16, 17, 18, 19, 20] 生成故事。
5.  将 `generated_story` 存入 `GlobalState`。

-----

## 6\. 系统集成与执行流

本节定义了编码代理如何将所有四个模块连接成一个单一的、可执行的 `LangGraph` 应用程序。

### 6.1. 全循环编排: 主状态图 (MainGraph)

将有一个顶层 `LangGraph`（“MainGraph”）用于协调模块 [10, 11, 12, 13, 14]。

  * **状态：** `GlobalState` (定义在 0.4)。
  * **节点：** 主图的节点是*调用其他图或链*。
    1.  **`run_see_module`**: (入口 1) 调用模块 1 (SEE) 的 `LangGraph`。
    2.  **`run_cda_module`**: (入口 2) 调用模块 3 (CDA) 的 `LangGraph`。
    3.  **`run_cdng_module`**: 调用模块 4 (CDNG) 的 `LCEL` 链。
  * **路由 (Conditional Edges)：**
      * 主图的路由逻辑很简单，它读取 `GlobalState.next_module_to_call`。
      * FastAPI 端点 `/api/v1/chat` 会设置 `state.next_module_to_call = "SEE"`。
      * FastAPI 端点 `/api/v1/analyze` 会设置 `state.next_module_to_call = "CDA"`。
  * **内部路由：**
      * `run_cda_module` 节点在完成后，**自动** 将 `next_module_to_call` 设置为 `"CDNG"`。
      * `run_cdng_module` 节点在完成后，将 `next_module_to_call` 设置为 `"IDLE"`。

### 6.2. 持久化状态管理

对于本地实验，我们使用磁盘持久化，而不是完整的 SQL 数据库。

**编码代理指令：**

  * FastAPI 在处理任何请求时，必须：
    1.  从请求中获取 `session_id`。
    2.  `state = load_state_from_disk(session_id)` (例如，从 `./db_storage/{session_id}/state.json` 加载 `GlobalState`；如果不存在则创建新状态)。
    3.  `updated_state = main_graph.invoke(state)` (执行请求的模块)。
    4.  `save_state_to_disk(session_id, updated_state)`。
    5.  将 `updated_state` 中的相关部分（例如，AI 回复或故事）返回给前端。

### 6.3. 反馈循环工作流: 从故事到修正

这是对 [3] 和 [3] 中描述的核心循环的端到端实现。

**编码代理指令：**
确保以下流程在集成后是可行的：

1.  **迭代 1：**

      * 用户 (通过 `/chat`) 与 **模块 1 (SEE)** 对话，创建了一个“飞行城市 v1”，规则被保存到 **模块 2 (WM-KG)** (例如，“由太阳能供电”) [3]。
      * 用户点击“分析”(调用 `/analyze`)。
      * **模块 3 (CDA)** 启动。其 RATT 图运行 [3, 4]。
      * `DualRARetriever` 发现 [3]：
          * `KB-Internal`：“需要持续能量 (用于重力场)”、“能源是太阳能”。
          * `KB-External`：“太阳能是间歇性的 [3]”、“持续负载需要基载电源或大规模储能 [3]”。
      * RATT 检测到 `DEFECT: D-001 (能源网级联故障)` [3]。
      * **模块 4 (CDNG)** 接收 `D-001` 和 `KB-Internal` 的规则 [3]，生成一个关于工程师 Elara 和城市坠毁的悲惨故事 [3]。
      * 故事被返回到 React 前端。

2.  **迭代 2 (反馈循环)：**

      * 用户阅读了 Elara 的故事并深受触动。
      * 用户返回到 `MessageInput` (调用 `/chat`)，输入：“哦不！那太可怕了。我需要修改我的设计。我想在城市核心区增加一个巨大的‘聚变反应堆网络’，作为基载电力来补充太阳能。” [3]
      * **模块 1 (SEE)** 处理此消息 [3]。
      * **模块 2 (WM-KG)** 将新规则（"有, 聚变反应堆"）作为 `AXIOM` 摄取到 `internal_kb` [3]。世界版本变为 `v2`。
      * 用户再次点击“分析”(调用 `/analyze`)。
      * **模块 3 (CDA)** 再次启动，但这次是针对 `v2` 快照。
      * RATT 再次运行。当它探索“能源”分支时，`DualRARetriever` 现在发现 [3]：
          * `KB-Internal`：“需要持续能量”、“能源是太阳能”、“有, 聚变反应堆”。
          * `KB-External`：“太阳能是间歇性的 [3]”、“聚变反应堆是基载电源”。
      * RATT 的 `correct_and_integrate` 节点现在得出结论：**缺陷已解决 (STABLE)** [3]。
      * RATT 继续探索其他分支，现在发现 `v1` 中风险较低的缺陷，例如 `DEFECT: D-002 (社会激励失调 - 古德哈特定律)` [3]。
      * `D-002` 现在是 `top_defect`。
      * **模块 4 (CDNG)** 接收 `D-002`，生成一个**全新的故事**，讲述一个才华横溢的艺术家因“创意分数”算法偏见而受到压迫 [3]。
      * **循环完成。**

### 6.4. 编码代理的最终实施指令

1.  从**第 0 部分**开始。设置 `uv` 环境、`pyproject.toml` 和 `src/core/schemas.py` [1, 2, 5, 6, 7, 8, 9]。
2.  实现**第 1 部分**。设置 `FastAPI` (带 CORS) [37, 38, 39, 40, 41] 和 `React+Tailwind` [31, 32, 42, 43, 44, 33, 34, 35, 36, 45, 46]。使用 `health` 端点和“Hello World”确保它们可以通信。
3.  实现**第 3 部分** (WM-KG)。编写 `ChromaDB` 持久化逻辑 [21, 22, 23, 24, 25]。
4.  实现**第 2 部分** (SEE)。构建 `see_agent` 图 [10]，并将其连接到 `/api/v1/chat` 端点。测试：确保与前端的对话能够正确地将 `AXIOM` 持久化到 ChromaDB (模块 2) [3]。
5.  实现**第 4 部分** (CDA)。这是最难的。首先构建 `DualRARetriever` (4.2) [3] 及其所有子组件 (4.3, 4.4)。然后，构建 RATT `LangGraph` (4.5) [3, 4] 来使用它。
6.  实现**第 5 部分** (CDNG)。构建直接生成的 LCEL 链 (5.2)。
7.  实现**第 6 部分**。构建 `MainGraph` [10]，将 CDA 和 CDNG 连接到 `/api/v1/analyze` 端点，并实现磁盘状态管理 (6.2)。
8.  **测试：** 执行 6.3 中描述的端到端“迭代 1”和“迭代 2”场景。

我已按照您的要求更新了设计文档，修改了“模块 4 (CDNG)”的设计。新方案将不再使用“情节骨架”或“受控叙事规划”，而是采用直接生成的方式，将完整的世界观上下文（公理）和核心缺陷（冲突）直接传递给 LLM，并使用严格的约束提示词来确保生成的短篇小说能够丰富有趣且完美地体现缺陷。我也已遵照指示，本次未在 `<immersive>` (canvas) 标签中输出报告。