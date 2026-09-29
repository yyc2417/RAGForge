> ⚠️ **历史快照（2026-06）**：本文为学习笔记，记录写作当时的机制与数字，部分已被 2026-09 修复取代（双预算终止、LANGSMITH_* 环境变量、45 题数据集、token 化分块等）。现行设计以 `docs/modules/` 与 `docs/adr/`（尤其 006）为准；逐项修复记录见本地 `docs/planning/fix-log-2026-09.md`（私有文档，不入库）。
>
> **本文具体过时点**：MetricsCollector 已从全局单例改为请求级实例 + ContextVar 注入；invoke_structured 改为显式追加独立 human 消息。

# Task 4：模块化 RAG 与基础管道

## 阶段概述

本阶段将 Task 1-3 的组件组装为完整的基础 RAG 管道，并实现 LLM 生成、Prompt 管理、性能指标收集等核心功能。Task 4 是"线性 RAG"（检索→生成）的实现，为 Task 5 的 Agent 状态机提供基础能力。

**前置依赖**：Task 1（配置/日志）、Task 2（文档处理）、Task 3（检索）  
**后续依赖**：Task 5（Agent 状态机）、Task 8（评估系统的 baseline 模式）

---

## 核心知识点

### 知识点 1：LCEL（LangChain Expression Language）

LCEL 是 LangChain 的声明式管道语法，用 `|` 操作符组合组件为 chain：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class LLMClient:
    def generate(self, query: str, context: list[Document]) -> str:
        """基于检索上下文生成回答（带重试）。"""
        context_text = self._format_context(context)
        chain = PromptManager.QA_PROMPT | self._llm  # LCEL 管道
        resp = chain.invoke({"context": context_text, "input": query})
        self._record_usage(resp)
        return resp.content if hasattr(resp, "content") else str(resp)
```

**管道语法 `prompt | llm`**：`|` 操作符将 `ChatPromptTemplate` 和 `ChatOpenAI` 组合为 chain。数据流向是：输入字典 → prompt 模板填充变量 → LLM 调用 → 输出响应。

**`invoke()` 的调用方式**：`chain.invoke({"context": ..., "input": ...})` 同步执行整个管道。LCEL 还提供 `stream()`（流式输出）和 `batch()`（批量执行）。

**与旧版 LLMChain 的区别**：
- **LLMChain（旧版）**：`LLMChain(llm=llm, prompt=prompt).run(context=..., input=...)`，组件通过构造函数绑定
- **LCEL（新版）**：`prompt | llm`，组件通过管道操作符组合，更灵活（可以动态组合不同组件）

**LCEL 的优势**：
1. **声明式**：管道结构一目了然，`prompt | llm | output_parser` 清晰表达数据流
2. **可组合**：可以 `chain1 | chain2` 串联多个 chain，或 `chain1 + chain2` 并行执行
3. **自动追踪**：LCEL chain 自动集成 LangSmith tracing，无需手动埋点

### 知识点 2：tenacity 重试机制

网络请求和 LLM API 调用可能因瞬时故障失败，tenacity 库提供声明式重试：

```python
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

class LLMClient:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(self, query: str, context: list[Document]) -> str:
        """基于检索上下文生成回答（带重试）。"""
        ...
```

**参数详解**：
- `stop=stop_after_attempt(3)`：最多重试 3 次（共 4 次尝试）
- `wait=wait_exponential(min=1, max=10)`：指数退避等待时间，第 1 次重试等 1s，第 2 次等 2s，第 3 次等 4s（上限 10s）
- `retry=retry_if_exception_type(Exception)`：任何异常都触发重试（生产环境建议缩小范围，如只重试 `ConnectionError`）
- `reraise=True`：重试耗尽后重新抛出异常（而非返回 None）

**指数退避的数学原理**：等待时间公式为 `wait = min(multiplier * 2^(attempt-1), max)`。例如 `min=1, max=10`：
- 第 1 次重试：`min(1 * 2^0, 10) = 1s`
- 第 2 次重试：`min(1 * 2^1, 10) = 2s`
- 第 3 次重试：`min(1 * 2^2, 10) = 4s`

**为什么用指数退避而非固定等待**：如果 API 因过载失败，固定等待（如每次都等 5s）可能导致"重试风暴"——多个客户端同时重试加剧服务端压力。指数退避让重试间隔逐渐增大，降低服务端负载。

### 知识点 3：PromptManager 设计

集中管理所有 Prompt 模板，避免分散在各节点中：

```python
from langchain_core.prompts import ChatPromptTemplate

class PromptManager:
    """Prompt 模板集中管理，供 LLMClient 与 Agent 节点复用。"""
    
    QA_PROMPT = ChatPromptTemplate.from_template(
        """你是一个知识库问答助手。请根据以下检索到的文档内容回答用户的问题。

回答规则：
1. 优先使用检索到的文档内容来组织回答
2. 如果文档中包含相关信息，即使需要推理或总结，也应尽力回答
3. 只有当文档完全不包含任何相关信息时，才说明无法从知识库中找到答案
4. 回答时引用文档中的关键信息，但不要逐字照搬
5. 不要编造文档中没有的具体数据或事实

检索到的文档内容：
{context}

用户问题：{input}

请用中文回答："""
    )
    
    # Task 5 补充：
    ANALYZE_PROMPT = ChatPromptTemplate.from_template(...)
    EVALUATE_PROMPT = ChatPromptTemplate.from_template(...)
    REFORMULATE_PROMPT = ChatPromptTemplate.from_template(...)
    VERIFY_PROMPT = ChatPromptTemplate.from_template(...)
```

**模板变量语法 `{var}`**：`ChatPromptTemplate.from_template()` 解析模板字符串，`{context}` 和 `{input}` 是占位符。调用 `chain.invoke({"context": ..., "input": ...})` 时，字典的键填充对应占位符。

**为什么集中管理 Prompt**：
1. **复用性**：多个节点可能使用相同 Prompt（如 `QA_PROMPT` 被 `generate_node` 和 `LLMClient.generate` 共用）
2. **一致性**：Prompt 变更只需改一处，避免"同一个 QA Prompt 在三个文件中各有一份副本"的问题
3. **可测试性**：可以单独测试 Prompt 模板（检查变量是否正确），而无需启动整个管道

**Prompt 设计原则**：
- **分层引导**：先要求"优先使用检索内容"，再允许"推理和总结"，最后才说"无法回答"
- **防幻觉约束**：明确"不要编造文档中没有的具体数据或事实"
- **输出格式要求**：Task 5 的结构化输出 Prompt 在末尾要求"请仅输出一个合法 JSON 对象"

### 知识点 4：MetricsCollector 百分位计算

性能指标需要统计百分位数（P50/P95/P99），`MetricsCollector` 用线性插值算法实现：

```python
import statistics

class MetricsCollector:
    @staticmethod
    def _percentiles(values: list[float]) -> dict[str, float]:
        """计算 P50/P95/P99；样本不足时回退到均值/最大值。"""
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0, "max": 0.0, "count": 0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        def _pct(p: float) -> float:
            # 简单线性插值百分位，样本极小时退化为端点值
            if n == 1:
                return sorted_vals[0]
            idx = p * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac
        
        return {
            "p50": round(_pct(0.50), 2),
            "p95": round(_pct(0.95), 2),
            "p99": round(_pct(0.99), 2),
            "avg": round(statistics.mean(values), 2),
            "max": round(max(values), 2),
            "count": n,
        }
```

**线性插值算法**：对于百分位 `p`（如 0.95），计算索引 `idx = p * (n - 1)`，然后在 `sorted_vals[lo]` 和 `sorted_vals[hi]` 之间线性插值：

```
idx = 0.95 * (n - 1)
lo = int(idx)
hi = lo + 1
frac = idx - lo
result = sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac
```

例如 `n=10, p=0.95`：`idx = 0.95 * 9 = 8.55`，`lo=8, hi=9, frac=0.55`，结果是第 8 个和第 9 个值的加权平均。

**为什么不用 numpy**：
1. **减少依赖**：numpy 是重型库（约 30MB），只为计算百分位引入不值得
2. **代码透明**：线性插值算法简单（10 行代码），便于理解和调试
3. **性能足够**：评估数据集通常只有 10-100 个样本，纯 Python 实现足够快

**P50/P95/P99 的含义**：
- **P50（中位数）**：50% 的请求延迟低于此值，反映"典型体验"
- **P95**：95% 的请求延迟低于此值，反映"大多数用户体验"
- **P99**：99% 的请求延迟低于此值，反映"最差体验"（长尾延迟）

---

## 设计模式与架构决策

**管道模式（Pipeline）**：`build_rag_pipeline()` 串联 parser → chunker → embedder → vector_store → llm，形成线性数据流。每个组件独立可测试，管道组装逻辑集中在一个函数中。

**重试模式（Retry）**：`@retry` 装饰器为 LLM 调用添加自动重试，指数退避避免重试风暴。装饰器方式比重试循环更声明式（逻辑与业务代码分离）。

**单例模式（MetricsCollector）**：进程内全局共享同一份统计数据，`reset()` 可在测试间清空状态。线程安全设计（double-check locking）确保并发环境下数据一致。

**工厂方法（PromptManager）**：`ChatPromptTemplate.from_template()` 是工厂方法，解析模板字符串生成 Prompt 对象。集中管理所有 Prompt，便于复用和一致性维护。

---

## 关键代码解读

### llm_client.py：LLM 客户端

```python
class LLMClient:
    """DeepSeek API 封装（OpenAI 兼容格式）。"""
    
    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.deepseek_model,
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            temperature=0.3,
        )
```

**设计意图**：DeepSeek 使用 OpenAI 兼容 API，所以用 `ChatOpenAI` 封装。`base_url` 指向 DeepSeek 端点（`https://api.deepseek.com`），`temperature=0.3` 控制生成随机性（低温度更确定性，适合知识问答）。

```python
def invoke_structured(self, prompt: ChatPromptTemplate, schema: type, **kwargs):
    """结构化输出：通过 prompt 引导 + JSON 解析返回符合 schema 的对象。
    
    DeepSeek 不支持 response_format=json_schema（会返回 400），因此
    不使用 with_structured_output，改为在 prompt 末尾要求纯 JSON 输出，
    再用 Pydantic 解析校验。
    """
    # 在 prompt 末尾追加 JSON 输出要求
    field_hints = ", ".join(f'"{name}"' for name in schema.model_fields)
    json_suffix = (
        f"\n\n请仅输出一个合法 JSON 对象（不要 markdown 代码块、不要多余文字），"
        f"包含字段：{field_hints}。"
    )
    
    # 修改 messages 末尾追加固 JSON 指令
    original_messages = prompt.messages
    last_msg = original_messages[-1]
    last_msg = last_msg + json_suffix
    new_prompt = ChatPromptTemplate.from_messages(
        [*original_messages[:-1], last_msg]
    )
    
    chain = new_prompt | self._llm
    resp = chain.invoke(kwargs)
    content = resp.content if hasattr(resp, "content") else str(resp)
    
    # 从回复中提取 JSON（容错：去除可能的 markdown 代码块包裹）
    try:
        cleaned = self._extract_json(content)
        data = json.loads(cleaned)
        return schema.model_validate(data)
    except Exception as e:
        logger.warning(f"[llm] 结构化输出解析失败：{e} | 原始回复: {content[:200]}")
        raise
```

**设计意图**：DeepSeek 不支持 OpenAI 的 `response_format={"type": "json_schema"}` 参数（会返回 400 错误），所以不能用 `with_structured_output()`。替代方案是在 Prompt 末尾追加"请仅输出 JSON"指令，然后用正则提取 JSON + Pydantic 校验。`_extract_json()` 容错处理 markdown 代码块包裹（如 ```json {...}```）。

```python
@staticmethod
def _extract_json(text: str) -> str:
    """从可能含 markdown 代码块或前后多余文字的文本中提取 JSON 对象。"""
    # 优先匹配 ```json ... ``` 或 ``` ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # 否则匹配第一个 {...} 块
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text.strip()
```

**设计意图**：LLM 可能不严格遵守"仅输出 JSON"指令，会在 JSON 前后加文字或用 markdown 代码块包裹。正则表达式依次尝试：1) 提取 markdown 代码块内的 JSON，2) 提取第一个 `{...}` 块，3) 直接返回原文（兜底）。

### main.py：基础管道入口

```python
def build_rag_pipeline() -> tuple[VectorStore, LLMClient]:
    """构建基础管道：ingestion → retrieval(VectorStore) → generation。
    
    Task 4 兼容入口（test_stage1.py 使用）。
    """
    parser = DocumentParser()
    docs = parser.parse_directory(DATA_DIR)
    chunker = TextChunker()
    chunks = chunker.split(docs)
    embedder = EmbeddingService()
    vector_store = VectorStore.load_or_build(chunks, embedder)
    llm = LLMClient()
    logger.info("[pipeline] RAG 基础管道就绪")
    return vector_store, llm
```

**设计意图**：`build_rag_pipeline()` 是 Task 4 的核心函数，串联 ingestion → retrieval → generation。返回 `(vector_store, llm)` 二元组，供 `answer()` 使用。这种设计让调用方可以复用已构建的管道（无需每次重新解析文档）。

```python
def answer(vector_store: VectorStore, llm: LLMClient, question: str) -> tuple[str, float]:
    """Task 4 线性管道的单问题回答（检索→生成），返回 (回答, 端到端延迟ms)。"""
    metrics = MetricsCollector()
    t0 = time.perf_counter()
    
    tr = time.perf_counter()
    docs = vector_store.search(question)
    metrics.record_retrieval_latency((time.perf_counter() - tr) * 1000)
    
    answer_text = llm.generate(question, docs)
    e2e = (time.perf_counter() - t0) * 1000
    metrics.record_e2e_latency(e2e)
    return answer_text, e2e
```

**设计意图**：`answer()` 是线性 RAG 的执行函数，流程是"检索 → 生成"。`time.perf_counter()` 提供高精度计时器（比 `time.time()` 更适合测量短时间间隔）。分别记录检索延迟和端到端延迟，便于性能分析。

### metrics.py：指标收集

```python
class MetricsCollector:
    """性能指标收集器（单例）。"""
    
    _instance: "MetricsCollector | None" = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # double-check locking
                    instance = super().__new__(cls)
                    instance._data = defaultdict(list)
                    instance._token_totals = {"prompt": 0, "completion": 0, "calls": 0}
                    cls._instance = instance
        return cls._instance
    
    def record_retrieval_latency(self, latency_ms: float) -> None:
        """记录单次检索延迟（毫秒）。"""
        self._data["retrieval_latency_ms"].append(float(latency_ms))
    
    def record_e2e_latency(self, latency_ms: float) -> None:
        """记录单次端到端延迟（毫秒）。"""
        self._data["e2e_latency_ms"].append(float(latency_ms))
    
    def record_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """记录单次 LLM 调用的 token 消耗。"""
        self._token_totals["prompt"] += int(prompt_tokens)
        self._token_totals["completion"] += int(completion_tokens)
        self._token_totals["calls"] += 1
```

**设计意图**：`MetricsCollector` 是进程内单例，全局共享同一份统计数据。`defaultdict(list)` 自动为每个指标键创建列表，`append()` 追加新样本。`_token_totals` 用字典累加 prompt/completion token，`calls` 计数便于计算平均值。

```python
def reset(self) -> None:
    """清空所有统计数据（用于新一轮评估）。"""
    self._data = defaultdict(list)
    self._token_totals = {"prompt": 0, "completion": 0, "calls": 0}

def export_report(self, path: str | Path, label: str = "") -> Path:
    """导出统计摘要为 JSON 报告文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "label": label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": self.get_summary(),
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
```

**设计意图**：`reset()` 在每轮评估前清空状态，避免历史数据污染。`export_report()` 将统计摘要导出为 JSON 文件，`label` 标记评估模式（如 "baseline" / "agent"），`generated_at` 记录生成时间便于追溯。

---

## 踩坑记录

### 问题 1：DeepSeek 不支持 `response_format=json_schema`

**问题现象**：调用 `llm.with_structured_output(schema)` 时返回 `400 Bad Request`。

**排查过程**：
1. 检查 DeepSeek API 文档，发现不支持 `response_format` 参数
2. 查看 LangChain 源码，`with_structured_output()` 内部会设置 `response_format={"type": "json_schema", "json_schema": ...}`

**根因**：DeepSeek 的 OpenAI 兼容 API 未实现 `json_schema` 格式。

**解决方案**：改用"Prompt 末尾追加 JSON 指令 + 正则提取 + Pydantic 校验"的替代方案。虽然不如 `with_structured_output()` 稳定（LLM 偶尔不严格遵守 schema），但有降级逻辑兜底。

### 问题 2：结构化输出解析偶发失败

**问题现象**：日志中多次出现 `[llm] 结构化输出解析失败` 警告，如 `Expecting value: line 1 column 1 (char 0)`（空回复）。

**排查过程**：
1. 打印 LLM 原始回复发现有时返回空字符串
2. 检查 Prompt 发现 JSON 指令在末尾，LLM 可能未完整生成

**根因**：LLM 生成可能因 token 限制截断，或模型未严格遵守"仅输出 JSON"指令。

**解决方案**：每个 LLM 节点都有降级逻辑——`evaluate` 节点 catch 后回退规则层，`analyze` 节点降级到 `factual + needs_retrieval=True`。这是设计内的容错，不影响整体流程。

---

## 与其他模块的交互

**输入接口**：
- `DocumentParser` / `TextChunker` / `EmbeddingService`：从 Task 2 注入
- `VectorStore`：从 Task 3 注入
- `settings.deepseek_*`：从 Task 1 配置读取 LLM 参数

**输出接口**：
- `build_rag_pipeline()` → `(VectorStore, LLMClient)`：供 test_stage1.py 使用
- `answer(vector_store, llm, question)` → `(str, float)`：供 test_stage1.py 使用
- `LLMClient.generate(query, context)` → `str`：供 Task 5 的 generate_node 使用
- `PromptManager.QA_PROMPT`：供 Task 5 的节点使用
- `MetricsCollector.record_*()`：供 Task 5-8 记录指标

**衔接方式**：
- Task 5 的 `build_agent_pipeline()` 复用 Task 4 的 ingestion 逻辑，但用 Agent 状态机替代线性管道
- Task 8 的 baseline 模式直接调用 `build_rag_pipeline()` + `answer()`，不经过 Agent
- Task 5 的 `generate_node` 调用 `LLMClient.generate()`，复用 Task 4 的生成逻辑

---

## 本阶段收获总结

1. **LCEL 是 LangChain 的声明式管道语法**：`prompt | llm` 比旧版 `LLMChain` 更灵活，支持动态组合
2. **tenacity 重试机制是 LLM 调用的必备保护**：指数退避避免重试风暴，声明式装饰器比重试循环更简洁
3. **集中管理 Prompt 提升复用性和一致性**：`PromptManager` 避免同一 Prompt 在多处维护副本
4. **百分位指标（P50/P95/P99）反映真实用户体验**：比平均值更能揭示长尾延迟问题
5. **DeepSeek 不支持 json_schema，需要替代方案**：Prompt 指令 + 正则提取 + Pydantic 校验，配合降级逻辑兜底