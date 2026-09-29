# ADR-006：检索分数语义按策略分治 + 评估口径统一

> **状态**：已采纳
> **日期**：2026-09-12
> **决策者**：宇诚

## 背景

2026-09 全面代码审查发现：自适应决策中枢（evaluate 节点）的「分数硬证据通道」建立在**错误的分数语义**之上，且索引与评估口径存在系统性失真。三组问题相互叠加：

### 问题 1：绝对阈值不适用于所有检索策略

`evaluate` 节点的规则层使用统一阈值 `SCORE_LOW_RECALL_THRESHOLD = 0.3` 判定 low_recall，但三种检索策略返回的分数**量纲完全不同**：

| 策略 | 分数语义 | 取值范围 | 0.3 阈值是否有意义 |
|------|----------|----------|:---:|
| vector | cosine relevance score | [0, 1] | ✅ |
| bm25 | TF-IDF 加权 | [0, +∞)，无上界 | ❌（几乎总 > 0.3，规则层失效） |
| hybrid | RRF 融合分 | 上限约 2/(rrf_k+1) ≈ **0.033** | ❌（必然 < 0.3，必误判） |

最严重的后果：hybrid 策略下**即使检索完美**（LLM 判定 sufficient），规则层也会强制覆盖为 low_recall → switch_strategy，而 switch 后策略仍是 hybrid → 相同查询 + 相同策略 = 相同结果 → 空转循环直到 `max_iterations` 强制生成。所有 complex 查询固定多花 3 轮检索 + 3 次评估 LLM 调用（约 4 倍成本）。

### 问题 2：分数流入 LLM 评估 prompt

检索分数被原样格式化进 EVALUATE_PROMPT，并明文提示「分数 < 0.3 → low_recall」。rerank 生效时分数被置 0，LLM 评估者看到全 0 分必然误诊——代码注释只考虑了规则层对 0 分的过滤，忽略了分数还在流入 LLM。

### 问题 3：索引与评估口径失真

- `Chroma.from_documents` 不传 `ids`：对已存在 collection 只追加不去重，重复 chunk 在 RRF 中重复计票
- `load_or_build` 仅以「目录非空」判断复用：语料或 embedding 模型变更后静默复用陈旧索引
- Chroma 默认 l2 距离下 relevance score 可能为负，无法与阈值比较
- 评估脚本报告 `recall_top5` 但实际按 `retrieval_k=3` 检索；忠实度双口径（baseline 关键词命中 vs agent LLM verify）且所有失败路径默认「忠实」

## 备选方案

### 方案 A：统一归一化所有分数到 [0,1]
- **优点**：单一阈值，规则层实现最简单
- **缺点**：RRF/BM25 是排名信号，强行归一化丢失「向量分数有绝对质量语义」这一信息——top-1 的平庸向量匹配（cosine 0.31）与 top-1 的强匹配（0.85）归一化后无法区分，恰恰破坏 low_recall 检测
- **决定**：不选，丢失向量路的绝对语义

### 方案 B：彻底移除分数规则，全靠 LLM 语义判断
- **优点**：实现最简单，无量纲问题
- **缺点**：丢失「零成本硬证据」——向量分数过低是 LLM 判断之外的独立信号；LLM 诊断失败时的规则兜底也失去依据
- **决定**：不选，保留 vector 策略的廉价硬证据通道

### 方案 C：按策略分治（✅ 采纳）
- **概述**：绝对阈值仅对量纲明确的 vector 策略生效；bm25/hybrid 依赖「空结果硬规则 + LLM 语义判断」；分数彻底退出所有 LLM prompt
- **决定**：采纳

## 决定

### 1. 分数规则按策略分治（evaluate.py）

- `strategy == "vector"`：保留绝对阈值规则（cosine relevance ∈ [0,1]，语义明确）
- `bm25` / `hybrid`：仅保留「检索返回 0 条文档 → low_recall + switch_strategy」硬规则，其余交给 LLM 语义诊断
- LLM 诊断失败的规则兜底分支同样按策略 gate
- `_format_docs` 不再向任何 LLM prompt 注入分数（彻底解决 rerank 置零与量纲误导）
- retrieve 节点：rerank 后按文档身份回映射原 RRF 分数（不再置 0），保留 sources 展示价值

### 2. 索引一致性三件套（vector_store.py）

- build_index 写入前删除同名 collection（`from_documents` 对已存在 collection 只追加）
- 确定性 ID：`uuid5(source:chunk_index)`，同语料重复构建幂等
- collection metadata 记录 `embedding_model` 指纹；`load_or_build` 校验 count + 指纹，不匹配即重建；显式声明 `hnsw:space=cosine`

### 3. 评估口径统一（eval.py）

- 所有模式统一检索 `k=5`，`recall_top5` 名副其实
- 忠实度统一由 LLM 裁决（verify 同款 prompt，四模式共用）；关键词仅作 LLM 失败时的 fallback 并标记 `faithfulness_source`
- verify 校验失败返回 `is_faithful=None`（「未验证」），不再默认 True 向「忠实」偏置；失败样本计入 `errors`，不进忠实度/延迟分母
- 新增 `--rebuild`（清库重建）与 `--retrieval-only`（免 API 离线检索评估）；报告写入索引指纹（chunk 数 / 模型 / 分块参数）

### 4. 双预算防死循环（graph.py / state.py）

- `iteration_count`：检索总次数上限，retrieve 节点唯一递增（修复原 reformulate/switch/verify 三处递增导致一轮循环加 2 的预算虚耗）
- `verify_failures`：幻觉重试上限，verify 节点唯一递增——覆盖「改写无效短路」后 verify→reformulate→generate 不经过 retrieve 的循环
- 无效动作短路：已是 hybrid 时的 switch_strategy、改写无效的 reformulate，均直达 generate

### 5. token 化分块（chunker.py）

- chunk_size/overlap 单位从字符改为 token（cl100k 计量，220/30）：embedding 模型窗口 256 token，按字符切 500 字符的中文长块后半段被静默截断
- MarkdownHeaderTextSplitter 先按标题切节，标题路径注入 metadata 并拼入正文前缀

## 结果

- evaluate 节点 4 场景 mock 验收：hybrid 高质量检索不再误判（修复前必空转 3 轮）
- 状态机最坏路径（恒幻觉 + 恒 switch）在双预算内收敛（iter=3）
- 索引重复构建幂等（count 稳定），指纹不匹配自动重建
- 离线 recall@5（12 样本）：vector/bm25/hybrid/reranker 四策略全部 100%
- 完整基线已于同日（2026-09-12）在 45 题数据集上重跑完成，结果见 README 评估表与 `reports/eval_report.json`

## 参考资料

- [Cormack et al., RRF, SIGIR 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)（RRF 分数的排名信号本质）
- [sentence-transformers all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)（max_seq_length=256）
- 内部：`docs/planning/fix-log-2026-09.md`（2026-09 六阶段修复日志；本地私有文档，`.gitignore` 排除不入库）
