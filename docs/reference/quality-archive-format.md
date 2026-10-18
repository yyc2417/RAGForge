# 优质库归档格式

## 路径

```text
data/library/quality/<message_id>.md
```

`message_id` 是 assistant 消息的 UUID；同一消息重复点赞不会生成多份。

## 模板

```markdown
# Q: <改写后的独立问题>

- 原始问题：<用户原始输入>
- 归档时间：<ISO8601>
- 会话：<session_id>
- 消息：<message_id>
- 模型：<model>

## 回答

<最终回答>

## 参考来源

1. `handbook.md` · page 1 · chunk 2 · score 0.9000
   > 来源片段短摘要
```

## 规则

- `# Q:` 是 h1；现有切块器会把问题前缀带到每个 chunk；
- 参考来源只存短摘要（≤200 字符），不复制全文，避免与原文档重复竞争；
- 空回答 / 空问题不归档；
- clarify 消息不归档；
- 归档后写 `source_type=qa_archive`，检索时进入加权 RRF 优质通道；
- 移除归档 = 删除文件 + 重建索引；对应会话消息标记 `archived=False`；
- 不做自动正确性校验；错误归档由人工移除。

## 加权通道

```text
quality_weight = 1.3
quality_top_k = 3
quality_min_relevance = 0.3
quality_max_per_source = 2
```

- 只对 `source_type=qa_archive` 的文档做 metadata filter；
- 按 `quality_weight / (rrf_k + rank)` 计入融合分；
- 低于最低相关度的优质文档不参与；
- 同一来源最多计入 `quality_max_per_source` 个 chunk；
- BM25 不参与优质通道；
- 优质库为空/通道失败时自动跳过，不影响普通 hybrid。

## 边界

- 点赞只代表用户认可，不代表事实正确；
- 归档答案可能过期或与后续文档冲突；
- 不提供自动过期、自动复核或版本回滚；移除后重建索引即可。
