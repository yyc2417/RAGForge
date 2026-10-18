# ADR-008：clarify 采用图内终止分支，而不是 interrupt

> **状态**：已采纳
> **日期**：2026-10-18
> **决策者**：宇诚

## 背景

通用 RAG 会遇到三类“先问清才能答”的问题：

1. 缺槽位：例如“它怎么配置？”，历史中没有“它”；
2. 多解释：同一问题有多个可能对象或意图；
3. 约束冲突：用户要求互相矛盾。

如果直接检索/生成，系统会把歧义猜成某个实体，浪费检索轮次，甚至给出错误答案。需要一种“先澄清、再继续”的机制。

## 决策

在 `analyze` 之后增加 **clarify 终止分支**：

```text
START → analyze ─(needs_clarification)→ clarify → END
                 └(else)→ decide → retrieve → evaluate → ...
```

- `analyze` 输出 `needs_clarification`、`ambiguity_type`、`clarification_question`；
- `route_after_analyze` 决定走 `clarify` 还是 `decide`；
- `clarify` 是终止节点：不检索、不生成，只返回澄清问题；
- 下一轮用户回答后，会话层把“原问题 + 澄清问题 + 用户回答”交给 condense，再重新进图；
- 使用 `clarify_enabled` 配置开关，便于开启/关闭对照。

### 节点数口径

- 旧 8 节点：含 `switch_strategy`；
- 固定 hybrid 后：7 节点主链路；
- 新增 clarify 后：8 节点（含 clarify，不含 switch）。

文档必须写清是哪一套，禁止无前缀的“8 节点”。

## 备选方案

### 方案 A：图外会话层检测（未选）

- 优点：图保持 7 节点；
- 缺点：澄清行为不可见；如果 analyze 已经做意图分析，会重复一次 LLM 调用；
- 结论：演示与可观测性较弱，不选。

### 方案 B：LangGraph interrupt / checkpointer（未选）

- 优点：能在同一图状态中暂停/恢复；
- 缺点：需要 checkpointer 与会话状态集成，复杂度高；
- 结论：当前用“终止节点 + 下一轮重新进图”已经满足需求，不引入。

### 方案 C：不澄清，直接回答（未选）

- 缺点：猜错实体、浪费检索、降低可信度；
- 结论：不选。

## 行为规则

- 只有缺槽位、多解释、约束冲突才问；
- 语料外问题走 refuse，不靠 clarify 拖延；
- 寒暄不问；
- 一轮最多澄清一次；用户说“随便/你决定”时按合理假设继续；
- 问题尽量带 2–3 个候选选项；
- condense 不应替用户消除真正的歧义。

## 后果

- 状态机多一个终止分支，节点数从 7 变 8；
- UI/API 需要区分“澄清”与“最终回答”；
- 新增 clarify 评测集（10–20 case），指标包括澄清准确率、过度澄清率、问题质量、补充后解决率；
- 过度澄清是主要风险，需要 `clarify_enabled` 与评测控制。

## 关联文档

- [ADR-007：固定 hybrid 与加权 RRF 优质通道](007-fixed-hybrid-and-quality-channel.md)
- [Agent 模块文档](../modules/agent.md)
- [会话模块文档](../modules/conversation.md)
- [UI 模块文档](../modules/ui.md)
