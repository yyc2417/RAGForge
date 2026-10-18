# 会话层（Conversation）

## 模块职责

会话层在图外维护多轮历史与查询改写，不修改 LangGraph 节点：

- 保存会话消息、来源、trace、反馈；
- 有历史时把追问 condense 成独立问题，再交给图；
- clarify 消息落盘，下一轮带上历史继续；
- 为 Streamlit UI 提供新建/切换/回放与反馈接口。

图仍是无状态的，每个 turn 用 standalone query 调用一次。

## 数据模型

```python
class SessionMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    message_type: Literal["answer", "clarify", "error"] = "answer"
    resolved_query: str | None
    sources: list[dict]
    trace: list[str]
    metrics: dict
    feedback: Literal["up", "down"] | None
    archived: bool

class Session(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[SessionMessage]
```

会话落盘为 `data/sessions/<session_id>.json`：

- 原子写：临时文件 + `os.replace`；
- 单进程锁；多进程后续换 SQLite；
- 文件损坏或 session_id 非法时跳过并告警，不阻塞列表。

## ConversationStore

| 方法 | 说明 |
|---|---|
| `create_session(title)` | 创建会话并落盘 |
| `list_sessions()` | 按 updated_at 倒序返回历史 |
| `load_session(session_id)` | 读取会话；不存在/损坏返回 None |
| `append_message(...)` | 追加消息；首条用户消息生成标题 |
| `update_feedback(...)` | 更新赞/踩与 archived 标记 |
| `delete_session(...)` | 删除会话文件 |

## ConversationService

### prepare_query

```python
standalone, reason = service.prepare_query(session_id, query)
```

- 无历史：直接返回原问题，不调用 LLM；
- 有历史：调用 `CONDENSE_PROMPT` + `CondenseResult`，把追问改写成独立问题；
- 只注入最近 `conversation_history_turns` 轮，并按 `conversation_token_budget` 从新到旧裁剪；
- LLM 失败/空输出：回退原问题，不阻塞主流程。

### record_assistant_state

把图最终状态写入 assistant 消息：

- `clarification_question` 存在 → `message_type=clarify`，content 为澄清问题；
- 否则 content 为最终回答；
- 保存 `sources`（最多 500 字符/条）、`trace`、`iteration_count`、`verify_failures`、`is_faithful`；
- 空回答不抛错，写入占位文案。

### record_feedback

`up` / `down` 更新会话消息；UI 点击反馈只改文件，不重新执行图。

## 配置

| 配置 | 默认 | 说明 |
|---|---|---|
| `session_dir` | `data/sessions` | 会话文件目录 |
| `conversation_history_turns` | 6 | 注入历史的轮数 |
| `conversation_token_budget` | 1200 | 历史文本 token 预算 |
| `clarify_enabled` | true | clarify 终止分支开关 |

## 与其他模块的边界

- 不依赖 `src/ui`；
- 依赖 `generation`（LLM / Prompt / Schema）与 `utils`；
- UI 依赖 conversation + agent graph；
- 不把会话状态塞进 `AgentState`，图保持无状态。

## 测试

`tests/test_conversation.py` 覆盖：

- 会话落盘/列表/删除/反馈；
- 损坏文件跳过；
- 有历史 condense、无历史跳过 LLM、LLM 失败回退；
- 回答来源落盘、clarify 消息落盘。
