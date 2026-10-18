# UI 层（Streamlit）

## 模块职责

Streamlit UI 是本地演示入口，进程内直调 LangGraph（不依赖 FastAPI）：

- 聊天式多轮对话；
- 新建会话 / 历史会话切换；
- clarify 消息展示；
- 来源面板与执行轨迹 expander；
- assistant 消息赞/踩反馈。

启动：

```bash
uv run --extra ui streamlit run src/ui/streamlit_app.py
```

## 页面结构

```text
侧栏：
  ＋ 新建会话
  历史会话列表
  运行参数 / max_iterations

主区：
  历史消息（user / assistant）
    assistant:
      - 澄清消息标记
      - 回答内容
      - 检索改写（如发生）
      - 来源 expander
      - 状态机细节 expander
      - 赞 / 踩
  st.chat_input
```

## 会话状态（st.session_state）

| key | 说明 |
|---|---|
| `current_session_id` | 当前会话；None 表示新会话，首条消息时创建 |
| `pending_query` | 预留；当前版本未使用 |

反馈按钮使用消息 id 生成唯一 key：

```text
up-<message_id>
down-<message_id>
```

点击后调用 `ConversationService.record_feedback`，然后 `st.rerun()`。
不会再次调用 LLM 或图。

## 一轮执行流程

1. `ConversationService.prepare_query`：有历史时 condense；
2. `add_user_message`：用户消息落盘；
3. `graph.astream_events`：流式执行 7 主链路 + clarify；
4. `EventView` 收集 token、来源与最终状态；
5. `record_assistant_state`：回答/澄清、来源、trace、指标落盘；
6. `st.rerun()`：从会话文件回放消息。

## clarify 展示

- `message_type == "clarify"` 时显示“需要澄清”提示；
- 内容为 `clarification_question`；
- 用户下一轮输入会进入 condense 历史，重新进图。

## 当前边界

- 知识库上传 / 来源列表 / 优质库移除尚未接入 UI（Phase 2/3）；
- 聊天页不再展示旧版流程轨道卡片；执行细节以 trace 文本 expander 呈现；
- 历史回放依赖会话 JSON 中的 `sources` / `trace` 快照，不重新检索；
- 单进程假设：会话文件锁为进程内锁。

## 测试

- `tests/test_conversation.py`：会话/反馈/来源落盘；
- `tests/test_ui.py`：EventView 事件蒸馏、8 节点状态、来源/诊断渲染；
- UI 入口本身需要手动 smoke test（需要真实 API key 或本地 LLM endpoint）。
