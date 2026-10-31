# Demo 1：Emotion-guided Memory Retrieval

验证一条窄因果链：

```text
Emotion → Memory Retrieval → LLM Context → Response
```

这个 Demo 只研究 Emotion 对记忆读取的影响，不研究 Emotion 对 Clock 的调节。

## 运行

需要 Python 3.10+。离线模式只使用标准库。

```bash
git clone https://github.com/entropy356/agent-runtime.git
cd agent-runtime/demos/demo1-emotion-memory
python -m agent_runtime serve
```

打开 `http://127.0.0.1:8765`，初始化示例实验后即可运行 A/B 对照。

默认状态保存在 `data/runtime.sqlite3`。开始新的独立实验时使用新的数据库路径。

## 实验

示例初始化同一个记忆库，并构造两个不同的 Emotion 状态：

- **A**：偏向“得到支持、愿意再次合作”的经历；
- **B**：偏向“遭到否定、担心再次受挫”的经历。

这些状态是实验夹具，不是 Agent 自己经历交互后学习出的结果。

输入：

```text
我们再一起试一次吧。
```

离线模式下应看到：

| 状态 | Emotion 读取开启 | Emotion 读取关闭 |
| --- | --- | --- |
| A | 优先检索支持合作的记忆 | 与 B 相同 |
| B | 优先检索受挫合作的记忆 | 与 A 相同 |

A/B 使用相同输入、相同记忆库和相同其他条件，只替换 Emotion。关闭 Emotion 读取调节后，两组检索结果应一致。

也可以通过快照从共同状态分叉，形成自己的 A/B；对照时只替换 Emotion，避免把记忆库差异误认为 Emotion 的作用。

## 机制

Emotion 由每轮认知产生的“感受描述”编码，并以惯性更新：

```text
V = unit(embed(feeling))
E_next = rho * E + (1 - rho) * V
```

默认 `rho=0.85`。

记忆检索同时考虑输入相关性、Emotion 匹配和历史写入强度：

```text
relevance_i = max(0, dot(unit(embed(input)), memory_i.vector))
match_i = dot(E, memory_i.affect)
read_i = sigmoid(4 * match_i)
score_i = relevance_i * read_i * memory_i.write_strength
```

关闭 Emotion 读取调节时，`read_i` 固定为 `0.5`。

每条新记忆也带有独立的 affect 向量。Emotion 与 affect 的匹配会影响其持久写入权重。读取本身不修改状态；认知与 embedding 全部成功后，才统一提交 Emotion、新记忆和使用记录。

记忆按实际引用次数暂分为 Cold / Warm / Hot，不使用现实时间衰减。Clock 在本 Demo 中只是固定间隔循环，不受 Emotion 调节。

## Provider

### offline

使用确定性的哈希字符向量和模板认知，用于验证工程链路。它没有真正的语义理解，不能据此评价真实对话效果。

### api

认知与嵌入可以来自不同服务商：两侧各有基础地址、密钥和协议，代理也可以只加在其中一侧。

```bash
# 认知：任意 OpenAI 兼容的 /chat/completions
export AGENT_CHAT_BASE='https://your-llm.example/v1'
export AGENT_CHAT_MODEL='your-chat-model'
export AGENT_CHAT_KEY='your-chat-key'

# 嵌入：openai（/embeddings）或 gemini（:embedContent）
export AGENT_EMBED_PROTOCOL='openai'
export AGENT_EMBED_BASE='https://your-embedding.example/v1'
export AGENT_EMBED_MODEL='your-embedding-model'
export AGENT_EMBED_KEY='your-embedding-key'
export AGENT_EMBED_DIMENSIONS='768'
export AGENT_EMBED_PROXY='http://127.0.0.1:8080'   # 可选，仅嵌入侧

python -m agent_runtime --mode api --db data/api.sqlite3 serve
```

`AGENT_EMBED_*` 省略时回落到 `AGENT_CHAT_*`；`AGENT_API_BASE` / `AGENT_API_KEY` 仍可顶替 `AGENT_CHAT_*`，即旧的单端点模式。

数据库绑定嵌入身份（协议、模型、维度），不绑定聊天模型：换聊天模型可复用数据库，换嵌入模型或维度必须新建。

API 模式用同一个 embedding 空间编码文本记忆、输入和感受描述。这里把语义向量暂时作为 Emotion 空间的实验性近似，并不假设“语义相近”等同于“情绪相近”。

LLM 不直接接收 Emotion 向量；Emotion 对回答的影响只通过记忆检索产生。

## 命令行

```bash
python -m agent_runtime seed
python -m agent_runtime compare
python -m agent_runtime step '我们再一起试一次吧。'
python -m agent_runtime save after-chat
python -m agent_runtime restore A
python -m agent_runtime state
python -m unittest discover -v
```

更多参数见：

```bash
python -m agent_runtime --help
```

## 局限

当前 Demo 只验证机制是否能形成可观察的检索差异。

- Emotion 空间仍是实验性近似；
- 示例 A/B 是人工构造的；
- 当前读写函数和参数只是暂定实现；
- 离线模式不能验证真实语义效果；
- 本 Demo 不验证 Emotion 对 Clock 的影响。
