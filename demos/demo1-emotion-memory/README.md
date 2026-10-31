# Demo 1：Emotion 与 Memory 的相互作用

研究 Emotion 与 Memory 之间的双向作用，而不是单向的「情绪影响检索」：

- Emotion 通过 `read_i` 决定哪些记忆进入上下文；
- Emotion 通过 `write_strength` 决定新记忆以后被读到的难易；
- 被读到的记忆改变认知，认知产生的感受又更新 Emotion。

关心的是这个闭环能否产生可观察、可复现的差异，以及差异出在哪一段。Clock 不在范围内。

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

分两层看。

**夹具对照**：固定记忆库和输入，只替换 Emotion，看读取排序是否随之改变。这层回答「`read_i` 是否真的起作用」。

**连续对话**：不设脚本，逐轮输入，让闭环自己跑。看的是检索结果如何进入认知、认知产生的感受如何更新 Emotion、下一轮又检索到什么。这层回答「闭环是否稳定、会不会漂移或退化」。

夹具对照的构造：同一个记忆库，两个不同的 Emotion 状态：

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

记忆检索同时考虑输入相关性、Emotion 匹配和历史写入强度。所有比较都在中心化空间里进行：查询、记忆内容、Emotion 与记忆 affect 各自减掉本空间的库内均值，再归一化。

```text
relevance_i = (1 + dot(q, v_i)) / 2       # 中心化余弦移位，0.5 = 与库内平均内容相同
match_i = dot(e, a_i)                     # 中心化余弦
read_i = sigmoid(4 * match_i)
score_i = relevance_i * read_i * memory_i.write_strength
```

中心化不是修饰。19 条 768 维记忆两两余弦均值 0.739、均值向量范数 0.868，不减掉这个公共分量时 `read_i` 恒落在 0.92–0.95，换哪条记忆都一样；减掉后跨度从 0.027 升到 0.353。

`relevance_i` 移位到 `[0, 1]` 而不是截断：均值按定义让约一半记忆落在负半轴，截断会把它们压成同一个 0，排序退化为按 ID。`0.5` 因此有明确含义——与库内平均内容相同。

均值取自当前记忆库，增删记忆会改变所有分数；库为空时没有均值，退化为未中心化。

关闭 Emotion 读取调节时，`read_i` 固定为 `0.5`。

每条新记忆也带有独立的 affect 向量。`write_strength` 同样以写入前记忆库的感受均值中心化，`0.5` 表示与库内既有感受无差别；它在写入时算定，之后不再改变。读取本身不修改状态；认知与 embedding 全部成功后，才统一提交 Emotion、新记忆和使用记录。

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

LLM 不直接接收 Emotion 向量，Emotion 对回答的影响经由检索间接产生。

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

这个 Demo 只验证闭环能否形成可观察、可复现的差异，不证明这种差异有用，也不证明它接近人类情绪。

- Emotion 空间是语义向量的实验性近似；
- 夹具 A/B 是人工构造的；
- 使用频次只输出 Cold / Warm / Hot 标记，不参与读写计算；
- 记忆一旦写入不会被覆盖、合并或失效，过期内容会持续参与检索；
- `write_strength` 在写入时算定，之后不随使用改变；
- Emotion 的范数在中心化后不进入任何公式；
- 离线模式不能验证真实语义效果；
- 不验证 Emotion 对 Clock 的影响。
