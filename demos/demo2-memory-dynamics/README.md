# Demo 2：Memory Dynamics

单独验证记忆的逐轮衰减、激活恢复与稳定性累积。它不依赖 Emotion，也不修改 Demo 1。

**不做前端。保留真实 LLM 与 Embedding 接入。**

## 要验证的问题

- 每条记忆的当前权重是否随 Runtime 轮次逐渐衰减？
- 成功使用记忆时，权重恢复是否能提高再次被检索到的机会？
- 激活是否同时提高稳定性，并降低后续衰减速度？
- 双状态模型相较于只有权重的模型，是否产生可观察的长期差异？
- 这些差异在真实 Embedding 检索与 LLM 认知链路中是否仍然存在？

## 候选模型

每轮认知开始时，所有已有记忆先衰减一次：

\[
w^- = w \exp[-\lambda(1-\gamma s)]
\]

只有 LLM 明确报告本轮实际使用了某条记忆，才对它执行激活：

\[
w' = w^- + (1-w^-)r
\]

\[
s' = s + (1-s)\beta(1-w^-)
\]

- \(w\)：当前权重，表示记忆此刻的可访问程度。
- \(s\)：稳定性，表示记忆抵抗后续衰减的能力。
- \(\lambda\)：基础逐轮衰减率。
- \(\gamma\)：稳定性对衰减的保护程度。
- \(r\)：成功激活后的权重恢复比例。
- \(\beta\)：稳定性增益。

稳定性增益随激活前权重降低而增大，是一个待验证假设：较难成功提取的记忆，可能从这次成功提取中获得更大的长期强化。它不是已经确定的人脑定律。

检索分数暂定为：

\[
score_i = \frac{1+\cos(q,m_i)}{2} \cdot w_i
\]

稳定性不直接给检索分数加成，只影响未来衰减。这让实验可以区分「当前更容易被找到」与「未来更不容易衰减」。

## 对照组

- **weight-only**：只有权重衰减与恢复，稳定性固定为 0。
- **two-state**：权重与稳定性独立变化；激活恢复权重并提高稳定性。

两组使用相同场景、Embedding、LLM、top-k 和参数。分别使用独立数据库，避免状态互相污染。

## 运行

需要 Python 3.10+；无第三方 Python 依赖。

### 离线机制测试

离线模式使用确定性哈希字符向量和固定模板认知，只验证状态更新、检索链路、持久化和边界条件，不代表真实语义检索能力。

在本目录运行：

\`\`\`bash
python -m unittest discover -v
python -m memory_dynamics --db data/offline.sqlite3 run scenarios/spacing-and-interference.json --output results/offline-two-state.json
\`\`\`

### 真实 API 测试

Chat 使用 OpenAI-compatible \`/chat/completions\`；Embedding 可以使用 OpenAI-compatible \`/embeddings\` 或 Gemini 原生 \`:embedContent\`。两者可使用不同服务商、模型、密钥、代理和超时。

\`\`\`bash
export AGENT_CHAT_BASE='https://your-chat.example/v1'
export AGENT_CHAT_MODEL='your-chat-model'
export AGENT_CHAT_KEY='your-chat-key'

export AGENT_EMBED_PROTOCOL='openai'  # 或 gemini
export AGENT_EMBED_BASE='https://your-embedding.example/v1'
export AGENT_EMBED_MODEL='your-embedding-model'
export AGENT_EMBED_KEY='your-embedding-key'
export AGENT_EMBED_DIMENSIONS='768'   # 可选，按服务商支持情况设置

# 可选：分别配置代理与超时
export AGENT_CHAT_PROXY='http://127.0.0.1:8080'
export AGENT_EMBED_PROXY='http://127.0.0.1:8080'
export AGENT_CHAT_TIMEOUT='120'
export AGENT_EMBED_TIMEOUT='60'
\`\`\`

先跑双状态组：

\`\`\`bash
python -m memory_dynamics --mode api --dynamics two-state --db data/api-two-state.sqlite3 run scenarios/spacing-and-interference.json --output results/api-two-state.json
\`\`\`

再跑单权重对照组：

\`\`\`bash
python -m memory_dynamics --mode api --dynamics weight-only --db data/api-weight-only.sqlite3 run scenarios/spacing-and-interference.json --output results/api-weight-only.json
\`\`\`

每组都从同一份场景重新初始化。Chat 请求使用 temperature 0，但服务端仍可能存在非确定性；比较时应保留两份完整 JSON 结果、模型标识、Embedding 标识和代码 commit。

也可以单独初始化后逐轮交互：

\`\`\`bash
python -m memory_dynamics --mode api --db data/interactive.sqlite3 init scenarios/spacing-and-interference.json
python -m memory_dynamics --mode api --db data/interactive.sqlite3 step 'Mira prefers what kind of update?'
python -m memory_dynamics --mode api --db data/interactive.sqlite3 tick
python -m memory_dynamics --mode api --db data/interactive.sqlite3 state
\`\`\`

每次 \`step\` 或 \`tick\` 都算一个 Runtime 轮次。现实时间不参与衰减。

## 观察什么

每轮 trace 会保留：

- 每条记忆衰减前后的权重与实际衰减率；
- 检索排序、相关性、权重和最终分数；
- LLM 的回答、实际引用的记忆 ID、新记忆提案；
- 被实际使用的记忆在激活前后的权重与稳定性；
- 新写入记忆的初始状态。

重点比较：

1. 同样经过若干轮后，未被使用的记忆是否逐渐退出检索结果；
2. 被成功使用的记忆是否恢复权重；
3. 重复成功提取是否逐渐提高稳定性并减缓后续衰减；
4. 间隔较久后重新成功提取的记忆，是否出现更大的稳定性增益；
5. 双状态组是否只是让旧记忆长期垄断上下文，或确实改善了有用记忆的再出现。

LLM 自报的 \`used_memory_ids\` 是实际使用的代理指标，不是认知过程的客观测量。需要人工抽查回答是否真的依赖对应记忆。

## 状态与边界

- 仅在成功完成认知并通过校验后，才原子提交本轮状态。
- 只检索到、但没有被 LLM 报告为实际使用的记忆，不会恢复权重或提高稳定性。
- 新记忆以 weight=1、stability=0 写入，不在创建轮立即衰减。
- 读取命令是只读的，不推进轮次。
- 数据库绑定 Embedding 模型与维度，也绑定动态参数；更换模型或参数时使用新数据库。
- API adapter 的单元测试使用 mock；真实服务连通性、模型行为与语义质量需要在可用网络环境中实测。

参数目前只是实验起点，不是拟合结果。不要把这一版公式直接当作最终记忆设计。
