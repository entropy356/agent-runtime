# Demo：情绪如何改变记忆调用

Demo 1 检验一条很窄的因果链：持续的 Emotion → 记忆读取排序 → 认知得到的上下文 → 回答。它聚焦情绪对记忆检索的影响，不实现情绪调节 Clock 频率。

## 直接运行

需要 Python 3.10 或以上。离线模式只用标准库，无须安装依赖或提供密钥。

```bash
git clone https://github.com/entropy356/agent-runtime.git
cd agent-runtime/demos/demo1-emotion-memory
python -m agent_runtime serve
```

打开 http://127.0.0.1:8765 ，点击「初始化示例实验」，再点击「运行对照」。也可以从空状态直接聊天。

默认数据库为当前目录下的 `data/runtime.sqlite3`。所有全局参数都放在子命令**之前**：

```bash
python -m agent_runtime --db data/another.sqlite3 --rho 0.85 serve --port 8766 --interval 10
```

初始化只接受空数据库；不会清空已有工作。开始新实验时使用新的 `--db` 路径。

## 示例应看到什么

初始化会原子写入四条人工构造的记忆和三个不可覆盖的快照：

- `baseline`：共同记忆库，尚未初始化 Emotion。
- `A`：同一个记忆库，情绪来自“被支持，期待再次合作”的描述。
- `B`：同一个记忆库，情绪来自“被否定，担心再次受挫”的描述。

当前工作状态切换到 A。例子的初始状态是测试夹具，**不是 Agent 经历交互后学出来的结果**。记忆内容不是固定情绪类别，向量维度也没有命名。

输入“我们再一起试一次吧。”时，离线模式应得到：

| 试验 | 首条检索结果 | 模板引用 |
| --- | --- | --- |
| A，情绪读取开启 | #1，得到支持的合作经历 | #1 |
| B，情绪读取开启 | #2，遭到否定的合作经历 | #2 |
| A，情绪读取关闭 | #1 | #1 |
| B，情绪读取关闭 | #1 | #1 |

所有试验都使用 baseline 的完整记忆、相同输入、相同写入权重和相同 top-k。A/B 只提供情绪向量；关闭读取调节时，两组检索分数与选择完全相同。API 模式即使上下文相同，回答也可能因服务端非确定性而不同。

对照不提交状态，不增加使用次数，不写入记忆或运行日志。网页会暂停自动时钟；API 仍有模型请求和费用。

## 用交互形成自己的 A/B

1. 在「与 Agent 交互」输入经历，或使用「无输入运行一轮」。
2. 保存快照 `shared`。
3. 继续一段交互，保存快照 `my-A`。
4. 恢复 `shared`，走另一段交互，保存快照 `my-B`。
5. 在对照面板选择共同记忆 `shared`、A 情绪 `my-A`、B 情绪 `my-B`。

两个分支新产生的记忆不会进入对照，只有各自的情绪向量会被使用。也可以给共同记忆选择其他已有快照。这能避免把“记忆库不同”误当成“情绪影响检索”。

## 两种 Provider

**offline** 使用 256 维、SHA-256 稳定哈希的字符二元组向量；相似度主要反映文字重合，不具备语义理解。认知使用确定性模板，回答参考检索第一名，并显式引用其 ID。它只验证工程链路，不能评价真实对话表现。

**api** 接入同时支持 `/embeddings` 和 `/chat/completions` 的 HTTPS JSON API。基础地址包含服务所需前缀（例如 `/v1`），不要包含端点名。聊天模型需要遵守文档中的 JSON 协议；不支持该协议时会报错，不会偷偷退回离线模式。

在本地 shell 设置：

```bash
export AGENT_API_BASE='https://your-provider.example/v1'
export AGENT_CHAT_MODEL='your-chat-model'
export AGENT_EMBEDDING_MODEL='your-embedding-model'
read -rsp 'API key: ' AGENT_API_KEY; echo
export AGENT_API_KEY
python -m agent_runtime --mode api --db data/api.sqlite3 serve
```

程序不自动读取 `.env`。密钥只来自环境变量，不进入网页、数据库或错误正文。不同 provider、基础地址或 embedding 模型必须使用不同数据库；向量维度改变也会拒绝提交。**同名模型在服务端被替换、但维度不变，无法自动检测**，此时也应新建数据库。

API 模式用同一 embedding 模型分别编码记忆内容、当前输入以及自由文本感受。感受描述的语义空间只是情绪空间的实验性近似；语义相近不必然等于情绪相近。

认知提示只包含固定系统指令、当前输入、检索出的记忆 ID 与内容。**不传入 Emotion 向量、情绪描述或完整聊天历史**，使当前情绪对回答的作用经由检索产生。认知同时返回简短主观感受、引用 ID 和最多三条待保存记忆；不要求输出内部推理过程。

每次普通运行请求：一次输入 embedding（无外部输入时省略）、一次认知、一次本轮感受 embedding，再为每条新记忆请求内容和感受 embedding。一次 A/B 对照请求一次输入 embedding、四次认知。默认自动运行关闭。

## 实现机制

所有 embedding 向量先归一化。Emotion 初始化为第一次认知感受的编码，之前是 `null`，不赋予零向量“中性”含义。

每轮完成认知后：

```text
V = unit(embed(本轮感受描述))
E_next = rho * E + (1 - rho) * V
```

默认 `rho=0.85`；没有外部输入也可以继续认知。Emotion 本身不再次归一化，所以范数可变化。这个凸组合只是惯性近似，不预设一个人类意义的中性状态，也不声称其范数已校准为情绪强度。

记忆 `i` 的检索公式：

```text
relevance_i = max(0, dot(unit(embed(input)), memory_i.vector))
match_i = dot(E, memory_i.affect)
read_i = sigmoid(4 * match_i)        # 关闭读取调节时固定为 0.5
score_i = relevance_i * read_i * memory_i.write_strength
```

无外部输入时所有记忆的 `relevance=1`，不从墙上时间生成检索权重。按总分降序、ID 升序打破平局，最多取三条。所有分数与实际选择都能在界面查看。暂不设相关性阈值，小记忆库可能读到低相关内容。

新记忆拥有独立的 `affect=unit(embed(该记忆的感受描述))`。其持久写入权重为 `sigmoid(3 * dot(E_next, affect))`，影响未来检索；每条合法记忆都会保存，没有随机丢弃。示例夹具的初始权重统一设为 0.5，以隔离初始情绪的影响。对照中的消融只关闭**本轮读取调节**，保留共同记忆的历史写入权重，不是关闭 Emotion 的全部历史作用。

读取函数不修改状态。认知结果和所有 embedding 校验成功后，才在一个 SQLite 事务中提交 Emotion、新记忆、引用计数和运行日志。模型引用了未提供的 ID、返回无效 JSON、embedding 失败，均不提交本轮。revision 检查防止并发写入覆盖新状态。

引用 ID 是模型自报的使用代理，不等于内部因果归因。只增加明确引用的 ID 的次数，不因为“被检索”而增加。Cold 为 0–2 次，Warm 为 3–7 次，Hot 为至少 8 次；这些暂定标签没有物理缓存机制，也没有现实时间衰减。

Clock 为可暂停的固定间隔串行循环，不让情绪调整频率。自动轮次无新的外部输入，不把上一条用户消息当作新消息重放。每次自动轮完成后再等待间隔，不追赶错过的轮次。退出程序后不会继续运行；重启恢复内部状态，但自动运行仍关闭。

## 命令行与检查

```bash
python -m agent_runtime seed
python -m agent_runtime compare
python -m agent_runtime step '我们再一起试一次吧。'
python -m agent_runtime save after-chat
python -m agent_runtime restore A
python -m agent_runtime step --no-emotion
python -m agent_runtime state
python -m unittest discover -v
```

命令行输出 JSON，方便比较结果或用其他界面观察。`state` 包含完整状态、快照名、最近二十条观察日志；日志不是认知上下文。恢复快照不会删除审计日志，日志里可能出现不同分支的相同轮次。

网页服务只监听 `127.0.0.1`，没有外部资源依赖。它是本地实验界面，不是带用户鉴权的公网服务。若要从其他机器访问，需要单独设计访问控制与部署。

单元测试覆盖持久恢复、只读对照、消融不变性、惯性更新、写入权重、引用计数、失效回滚、并发版本冲突及 API 适配协议；HTTP 集成测试覆盖交互、快照、时钟、错误请求与来源检查。真实模型的语义表现需要配置 API 后实际观察，不能从离线测试结果外推。
