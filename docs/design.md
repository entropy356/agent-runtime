# Agent Runtime Design

## 核心概念

Agent Runtime 持续维护内部状态。每次运行都可以基于当前状态、外部输入和认知结果更新系统状态。

- **Emotion**：高维情绪状态，影响记忆读写强度和 Runtime 运行频率。
- **Memory**：由多条独立记忆组成，每条记忆附带情绪向量，并按实际使用频次动态分层。
- **Clock**：Runtime 的整体运行频率。
- **LLM**：参与理解、联想、推理、生成和决策的认知过程。

## Emotion

情绪表示为高维向量：

\[
E_k \in \mathbb{R}^n
\]

情绪空间不预设为固定的人类情绪标签，也不设定固定的中性基准状态。情绪可以在每次 Runtime 运行中更新，并允许自身影响自身及向量的其他维度。

情绪具有惯性：

\[
E_{k+1}=\rho E_k+\Delta E_k
\]

其中 \(\rho E_k\) 表示当前情绪对下一轮的延续，\(\Delta E_k\) 表示本轮认知与交互带来的变化。情绪变化量可以依赖当前情绪及本轮其他影响因素：

\[
\Delta E_k=f_E(E_k,\Omega_k)
\]

\(\Omega_k\) 表示外部输入、认知结果、行动结果及其他相关状态。

## Memory

Memory 由多条具有独立身份的记忆组成。每条记忆附带一个情绪向量：

\[
A_i \in \mathbb{R}^n
\]

当前情绪与记忆情绪的点积决定两者的匹配程度：

\[
s_i(k)=E_k^T A_i
\]

匹配程度影响记忆的读取和写入强度：

\[
r_i(k)=g_r(s_i(k)),\qquad
w_i(k)=g_w(s_i(k))
\]

例如，可以使用以下映射函数：

\[
g_r(s)=\sigma(\alpha s),\qquad
g_w(s)=\sigma(\beta s)
\]

读取与写入可以采用不同的映射函数。

记忆按实际使用频次动态分层，例如 Hot / Warm / Cold。层级描述同一种记忆的不同访问层次，而不是不同的记忆类型。记忆使用频次由实际使用行为产生，不依赖现实时间衰减。

**读取与状态更新分开：**

- 情绪影响记忆的激活和读取强度。
- 记忆被实际使用时，更新其使用频次。
- 读取本身不直接改变当前情绪或记忆内容。
- LLM 完成本轮认知后，认知结果进入统一的状态更新；此时可以更新 Emotion、Memory 及其他系统状态。

## Clock

Clock 表示 Agent Runtime 的整体运行频率。情绪可以影响运行频率：

\[
f_k=g_c(E_k)
\]

其中 \(f_k\) 表示 Runtime 的运行频率。Clock 控制系统运行节奏，不直接规定记忆检索、写入或分层算法。

现实时间用于外部调度；内部 Clock 表示 Runtime 自身的运行频率。

## LLM：认知过程

LLM 是认知过程，不是与 Emotion、Memory 并列的状态模块。它基于外部输入和当前系统状态进行理解、联想、推理、生成与决策，并产生认知结果、记忆内容、行动或状态更新。

\[
Y_k=LLM(X_k,S_k)
\]

\[
S_{k+1}=Update(S_k,Y_k,X_k)
\]

其中：

- \(X_k\)：本轮外部输入；
- \(S_k\)：当前系统状态，包含 Emotion、Memory 等；
- \(Y_k\)：LLM 产生的认知结果；
- \(S_{k+1}\)：更新后的系统状态。

LLM 的模型权重和上下文窗口不等同于 Agent 的持续记忆；LLM 生成的情绪描述也不等同于 Emotion 状态本身。

## 运行关系

```text
Current State + External Input
              ↓
        Memory Retrieval
              ↓
         LLM Cognition
              ↓
          State Update
        ┌─────┼─────┐
        ↓     ↓     ↓
     Emotion Memory Other State
        │
        └──────→ Clock Frequency
```

每次 Runtime 运行都可以更新系统状态。Clock 决定 Runtime 的运行频率；Emotion 影响 Memory 的读写强度和 Clock 频率；LLM 参与认知并产生状态更新所需的结果。

## 记号

- \(k\)：Runtime 运行轮次，不表示现实时间。
- \(i\)：一条具体记忆。
- \(E_k\)：第 \(k\) 轮的当前情绪向量。
- \(A_i\)：记忆 \(i\) 附带的情绪向量。
- \(\rho\)：情绪惯性参数。
- \(\Delta E_k\)：第 \(k\) 轮运行带来的情绪变化量。
