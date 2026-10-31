# Agent Runtime

一个探索具有持续内部状态的 Agent Runtime 的实验项目。

## 项目愿景

- [信息化历史](docs/vision.md)

## 设计

- [核心设计](docs/design.md)

## Demos

- [Demo 的组织与演化](docs/demos.md)

### Demo 1 — Emotion-guided Memory Retrieval

验证情绪向量如何影响记忆检索。

- [实验说明与运行方式](demos/demo1-emotion-memory/README.md)

运行（Python 3.10+，离线模式无需额外依赖）：

```bash
cd demos/demo1-emotion-memory
python -m agent_runtime serve
```

每个 Demo 都是独立实验，拥有自己的实现、说明和测试。
