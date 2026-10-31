# Agent Runtime

一个探索持续内部状态的 Agent Runtime 实验项目。

## 文档

- [核心设计](docs/design.md)
- [项目愿景](docs/vision.md)
- [Demo 规则](docs/demos.md)

## 当前 Demo

### Demo 1 — Emotion-guided Memory Retrieval

验证 Emotion 如何影响记忆检索。

```bash
cd demos/demo1-emotion-memory
python -m agent_runtime serve
```

Python 3.10+；离线模式无需额外依赖。详见 [Demo 说明](demos/demo1-emotion-memory/README.md)。
