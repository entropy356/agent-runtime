# Agent Runtime

一个探索持续内部状态的 Agent Runtime 实验项目。

目标是**像人**，不是可靠或有用：它允许记错、执着于旧结论、因为当下情绪而想起不同的事。

## 文档

- [核心设计](docs/design.md)
- [项目愿景](docs/vision.md)
- [Demo 规则](docs/demos.md)

## 当前 Demo

### Demo 1 — Emotion 与 Memory 的相互作用

观察 Emotion 如何影响记忆的读写，以及被读到的记忆如何反过来改变 Emotion。

```bash
cd demos/demo1-emotion-memory
python -m agent_runtime serve
```

Python 3.10+；离线模式无需额外依赖。详见 [Demo 说明](demos/demo1-emotion-memory/README.md)。
