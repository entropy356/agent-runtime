# Demo 的组织与演化

Demo 是围绕一个稳定问题持续演化的实验单元。

- **问题不变**：直接修改原 Demo。算法、参数、模型、实现和重构都由 Git 记录，不为它们创建新 Demo。
- **问题改变**：创建新的 Demo。
- **不过早抽象**：不同 Demo 可以暂时重复实现，只有稳定的共同需求出现后再提取共享模块。

## 实验记录放在哪里

实验在哪里发生，记录就跟随哪个 Demo。

```text
demos/
└── demo1-emotion-memory/
    ├── README.md
    ├── experiments/
    ├── results/
    ├── agent_runtime/
    └── tests/
```

- `README.md`：当前实验目标、实现方式、主要观察、结论、局限和运行方式。
- `experiments/`：有长期比较价值的实验记录，例如不同算法、参数或模型的对照。
- `results/`：需要长期保留的原始结果、指标或结构化数据。
- 普通函数调整、Bug 修复和重构只由 Git 记录，不额外生成实验记录。

实验记录应尽量关联对应的代码版本、配置、模型和测试条件，使后续结果可以比较和复现。

跨 Demo 的比较不放回某一个 Demo，而放在项目级 `docs/` 中，例如：

```text
docs/
└── comparisons/
    └── emotion-memory.md
```

这里记录跨实验、跨 Demo 得到的比较结果和项目级结论。仍然有效的结论可以进一步合并进 `docs/design.md`。

## 新建 Demo

Demo 使用稳定编号和简短主题名，例如：

```text
demos/
├── demo1-emotion-memory/
├── demo2-emotion-inertia/
└── demo3-emotion-clock/
```

编号一经使用保持不变。内部实现可以持续演化，不因函数或算法调整改变编号。
