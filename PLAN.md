# UFO v2 Paper Plan

## 核心故事

**传统 multi-turn / ICL 方法需要 domain-specific feedback（代码要 runtime error，定理要 type-check error，搜索要 retrieval result）。UFO 只用一个 domain-agnostic 的 "Try Again"，在所有 domain 上都能 work。**

一句话：**One Feedback Fits All.**

---

## Paper 结构

### 1. 实验层：Universal Effectiveness

**目标**：证明 "Try Again" 在每个 domain 都有效，不只是数学。

**环境**（6 个代表性 domain）：

| 类别 | 环境 | 为什么选 |
|------|------|---------|
| Math | MetaMathQA | 论文主实验，已有数据 |
| Planning | Sokoban | 空间推理，和数学完全不同 |
| QA | HotpotQA | 多跳推理 |
| Constraint | Sudoku | 约束求解 |
| Stochastic | FrozenLake | 有随机性，失败不完全是模型的错 |
| Interactive | WebShop | 长链决策，agentic 任务 |

**实验设计**：
- 每个环境用 Qwen2.5-3B 训 200 步
- 对比 base model vs single-turn RL vs UFO (5-turn)
- 统一指标：Succ@5、Succ@1、conditional success rate

**预期输出**：一张大表（类似原论文 Table 1 但覆盖更多类型的 domain）

**要做的事**：
- [ ] 跑 5 个新环境的训练（MetaMathQA 已有数据）
- [ ] 跑对应的 evaluation
- [ ] 汇总结果

---

### 2. 迁移层：Cross-Domain Transfer Matrix

**目标**：证明 revision skill 是 domain-agnostic 的，能跨域迁移。

**实验设计**：
- 在每个环境上训练一个 UFO 模型
- 用每个训好的模型去测所有其他环境
- 得到 6×6 的迁移矩阵

**预期发现**：
- 对角线（in-domain）效果最好
- 但 off-diagonal 也有明显提升 → revision skill 可迁移
- 某些 domain pair 迁移性更强（比如 Math → QA 可能比 Math → Planning 强）
- 矩阵本身就是一个发现：可以分析 revision skill 的结构

**要做的事**：
- [ ] 每个环境训一个模型（实验层已做）
- [ ] 每个模型在所有 6 个环境上 eval（6×6 = 36 次 eval）
- [ ] 画迁移矩阵热力图
- [ ] 分析哪些 domain pair 迁移性强/弱，给出解释

---

### 3. 分析层：In-Context Learning 视角

**目标**：解释 WHY — UFO 到底在教模型什么？

#### 3a. ICL Decomposition

**核心问题**：UFO 的提升有多少来自权重更新，多少来自变成更好的 in-context learner？

**实验设计**：
- Base model + ICRL prompting（把错误尝试 + "Try Again" 放进 context，不训练）
- UFO-trained model + 正常多轮交互
- 对比两者的 Succ@5

**预期发现**：
- Base model 用 ICRL prompting 也能有一些提升（LLM 本身就有 in-context revision 能力）
- UFO 训练后提升更大 → RL 训练增强了 in-context revision 能力
- 差值 = RL 训练的增量贡献

**MetaMathQA 已有数据**（Qwen2.5-3B, 5 轮, Succ@5）：
- Base + no feedback: 53.9%
- Base + unary feedback: 61.5% → unary feedback 信号的 ICL 贡献 +7.6pp
- UFO-trained + unary feedback: 92.7% → RL 训练贡献 +31.2pp

**要做的事**：
- [x] MetaMathQA 的 ICL decomposition（已有数据）
- [ ] 在其他 5 个新环境上跑同样的 decomposition（需要 base model 多轮 eval）
- [ ] 画柱状图：base / base+unary feedback / UFO-trained

#### 3b. Feedback Prompt Robustness

**核心问题**：模型学到的是 "Try Again" 这个字符串，还是 "被告知失败" 这个信号？

**实验设计**：

*推理时换 prompt*（原论文 Fig 9 已有部分数据）：
- 训练用 "Try Again"，推理换成 "Incorrect"、"Please think again"、"Wrong, try differently" 等
- 验证推理时 robustness

*训练时随机化 prompt*（新实验）：
- 训练时从一组同义 prompt 中随机抽取：
  - "Try again"
  - "That's incorrect, please try again"
  - "Wrong answer, think differently"
  - "Not quite right, give it another shot"
  - "Incorrect. Please reconsider."
  - ...
- 对比 fixed prompt vs random prompt 训练效果

**预期发现**：
- 训练时随机化不影响甚至略微提升性能
- 进一步证明模型学到的是 revision skill 而非特定字符串

**要做的事**：
- [ ] 在 ctx_manager.py 中实现 random feedback prompt 功能
- [ ] 在 MetaMathQA 上跑 random prompt 训练实验
- [ ] 对比 fixed vs random prompt

---

### 4. 理论层：Feedback Information Scaling

**目标**：从信息论角度理解 "为什么 1 bit 就够了"。

**实验设计**：
在同一个环境（MetaMathQA）上对比不同信息量的 feedback：

| 信息量 | Feedback 内容 | 已有数据？ |
|--------|-------------|-----------|
| 0 bit | 无反馈（空 observation） | ✅ 已有 |
| ~1 bit | "Try Again" | ✅ 已有 |
| ~3 bit | "Wrong, try a different approach" | 需要跑 |
| ~5 bit | "Wrong, check your arithmetic" | 需要跑 |
| ~10+ bit | 完整 critique | ✅ 已有部分 |

**预期发现**：
- 0 → 1 bit 的提升最大（从无到有）
- 1 bit → 多 bit 的提升边际递减
- Pareto curve 表明 "Try Again" 接近最优信息效率点

**要做的事**：
- [ ] 实现 2-3 种中间粒度的 feedback
- [ ] 在 MetaMathQA 上训练对应模型
- [ ] 画 information bits vs Succ@5 的 Pareto curve

---

## 实验优先级

| 优先级 | 实验 | 成本 | 依赖 |
|--------|------|------|------|
| **P0** | 6 个环境各训 200 步 | 6 次训练 | 无 |
| **P0** | 6×6 迁移矩阵 eval | 36 次 eval | P0 训练完 |
| **P1** | Base model ICRL prompting | 6 次 eval（不需训练） | 无 |
| **P1** | Random feedback prompt 训练 | 1 次训练 | 改 ctx_manager |
| **P2** | 中间粒度 feedback 实验 | 2-3 次训练 | 实现新 feedback 类型 |

---

## 需要实现的代码改动

### 已就绪
- [x] 13 个环境全部可用
- [x] 28 个 per-env config
- [x] 通用 train.sh / eval.sh
- [x] 数据下载脚本

### 需要实现
- [ ] Random feedback prompt 功能（改 ctx_manager.py，加一个 prompt pool）
- [ ] ICRL prompting 评估模式（base model 不训练，只在 context 中放历史）
- [ ] 中间粒度 feedback 实现（2-3 种新的 feedback template）
- [ ] 迁移矩阵可视化脚本

---

## Timeline（建议）

**Week 1-2**：跑 P0 实验（6 个环境训练 + 迁移矩阵 eval）
**Week 3**：跑 P1 实验（ICRL decomposition + random prompt）
**Week 4**：跑 P2 实验（feedback information scaling）
**Week 5-6**：分析结果 + 写 paper

---

## 预期 Figures

1. **Table 1（大表）**：6 个 domain × {base, single-turn RL, UFO} 的 Succ@1/5
2. **Fig 1（迁移矩阵）**：6×6 热力图
3. **Fig 2（ICL Decomposition）**：base / base+ICRL / UFO 的柱状图
4. **Fig 3（Prompt Robustness）**：fixed vs random prompt 的对比
5. **Fig 4（Information Scaling）**：feedback bits vs Succ@5 的 Pareto curve

---

## Related Work 要提的

- **Guru**（UCSD, 2506.14965）：single-turn RL 的跨域迁移，发现有些 domain 必须 in-domain。我们研究 multi-turn + minimal feedback 的跨域迁移
- **SCoRe**（ICLR 2025）：multi-turn self-correction via RL。用自生成纠错，我们用 external minimal feedback
- **ERL**（2026）：experience-reflection-consolidation loop。用显式 self-reflection，我们用隐式 "Try Again"
- **"Reward Is Enough"**（2025）：LLM 在推理时就能做 ICRL。我们通过 RL 训练增强这个能力
- **"No Need for Explanations"**（2025）：LLM 能从错误中隐式学习，不需要解释。直接支持 UFO 的设计
- **S²R**（ACL 2025）：self-verify + self-correct via RL
