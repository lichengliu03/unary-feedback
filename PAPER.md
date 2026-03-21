# Paper Draft: One-Bit Feedback is All You Need

---

## Abstract

LLM 的多轮交互通常依赖 domain-specific 的反馈信号——代码任务用 runtime error [1]，搜索任务用检索结果 [3]，embodied 任务用环境反馈驱动显式 self-reflection [2]。我们研究一个基本问题：多轮推理究竟需要多少反馈信息？我们发现，**one-bit feedback**——仅告知模型"你的答案不正确"的单一信号，不包含任何关于错误原因或修正方向的信息——就足以在多种不同类型的任务上实现有效的多轮推理提升，涵盖数学推理、空间规划、约束求解、多跳问答、随机导航和交互式网页任务。通过 8 个代表性环境的实验，我们证明：（1）one-bit feedback 在每个 domain 上都能持续提升 Succ@5；（2）这个能力可以跨域迁移——在数学上训练的模型，在规划任务上依然能有效利用 one-bit feedback；（3）这种提升源于模型固有的 in-context learning from failure 能力，RL 训练通过 RL 强化了这个能力。我们的发现表明，从最小失败信号中学习是语言模型的一种基本的、domain-agnostic 的能力。

---

## 1. Introduction

**第一段：背景**

LLM 的多轮问题求解是一个重要但具有挑战性的能力。现有的多轮方法通常依赖 domain-specific 的反馈信号来实现 in-context 的迭代纠错：SDPO [1] 利用 runtime error 作为文本反馈来蒸馏代码生成策略；ERL [2] 使用环境反馈驱动显式的 self-reflection 来指导第二次尝试；Search-R1 [3] 利用检索结果作为多轮搜索反馈；MINT [4] 综合使用 tool execution 和 language feedback 进行多轮评估。这些方法虽然有效，但每个 domain 都需要单独设计反馈机制（代码要 runtime error，搜索要 retrieval result，embodied 要 simulator signal），限制了方法的通用性。

**第二段：核心问题**

这引出了一个基本问题：**多轮推理到底需要多少反馈信息？** 是否一定需要 domain-specific 的、包含丰富纠错信息的反馈？还是说，仅仅告知模型 "你错了" 就够了？我们研究 **one-bit feedback**——最小的反馈信号，仅传递 1 bit 的信息（correct/incorrect），不包含任何关于错误原因、位置或修正方向的信息。我们发现，这个 domain-agnostic 的信号在 8 种完全不同类型的任务上都能产生显著的多轮推理提升：数学推理（MetaMathQA [11]）、组合数学（Countdown [24]）、空间规划（Sokoban [21]）、多跳问答（HotpotQA [12]）、约束求解（Sudoku [15]）、随机导航（FrozenLake [22]）、交互式网页决策（WebShop [13]）和 embodied 家务任务（ALFWorld [25]）。更值得注意的是，在一个 domain 上训练的模型，迁移到其他 domain 后依然能有效响应 one-bit feedback。

**第三段：为什么 1 bit 就够**

需要澄清的是：one-bit feedback 本身只传递 1 bit 的外部信息（"你错了"），但模型在多轮交互中能看到自己之前的完整回答。因此模型可利用的总信息远不止 1 bit——它可以结合 context 中的失败历史进行自我诊断，推断哪里可能出错。我们的发现恰恰是：**这种自我诊断的能力是 LLM 固有的，外部只需要提供一个最小的触发信号。** 在没有 one-bit feedback 的条件下，模型同样能看到自己的历史回答，但它不知道这些回答是错的，因此可能重复同样的错误。One-bit feedback 的 1 bit 信息告诉模型"你错了"，触发它去利用 context 中的失败历史进行 self-correction——真正的纠错工作由模型自身完成。

已有研究表明，LLM 具备从 context 中隐式学习的能力——即使没有显式的纠错解释，模型也能从错误样例中推断出正确方向 [6]，甚至在多轮交互中展现出类似 RL 的 in-context 学习行为 [5]。我们的 ICL decomposition 实验（§3.3）进一步验证了这一点：未经训练的 base model 接收到 one-bit feedback 后就能产生一定提升，而 RL 训练显著放大了这个能力。从认知科学的角度看，one-bit feedback 的作用类似于 impasse signal [8]——它不提供解题信息，但触发 learner 从 exploitation（重复当前策略）切换到 exploration（搜索替代方案）。这解释了为什么一个不包含任何 domain knowledge 的信号能在所有 domain 上生效：它激活的不是 domain-specific 的纠错，而是模型固有的 self-correction 能力。

**第四段：贡献**
- 提出并回答了一个基本问题：多轮推理需要多少外部反馈信息？1 bit 就够了——仅告知模型"你错了"，不提供任何纠错细节，就足以触发有效的多轮推理提升
- 在 8 种涵盖数学、规划、QA、约束求解、交互决策等完全不同类型的 domain 上验证了 one-bit feedback 的 universal effectiveness
- 通过 8×8 跨域迁移矩阵证明模型学到的是 domain-agnostic 的 self-correction 能力，而非特定 domain 的纠错技巧
- 通过 ICL decomposition 和 self-correction analysis 揭示了机制：LLM 天然具备从失败中自我修正的能力，RL 训练放大了这一能力
- Information scaling 实验表明 one-bit feedback 已接近反馈信息的效率最优点，更丰富的 domain-specific 反馈仅带来有限的边际收益

> **[Fig 1]** 放在 Introduction
> 内容：8 个 domain 的 Succ@k 曲线（k=1~5）并排展示
> Insight：一个不包含任何 domain knowledge 的 1-bit 信号，在从数学到规划到交互决策的 8 种完全不同的任务上都能持续提升多轮推理成功率——one-bit feedback 是 universal 的

---

## 2. Method

简要回顾我们的 one-bit feedback 框架：
- MDP formulation：state = question + history，action = answer，reward = binary
- One-bit feedback as observation：错误时给 one-bit feedback（仅 "incorrect" 信号），正确时终止 episode
- 训练时 one-bit feedback 的具体措辞从一个 prompt pool 中随机抽取（如 "Incorrect"、"That's wrong, try again"、"Not quite right" 等），确保模型学到的是 1-bit 信号本身而非特定字符串
- PPO 训练 [9] + reward decay + repetition penalty

**这部分简短，不是本文重点。**

---

## 3. Experiments

我们通过五组实验系统回答以下问题：

| 小节 | 问题 | 回答 |
|------|------|------|
| §3.1 | 有效吗？ | 8 个 domain 都有效 |
| §3.2 | 跨域吗？ | 跨域迁移有效 |
| §3.3 | 提升从哪来？ | 固有 ICL + RL 放大 |
| §3.4 | RL 具体增强了什么能力？ | self-correction（从失败中修正） |
| §3.5 | 需要更多信息吗？ | 不需要，1 bit 接近最优 |

### 3.1 Universal Effectiveness（核心实验）

**设定**：8 个环境，每个用 Qwen2.5-3B [10] 用 one-bit feedback 训 200 步

| 类别 | 环境 | 任务类型 |
|------|------|---------|
| Math | MetaMathQA [11] | 数学推理 |
| Combinatorial | Countdown [24] | 组合数学 |
| Planning | Sokoban [21] | 空间规划 |
| QA | HotpotQA [12] | 多跳推理 |
| Constraint | Sudoku [15] | 约束求解 |
| Stochastic | FrozenLake [22] | 随机性规划 |
| Interactive | WebShop [13] | 交互式网页决策 |
| Embodied | ALFWorld [25] | 家务任务 |

> **[Table 1]** 主表
> 行：8 个环境
> 列：Base model Succ@5 | In-domain trained Succ@5 | Δ
> 每个环境的 trained 模型 = 在该环境上用 one-bit feedback 训练 200 步（Qwen2.5-3B）
> Insight：每个 domain 都有显著提升——one-bit feedback 不是只在某些任务上碰巧有效，而是一种 domain-agnostic 的能力激活机制

> **[Fig 2]** 每个环境的 Succ@k 详细曲线（k=1~5），8 个子图
> 每个子图展示 base vs trained 的 Succ@k
> Insight：不只是最终的 Succ@5 提升了，Succ@k 曲线的形状跨域一致——说明 one-bit feedback 在不同 domain 上激活的是同一种多轮改进机制，而非各 domain 各自不同的纠错方式

#### Model Scaling

上述实验均使用 Qwen2.5-3B。为验证 one-bit feedback 的效果不依赖于特定模型规模，我们在 Qwen2.5-1.5B / 3B / 7B 上重复核心实验。

> **[Table 2]** 行：模型规模（1.5B / 3B / 7B），列：Base Succ@5 | Trained Succ@5 | Δ
> 预期：所有规模都有提升；更大模型的 base 和 trained 都更高（更强的 ICL 能力 → 更强的 self-correction）

#### Turn Scaling

我们考察训练时的交互轮数对性能的影响。分别用不同的最大轮数（T=1, 3, 5, 7）训练模型，评估时使用对应的轮数。

> **[Fig 2b]** 折线图：X 轴为训练轮数 T，Y 轴为 Succ@T
> 预期：从 T=1 到 T=3/5 有显著提升，之后收益递减

### 3.2 Cross-Domain Transfer

**设定**：在环境 A 上训练的模型，拿到环境 B 上评估（推理时给 one-bit feedback，不重新训练）

> **[Fig 3]** 8×8 迁移矩阵热力图
> 横轴：评估环境，纵轴：训练环境
> 颜色：Succ@5 相对 base model 的提升
> Insight：off-diagonal 大部分为正，说明在 domain A 上学到的 "从 one-bit feedback 中改进" 的能力可以直接迁移到 domain B——模型学到的不是某个任务的纠错技巧，而是一种通用的 "从失败中探索替代方案" 的 meta-ability

值得注意的是，这 8 个环境涵盖了数学推理、空间规划、多跳问答、约束求解、随机导航、交互式网页决策等完全不同的任务类型，彼此之间几乎没有内容关联。在这种情况下迁移依然普遍有效——这一现象的机制解释见 §4.2。

### 3.3 ICL Decomposition

**核心问题**：one-bit feedback RL 的提升有多少来自模型本身的 ICL 能力，多少来自 RL 训练？

四条线对比（MetaMathQA, Qwen2.5-3B, 5 轮，Succ@5）：

1. Base model + no feedback（水平虚线）
2. Base model + one-bit feedback（水平虚线）→ 两条虚线的 gap = one-bit feedback 的 ICL 贡献
3. No feedback trained（训练曲线，step 50 → 200）
4. One-bit trained（训练曲线，step 50 → 200）→ 曲线和虚线的 gap = RL 训练贡献

> **[Fig 4]** 折线图：X 轴为训练步数，Y 轴为 Succ@5，四条线
> Insight：
> - 两条 base 虚线的 gap → 未经训练的 base model 就能从 one-bit feedback 中获益，说明 LLM 天然具备一定的 in-context learning from failure 能力 [5][6]
> - 训练曲线远高于对应的 base 虚线 → RL 训练大幅放大了 context 利用能力
> - One-bit trained 始终高于 no feedback trained → one-bit feedback 在训练全程都持续提供额外收益
> - 核心信息：one-bit feedback 不是一个 prompt trick——如果只是 prompt trick，base 虚线就应该足够好了。真正的价值来自 RL 训练教会模型如何系统性地利用这个最小信号

### 3.4 Self-Correction Analysis

§3.3 表明 RL 训练放大了模型的 context 利用能力。那么 RL 具体增强了什么能力？我们用 Succ@2|fail@1（第一次错了，第二次做对的概率）来衡量模型在接收到失败信号后的 self-correction 能力，对比三种模型：base model、single-turn RL trained、multi-turn one-bit trained。

> **[Fig 7]** 折线图：X 轴为训练步数（50, 100, 150, 200），Y 轴为 Succ@2|fail@1
> 三条线：base model（水平虚线）、single-turn RL（训练曲线）、multi-turn one-bit（训练曲线）
> Insight：
> - **Single-turn RL（下降趋势）**：越训越差，训练在**持续压制**模型的 self-correction 能力——模型被优化为"一次做对"，逐渐丧失了从失败中修正的能力
> - **Multi-turn one-bit（上升趋势）**：逐步超过 base 并持续上升，训练在**逐步增强**模型的 self-correction 能力——模型学会了看到 "incorrect" 后主动修正推理策略
> - 两条曲线走势完全相反——同样是 RL 训练，训练范式的差异决定了模型是获得还是失去 self-correction 能力
> - 核心信息：**one-bit feedback 训练增强的核心能力是 self-correction——模型不需要外部告知哪里错了，仅凭 "incorrect" 信号就能自主诊断并修正**

### 3.5 Feedback Signal Analysis

#### Signal Robustness

训练时每一步的 one-bit feedback 从一个 prompt pool 中随机抽取（§2），因此模型从未对某个特定措辞过拟合。为验证模型确实响应的是 1-bit signal 本身而非特定字符串，我们在推理时测试了 N 种不同的 feedback 措辞（如 "Incorrect."、"That's wrong, try again."、"Not quite right." 等），Succ@5 在 X% 到 Y% 之间，标准差仅 Z%。这表明具体措辞不影响性能，模型响应的是 correct/incorrect 这一 1-bit 信号本身。

#### Information Scaling

核心问题：如果我们给模型更多的反馈信息，性能还能提升多少？

我们设计了 4 个 feedback level，信息量从无到有逐步递增，在 MetaMathQA 上对比（均为 Qwen2.5-3B, 5 轮）：

| Level | 名称 | 反馈内容示例 | 是否需要 domain knowledge |
|-------|------|------------|------------------------|
| 0 | No feedback | （空） | 否 |
| 1 | One-bit feedback | "Incorrect." | 否 |
| 2 | Generic feedback | "Incorrect. 是否理解对了题意？有没有计算错误？方法对不对？"（固定的自检模板，每道题一样） | 是（模板需要针对数学任务设计） |
| 3 | Specific feedback | "Incorrect. The correct answer should be larger."（基于 ground truth 的方向性提示，每道题不同） | 是（需要访问正确答案） |

从 Level 0 到 Level 3，反馈信息量递增：
- Level 0 → 1：从 "没有任何信号" 到 "知道自己错了"（domain-agnostic）
- Level 1 → 2：从 "知道自己错了" 到 "被引导反思可能的错误类型"（domain-specific 模板）
- Level 2 → 3：从 "通用的反思引导" 到 "基于正确答案的具体提示"（domain-specific + ground truth）

> **[Fig 6]** 折线图：X 轴为 feedback level（0 → 1 → 2 → 3），Y 轴为 Succ@5
> 预期曲线形状：Level 0 → 1 有一个大跳跃，之后基本平坦
> Insight：最大的收益来自 domain-agnostic 的 one-bit feedback（Level 0 → 1）。跨过 domain-agnostic 到 domain-specific 的边界（Level 1 → 2 → 3），需要额外的工程成本（设计模板、访问 ground truth），但性能收益极为有限。这从实验上回答了论文的核心问题：**多轮推理需要多少反馈信息？1 bit 就接近最优。** 形式化分析见 §4.1。

#### Comparison with Domain-Specific Methods

上述 information scaling 使用的是同一框架下不同 feedback level 的对比。我们进一步与专门为特定 domain 设计的多轮方法进行对比：在代码任务上对比 SDPO [1]（利用 runtime error），在搜索任务上对比 Search-R1 [3]（利用 retrieval result）。

> **[Table 3]** 行：方法（one-bit feedback / domain-specific method），列：各 domain 的 Succ@5
> Insight：one-bit feedback 无需任何 domain-specific 工程，但能达到 domain-specific 方法的大部分收益。这验证了 information scaling 的结论在实际方法对比中同样成立。

---

## 4. Analysis & Discussion

### 4.1 为什么 1 bit 就够？

§3.3 表明未经训练的 base model 接收 one-bit feedback 后就能产生提升——说明提升的来源是模型自身的 ICL 能力 [5][6]，而非 feedback 携带的信息量。§3.4 进一步表明 RL 增强的具体能力是 self-correction——模型学会了仅凭 "incorrect" 信号就自主诊断并修正，不需要外部提供纠错细节。两者共同说明：one-bit feedback 的作用不是传递纠错信息，而是触发模型已有的 self-correction 能力。§3.5 的 information scaling 实验从另一侧验证了这一点：跨过 one-bit 之后，更丰富的 domain-specific 反馈几乎不带来额外收益。

**形式化分析。** 我们在 Appendix A 中将模型的推理过程建模为从有限策略库中选择（Proposition 1）。没有 feedback 时，模型可能反复使用同一种失败策略（有放回采样）；one-bit feedback 让模型知道当前策略失败了，从而排除它、转向其他策略（无放回采样）。这个从"有放回"到"无放回"的转变是结构性的——它直接消除了重复失败的可能性。而更丰富的反馈只是在此基础上优化"先尝试哪条新策略"的顺序，边际收益受限于策略库内质量方差——当各策略质量相近时，尝试顺序无关紧要，richer feedback 的额外收益趋近于零。

**认知科学视角。** 这与两个理论框架一致。**Impasse-driven learning** [8]：impasse signal 不包含解题信息，但触发 learner 从 exploitation（重复当前策略）切换到 exploration（搜索替代方案）。One-bit feedback 正是这样一种最小化的 impasse signal。**Desirable difficulties** [23]：故意减少反馈细节，反而促进 learner 更深层的加工和更强的迁移能力——这解释了为什么 one-bit feedback 训练出的模型能跨域迁移（§3.2），而更丰富的 domain-specific 反馈反而可能将模型锚定在特定 domain 的纠错模式上。

### 4.2 为什么跨域迁移有效？

§3.2 的迁移矩阵表明，在 8 个彼此几乎没有内容关联的环境之间，跨域迁移依然普遍有效。这直接排除了"模型学到的是 domain-specific 纠错技巧"的解释——如果是 domain-specific 的，数学上训练的模型不应该在空间规划上也能利用 one-bit feedback。

结合 §3.3 和 §3.4 的分析，我们可以解释这一现象。§3.3 的关键发现是：base model 在**多个不同 domain** 上都能响应 one-bit feedback——这说明被 RL 强化的 ICL 能力本身就不绑定任何特定 domain，是一种跨域共享的基础能力。§3.4 的关键发现是：RL 增强的具体行为模式是 self-correction，而 self-correction 的每一步——识别 "incorrect" 信号、利用 context 中的失败历史、生成不同的回答——都发生在 context processing 层面，不依赖 domain knowledge。换言之，§4.1 回答的是"为什么不需要更多信息"（因为 self-correction 靠模型自身能力，不靠 feedback 内容），§4.2 回答的是"为什么能跨域"（因为 self-correction 的行为模式本身就是 domain-agnostic 的）——同样的 §3.3/§3.4 发现，服务于不同的论证方向。

---

## 5. Related Work

- **多轮 RL for LLMs**：RAGEN [15], ArCHer [16], CollabLLM [17]
- **Self-correction via RL**：SCoRe [18]（ICLR 2025）, S²R [19]（ACL 2025）
- **跨域 RL 迁移**：Guru [20]（UCSD, 2025）——研究 single-turn RL 的跨域迁移
- **In-context RL**："Reward Is Enough" [5], "No Need for Explanations" [6]
- **Experiential RL**：ERL [2]（2026）——显式 self-reflection + self-distillation
- **认知科学**：Impasse-driven learning [8], Desirable difficulties [23]

---

## 6. Conclusion

我们提出了一个基本问题：多轮推理需要多少反馈信息？通过 8 种不同类型 domain 的系统实验，我们的回答是：1 bit 就够了。One-bit feedback——仅传递 correct/incorrect 信号——足以在所有测试 domain 上实现有效的多轮推理提升，且这个能力可以跨域迁移。分析表明，one-bit feedback 激活的是语言模型固有的 in-context learning from failure 能力，RL 训练通过 RL 强化了这个能力。这一发现为多轮 RL 训练提供了一个 domain-agnostic 的范式：不需要为每个 domain 设计反馈机制，一个最小信号就够了。

---

## References

[1] Song et al. "SDPO: Reinforcement Learning via Self-Distillation." 2026. https://arxiv.org/abs/2601.20802
[2] "Experiential Reinforcement Learning." 2026. https://arxiv.org/abs/2602.13949
[3] Jin et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning." 2025. https://arxiv.org/abs/2503.09516
[4] Wang et al. "MINT: Evaluating LLMs in Multi-Turn Interaction with Tools and Language Feedback." 2024. https://arxiv.org/abs/2309.10691
[5] Song et al. "Reward Is Enough: LLMs Are In-Context Reinforcement Learners." 2025. https://arxiv.org/abs/2506.06303
[6] Alazraki et al. "No Need for Explanations: LLMs Can Implicitly Learn from Mistakes In-Context." 2025. https://arxiv.org/abs/2502.08550
[8] VanLehn, K. "Rule-Learning Events in the Acquisition of a Complex Skill: An Evaluation of Cascade." Journal of the Learning Sciences, 8(2), 1999.
[9] Schulman et al. "Proximal Policy Optimization Algorithms." 2017. https://arxiv.org/abs/1707.06347
[10] Yang et al. "Qwen2.5 Technical Report." 2024. https://arxiv.org/abs/2412.15115
[11] Yu et al. "MetaMathQA: A Dataset for Mathematical Reasoning with Large Language Models." 2024. https://arxiv.org/abs/2405.17633
[12] Yang et al. "HotpotQA: A Dataset for Diverse, Explainable Multi-Hop Question Answering." 2018. https://arxiv.org/abs/1809.09600
[13] Yao et al. "WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents." 2023. https://arxiv.org/abs/2207.01206
[14] Kapur, M. "Productive Failure." Cognition and Instruction, 26(3), 2008.
[15] Wang et al. "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning." 2025. https://arxiv.org/abs/2504.20073
[16] Zhou et al. "ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL." 2024. https://arxiv.org/abs/2402.19446
[17] Wu et al. "CollabLLM: From Passive Responders to Active Collaborators." ICML 2025.
[18] Kumar et al. "Training Language Models to Self-Correct via Reinforcement Learning." ICLR 2025. https://arxiv.org/abs/2409.12917
[19] S²R. "Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning." ACL 2025. https://arxiv.org/abs/2502.12853
[20] Cheng et al. "Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective." 2025. https://arxiv.org/abs/2506.14965
[21] Schrader, M. "gym-sokoban: Reinforcement Learning Environment for the Game of Sokoban." 2018. https://github.com/mpSchrader/gym-sokoban
[22] Brockman et al. "OpenAI Gym." 2016. https://arxiv.org/abs/1606.01540 (FrozenLake environment)
[23] Bjork, R.A. "Memory and Metamemory Considerations in the Training of Human Beings." In Metacognition: Knowing about Knowing, MIT Press, 1994.
[24] Pan, J. et al. "Countdown Tasks: A Dataset for Combinatorial Number Game." Based on Jiayi-Pan/Countdown-Tasks-3to4 on HuggingFace.
[25] Shridhar et al. "ALFWorld: Aligning Text and Embodied Environments for Interactive Learning." 2021. https://arxiv.org/abs/2010.03768

---

## Appendix

### A. Diminishing Returns of Feedback Information

我们形式化分析为什么 one-bit feedback（1 bit）捕获了 feedback 信息的绝大部分价值，而更丰富的反馈只带来有限的边际收益。

**设定.** 对于给定问题 q，模型有 k 个推理策略 S = {s₁, ..., sₖ}，策略 sᵢ 以概率 rᵢ ∈ [0,1] 产生正确答案。模型有 T 轮尝试机会。我们比较三种 policy：

**Definition 1 (Parallel Policy / 无反馈).** 每轮独立按分布 p = (p₁, ..., pₖ) 采样策略。同一策略可被重复选择。
$$\text{Succ@T}_{\text{par}} = 1 - (1 - \mu)^T, \quad \mu = \sum_i p_i r_i$$

**Definition 2 (Elimination Policy / one-bit feedback).** 每轮排除已尝试过且失败的策略，按某种顺序 σ 从剩余策略中选择。
$$\text{Succ@T}_{\text{elim}} = 1 - \prod_{t=1}^{\min(T,k)} (1 - r_{\sigma(t)})$$

**Definition 3 (Optimal Policy / 完美信息).** 排除已失败策略，且按 rᵢ 从大到小排列剩余策略（即拥有完美信息来决定最优尝试顺序）。
$$\text{Succ@T}_{\text{opt}} = 1 - \prod_{t=1}^{\min(T,k)} (1 - r_{(t)})$$
其中 r₍₁₎ ≥ r₍₂₎ ≥ ... ≥ r₍ₖ₎ 是正确率的降序排列。

**Proposition 1 (One-bit feedback 的价值 vs 额外信息的边际收益).**

定义：
- ΔU = Succ@T_elim - Succ@T_par：one-bit feedback 的价值（1st bit）
- ΔR = Succ@T_opt - Succ@T_elim：更丰富反馈的边际价值（additional bits）

则：

**(a) One-bit feedback 的价值可以任意大.** 对于任意 δ > 0，存在参数设定使得 ΔU > 1 - δ。

> *证明.* 取 k = 2, r₁ = 0, r₂ = 1, p₁ = 1-ε, T = 2。Parallel policy 以 (1-ε)² 的概率两轮都选择 s₁ 而失败；Elimination policy 第二轮必定选择 s₂ 而成功。当 ε → 0 时，ΔU → 1。□

**(b) 额外信息的收益受限于策略方差.** 定义策略质量方差 V = Var(r₁, ..., rₖ)。当 V = 0（即所有 rᵢ = r）时：

$$\Delta R = 0$$

> *证明.* 当所有策略正确率相同（rᵢ = r, ∀i）时，任何排列 σ 下的 Succ@T_elim 都等于 Succ@T_opt = 1 - (1-r)^{min(T,k)}。排列顺序不影响结果，因此完美信息（知道最优排列）不比 one-bit feedback 多带来任何收益。□

**(c) 一般情况下的上界.** 对于 T ≤ k，设 r_max = max_i r_i，r_min = min_i r_i，则：

$$\Delta R \leq \prod_{t=1}^{T}(1-r_{\min}) - \prod_{t=1}^{T}(1-r_{\max}) \cdot \prod_{t=1}^{T}\frac{1-r_{\min}}{1-r_{\max}}$$

更直观地，当策略质量差距小（r_max - r_min = ε → 0）时，ΔR = O(Tε)，而 ΔU 不依赖于 ε。

**Remark.** 这个结果解释了为什么 one-bit feedback 在实验中接近信息效率的最优点：
- 1st bit（one-bit feedback）实现了从 "有放回采样" 到 "无放回采样" 的结构性转变，价值不受策略方差限制
- Additional bits 只优化无放回采样的顺序，价值受限于策略方差
- 当模型的替代策略质量相近时（对一个 well-calibrated LLM 来说是合理假设），额外信息几乎无用

### B. 其他可能的附录内容

- 各环境的详细实验结果（per-step Succ@k curves）
- 迁移矩阵的完整数据表
- Signal robustness 实验的所有 prompt 变体
- 理论分析（collision probability, sequential vs parallel policy）

---

## Figure 清单

| Figure | 位置 | 内容 | Insight |
|--------|------|------|---------|
| Fig 1 | §1 Intro | 8 个 domain 的 Succ@k 曲线并排 | One-bit feedback is universal |
| Table 1 | §3.1 | 8 env × {Base, Trained} 的 Succ@5 | 每个 domain 都有显著提升 |
| Table 2 | §3.1 | 1.5B / 3B / 7B × {Base, Trained} 的 Succ@5 | 跨模型规模有效 |
| Fig 2 | §3.1 | 每个 env 的 Succ@k 详细曲线 | 提升模式跨域一致 → 同一种机制 |
| Fig 2b | §3.1 | 不同训练轮数 T（1/3/5/7）的 Succ@T | 收益集中在前几轮，之后递减 |
| Fig 3 | §3.2 | 8×8 迁移矩阵热力图 | 跨域迁移有效 → domain-agnostic meta-ability |
| Fig 4 | §3.3 | 四条线：2 条 base 虚线 + 2 条训练曲线（no feedback / one-bit） | ICL 贡献 vs RL 训练贡献的拆解 |
| Fig 5 | §3.4 | Base / Single-turn RL / Multi-turn one-bit 的 Succ@2\|fail@1 | Single-turn RL 削弱 self-correction，multi-turn one-bit 增强 |
| Fig 6 | §3.5 | 4-level feedback scaling（no / one-bit / generic / specific） | domain-agnostic 的 one-bit 就接近最优 |
| Table 3 | §3.5 | one-bit vs domain-specific methods（SDPO, Search-R1 等） | 无需 domain 工程即可达到大部分收益 |

---

## → 对应 PLAN.md 的实验

| Paper 内容 | PLAN.md 实验 | 优先级 |
|-----------|-------------|--------|
| Table 1 + Fig 1, 2 | 实验层：6 env 各训 200 步 | P0 |
| Fig 3 | 迁移层：6×6 eval | P0 |
| Fig 4 | 分析层：ICL decomposition | P1 |
| Fig 5 | 分析层：self-correction analysis | P1 |
| §3.5 文字 | 分析层：signal robustness（推理时多 prompt 测试） | P1 |
| Fig 6 | 理论层：information scaling | P2 |
