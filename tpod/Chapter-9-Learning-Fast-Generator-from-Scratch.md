# 从零学习快速生成器（Learning Fast Generators from Scratch）

> *真理往往蕴含于简洁之中，而非纷繁与混乱。* — Isaac Newton

在第八章中我们看到，扩散模型中的慢速迭代采样器可以通过蒸馏被压缩成少步生成器。从工程角度看，两阶段流水线是实用的，因为它将复杂的生成式训练任务拆成清晰、独立的目标：第一阶段学习数据分布，第二阶段加速采样或提升质量。这种分离使每个阶段可独立优化，整体系统更易管理、更稳定、更可靠。

然而本章将焦点转向驱动深度生成建模发展的一个核心问题：

**我们能否设计一种独立的生成原理，使其训练稳定高效、采样快速，并让用户易于引导或控制生成结果？**

本章沿此方向，讨论另一种思路：**不依赖**预训练模型地训练少步、基于扩散的生成器。我们的重点是**流映射模型（flow map model）**，它通过学习直接变换来近似 PF-ODE 的 Oracle 流映射，使样本随时间迁移。该形式化为从先验分布 $p_{\mathrm{prior}}$ 到数据分布 $p_{\mathrm{data}}$ 的概率质量传输提供了原则性方式，同时在各中间时刻保持前向扩散过程所指定的边缘分布 $p_t$。

---

## 序言（Prologue）

**流映射建模时间线示意。** 图中用蓝色表示特殊情形 $\boldsymbol{\Psi}_{s \to 0}$，用橙色表示一般映射 $\boldsymbol{\Psi}_{s \to t}$。（注：原图为 TikZ 时间线，此处仅保留 caption 说明。）

**流映射模型的动机。** 在第八章中我们说明了 Oracle 流映射损失 $\mathcal{L}_{\mathrm{oracle}}(\boldsymbol{\theta})$（见统一目标式）中不可直接计算的回归目标，如何通过从预训练扩散模型蒸馏知识来估计，从而得到少步生成器。这条路线有效且实用：两阶段流水线便于工程上的鲁棒性，在数据和计算效率上往往仍具竞争力。

本章将焦点转向深度生成建模的核心挑战：**能否建立一种独立的生成原理，在不依赖预训练模型的条件下，实现稳定、可扩展且高效的训练、快速采样，以及易于被用户意图引导的生成？** 设计这类独立原理正是生成建模的中心议题。

扩散模型提供了一种有用的设计原则：以前向过程为参考，将数据逐渐变为简单先验（噪声），并将建模任务表述为学习逆时传输以恢复该过程并匹配期望的边缘分布。这种依赖时间的表述也使得在中间步骤上引导生成过程比单步生成映射更容易。专就扩散类方法而言，这引出如下问题：

**能否在没有预训练模型的情况下，用网络 $\mathrm{G}_{\boldsymbol{\theta}}(\cdot,s,t)$（流映射模型）学习流映射 $\boldsymbol{\Psi}_{s\to t}(\cdot)$，并保持高保真生成？**

本章围绕单一目标发展方法，该目标也支撑蒸馏并对流映射表述给出统一视角：

$$
\mathcal{L}_{\mathrm{oracle}}(\boldsymbol{\theta})
:= \mathbb{E}_{s,t}\,\mathbb{E}_{\mathbf{x}_s \sim p_s}\Big[
   w(s,t)  d\big(\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t),  \boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s)\big)
\Big].
$$

其中 $s,t$ 从某时间分布（如均匀）采样，$w(s,t) \ge 0$ 为时间对 $(s,t)$ 赋权，$d(\cdot,\cdot)$ 为差异度量（如 $\ell_2$ 范数平方）。Oracle 流映射 $\boldsymbol{\Psi}_{s\to t}$ 表示将时刻 $s$ 的状态 $\mathbf{x}_s$ 直接传输到时刻 $t$ 的理想变换：

$$
\boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s) = \mathbf{x}_s + \int_s^t \mathbf{v}^*(\mathbf{x}_u,u)\,\mathrm{d}u,
$$

其中 Oracle 漂移为

$$
\mathbf{v}^*(\mathbf{x}_u,u) = \mathbb{E} \left[\alpha_u'\mathbf{x}_0+ \sigma_u'\boldsymbol{\epsilon} | \mathbf{x}_u\right],
$$

等价的参数化形式也可用（见等价参数化章节），常见选择包括 $\mathbf{x}$-预测与 $\mathbf{v}$-预测形式。

在 Oracle 损失的最优处，所学模型精确恢复真实流映射：

$$
\mathrm{G}^*(\mathbf{x}_s, s, t) = \boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s), \quad \text{对所有 }\, s,t \text{ 及 }\, \mathbf{x}_s \sim p_s.
$$

由于流映射 $\boldsymbol{\Psi}_{s\to t}$ 无闭式，必须用近似。一种做法（第八章）是依赖预训练扩散模型；另一种如本章所示，是引入新的、更易处理的替代目标。为清晰起见，现有方法可按训练过程是否在循环中调用教师来大致分类：**蒸馏**（显式调用教师模型）与**从零训练**（通过构造自包含的替代目标而避免教师调用）。

在上述原则性目标的基础上，我们转而系统介绍学习流映射模型的方法，目标是既实用又能更准确反映真实数据分布且计算高效。先给出该范式的高层介绍。

**特殊流映射：一致性函数。** Consistency Models 是流映射学习的早期代表之一。它们学习少步去噪器 $\mathbf{f}_{\boldsymbol{\theta}}(\cdot,s)$，近似流映射到原点的特殊情形 $\boldsymbol{\Psi}_{s\to 0}(\cdot)$，$s \in (0,T]$。核心思想是：每个带噪样本 $\mathbf{x}_s$ 都应映射回其轨迹末端的干净数据点 $\mathbf{x}_0$。形式上，CM 族的 Oracle 训练目标为

$$
\mathcal{L}_{\mathrm{oracle}}^{\mathrm{CM}}(\boldsymbol{\theta}) 
:= \mathbb{E}_{s} \mathbb{E}_{\mathbf{x}_s\sim p_s} \left[ w(s)  d \left(\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}_s,s), \boldsymbol{\Psi}_{s\to 0}(\mathbf{x}_s)\right)\right].
$$

实践中 Oracle $\boldsymbol{\Psi}_{s\to 0}(\mathbf{x}_s)$ 不可得，因此用**止梯度**目标 $\mathbf{f}_{\boldsymbol{\theta}^-}$ 替代，取自同一轨迹上略早一步 $\boldsymbol{\Psi}_{s\to s-\Delta s}(\mathbf{x}_s)$：

$$
\boldsymbol{\Psi}_{s\to 0}(\mathbf{x}_s) \approx \mathbf{f}_{\boldsymbol{\theta}^-} \left(\boldsymbol{\Psi}_{s\to s-\Delta s}(\mathbf{x}_s),\,s-\Delta s\right), \quad \Delta s > 0,
$$

其中 $\boldsymbol{\Psi}_{s\to s-\Delta s}(\mathbf{x}_s)$ 本身也需近似。两种实用策略为：(i) **蒸馏**，依赖预训练扩散模型；(ii) **从零训练**，使用无教师引导的单点估计。

**一般流映射。** 两个代表方法是 **Consistency Trajectory Model (CTM)** 与 **Mean Flow (MF)**。

*Consistency Trajectory Models.* CTM 是首个学习任意起止时间的一般流映射 $\boldsymbol{\Psi}_{s\to t}$ 的工作，可视为统一目标下的具体实例。CTM 采用 Euler 风格的参数化，将 Oracle 流映射写成

$$
\boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s)
:= \mathbf{x}_s + \int_s^t \mathbf{v}^*(\mathbf{x}_u,u)\mathrm{d} u 
= \frac{t}{s} \mathbf{x}_s
+ \frac{s-t}{s}
\Big[\mathbf{x}_s + \frac{s}{s-t} \int_s^t \mathbf{v}^*(\mathbf{x}_u,u) \mathrm{d} u\Big],
$$

从而得到神经网络参数化

$$
\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t)
:= \frac{t}{s}\,\mathbf{x}_s + \frac{s-t}{s}\,\mathbf{g}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t),
$$

其中 $\mathbf{g}_{\boldsymbol{\theta}}$ 为神经网络，训练使 $\boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s)\approx \mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t)$。

由于 Oracle $\boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s)$ 不可得，CTM 用中间时刻 $u$ 处的**止梯度**目标训练：

$$
\boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s)
 \approx 
\mathrm{G}_{\boldsymbol{\theta}^-} \big(\boldsymbol{\Psi}_{s\to u}(\mathbf{x}_s),\,u,\,t\big),
\qquad u\in[t,s],
$$

中间状态 $\boldsymbol{\Psi}_{s\to u}(\mathbf{x}_s)$ 的近似有两种：(i) **蒸馏**：用预训练扩散教师的少步求解器；(ii) **从零训练**：通过 $\mathrm{G}_{\boldsymbol{\theta}}$ 参数化直接构造自诱导教师。

*Mean Flow.* Mean Flow (MF) 对流匹配进行推广，对区间 $[t,s]$（$t\le s$）建模**平均漂移**：

$$
\mathbf{h}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t)
 \approx 
\mathbf{h}^*(\mathbf{x}_s,s,t)
:= \frac{1}{\,t-s\,}\int_s^t \mathbf{v}^*(\mathbf{x}_u,u)\,\mathrm{d} u,
$$

也与统一目标一致。对恒等式 $(t-s)\,\mathbf{h}^*(\mathbf{x}_s,s,t)  =  \int_s^t \mathbf{v}^*(\mathbf{x}_u,u)\,\mathrm{d} u$ 关于 $s$ 求导可得自指关系，从而得到 MF 目标

$$
\mathcal{L}_{\mathrm{MF}}(\boldsymbol{\theta})
:= \mathbb{E}_s\,\mathbb{E}_{\mathbf{x}_s\sim p_s} \Big[w(s)\,\big\|
\mathbf{h}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t)-\mathbf{h}_{\boldsymbol{\theta}^-}^{\mathrm{tgt}}(\mathbf{x}_s,s,t)
\big\|_2^2\Big],
$$

止梯度目标为

$$
\mathbf{h}_{\boldsymbol{\theta}^-}^{\mathrm{tgt}}(\mathbf{x}_s,s,t)
:= \mathbf{v}^*(\mathbf{x}_s,s)  -  (s-t) \left(\mathbf{v}^*(\mathbf{x}_s,s)\,\partial_\mathbf{x} \mathbf{h}_{\boldsymbol{\theta}^-} + \partial_s \mathbf{h}_{\boldsymbol{\theta}^-}\right).
$$

实践中 Oracle 速度 $\mathbf{v}^*(\mathbf{x}_s,s)$ 也需近似，常见两种策略：蒸馏（利用预训练 flow matching 扩散模型）或从零训练（使用前向过程 $\mathbf{x}_s = \alpha_s \mathbf{x}_0+\sigma_s \boldsymbol{\epsilon}$ 的单点条件速度 $\alpha'_s\,\mathbf{x}_0+\sigma'_s\,\boldsymbol{\epsilon}$）。

*CTM 与 MF 的关系。* CTM 与 MF 近似同一路径积分，但对它的不同替代进行参数化：CTM 通过 $\mathbf{g}_{\boldsymbol{\theta}}$ 学习**斜率位移**，MF 学习**平均漂移** $\mathbf{h}_{\boldsymbol{\theta}}$；二者都是近似定义 $\boldsymbol{\Psi}_{s\to t}$ 的同一积分的一致方式。

**后续内容。** 我们先从 CM 族入手，它关注特殊流映射 $\boldsymbol{\Psi}_{s\to 0}$，包括离散时间起源与连续时间推广。然后讨论一般流映射的两个代表 CTM 与 MF，其参数化、训练策略与实用近似分别在相应小节给出。

EDM 为 $\mathbf{x}$-预测模型的网络参数化提供了系统设计指南并表现出强经验性能；该节可视为可选，但 EDM 表述是 CM 类模型的有用基础。

为便于后文叙述，我们不严格按时间顺序，而是按概念关系组织；同时用时间线图标注原创性与时间线。

---

## 特殊流映射：离散时间一致性模型

**流映射的一条重要性质：半群性质。**

![流映射半群性质示意](../arXiv-2510.21890v1/Images/PartD/semigroup.pdf)

*图：流映射半群性质示意。该性质表示：从 $s$ 到 $u$ 再从 $u$ 到 $t$ 的转移，等价于直接从 $s$ 到 $t$。*

Consistency Models 及其推广 Consistency Trajectory Model 通过利用流映射的关键数学结构来定义回归目标，即**半群性质**：

$$
\boldsymbol{\Psi}_{u\to t}\circ \boldsymbol{\Psi}_{s\to u}=\boldsymbol{\Psi}_{s\to t},
\quad \boldsymbol{\Psi}_{s\to s}=\mathrm{I}, \quad \text{对所有 }\, s, u, t \in [0,T].
$$

直观上，若先将状态从 $s$ 演化到 $u$（经 $\boldsymbol{\Psi}_{s\rightarrow u}$），再从 $u$ 到 $t$（经 $\boldsymbol{\Psi}_{u\rightarrow t}$），与直接从 $s$ 演化到 $t$ 到达同一点。这正是 ODE 求解的基本原理：一旦流的起点给定，其后续演化由唯一性完全确定，沿一条确定的路径。无论一步走完还是分成多段，都沿同一轨迹到达同一终点。

考虑 PF-ODE 的解轨迹 $\{\mathbf{x}(s)\}_{s \in [0,T]}$：

$$
\frac{\mathrm{d} \mathbf{x}(\tau)}{\mathrm{d} \tau} = \mathbf{v}^*(\mathbf{x}(\tau), \tau),
$$

固定终端时刻 $t=0$ 时，相应流映射可简写为 $\mathbf{f}^*(\cdot, s) := \boldsymbol{\Psi}_{s\to 0}(\cdot)$，称为**一致性函数**。由半群恒等式（取 $t=0$）可直接得到：

(i) **全局一致性**：轨迹上每点都映射到同一干净端点 $\mathbf{f}^*(\mathbf{x}(s), s) = \mathbf{x}(0)$，$s \in [0,T]$。  
(ii) **自洽性**：同一轨迹上任意两点输出相同，$\mathbf{f}^*(\mathbf{x}(s), s) = \mathbf{f}^*(\mathbf{x}(u), u)$，$s, u \in [0,T]$。  
(iii) **局部一致性**：沿轨迹求值时一致性函数对 $s$ 不变，$\frac{\mathrm{d}}{\mathrm{d} s}\,\mathbf{f}^*(\mathbf{x}(s),s) = 0$，$\mathbf{f}^*(\mathbf{x}(0),0) = \mathbf{x}(0)$。

三者等价，都表示沿任意解轨迹 $s\mapsto \mathbf{x}(s)$，流到原点/一致性映射 $\mathbf{f}^*(\mathbf{x}(s),s)=\boldsymbol{\Psi}_{s\to 0}(\mathbf{x}(s))$ 给出同一终点 $\mathbf{x}(0)$，与起始时刻无关。

**一致性模型的目标。** CM 旨在训练神经网络 $\mathbf{f}_{\boldsymbol{\theta}} \colon \mathbb{R}^D \times [0,T] \to \mathbb{R}^D$ 近似特殊流映射 $\boldsymbol{\Psi}_{s\to 0}$（即一致性函数）。核心思想是在 PF-ODE 的多条轨迹上强制半群性质，使同一数据点的不同带噪版本一致地映射回同一干净原点（对应半群中 $t=0$、$u=s-\Delta s$ 的特殊情况）。

实现方式有多种，取决于是否有预训练扩散模型以及采用离散时间还是连续时间。下表概括这些变体，后续小节逐步展开各方法细节。

|  | **蒸馏** | **从零训练** |
|--|----------|--------------|
| **离散时间** | $\mathcal{L}_{\text{CD}}^N$（式 CD 离散近似） | $\mathcal{L}_{\text{CT}}^N$（式 CT 离散近似） |
| **连续时间** | $\mathcal{L}_{\mathrm{CD}}^\infty$ | $\mathcal{L}_{\mathrm{CT}}^\infty$ |

*表：一致性模型的训练目标*

### 学习一致性函数的离散时间近似

原则上可通过最小化 Oracle 损失来学习一致性函数：

$$
\mathcal{L}_{\mathrm{oracle}}^{\mathrm{CM}}(\boldsymbol{\theta}) 
:= \mathbb{E}_{s} \mathbb{E}_{\mathbf{x}_s\sim p_s} \left[ w(s)  d \left(\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}_s,s), \boldsymbol{\Psi}_{s\to 0}(\mathbf{x}_s)\right)\right].
$$

该目标要求每个带噪样本 $\mathbf{x}_s$ 被映射回其干净端点 $\boldsymbol{\Psi}_{s\to 0}(\mathbf{x}_s)$。难点在于 Oracle 映射 $\boldsymbol{\Psi}_{s\to 0}(\mathbf{x}_s)$ 在实践中不可得。为此可利用**半群性质**：同一 PF-ODE 轨迹上任意带噪状态与其相邻步必须映射到同一干净端点。具体地，用轨迹上略早一点的**止梯度**目标替代 Oracle 目标：

$$
\boldsymbol{\Psi}_{s\to 0}(\mathbf{x}_s) 
= \boldsymbol{\Psi}_{s-\Delta s \to 0} \left(\boldsymbol{\Psi}_{s\to s-\Delta s}(\mathbf{x}_s)\right) 
\approx \mathbf{f}_{\boldsymbol{\theta}^-} \left(\boldsymbol{\Psi}_{s\to s-\Delta s}(\mathbf{x}_s),\,s-\Delta s\right), 
\quad \Delta s > 0,
$$

其中 $\boldsymbol{\theta}^-$ 为止梯度参数。进一步困难是中间状态 $\boldsymbol{\Psi}_{s\to s-\Delta s}(\mathbf{x}_s)$ 也无闭式，需近似。两种实用方案如下。

**有预训练扩散模型（Consistency Distillation）。** 若有预训练扩散模型，Consistency Distillation (CD) 用教师模型仅模拟一步反向 ODE 来近似中间状态：

$$
\boldsymbol{\Psi}_{s\to s-\Delta s}(\mathbf{x}_s)  \approx  \mathtt{Solver}_{s \to s-\Delta s}(\mathbf{x}_s).
$$

具体地，预训练扩散模型给出 score 的估计 $\mathbf{s}_{\boldsymbol{\phi}^\times}(\mathbf{x}_s, s) \approx \nabla_{\mathbf{x}_s}\log p_s(\mathbf{x}_s)$，可用其从 $\mathbf{x}_s$ 做一步 DDIM 更新得到 $s' = s-\Delta s$ 处的近似：

$$
\boldsymbol{\Psi}_{s\to s-\Delta s}(\mathbf{x}_s) 
\approx \frac{\alpha_{s'}}{\alpha_s}\,\mathbf{x}_s 
+ \sigma_s^2 \left( \frac{\alpha_{s'}}{\alpha_s} - \frac{\sigma_{s'}}{\sigma_s} \right)\mathbf{s}_{\boldsymbol{\phi}^\times}(\mathbf{x}_s, s) 
:= \tilde{\mathbf{x}}_{s'}^{\boldsymbol{\phi}^\times}.
$$

结合止梯度目标即得 Oracle 损失 $\mathcal{L}_{\mathrm{oracle}}^{\mathrm{CM}}(\boldsymbol{\theta})$ 的**离散时间**替代。在划分 $0 = s_1 < s_2 < \cdots < s_N = T$ 上，CD 训练目标为

$$
\mathcal{L}_{\text{CD}}^N(\boldsymbol{\theta}, \boldsymbol{\theta}^-; \boldsymbol{\phi}^\times) 
:= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, i} \Big[\, \omega(s_i)\,
d\big(\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}_{s_{i+1}}, s_{i+1}),\, 
      \mathbf{f}_{\boldsymbol{\theta}^-}(\tilde{\mathbf{x}}_{s_i}^{\boldsymbol{\phi}^\times}, s_i)\big)\Big].
$$

其中 $\omega(\cdot)$ 为时间权重，$d(\cdot,\cdot)$ 为距离度量，$\boldsymbol{\theta}^-$ 表示止梯度参数，用于防止坍缩到平凡解（如常数预测）。

**无预训练扩散模型（Consistency Training）。** 当没有预训练扩散模型时，Oracle score $\nabla_{\mathbf{x}}\log p_s(\mathbf{x}_s)$ 仍可直接用简单单点估计（尽管方差较大）。其条件期望形式为

$$
\nabla_{\mathbf{x}_s} \log p_s(\mathbf{x}_s)
= \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0|\mathbf{x}_s)} \left[ -\frac{\mathbf{x}_s - \alpha_s \mathbf{x}_0}{\sigma_s^2} \right].
$$

若 $\mathbf{x}_s$ 由配对样本 $(\mathbf{x}_0,\boldsymbol{\epsilon})$ 经 $\mathbf{x}_s = \alpha_s \mathbf{x}_0 + \sigma_s \boldsymbol{\epsilon}$ 得到，则

$$
\widehat{\nabla_{\mathbf{x}}\log p_s}(\mathbf{x}_s)
:=-\frac{\boldsymbol{\epsilon}}{\sigma_s}
=-\frac{\mathbf{x}_s-\alpha_s \mathbf{x}_0}{\sigma_s^2}
$$

是 $\mathbf{x}_s$ 处 score 的**无偏**估计（对 $p(\mathbf{x}_0|\mathbf{x}_s)$ 条件无偏），对应去噪分数匹配中用作回归目标的条件分数。

将该估计代入从 $s$ 到 $s' = s-\Delta s$ 的 DDIM 一步更新可得

$$
\boldsymbol{\Psi}_{s\to s'}(\mathbf{x}_s)
\approx \frac{\alpha_{s'}}{\alpha_s}\,\mathbf{x}_s 
-\left(\frac{\alpha_{s'}}{\alpha_s}-\frac{\sigma_{s'}}{\sigma_s}\right)(\mathbf{x}_s-\alpha_s\mathbf{x}_0)
= \alpha_{s'}\mathbf{x}_0+\sigma_{s'}\boldsymbol{\epsilon},
$$

其中 $\mathbf{x}_0$ 与构造 $\mathbf{x}_s$ 时相同，$\boldsymbol{\epsilon}$ 为同一高斯噪声。

由此得到 Oracle 目标 $\mathcal{L}_{\mathrm{oracle}}^{\mathrm{CM}}$ 的无教师离散时间替代：

$$
\mathcal{L}_{\text{CT}}^N(\boldsymbol{\theta}, \boldsymbol{\theta}^-) 
:= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, i} \left[ 
   \omega(s_i)\, d \left(\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}_{s_{i+1}}, s_{i+1}),\, 
   \mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_{s_i}, s_i)\right)
\right],
$$

其中 $\mathbf{x}_{s_i} = \alpha_{s_i}\mathbf{x}_0 + \sigma_{s_i}\boldsymbol{\epsilon}$，$\mathbf{x}_{s_{i+1}} = \alpha_{s_{i+1}}\mathbf{x}_0 + \sigma_{s_{i+1}}\boldsymbol{\epsilon}$。

直接使用 $\alpha_{s'}\mathbf{x}_0+\sigma_{s'}\boldsymbol{\epsilon}$ 作为 $\boldsymbol{\Psi}_{s\to s'}(\mathbf{x}_s)$ 的近似而不取期望会引入高方差。但类比去噪分数匹配，单次条件分数作为训练目标在损失中对 $\mathbf{x}_0,\boldsymbol{\epsilon}$ 求期望后是无偏的。同理，$\mathcal{L}_{\mathrm{CT}}^N$ 中对 $\mathbf{x}_0$ 和 $\boldsymbol{\epsilon}$ 的期望会平均掉采样噪声，得到损失层面的无偏近似。下列定理形式化了该单点估计在期望层面的合理性。

**定理（CM-CT 等价，误差 $\mathcal{O}(\Delta s^2)$）。** 令 $s' := s - \Delta s$，定义

$$
\mathcal{L}_{\mathrm{CM}}(\boldsymbol{\theta},\boldsymbol{\theta}^-)
:= \mathbb{E}_{s,\mathbf{x}_0,\boldsymbol{\epsilon}} \big[w(s)\,d\big(\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}_s,s), \mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_{s'}^{\mathrm{DDIM}},s')\big)\big],\quad
\mathcal{L}_{\mathrm{CT}}(\boldsymbol{\theta},\boldsymbol{\theta}^-)
:= \mathbb{E}_{s,\mathbf{x}_0,\boldsymbol{\epsilon}} \big[w(s) d\big(\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}_s,s), \mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_{s'},s')\big)\big],
$$

其中 $\mathbf{x}_{s'}^{\mathrm{DDIM}}$ 为使用 Oracle score 的 DDIM 更新，$\mathbf{x}_s$ 与 $\mathbf{x}_{s'}$ 共享同一对 $(\mathbf{x}_0,\boldsymbol{\epsilon})$，$\mathbf{x}_0 \sim p_{\mathrm{data}}$，$\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0},\mathrm{I})$。则

$$
\mathcal{L}_{\mathrm{CM}}(\boldsymbol{\theta},\boldsymbol{\theta}^-)
= \mathcal{L}_{\mathrm{CT}}(\boldsymbol{\theta},\boldsymbol{\theta}^-) + \mathcal{O}(\Delta s^2).
$$

*证明要点：* Oracle score 下的 DDIM 更新等于条件均值 $\mathbf{x}_{s'}^{\mathrm{DDIM}} = \mathbb{E}[\mathbf{x}_{s'} | \mathbf{x}_s]$。将 $d(\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}_s,s), \mathbf{f}_{\boldsymbol{\theta}^-}(\cdot,s'))$ 在 $\mathbf{x}_{s'}^{\mathrm{DDIM}}=\mathbb{E}[\mathbf{x}_{s'}|\mathbf{x}_s]$ 处泰勒展开，一阶项在对 $\mathbf{x}_{s'}|\mathbf{x}_s$ 求期望时为零（$\mathbb{E}[\mathbf{x}_{s'}-\mathbf{x}_{s'}^{\mathrm{DDIM}}|\mathbf{x}_s]=0$）；用 $\mathbf{x}_{s'}=\alpha_{s'}\mathbf{x}_0+\sigma_{s'}\boldsymbol{\epsilon}$ 重参数化后，单点 score 的 DDIM 更新在同一条 $(\mathbf{x}_0,\boldsymbol{\epsilon})$ 路径上精确得到 $\mathbf{x}_{s'}$，故内层期望不变，剩余项为二阶 $\mathcal{O}(\Delta s^2)$。

综上，CD 利用教师做初始化与引导，通常能稳定优化并降低方差；Consistency Training (CT) 则无需预训练模型，可完全从零训练，并作为独立的生成模型。

**实用考虑。** 实践中采用 EDM 表述，前向扰动核为 $\mathbf{x}_s = \mathbf{x}_0 + s \boldsymbol{\epsilon}$，网络参数化为

$$
\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}, s)
= c_{\mathrm{skip}}(s) \mathbf{x}
  + c_{\mathrm{out}}(s)\,
    \mathrm{F}_{\boldsymbol{\theta}} \left(c_{\mathrm{in}}(s)\,\mathbf{x},\, c_{\mathrm{noise}}(s)\right),
$$

其中 $\mathrm{F}_{\boldsymbol{\theta}}$ 为神经网络，系数遵循 EDM 简单系数。该参数化具有重要边界性质 $\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}, 0) = \mathbf{x}$，在零时刻强制一致性，并保证无噪声时网络输出等于输入。

### 一致性模型的采样

一旦一致性模型 $\mathbf{f}_{\boldsymbol{\theta}^\times}$ 在离散或连续时间下训练完成，即可用于单步或少步生成。算法如下。

**单步生成。** 给定从先验（实践中为 $\mathcal{N}(\boldsymbol{0}, T^2\mathrm{I})$）采样的初始潜变量 $\hat{\mathbf{x}}_T$，一次函数求值即可生成干净样本：

$$
\mathbf{f}_{\boldsymbol{\theta}^\times}(\hat{\mathbf{x}}_T, T).
$$

**多步生成。** 选取时间序列 $T>\tau_1 > \tau_2 > \cdots > \tau_{M-1}=0$，从初始噪声 $\hat{\mathbf{x}}_T$ 出发，在较早时间点交替进行噪声注入与一致性模型的大步“去噪”，逐步细化样本：

$$
\hat{\mathbf{x}}_T \xrightarrow{\text{大步去噪}} \mathbf{f}_{\boldsymbol{\theta}^\times}(\hat{\mathbf{x}}_T, T)
\xrightarrow{\text{加噪到 }\tau_1} \hat{\mathbf{x}}_{\tau_1}
\xrightarrow{\text{大步去噪}} \mathbf{f}_{\boldsymbol{\theta}^\times}(\hat{\mathbf{x}}_{\tau_1}, \tau_1)
\xrightarrow{\text{加噪到 }\tau_2} \cdots.
$$

**算法：CM 的单步或多步采样**

- **输入**：一致性模型 $\mathbf{f}_{\boldsymbol{\theta}^\times}(\cdot, \cdot)$，时间序列 $T>\tau_1 > \tau_2 > \cdots > \tau_{M-1}=0$，初始噪声 $\hat{\mathbf{x}}_T$。
- 若为单步：$\mathbf{x} \leftarrow \mathbf{f}_{\boldsymbol{\theta}^\times}(\hat{\mathbf{x}}_T, T)$。
- 否则：$\mathbf{x} \leftarrow \mathbf{f}_{\boldsymbol{\theta}^\times}(\hat{\mathbf{x}}_T, T)$；对 $m=1$ 到 $M-1$：采样 $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathrm{I})$，$\hat{\mathbf{x}}_{\tau_m} \leftarrow \alpha_{\tau_m}\mathbf{x} + \sigma_{\tau_m} \boldsymbol{\epsilon}$，$\mathbf{x} \leftarrow \mathbf{f}_{\boldsymbol{\theta}^\times}(\hat{\mathbf{x}}_{\tau_m}, \tau_m)$。
- **输出**：$\mathbf{x}$。

---

## 特殊流映射：连续时间一致性模型

我们超越离散时间设定，从连续时间角度考虑一致性模型。与固定时间网格、仅在采样点上训练的离散做法不同，连续表述将流映射视为对所有时间都有定义。这一转变去掉对任意离散化的依赖，与底层动力学更一致，有助于减少离散积分带来的近似误差，并保证一致性在全局而不仅在选定的步上成立。

### 连续时间一致性模型

为引出连续时间表述，先回顾局部一致性条件 $\frac{\mathrm{d}}{\mathrm{d} s}\,\mathbf{f}^*(\mathbf{x}(s),s) = 0$。由链式法则可得

$$
\frac{\mathrm{d} }{\mathrm{d} s} \mathbf{f}^*(\mathbf{x}(s), s) = 0
\Longleftrightarrow 
\left(\nabla_{\mathbf{x}} \mathbf{f}^*\right)(\mathbf{x}(s), s) \cdot \frac{\mathrm{d} }{\mathrm{d} s} \mathbf{x}(s) + \left(\frac{\partial }{\partial s} \mathbf{f}^*\right)(\mathbf{x}(s), s) = 0,
$$

其中轨迹 $\mathbf{x}(s)$ 满足 PF-ODE $\frac{\mathrm{d} }{\mathrm{d} s}\mathbf{x}(s) =  \mathbf{v}^*(\mathbf{x}(s), s)$。该关系表明一致性函数 $\mathbf{f}^*$ 沿 ODE 的任意解轨迹保持不变。速度场 $\mathbf{v}^*$ 在实践中可由预训练扩散模型估计（若有），或由单点近似（如 $\alpha_s' \mathbf{x}_0 + \sigma_s' \boldsymbol{\epsilon}$）得到。

由此可自然设计连续时间训练目标。一种做法是类似 PINN 最小化残差以强制该微分条件。实践中观察到另一种表述更利于训练：考虑离散近似的连续时间极限 $\Delta s \to 0$：

$$
\mathcal{L}_{\mathrm{CM}}^{\Delta s}(\boldsymbol{\theta},\boldsymbol{\theta}^-)
:= \mathbb{E} \left[\omega(s) \big\|
\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}_s,s) -
\mathbf{f}_{\boldsymbol{\theta}^-} \big(\boldsymbol{\Psi}_{s\to s-\Delta s}(\mathbf{x}_s), s-\Delta s\big)
\big\|_2^2\right].
$$

令 $\Delta s \to 0$ 等价于在离散时间目标中令步数 $N \to \infty$。

**命题（连续时间一致性训练）。** 下列收敛成立：

$$
\lim_{\Delta s\to 0}\frac{1}{\Delta s}\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\mathrm{CM}}^{\Delta s}(\boldsymbol{\theta}, \boldsymbol{\theta}^-) = \nabla_{\boldsymbol{\theta}} \mathcal{L}^\infty_{\mathrm{CM}}(\boldsymbol{\theta}, \boldsymbol{\theta}^-),
$$

其中

$$
\mathcal{L}^\infty_{\mathrm{CM}}(\boldsymbol{\theta},\boldsymbol{\theta}^-) := \mathbb{E}_{s,\mathbf{x}_0, \boldsymbol{\epsilon}}\left[2\omega(t) \mathbf{f}_{\boldsymbol{\theta}}^\top(\mathbf{x}_s, s) \cdot \frac{\mathrm{d} }{\mathrm{d} s} \mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s, s) \right],
$$

且全微分恒等式为

$$
\frac{\mathrm{d}}{\mathrm{d} s}\mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s)
= \partial_s \mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s)
+ \big(\partial_{\mathbf{x}}\mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s)\big) \mathbf{v}^*(\mathbf{x}_s,s).
$$

（证明：对止梯度目标在 $(\mathbf{x}_s, s)$ 处一阶泰勒展开可知，$\mathcal{L}_{\mathrm{CM}}^{\Delta s}$ 在 $\mathcal{O}(\Delta s^2)$ 内表现为学生更新与切向变化的内积；缩放梯度即得所述等式。）

在 $\nabla_{\boldsymbol{\theta}}$ 下 $\boldsymbol{\theta}^-$ 视为常数，故涉及 $\boldsymbol{\theta}^-$ 的项消失。$\tfrac{\mathrm{d}}{\mathrm{d} s}\mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s)$ 表示沿 Oracle 轨迹的全导数，而非仅对时间的偏导。

综上，连续时间一致性模型可通过最小化下列目标（忽略常数因子 2）训练：

$$
\mathcal{L}^\infty_{\mathrm{CM}}(\boldsymbol{\theta},\boldsymbol{\theta}^-) := \mathbb{E}_{s,\mathbf{x}_0, \boldsymbol{\epsilon}}\left[\omega(s)\mathbf{f}_{\boldsymbol{\theta}}^\top(\mathbf{x}_s, s) \cdot \frac{\mathrm{d} }{\mathrm{d} s} \mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s, s) \right].
$$

### 连续时间一致性模型的训练

与离散情形类似，需明确上式切向项中不可得的 Oracle 速度 $\mathbf{v}^*$ 的实用近似：

$$
\frac{\mathrm{d}}{\mathrm{d} s}\mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s)
= \partial_s \mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s)
+ \big(\partial_{\mathbf{x}}\mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s)\big)\,\mathbf{v}^*(\mathbf{x}_s,s).
$$

训练完成后，采样方式与离散时间一致。

**连续时间一致性蒸馏。** 若有预训练扩散模型 $\mathbf{v}_{\boldsymbol{\phi}^\times} \approx \mathbf{v}^*$，则切向量 $\tfrac{\mathrm{d}}{\mathrm{d} s}\mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s)$ 可用替代

$$
\frac{\mathrm{d}}{\mathrm{d} s}\mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s) 
 \approx 
\partial_s \mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s)
+ \big(\partial_{\mathbf{x}}\mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s)\big)\,
  \mathbf{v}_{\boldsymbol{\phi}^\times}(\mathbf{x}_s,s).
$$

所得目标记为 $\mathcal{L}^\infty_{\mathrm{CM}}(\boldsymbol{\theta},\boldsymbol{\theta}^-;\boldsymbol{\phi}^\times)$。相应地，离散 CD 目标满足 $N\to\infty$ 时 $N \nabla_{\boldsymbol{\theta}}\mathcal{L}_{\mathrm{CD}}^{N} \to \nabla_{\boldsymbol{\theta}}\mathcal{L}_{\mathrm{CD}}^{\infty}$。

**连续时间一致性训练（从零）。** 若无预训练扩散模型，可用单点条件估计 $\alpha_s'\mathbf{x}_0 + \sigma_s'\boldsymbol{\epsilon}$ 近似 Oracle 速度 $\mathbf{v}^*$。此时切向量替换为

$$
\frac{\mathrm{d}}{\mathrm{d} s}\mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s) 
 \approx 
\partial_s \mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s)
+ \big(\partial_{\mathbf{x}}\mathbf{f}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s)\big) 
  \left(\alpha_s'\mathbf{x}_0 + \sigma_s'\boldsymbol{\epsilon}\right).
$$

所得目标记为 $\mathcal{L}^\infty_{\mathrm{CT}}(\boldsymbol{\theta},\boldsymbol{\theta}^-)$，对应从零训练。类似地，$N\to\infty$ 时 $N \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{CT}}^{N} \to \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{CT}}^{\infty}$。

至此，表中列出的实现一致性函数 $\boldsymbol{\Psi}_{s\to 0}$ 学习的基本方法均已介绍。不同损失之间的关系、是否依赖预训练扩散模型以及离散/连续时间目标，可参见原文献中的关系图。

然而，切向量 $\tfrac{\mathrm{d} }{\mathrm{d} s}\mathbf{f}_{\boldsymbol{\theta}^-}$ 常导致训练不稳定。下面可选小节介绍 sCM 中的稳定化技术。

### （可选）连续时间一致性训练的实用考虑

我们关注从零训练情形，因其得到不依赖外部预训练扩散模型的独立生成模型。

直接使用上述连续时间损失训练往往不稳定，因为 $\tfrac{\mathrm{d}}{\mathrm{d} s}\mathbf{f}_{\boldsymbol{\theta}^-}$ 可能很大或无界，导致梯度爆炸。因此需要合适的参数化与稳定化策略。影响稳定训练的主要因素包括**扩散过程**、**参数化选择**、**时间权重函数**与**时间采样分布**，在连续时间 CM 中也需仔细设计并解耦。

**扩散过程。** 采用三角调度 $\mathbf{x}_s := \cos(s) \mathbf{x}_0 + \sin(s) \mathbf{z}$，$\mathbf{z} \sim \mathcal{N}(\boldsymbol{0}, \sigma_{\mathrm{d}}^2 \mathrm{I})$，并纳入数据分布 $p_{\mathrm{data}}$ 的标准差 $\sigma_{\mathrm{d}}$。该形式与标准 $\alpha_s,\sigma_s$ 形式数学等价，但提供更清晰的结构与目标中更好的分离，有利于训练稳定（直观上，三角函数及其导数有界，有助于控制 $\frac{\mathrm{d}}{\mathrm{d} s} \mathbf{f}_{\boldsymbol{\theta}^-}$ 等项的尺度）。

**参数化。** 类似 EDM 的设计准则，采用

$$
\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}, s) := c_{\mathrm{skip}}(s)\mathbf{x} + c_{\mathrm{out}}(s)\mathrm{F}_{\boldsymbol{\theta}}\left(c_{\mathrm{in}}(s)\mathbf{x}, c_{\mathrm{noise}}(s)\right),
$$

其中 $c_{\mathrm{skip}}(s) = \cos(s)$，$c_{\mathrm{out}}(s) = -\sigma_{\mathrm{d}} \sin(s)$，$c_{\mathrm{in}}(s) \equiv 1/\sigma_{\mathrm{d}}$，默认 $c_{\mathrm{noise}}(s) = s$（$\partial_s c_{\mathrm{noise}}(s)$ 有界以保证稳定）。三角调度下的参数化为

$$
\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}, s) = \cos(s)\mathbf{x} - \sin(s)\sigma_{\mathrm{d}}\mathrm{F}_{\boldsymbol{\theta}}\left(\frac{\mathbf{x}}{\sigma_{\mathrm{d}}}, c_{\mathrm{noise}}(s)\right).
$$

该参数化保证边界条件 $\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}, 0) \equiv \mathbf{x}$。

**切向训练稳定技巧。** 在上述三角调度与网络参数化下，损失梯度涉及项 $\frac{\mathrm{d} \mathbf{f}_{\boldsymbol{\theta}^-}}{\mathrm{d} s}(\mathbf{x}_s, s)$，可分解为若干项；其中 $\sin(s) \partial_s \mathrm{F}_{\boldsymbol{\theta}^-}$ 易引发不稳定，进一步可写为 $\sin(s) \cdot \frac{\partial c_{\mathrm{noise}}}{\partial s} \cdot \frac{\partial \text{emb}}{\partial c_{\mathrm{noise}}} \cdot \frac{\partial \mathrm{F}_{\boldsymbol{\theta}^-}}{\partial \text{emb}}$（$\text{emb}$ 为时间嵌入）。为缓解不稳定，可采用：

- **A. 切向归一化**：用 $\frac{\frac{\mathrm{d} }{\mathrm{d} s} \mathbf{f}_{\theta^-}}{\left\| \frac{\mathrm{d} }{\mathrm{d} s} \mathbf{f}_{\theta^-} \right\|_2 + c}$ 替代 $\frac{\mathrm{d} }{\mathrm{d} s}\mathbf{f}_{\theta^-}$，$c>0$ 为常数；或将切向裁剪到 $[-1,1]$。
- **B. 切向预热**：将系数 $\sin(s)$ 替换为 $r \cdot \sin(s)$，$r$ 在训练初期从 0 线性增至 1。
- **C. 时间嵌入**：选用较小幅值控制 $\frac{\partial \text{emb}}{\partial c_{\mathrm{noise}}}$；$c_{\mathrm{noise}}(s)=s$ 时 $\partial_s c_{\mathrm{noise}}(s)=1$ 为有界常数。

此外，为稳定性和高效计算 $\frac{\mathrm{d}}{\mathrm{d} s}\mathbf{f}_{\theta^-}$，常需架构上的归一化改进与基于 JVP 的计算，此处不展开。

**时间权重函数。** 人工设计 $\omega(s)$ 可能次优。可借鉴 EDM-2 学习自适应权重 $\omega_{\boldsymbol{\varphi}}(s)$，使不同时间 $s$ 的加权损失方差更均衡。最优 $\omega^*(s)$ 满足：经 $e^{\omega^*(s)}/D$ 缩放后，不同 $s$ 的期望（加权）损失为 1，从而在时间步上平衡并稳定训练。

**时间采样分布。** 从对数正态提议分布采样 $\tan(s)$，即 $e^{\sigma_{\mathrm{d}} \tan(s)} \sim \mathcal{N}(\cdot;P_{\text{mean}}, P_{\text{std}}^2)$，$P_{\text{mean}}$、$P_{\text{std}}$ 为超参数。

**训练目标小结。** 综合上述讨论，最终训练损失为

$$
\mathcal{L}_{\text{sCM}}(\boldsymbol{\theta}, \boldsymbol{\varphi})
:= \mathbb{E}_{s, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[
\frac{e^{\omega_{\boldsymbol{\varphi}}(s)}}{D}
\left\|
\mathrm{F}_{\boldsymbol{\theta}}\left(\frac{\mathbf{x}_s}{\sigma_{\mathrm{d}}}, s\right)
- \mathrm{F}_{\boldsymbol{\theta}^-}\left(\frac{\mathbf{x}_s}{\sigma_{\mathrm{d}}}, s\right)
- \cos(s) \frac{\mathrm{d} \mathbf{f}_{\boldsymbol{\theta}^-}}{\mathrm{d} s}\left(\mathbf{x}_s, s\right)
\right\|_2^2
- \omega_{\boldsymbol{\varphi}}(s)
\right].
$$

其中 $s$ 按上述分布采样，$\mathbf{x}_s$ 由三角形式计算。以此损失训练的模型称为 sCM，其训练流程见原文献算法。

---

## 一般流映射：Consistency Trajectory Model (CTM)

Consistency Trajectory Model (CTM) 是首批学习**一般**流映射 $\boldsymbol{\Psi}_{s\to t}$ 的方法之一。

**实践中 CTM 的设定。** 与 CM 族类似，CTM 最初采用 EDM 表述，使用 $\mathbf{x}$-预测形式的 PF-ODE，噪声调度 $\alpha_t = 1$、$\sigma_t = t$。在此设定下 PF-ODE 为

$$
\frac{\mathrm{d} \mathbf{x}(\tau)}{\mathrm{d} \tau} 
= \frac{\mathbf{x}(\tau) - \mathbb{E}[\mathbf{x}|\mathbf{x}(\tau)]}{\tau}.
$$

从时刻 $s$ 的 $\mathbf{x}_s$ 演化到 $t \le s$ 时，相应流映射可写为 $\mathbf{x}_s + \int_s^t \frac{\mathbf{x}_\tau - \mathbb{E}[\mathbf{x} |\mathbf{x}_\tau]}{\tau}\, \mathrm{d} \tau$。CTM 采用 Euler 风格参数化：对 PF-ODE 做单步 Euler（等价于 DDIM）得

$$
\mathbf{x}_{s\to t}^{\mathrm{Euler}}
= \frac{t}{s}\,\mathbf{x}_s + \Big(1 - \frac{t}{s}\Big)\mathbb{E}[\mathbf{x}|\mathbf{x}_s],
$$

$\mathbf{x}_{s\to t}^{\mathrm{Euler}}$ 为给定 $s$ 时刻状态 $\mathbf{x}_s$ 时 $t$ 时刻解的近似。CTM 也允许更一般的由任意线性高斯前向核 $(\alpha_t, \sigma_t)$ 定义的噪声调度，并以 $\mathbf{v}$-预测形式表达 PF-ODE：$\boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s) = \mathbf{x}_s + \int_s^t \mathbf{v}^*(\mathbf{x}_u,u) \mathrm{d} u$。下面对此一般形式进行讨论。

### CTM 的灵活转移学习参数化

沿用上述 PF-ODE 的单步 Euler 求解器，CTM 将 Oracle 流映射 $\boldsymbol{\Psi}_{s\to t}$ 重写为输入 $\mathbf{x}_s$ 与残差函数 $\mathbf{g}^*$ 的凸组合：

$$
\boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s)
= \frac{t}{s}\,\mathbf{x}_s
+ \frac{s-t}{s}
\Big[\mathbf{x}_s + \frac{s}{s-t} \int_s^t \mathbf{v}^*(\mathbf{x}_u,u) \mathrm{d} u\Big],
$$

残差项定义为 $\mathbf{g}^*(\mathbf{x}_{s}, s, t) := \mathbf{x}_s + \frac{s}{s-t} \int_s^t \mathbf{v}^*(\mathbf{x}_u,u)\mathrm{d} u$。由此得到神经网络参数化

$$
\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t)
:= \frac{t}{s}\,\mathbf{x}_s + \frac{s-t}{s}\,\mathbf{g}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t),
$$

其中 $\mathbf{g}_{\boldsymbol{\theta}}$ 旨在逼近 $\mathbf{g}^*$，故 $\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t)$ 被训练以近似 Oracle 流映射 $\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t) \approx \boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s)$，自然落在统一流映射框架内。该形式天然满足初始条件 $\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,s)=\mathbf{x}_s$，无需在训练中显式强制。

**CTM 参数化的优点。** 当 $t \to s$ 时，$\mathbf{g}^*$ 的性质为：(i) 恢复扩散模型：$\mathbf{g}^*(\mathbf{x}_s, s, s)=\mathbf{x}_s - s\mathbf{v}^*(\mathbf{x}_s,s)$；(ii) 积分表示：$\mathbf{g}^*(\mathbf{x}_{s},s,t)=\mathbf{x}_s - s\mathbf{v}^*(\mathbf{x}_s,s)+\mathcal{O}(|t-s|)$。因此，(1) 估计 $\mathbf{g}^*$ 既能近似有限 $s\to t$ 转移，也能在无穷小 $s\to s$ 极限下刻画瞬时速度 $\mathbf{v}^*$；(2) $\mathbf{g}^*(\mathbf{x}_{s},s,t)$ 可理解为 Oracle 速度加泰勒余项。因此通过 CTM 参数化，学习 $\mathbf{G}_{\boldsymbol{\theta}} \approx \boldsymbol{\Psi}_{s\to t}$（即 $\mathbf{g}_{\boldsymbol{\theta}} \approx \mathbf{g}^*$）同时获得 $\mathbf{G}_{\boldsymbol{\theta}}$ 的长跳能力与通过 $\mathbf{g}_{\boldsymbol{\theta}}$ 恢复扩散模型速度（或 score/去噪器）的能力，在一个框架下统一扩散模型与一致性模型（特殊流映射）的优势。

下面两小节分别介绍 CTM 的一致性损失（支持蒸馏与从零训练，并利用半群性质达到 $\mathbf{G}_{\boldsymbol{\theta}}(\cdot, s, t) \approx \boldsymbol{\Psi}_{s\to t}(\cdot)$）与由该参数化自然产生的辅助损失（扩散模型损失与 GAN 损失），以进一步提升性能。

### CTM 的一致性损失

CTM 旨在近似 Oracle 解映射 $\mathbf{G}_{\boldsymbol{\theta}}(\cdot, s, t) \approx \boldsymbol{\Psi}_{s\to t}(\cdot)$，$s \ge t$。因 Oracle $\boldsymbol{\Psi}_{s\to t}$ 通常无闭式，CTM 通过强制**半群性质** $\boldsymbol{\Psi}_{u\to t}\circ \boldsymbol{\Psi}_{s\to u} = \boldsymbol{\Psi}_{s\to t}$（$s \ge u \ge t$）构造可行的回归目标。根据是否有预训练扩散模型，流映射 $\boldsymbol{\Psi}_{s\to t}$ 可用不同方式近似。以下设 $s \ge u \ge t \in [0,T]$。

**蒸馏训练。** 设已有预训练扩散模型 $\mathbf{v}_{\boldsymbol{\phi}^\times}(\mathbf{x}_s, s) \approx \mathbf{v}^*(\mathbf{x}_s, s)$，则 PF-ODE 由经验动力学近似。CTM 训练 $\mathbf{G}_{\boldsymbol{\theta}}$ 匹配应用于该经验 ODE 的数值求解器 $\mathtt{Solver}_{s\to t}(\mathbf{x}_s;\boldsymbol{\phi}^\times)$，作为 Oracle 的可计算替代。当教师足够强时，求解器可恢复到离散化误差内的 $\boldsymbol{\Psi}_{s\to t}$。若在训练中对整个区间 $[t,s]$ 求解，当 $s$ 与 $t$ 相距较远时成本较高。为提高效率并提供更平滑的信号，CTM 引入**软一致性匹配**：比较 $t$ 时刻的两项预测——直接的学生输出 $\mathbf{G}_{\boldsymbol{\theta}}(\mathbf{x}_s, s, t)$，以及先由教师从 $s$ 推进到随机 $u \sim \mathcal{U}[t,s)$，再由学生在 $u$ 跳到 $t$ 的混合路径 $\mathbf{G}_{\boldsymbol{\theta}^-} \big(\mathtt{Solver}_{s\to u}(\mathbf{x}_s;\boldsymbol{\phi}^\times),\, u,\, t\big)$。学生被训练以匹配该复合预测；$\boldsymbol{\theta}^-$ 为 $\mathbf{G}_{\boldsymbol{\theta}}$ 的止梯度副本。通过变化 $u$，在全局一致性（$u=s$，学生模仿整段教师）与局部一致性（$u = s - \Delta s$，学生从 $s$ 附近短教师步学习）之间插值。为同时强化样本质量，将两项预测用止梯度学生映射到 $t=0$ 后在特征空间度量 $d$ 下比较，定义 $\mathbf{x}_{\mathrm{est}}$ 与 $\mathbf{x}_{\mathrm{target}}$，CTM 一致性损失为

$$
\mathcal{L}_{\mathrm{consist}}(\boldsymbol{\theta}; \boldsymbol{\phi}^\times)
:=
\mathbb{E}_{s,t,u,\mathbf{x}_0,\mathbf{x}_s|\mathbf{x}_0}\Big[ d\big(\mathbf{x}_{\mathrm{est}}, \mathbf{x}_{\mathrm{target}}\big) \Big].
$$

**从零训练。** 利用 CTM 参数化性质 $\mathbf{g}^*(\mathbf{x}_\tau, \tau, \tau) = \mathbf{x}_\tau - \tau\,\mathbf{v}^*(\mathbf{x}_\tau,\tau)$，可用 CTM 自身的估计 $\mathbf{g}_{\boldsymbol{\theta}^-}(\cdot,\tau,\tau)$ 替代 Oracle 残差，得到自诱导的经验 PF-ODE：

$$
\frac{\mathrm{d} \mathbf{x}(\tau)}{\mathrm{d} \tau}
= \frac{\mathbf{x}(\tau) - \mathbf{g}_{\boldsymbol{\theta}^-} \left(\mathbf{x}(\tau),\tau,\tau\right)}{\tau}.
$$

然后用该 ODE 的求解器近似 Oracle 解映射，并训练学生匹配求解器输出。与蒸馏情形类似，为降低成本，通过半群性质使用较短监督路径：$\mathbf{G}_{\boldsymbol{\theta}}(\mathbf{x}_s, s, t) \approx \mathbf{G}_{\boldsymbol{\theta}^-} \big(\mathtt{Solver}_{s\to u}(\mathbf{x}_s;\boldsymbol{\theta}^-),\, u,\, t\big)$，$u \sim \mathcal{U}[t,s)$。唯一变化是用自诱导教师 $\mathbf{g}_{\boldsymbol{\theta}^-}$ 替代外部教师 $\mathbf{v}_{\boldsymbol{\phi}^\times}$。无预训练模型时的目标为 $\hat{\mathbf{x}}_{\mathrm{target}} := \mathbf{G}_{\boldsymbol{\theta}^-} \big( \mathbf{G}_{\boldsymbol{\theta}^-}(\mathtt{Solver}_{s\to u}(\mathbf{x}_s;\boldsymbol{\theta}^-),\, u,\, t),\, t,\, 0\big)$，代入即得从零训练的一致性损失 $\mathcal{L}_{\mathrm{consist}}(\boldsymbol{\theta}; \boldsymbol{\theta}^-)$。概念上这是 CTM 内部的自蒸馏：模型为自己提供短视距教师信号，学生学习完整转移。

![CTM 半群性质示意](../arXiv-2510.21890v1/Images/PartD/ctm-target.pdf)

*图：CTM 半群性质示意。对任意 $s \ge u \ge t$，CTM 强制 $\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_s, s, t) \approx \mathrm{G}_{\boldsymbol{\theta}^-} \big(\mathtt{Solver}_{s\to u}(\mathbf{x}_s), u, t\big)$，即短求解段 $s \to u$ 再接 CTM“跳跃”到 $t$ 与直接 CTM 映射 $s\to t$ 一致。求解器可为预训练扩散或 CTM 自诱导教师。*

### CTM 的辅助损失

（自）蒸馏可能仅优化教师生成的目标而缺乏真实数据的直接监督，从而不如教师。CTM 可自然加入数据驱动的正则，例如去噪分数匹配与对抗（GAN）项，以更好地学习流映射。

**扩散损失的自然整合。** 扩散模型损失（更确切地说，条件流匹配损失）自然融入 CTM，提供固定回归目标以辅助流映射学习。由 $\mathbf{v}^*(\mathbf{x}_s, s) = \frac{\mathbf{x}_s - \mathbf{g}^*(\mathbf{x}_s, s, s)}{s}$ 及 $\mathbf{g}^* \approx \mathbf{g}_{\boldsymbol{\theta}}$ 可诱导速度参数化 $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_s, s) := \frac{1}{s}\bigl(\mathbf{x}_s - \mathbf{g}_{\boldsymbol{\theta}}(\mathbf{x}_s, s, s)\bigr)$。在线性高斯路径 $\mathbf{x}_s = \alpha_s \mathbf{x}_0 + \sigma_s \boldsymbol{\epsilon}$ 下，扩散模型损失为

$$
\mathcal L_{\mathrm{DM}}(\boldsymbol{\theta})
:= 
\mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, s}
\Bigl[
w(s)
\bigl\|
 \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_s, s) - \left(\alpha_s' \mathbf{x}_0 + \sigma_s' \boldsymbol{\epsilon}\right)
\bigr\|_2^2
\Bigr].
$$

当 $t$ 接近 $s$ 时，$\mathcal L_{\mathrm{DM}}$ 通过显式监督轨迹上的小跳跃提高精度；此时 $\mathrm{G}_{\boldsymbol{\theta}}$ 中因子 $1-\tfrac{t}{s}$ 接近零，梯度较弱，$\mathcal L_{\mathrm{DM}}$ 提供更强的局部信号并稳定训练。一致性损失强制轨迹匹配（零阶），扩散损失强制斜率匹配（一阶）。

**（可选）GAN 损失。** 为减轻过度平滑，CTM 可加入对抗项，用判别器 $D_{\boldsymbol{\zeta}}$ 区分真实 $\mathbf{x}_{0}\sim p_{\mathrm{data}}$ 与生成 $\mathbf{x}_{\mathrm{est}}(\mathbf{x}_{s},s,t)$，目标为

$$
\mathcal{L}_{\mathrm{GAN}}(\boldsymbol{\theta}, \boldsymbol{\zeta}) := \mathbb{E}_{\mathbf{x}_0} \big[\log D_{\boldsymbol{\zeta}}(\mathbf{x}_0)\big] + \mathbb{E}_{s,t,\mathbf{x}_0,\mathbf{x}_s \vert \mathbf{x}_0} \big[\log (1 - D_{\boldsymbol{\zeta}}( \mathbf{x}_{\mathrm{est}}(\mathbf{x}_s, s, t)))\big],
$$

$D_{\boldsymbol{\zeta}}$ 最大化、$\mathrm{G}_{\boldsymbol{\theta}}$ 最小化。判别器充当自适应感知距离，鼓励真实细节；理论上 GAN 项驱动 $p_{\mathrm{data}}$ 与 $\mathrm{G}_{\boldsymbol{\theta}}$ 诱导的分布之间的分布匹配（Jensen–Shannon 散度），可提升保真度。

**CTM 总体目标。** CTM 将（自）蒸馏、扩散与 GAN 损失统一为单一训练框架：

$$
\mathcal{L}_{\mathrm{CTM}}(\boldsymbol{\theta},\boldsymbol{\zeta})
:=  \mathcal{L}_{\mathrm{consist}}(\boldsymbol{\theta};  \boldsymbol{\phi}^\times/\boldsymbol{\theta}^-)
+ \lambda_{\mathrm{DM}} \mathcal{L}_{\mathrm{DM}}(\boldsymbol{\theta})
+ \lambda_{\mathrm{GAN}} \mathcal{L}_{\mathrm{GAN}}(\boldsymbol{\theta},\boldsymbol{\zeta}),
$$

教师为外部预训练 $\boldsymbol{\phi}^\times$ 或自诱导 $\boldsymbol{\theta}^-$。回归式成分 $\mathcal{L}_{\mathrm{consist}}$ 与 $\mathcal{L}_{\mathrm{DM}}$ 作为强正则，可选 GAN 项在不牺牲稳定性的前提下改善细尺度细节。

### CTM 的灵活采样

CTM 学习任意 $s>t$ 的一般流映射 $\boldsymbol{\Psi}_{s\to t}$，因此支持任意时刻到任意时刻的转移，可实现灵活采样策略。例如 CTM 提出 **$\gamma$ 采样**，超参数 $\gamma$ 控制生成过程中的随机性。CTM 也可复用扩散模型的标准推断技术（如 ODE 求解器与精确似然计算）。下面固定采样时间网格 $T=\tau_0 > \tau_1 > \cdots > \tau_{M}=0$。

**算法：CTM 的 $\gamma$-采样**

- **输入**：训练好的 CTM $\mathbf{G}_{\boldsymbol{\theta}^\times}$，$\gamma\in[0,1]$，$T=\tau_0>\tau_1 > \cdots > \tau_{M}=0$。
- 从 $\mathbf{x}_{\tau_{0}}\sim p_{\mathrm{prior}}=\mathcal{N}(\boldsymbol{0}, T^2\mathrm{I})$ 开始。
- 对 $n=0$ 到 $M-1$：$\tilde{\tau}_{n+1}\leftarrow \sqrt{1-\gamma^{2}}\tau_{n+1}$；去噪 $\mathbf{x}_{\tilde{\tau}_{n+1}}\leftarrow \mathbf{G}_{\boldsymbol{\theta}^\times}(\mathbf{x}_{\tau_{n}},\tau_{n},\tilde{\tau}_{n+1})$；扩散 $\mathbf{x}_{\tau_{n+1}}\leftarrow\mathbf{x}_{\tilde{\tau}_{n+1}}+\gamma \tau_{n+1}\boldsymbol{\epsilon}$，$\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0}, \mathrm{I})$。
- **输出**：$\mathbf{x}_{\tau_{M}}$。

**$\gamma$-采样的方法意义。** $\gamma$-采样给出了学习一般流映射模型时自然出现的一族采样器。它涵盖 CM 的多步采样与类似 ODE 时间步进的采样。$\gamma$ 直接控制生成过程中的语义变化程度，使 $\gamma$ 采样成为面向不同下游任务的灵活、任务感知策略。

![不同 $\gamma$ 的 $\gamma$-采样示意](../arXiv-2510.21890v1/Images/PartD/ctm_gamma.pdf)

*图：不同 $\gamma$ 下的 $\gamma$-采样示意。流程在“去噪”与“加噪”之间交替：$\tau_{n}\xrightarrow{\text{去噪}} \sqrt{1-\gamma^{2}}\tau_{n+1}\xrightarrow{\text{加噪}} \tau_{n+1}$。左：$\gamma=1$，完全随机；右：$\gamma=0$，完全确定性；中：$0<\gamma<1$ 介于两者之间。*

- $\gamma=1$：与 CM 的多步采样一致（特殊流映射 $\boldsymbol{\Psi}_{s\to 0}$），完全随机，步数变化会产生语义变化。  
- $\gamma=0$：退化为完全确定性时间步进，估计 PF-ODE 的解轨迹；与常规基于 ODE 的时间步进相比，CTM 避免了数值求解器的离散化误差。  
- $0 < \gamma < 1$：在两者之间插值，允许受控的随机性。  

只有模型学习一般流映射 $\boldsymbol{\Psi}_{s\to t}$ 时才能实现 $\gamma \in (0,1]$ 的采样器。

**$\gamma$-采样的分析。** CTM 经验上观察到 CM 的多步采样在步数 $M \geq 4$ 时质量下降。原因可概括为：当 $\gamma \neq 0$ 时，每次神经“跳跃”会引入小失配，这些失配在向零时刻迭代映射时累积。形式上有关于 2 步 $\gamma$-采样的 TV 误差界（见原文献定理 8）。当 $\gamma=1$ 时误差量级为 $\mathcal{O}(\sqrt{T + \tau_1 + \cdots + \tau_M})$；当 $\gamma=0$ 时时间重叠被消除，误差界更紧，为 $\mathcal{O}(\sqrt{T})$。经验上 CTM 在 $\gamma=0$ 时在采样速度与样本质量之间取得较好折中，增加步数可提升质量而不引入不稳定。

**CTM 支持扩散推断。** 因 CTM 通过 $\mathbf{g}_{\boldsymbol{\theta}}$ 直接学习 score（或去噪器），与其参数化一致，故与扩散模型的推断技术兼容，例如精确似然计算或 DDIM、DPM 等高级采样器，只需使用 $\mathbf{g}_{\boldsymbol{\theta}}(\cdot, s, s)$。

---

## 一般流映射：Mean Flow (MF)

正如扩散模型存在多种等价参数化与训练目标，一般流映射 $\boldsymbol{\Psi}_{s\to t}$ 也可用多种方式学习。本节介绍 **Mean Flow (MF)**，作为一般流映射族 $\boldsymbol{\Psi}_{s\to t}$ 的后来代表，从另一种同样有原则的视角说明如何有效学习该映射。

### Mean Flow 的建模与训练

与基于 EDM 的 CM、CTM 不同，MF 基于流匹配表述（$t\in[0,1]$ 时 $\alpha_t = 1-t$，$\sigma_t  = t$）。MF 不直接参数化流映射，而是学习区间 $[t,s]$（$t<s$）上的**平均漂移**：

$$
\mathbf{h}_{\boldsymbol{\theta}}(\mathbf{x}_s, s, t)  \approx 
\mathbf{h}^*(\mathbf{x}_s, s, t) := \frac{1}{t-s}\int_s^t \mathbf{v}^*(\mathbf{x}_u,u) \mathrm{d} u.
$$

对应 Oracle 损失为

$$
\mathbb{E}_{t<s} \mathbb{E}_{\mathbf{x}_s\sim p_s}\Big[w(s) 
\|\mathbf{h}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t) - \mathbf{h}^*(\mathbf{x}_s,s,t)\|_2^2\Big].
$$

特别地，当 $s \to t$ 时，损失退化为流匹配损失，学习瞬时速度。Oracle 目标 $\mathbf{h}^*(\mathbf{x}_s,s,t)$ 一般无闭式，MF 通过对 $(t-s)\,\mathbf{h}^*(\mathbf{x}_s,s,t) = \int_s^t \mathbf{v}^*(\mathbf{x}_u,u)\,\mathrm{d} u$ 关于 $s$ 求导得到的恒等式构造替代，得到实用训练目标

$$
\mathcal{L}_{\mathrm{MF}}(\boldsymbol{\theta})
:= \mathbb{E}_{t<s} \mathbb{E}_{\mathbf{x}_s\sim p_s}\Big[w(s)\,
\|\mathbf{h}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t) - \mathbf{h}_{\boldsymbol{\theta}^-}^{\mathrm{tgt}}(\mathbf{x}_s,s,t)\|_2^2\Big],
$$

回归目标为

$$
\mathbf{h}_{\boldsymbol{\theta}^-}^{\mathrm{tgt}}(\mathbf{x}_s,s,t)
:= \mathbf{v}^*(\mathbf{x}_s,s) - (s-t)\Big[(\partial_{\mathbf{x}}\mathbf{h}_{\boldsymbol{\theta}^-})(\mathbf{x}_s,s,t) \mathbf{v}^*(\mathbf{x}_s,s)
+ \partial_s \mathbf{h}_{\boldsymbol{\theta}^-}(\mathbf{x}_s,s,t)\Big].
$$

实践中 Oracle 速度 $\mathbf{v}^*$ 无闭式，需近似。两种常见策略：**蒸馏**（使用 flow matching 骨干的预训练扩散模型 $\mathbf{v}_{\boldsymbol{\phi}^\times} \approx \mathbf{v}^*$）或**从零训练**（使用前向 $\mathbf{x}_s=\alpha_s\mathbf{x}_0+\sigma_s\boldsymbol{\epsilon}$ 的单点条件速度 $\alpha_s'\mathbf{x}_0 + \sigma_s'\boldsymbol{\epsilon}$）。无论哪种，都需要计算目标网络 $\mathbf{h}_{\boldsymbol{\theta}^-}$ 的**雅可比–向量积 (JVP)**。

### Mean Flow 的采样

训练好的 MF $\mathbf{h}_{\boldsymbol{\theta}^\times}$ 自然给出流映射的近似：对任意起点 $\mathbf{x}_s$，从 $s$ 到 $t$ 的映射近似为

$$
\boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s) = \mathbf{x}_s + (t-s)\,\mathbf{h}^*(\mathbf{x}_s,s,t)
  \approx   \mathbf{x}_s + (t-s)\,\mathbf{h}_{\boldsymbol{\theta}^\times}(\mathbf{x}_s,s,t).
$$

因此支持单步与多步采样。例如从 $\mathbf{x}_T\sim p_{\mathrm{prior}}$ 出发，单步生成干净样本为 $\mathbf{x}_0  \leftarrow  \mathbf{x}_T + T\,\mathbf{h}_{\boldsymbol{\theta}^\times}(\mathbf{x}_T,T,0)$。多步生成可沿时间网格顺序应用该映射，与 CTM 的时间步进方式相同。因 MF 学习一般流映射，也支持类似 CTM 的 $\gamma$-采样，通过超参数 $\gamma$ 在采样中注入随机性。

### CTM 与 MF 的等价性

初看 CTM 与 MF 似乎无关。事实上二者只是同一 Oracle 流映射 $\boldsymbol{\Psi}_{s\to t}$ 的不同参数化，训练损失（CTM 的一致性损失与 MF 的 Oracle 损失）仅时间权重不同。

**参数化关系。** 流映射可等价写成：定义式 $\mathbf{x}_s + \int_s^t \mathbf{v}^*(\mathbf{x}_u, u)\mathrm{d} u$；CTM 形式 $\frac{t}{s}\mathbf{x}_s + \frac{s-t}{s}[\mathbf{x}_s + \frac{s}{s-t}\int_s^t \mathbf{v}^*(\mathbf{x}_u,u)\,\mathrm{d} u]$（括号内由 $\mathbf{g}_{\boldsymbol{\theta}}$ 近似）；MF 形式 $\mathbf{x}_s + (t-s)[\frac{1}{t-s}\int_s^t \mathbf{v}^*(\mathbf{x}_u,u)\,\mathrm{d} u]$（括号内由 $\mathbf{h}_{\boldsymbol{\theta}}$ 近似）。

**训练损失关系。** 令 $\mathbf{g}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t) := \mathbf{x}_s - s \mathbf{h}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t)$，取 $d(\mathbf{x},\mathbf{y}):=\|\mathbf{x}-\mathbf{y}\|^2$，代入统一目标并视 $\mathrm{G}_{\boldsymbol{\theta}}$ 为 CTM 的流映射参数化，可证

$$
\frac{1}{s^2} \Big\|\mathbf{g}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t)-\mathbf{g}^*(\mathbf{x}_s,s,t) \Big\|^2 = \Big\|\mathbf{h}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t)-\mathbf{h}^*(\mathbf{x}_s,s,t) \Big\|^2.
$$

因此 CTM 与 MF 的损失在权重意义下等价。令 $t=0$ 即恢复 CM 情形 $\boldsymbol{\Psi}_{s\to 0}$。

**实践中的辅助损失。** CTM 将一致性损失与自带的扩散模型损失联合训练；MF 类似地控制 $s \neq t$ 与 $s=t$ 样本对的比例，整体优化为 MF 目标与流匹配目标的混合。两种参数化都能从学习瞬时速度（固定回归目标的扩散训练）平滑过渡到使用止梯度伪回归目标的流映射学习。

**CTM 与 MF 参数化均支持灵活推断。** CTM（$\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t)$）与 MF（$\mathbf{x}_s+(t-s)\mathbf{h}_{\boldsymbol{\theta}}(\mathbf{x}_s,s,t)$）都近似流映射 $\boldsymbol{\Psi}_{s\to t}$。因二者在无穷小极限 $t\to s$ 下都恢复瞬时扩散漂移（$\mathbf{g}^*(\mathbf{x}_s,s,s)=\mathbf{x}_s-\mathbf{v}^*(\mathbf{x}_s,s)$，$\mathbf{h}^*(\mathbf{x}_s,s,s)=\mathbf{v}^*(\mathbf{x}_s,s)$），故自然支持 CTM 的 $\gamma$-采样，并与扩散模型的引导、精确似然、高阶求解器加速等推断技术兼容。这是仅针对特殊流映射 $\boldsymbol{\Psi}_{s\to0}$ 的 CM 族所不具备的。

**小结。** CTM 与 MF 的等价性类似于扩散模型中不同参数化描述同一 Oracle 目标的情形。原则上二者数学一致；实践中因损失权重、网络设计或优化动力学的差异，在特定条件下可能一方更优。该视角表明 CTM 与 MF 并非唯一可能，流映射的其他参数化也可能实现高效稳定训练，为新的独立生成模型打开空间。

---

## 结语

本章将讨论收束到一个新的生成建模范式：从零学习快速、少步生成器。在改进数值求解器或蒸馏预训练模型之外，我们聚焦于设计既在原理上站得住脚、又在设计上高效的独立训练原则。

核心创新是直接学习底层概率流 ODE 的流映射 $\boldsymbol{\Psi}_{s\rightarrow t}$。在没有教师的情况下使其可解的关键，在于利用 ODE 流的**半群性质**：长轨迹可分解为更短的段落，从而为训练提供强有力的自监督信号。

我们从 Consistency Models (CM) 开始，它们通过学习将任意带噪状态映射回干净原点的特殊流映射 $\boldsymbol{\Psi}_{s\rightarrow 0}$ 开创了这一方向。随后 Consistency Trajectory Model (CTM) 与 Mean Flow (MF) 将这一思想推广到学习对满足 $s\ge t$ 的任意 $s,t$ 的完整“任意时刻到任意时刻”流映射 $\boldsymbol{\Psi}_{s\rightarrow t}$。尽管参数化形式不同，我们说明了这些方法本质上是近似定义流映射的同一路径积分的等价方式。

这些流映射模型代表了本专著所发展原理的有力综合：它们继承了 Score SDE 框架的连续时间基础与 Flow Matching 的确定性传输视角，但将训练目标重新表述为自包含且高效的形式。

通过直接学习解映射，这些独立模型成功地将迭代扩散过程的高样本质量与单步生成器的推断速度结合在一起，在保真度与速度之间取得了根本性的折中，标志着生成建模的重要里程碑。这一成就并非终点，而是设计强大、高效、可控的生成式 AI 的新篇章的开端。

> *重要的是不要停止发问。好奇心自有其存在的理由。* — Albert Einstein
