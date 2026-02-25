# 第 4 章 （选读）扩散模型与最优传输

将一个分布映射到另一个分布（以生成为特例）是一个核心挑战。流匹配通过学习一个随时间变化的速度场来解决这一问题，该速度场将质量从源输运到目标。这自然与传输理论相联系：经典最优传输寻求分布之间成本最小的路径，而其熵正则化形式——Schrödinger 桥——则相对于参考过程（如布朗运动）选择最可能的受控扩散。

本章我们回顾最优传输、熵正则最优传输以及 Schrödinger 桥作为分布到分布问题的表述基础。由此引出一个核心问题：扩散模型在多大程度上实现了这类最优传输？它们有两种视角：作为**随机过程**（由前向与逆向 SDE 定义），以及作为**确定性过程**（由 PF-ODE 给出）。随机视角直接对应熵正则最优传输，而 PF-ODE 一般并不对应任何已知的传输目标。这一差距留下一个开放问题：在什么条件下扩散模型可以被视为在求解最优传输问题？

---

## 4.1 分布到分布翻译的前奏

扩散模型将终端分布固定为标准高斯 $p_{\mathrm{prior}}$。然而，许多应用需要**分布到分布**的翻译：将源分布 $p_{\mathrm{src}}$ 变换为不同的目标 $p_{\mathrm{tgt}}$。例如将草图转为逼真图像，或在艺术风格之间翻译。

现代扩散方法提供了实现这一点的实用途径。**单端点**方法如 SDEdit 从 $t=0$ 的源图像出发，将其扩散到中间步 $t$，然后使用目标域上预训练的扩散模型逆向该过程，得到在风格与内容上匹配目标分布的输出。

**双端点**方法如 Dual Diffusion Bridge 则通过共享的潜在分布（通常是 $t=1$ 处的高斯）连接两个域。前向概率流 ODE 将 $p_{\mathrm{src}}$ 的样本输运到该潜在空间，而在目标域上训练的反向 ODE 再将它们映射回 $p_{\mathrm{tgt}}$。除这类采样时方法外，流匹配框架提供了一种基于训练的替代：直接学习一个 ODE 流，连续地将质量从 $p_{\mathrm{src}}$ 输运到 $p_{\mathrm{tgt}}$。

关键在于，在分布之间变换需要的不仅是两个分别训练的模型，而是一种在**两个**端点都对齐动力学、并以“最省”（成本最优）方式做到这一点的有原则的映射。

本节中，我们不逐一综述众多基于扩散的翻译应用，而是将焦点转向这一经典分布到分布问题的数学基础。特别地，我们突出最优传输（OT）及其熵正则变体——Schrödinger 桥（SB）——它们在理论界长期被视为成本高效（数学意义上）分布变换的典范表述。

其核心，根本问题是：

**问题.** 给定两个概率分布，在最小化总**成本**的前提下，将其中一个变换为另一个的最有效方式是什么？

这里，成本 $c(\mathbf{x}, \mathbf{y})$ 是一个非负函数，表示将单位质量从点 $\mathbf{x}$ 移动到点 $\mathbf{y}$ 的惩罚。常见选择是平方距离 $c(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|^2$。

本节给出简要概述，以澄清包括流匹配在内的基于扩散方法如何与经典及正则化最优传输相联系。我们旨在探讨的核心问题是：

**问题.** 扩散模型是否是连接 $p_{\text{data}}$ 与 $p_{\text{prior}}$ 的一种最优传输形式，且在何种意义上？

为回答这一问题，我们首先在 4.2 节澄清“最优性”的含义。我们回顾经典**最优传输（OT）**的 Monge–Kantorovich 静态形式（式 (eq:ot)）及其动态 Benamou–Brenier 表述（式 (eq:bb-ot)；在连续性方程约束下最小化动能），以及 4.2 节中的熵正则变体（**熵正则 OT**）式 (eq:eot-epsilon)，它与 **Schrödinger 桥问题**（式 (eq:sb-kinetic)）等价。在动态视角下，OT 诱导满足连续性方程的确定性流，而 SB 诱导边缘由 Fokker–Planck 方程演化的受控扩散。我们在 4.3 节给出这些表述之间的高层对应关系。

随后我们将讨论分为两部分。第一，在 4.4 节我们说明：标准扩散模型中使用的固定前向加噪 SDE 本身并不是任意 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 之间的 Schrödinger 桥——前向过程是选定的参考扩散，无论前向还是逆向时间 SDE 一般都不会强制端点精确匹配到给定的目标。因此除非针对这些端点显式求解 SB 问题，否则它不是熵 OT 最优的；而由于它以一个起点为锚，它是半桥问题的最优解。

第二，在 4.5 节我们回到生成设定 $p_{\mathrm{src}}=p_{\mathrm{prior}}$（高斯）与 $p_{\mathrm{tgt}}=p_{\mathrm{data}}$。PF-ODE 按构造定义了一个将 $p_{\mathrm{prior}}$ 输运到 $p_{\mathrm{data}}$ 的确定性映射。然而，该流一般不是给定传输成本（例如二次 $W_2$）下的 OT 映射：它实现的是众多可行确定性耦合之一，并不最小化 Benamou–Brenier 作用量。接着我们讨论“整流流”程序（4.5.2 节）是否能得到 OT 映射；但一般而言没有这样的理论保证。因此，扩散模型 PF-ODE 映射与 OT 之间的精确刻画仍然是一个困难且未解决的问题。

---

## 4.2 问题设定的分类

本节介绍何谓将质量从 $p_{\mathrm{src}}$ 输运到 $p_{\mathrm{tgt}}$ 的“最有效”或“最优”方式。其中包括经典最优传输（OT）及其熵正则变体，后者有一个等价表述即 Schrödinger 桥。这一分类为后文澄清与扩散模型的联系提供背景。

### 4.2.1 最优传输（OT）

**Monge–Kantorovich（静态）形式的 OT 问题.**

我们固定一个成本函数 $c:\mathbb{R}^D\times\mathbb{R}^D\to\mathbb{R}$，用于指定将概率质量从 $\mathbf{x}$ 输运到 $\mathbf{y}$ 的代价。目标是以尽可能低的成本将源分布 $p_{\mathrm{src}}$ 变换为目标分布 $p_{\mathrm{tgt}}$。

要定义成本，我们必须知道哪些配对 $(\mathbf{x},\mathbf{y})$ 被匹配。这一角色由**耦合**扮演：$\mathbb{R}^D\times\mathbb{R}^D$ 上的联合分布 $\gamma$，其边缘为 $p_{\mathrm{src}}$ 和 $p_{\mathrm{tgt}}$。换言之，从 $\gamma$ 抽样 $(\mathbf{x},\mathbf{y})$ 表示我们将源中的 $\mathbf{x}$ 与目标中的 $\mathbf{y}$ 匹配。若 $\gamma$ 关于 Lebesgue 测度有密度 $\gamma(\mathbf{x},\mathbf{y})$，则边缘约束为
$$
\int_{\mathbb{R}^D} \gamma(\mathbf{x},\mathbf{y})\,\mathrm{d} \mathbf{y} = p_{\mathrm{src}}(\mathbf{x}), 
\qquad 
\int_{\mathbb{R}^D} \gamma(\mathbf{x},\mathbf{y})\,\mathrm{d} \mathbf{x} = p_{\mathrm{tgt}}(\mathbf{y}).
$$
即对 $\mathbf{y}$ 积分得到 $\mathbf{x}$ 处的源密度，对 $\mathbf{x}$ 积分得到 $\mathbf{y}$ 处的目标密度。

我们给出两个标准例子作为说明：

1. **离散支撑.** 若 $p_{\mathrm{src}}$ 和 $p_{\mathrm{tgt}}$ 支撑在有限个点上，则耦合由一个非负矩阵 $(\gamma_{ij})$ 表示，其行和为 $p_{\mathrm{src}}(i)$、列和为 $p_{\mathrm{tgt}}(j)$。每个元素 $\gamma_{ij}$ 表示从 $i$ 输运到 $j$ 的质量。
2. **确定性映射.** 若存在可测映射 $\mathbf{T}$ 使得 $\mathbf{T}_\#p_{\mathrm{src}}=p_{\mathrm{tgt}}$，则 $\gamma=(\mathbf{I},\mathbf{T})_\#p_{\mathrm{src}}$ 是一个确定性耦合，将每点 $\mathbf{x}$ 直接映到 $\mathbf{T}(\mathbf{x})$。

一旦固定耦合 $\gamma$，传输成本就是该方案下的平均单位成本：
$$
\int c(\mathbf{x},\mathbf{y})\,\mathrm{d}\gamma(\mathbf{x},\mathbf{y})
=\mathbb{E}_{(\mathbf{x},\mathbf{y})\sim\gamma}\!\big[c(\mathbf{x},\mathbf{y})\big].
$$
离散情形下这化为 $\sum_{i,j} c_{ij}\gamma_{ij}$，连续情形下则为二重积分。下文我们只考虑连续情形。

最优传输问题就是在所有可行耦合中选出使该期望成本最小的那个：
$$
\mathrm{OT}\big(p_{\mathrm{src}},p_{\mathrm{tgt}}\big)
:= \inf_{\gamma\in\Gamma(p_{\mathrm{src}},p_{\mathrm{tgt}})}
\int c(\mathbf{x},\mathbf{y})\,\mathrm{d}\gamma(\mathbf{x},\mathbf{y}),
$$
其中可行集仅施加边缘（或质量守恒）约束：
$$
\Gamma(p_{\mathrm{src}},p_{\mathrm{tgt}})
= \Big\{\gamma \in \mathcal{P}(\mathbb{R}^D\times\mathbb{R}^D): 
\int \gamma(\mathbf{x},\mathbf{y}) \mathrm{d} \mathbf{y} = p_{\mathrm{src}}(\mathbf{x}),\,
\int \gamma(\mathbf{x},\mathbf{y})\mathrm{d} \mathbf{x} = p_{\mathrm{tgt}}(\mathbf{y})\Big\},
$$
其中 $\mathcal{P}(\mathbb{R}^D\times\mathbb{R}^D)$ 表示 $\mathbb{R}^D\times\mathbb{R}^D$ 上所有概率测度的集合。

**特例：Wasserstein-2 距离.**

Wasserstein-2 距离是二次成本 $c(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|^2$ 下 Monge–Kantorovich 问题的特例。它将两个概率分布之间的距离度量为：
$$
\mathcal{W}_2^2(p_{\mathrm{src}}, p_{\mathrm{tgt}}) := \inf_{\gamma \in \Gamma(p_{\mathrm{src}}, p_{\mathrm{tgt}})} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma} \left[ \|\mathbf{x} - \mathbf{y}\|^2 \right].
$$

在 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 满足适当假设时，Brenier 定理（见 4.5.1 节）保证二次成本下的最优耦合 $\gamma$ 集中在某个确定性映射 $\mathbf{T}:\mathbb{R}^D \rightarrow \mathbb{R}^D$ 的图上。因此 Wasserstein-2 距离可等价地写为：
$$
\mathcal{W}_2^2(p_{\mathrm{src}}, p_{\mathrm{tgt}}) = \inf_{\substack{\mathbf{T}:\mathbb{R}^D \rightarrow \mathbb{R}^D,\\ \text{ s.t. }\mathbf{T}_\#p_{\mathrm{src}} = p_{\mathrm{tgt}}}} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{src}}} \left[ \| \mathbf{T}(\mathbf{x}) - \mathbf{x} \|^2 \right].
$$
这里 $\mathbf{T}_\#p_{\mathrm{src}} = p_{\mathrm{tgt}}$ 表示 $\mathbf{T}$ 将 $p_{\mathrm{src}}$ 前推到 $p_{\mathrm{tgt}}$，即 $\mathbf{x} \sim p_{\mathrm{src}}$ 时 $\mathbf{T}(\mathbf{x}) \sim p_{\mathrm{tgt}}$。

因此，Wasserstein-2 距离表示在所有匹配给定边缘的耦合或传输映射中，最小的期望平方传输成本。记为 $\mathbf{T}^*(\mathbf{x})$ 的最优传输映射称为 **Monge 映射**，给出了将 $p_{\mathrm{src}}$ 变换为 $p_{\mathrm{tgt}}$ 的最有效方式。

**Benamou–Brenier（动态）形式的 OT.**

![OT 动态视角示意图](../arXiv-2510.21890v1/Images/PartB/ot_illustration.pdf)

**图 4.1：OT 动态视角示意图。** 插值 $p_t^{\mathrm{OT}}$ 随时间连续演化，提供将 $p_{\mathrm{src}}$ 确定性映射到 $p_{\mathrm{tgt}}$ 的成本最小传输方案（McCann 位移插值）。

与 Monge–Kantorovich 形式中直接静态地映射分布不同，传输也可以建模为连续时间流：
$$
p_0 := p_{\mathrm{src}} \to p_t \to p_1 := p_{\mathrm{tgt}}, \quad t \in [0,1].
$$
Benamou 与 Brenier 引入的这一最优传输动态表述，寻求一个光滑速度场 $\mathbf{v}_t(\mathbf{x})$，描述 $p_t(\mathbf{x})$ 中的质量如何随时间演化。

Benamou–Brenier 形式表明，对二次成本 $c(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2^2$（即 $\mathcal{W}_2$ 距离），式 (eq:ot) 中静态 OT 问题的最优值等于下列动能最小化问题的最优值：
$$
\mathcal{W}_2^2(p_{\mathrm{src}}, p_{\mathrm{tgt}}) = \min_{\substack{ (p_t, \mathbf{v}_t) \text{ s.t. }
    \partial_t p_t + \nabla \cdot (p_t \mathbf{v}_t) = 0, \\
    p_0 = p_{\mathrm{src}},\,\, p_1 = p_{\mathrm{tgt}}
}}
\int_0^1 \int_{\mathbb{R}^D} \|\mathbf{v}_t(\mathbf{x})\|^2 p_t(\mathbf{x}) \mathrm{d}\mathbf{x} \mathrm{d} t
$$
其中对每个 $t \in [0,1]$，$p_t$ 是 $\mathbb{R}^D$ 上的概率分布。特别地，最优传输流 $p_t(\mathbf{x})$ 遵循 **McCann 位移插值**：
$$
\mathbf{T}_t^*(\mathbf{x}) = (1 - t)\mathbf{x} + t\mathbf{T}^*(\mathbf{x}),
$$
其中 $\mathbf{T}^*(\mathbf{x})$ 是将 $p_{\mathrm{src}}$ 输运到 $p_{\mathrm{tgt}}$ 的 OT 映射。该线性插值使质量沿直线以恒定速度移动：对每个 $t \in [0,1]$ 有 $p_t = \mathbf{T}_t^*\#p_{\mathrm{src}}$。

最优传输映射 $\mathbf{T}^*$ 满足 **Monge–Ampère 方程**：
$$
p_{\mathrm{tgt}}\left(\nabla \psi(\mathbf{x})\right) \det\left(\nabla^2 \psi(\mathbf{x})\right) = p_{\mathrm{src}}(\mathbf{x}),
$$
其中由 Brenier 定理，对某个凸函数 $\psi$ 有 $\mathbf{T}^*(\mathbf{x}) = \nabla \psi(\mathbf{x})$。然而该非线性 PDE 通常难以得到显式解。注意这正是归一化流使用的变量变换关系（参见式 (eq:nf-change-of-var)）：流用可处理的 Jacobian 行列式参数化可逆传输映射，但一般不施加梯度势结构 $\mathbf{T}^*=\nabla\psi$；因此训练得到的流可能与 Brenier/OT 映射相差很大。

### 4.2.2 熵正则最优传输（EOT）

为具体动机 EOT，考虑由样本构造的经验分布。设 $p_{\mathrm{src}}$ 支撑在点 $\{\mathbf{x}^{(i)}\}_{i=1}^n \subset \mathbb{R}^D$ 上、权重为 $a_i$，$p_{\mathrm{tgt}}$ 支撑在 $\{\mathbf{y}^{(j)}\}_{j=1}^n \subset \mathbb{R}^D$ 上、权重为 $b_j$。则耦合是一个 $n\times n$ 非负矩阵 $\gamma=(\gamma_{ij})$，其行和匹配 $a$、列和匹配 $b$。每个元素 $\gamma_{ij}$ 表示从 $\mathbf{x}^{(i)}$ 输运到 $\mathbf{y}^{(j)}$ 的质量。

**为何正则化 OT？**

在这种离散设定下，经典 OT（在连续表述式 (eq:ot) 中取计数测度得到）化为最小化
$$
\min_{\gamma=(\gamma_{ij})}\sum_{i,j} C_{ij}\,\gamma_{ij},
$$
在所有可行耦合 $\gamma=(\gamma_{ij})$ 上，其中 $C_{ij} = c(\mathbf{x}^{(i)},\mathbf{y}^{(j)})$ 是将单位质量从源点 $\mathbf{x}^{(i)}$ 移到目标点 $\mathbf{y}^{(j)}$ 的成本，$c:\mathbb{R}^D\times\mathbb{R}^D\to\mathbb{R}_{\ge 0}$ 为给定的地面成本（例如 $c(\mathbf{x},\mathbf{y})=\|\mathbf{x}-\mathbf{y}\|_2^2$）。

会出现两个主要问题：

1. **非唯一性与不稳定性：** 最小化元 $\gamma^*$ 不必唯一。例如若两种传输方案达到相同最小成本，求解器可能任选其一。因此输入的微小变化 $(a,b,C)$（如移动样本、调整权重或轻微扰动成本）可能导致解的突变。
2. **高计算成本：** 该问题是具有 $n^2$ 个变量和 $2n$ 个约束的线性规划。实用求解器（如匈牙利算法、网络单纯形）通常具有 $\mathcal{O}(n^3)$ 的规模，对大 $n$ 不可行。

为克服这些瓶颈，EOT 目标函数在经典 OT 问题上加入由参数 $\varepsilon > 0$ 控制的正则项：
$$
\text{EOT}_\varepsilon(p_{\mathrm{src}}, p_{\mathrm{tgt}}) := \min_{\gamma \in \Gamma(p_{\mathrm{src}}, p_{\mathrm{tgt}})} \int c(\mathbf{x},  \mathbf{y})  \mathrm{d}\gamma(\mathbf{x}, \mathbf{y}) 
+ \varepsilon \mathcal{D}_{\text{KL}}\left(\gamma \Vert M \right).
$$
参考测度 $M$ 通常取为边缘的乘积 $p_{\mathrm{src}} \otimes p_{\mathrm{tgt}}$。KL 散度项与传输方案 $\gamma$ 的 Shannon 熵直接相关：
$$
\mathcal{D}_{\text{KL}}(\gamma \,\Vert\, p_{\mathrm{src}} \otimes p_{\mathrm{tgt}}) = -\mathcal{H}(\gamma) + \text{常数},
$$
其中 $\mathcal{H}(\gamma) := -\int \gamma(\mathbf{x}, \mathbf{y}) \log \gamma(\mathbf{x}, \mathbf{y}) \mathrm{d} \mathbf{x} \mathrm{d} \mathbf{y}$。

**为何熵正则项有帮助？**

加入该正则项带来若干理论与实用上的好处，简述如下：

1. **质量铺开.** 由于 $t\mapsto t\log t$ 凸且在 $t$ 大时增长迅速，最小化 $\int \gamma\log\gamma$ 会惩罚**尖峰**耦合（部分 $\gamma(\mathbf{x},\mathbf{y})$ 很大、多数接近零）。在总质量 $\int \gamma=1$ 固定时，它倾向于 $\gamma(\mathbf{x},\mathbf{y})$ 在 $(\mathbf{x},\mathbf{y})\in\mathbb{R}^D\times\mathbb{R}^D$ 上更均匀分布的计划。等价地，最大化 Shannon 熵促进更高的“不确定性”（扩散度）。

2. **严格凸性与唯一性.** 因为 $\mathcal{H}$ 严格凹，式 (eq:eot-epsilon) 中的目标对 $\gamma$ 严格凸，从而得到**唯一**最小化元 $\gamma^*_\varepsilon$，且连续依赖于 $(p_{\mathrm{src}},p_{\mathrm{tgt}},c)$。

3. **Sinkhorn 形式与正性.** 在温和条件下，最优解具有 **Schrödinger/Sinkhorn 形式**
   $$
   \gamma^*_\varepsilon(\mathbf{x},\mathbf{y}) = u(\mathbf{x})\, \exp \bigl(-\tfrac{c(\mathbf{x},\mathbf{y})}{\varepsilon}\bigr)  \,v(\mathbf{y})p_{\mathrm{src}}(\mathbf{x}) p_{\mathrm{tgt}}(\mathbf{y}),
   $$
   其中 $u,v$ 为正的缩放函数（在整体因子意义下唯一）。实践中，连续表述用有限样本近似，将 EOT 化为有限（采样）Sinkhorn 迭代。熵目标是严格凸的，缩放（Sinkhorn/IPFP）算法可高效求解。对每边缘 $n$ 个支撑点的稠密问题（$n\times n$ 核），每次 Sinkhorn 迭代的时间与内存均为 $\mathcal{O}(n^2)$，使方法更具可扩展性与实用性。

4. **$\varepsilon$ 的极限.** 当 $\varepsilon \to 0$ 时，最优方案 $\gamma^*_\varepsilon$ 越来越集中，趋于（可能奇异的）经典 OT 耦合（我们将在 4.3.2 节重访这一联系）。当 $\varepsilon$ 增大时，$\gamma^*_\varepsilon$ 逐渐铺开并趋于独立耦合 $p_{\mathrm{src}}\otimes p_{\mathrm{tgt}}$。

### 4.2.3 Schrödinger 桥（SB）

**SB 的 KL 表述.**

Schrödinger 桥（SB）问题由 Erwin Schrödinger 在 1930 年代提出，问的是如下问题。假设粒子按某种简单参考动力学（如布朗运动）运动。设想我们在两个时刻观测粒子：$t=0$ 时其分布为 $p_{\mathrm{src}}$，$t=1$ 时为 $p_{\mathrm{tgt}}$。在连接这两个分布的所有可能随机过程中，哪一个与参考动力学偏离最小？这里“偏离”用 KL 散度度量，因此 SB 问题的解是将布朗运动变形为满足给定边界条件的、最可能的方式。

为严格表述，记 $\mathbf{x}_{0:T}:=\{\mathbf{x}_t\}_{t\in[0,T]}$ 为过程的完整轨迹。用 $P$ 表示**轨迹律**，即整条样本路径上的概率分布。$P$ 在时刻 $t$ 的边缘记为 $p_t$（或 $P_t$），描述单时刻状态 $\mathbf{x}_t$ 的分布。形式地，对可测集 $A\subseteq\mathbb R^D$，
$$
p_t(A) = P(\mathbf{x}_t \in A).
$$
换言之，$p_t$ 可视为从 $P$ 抽取多条完整轨迹、再收集时刻 $t$ 的状态得到的经验分布——例如若状态为一维则可视为直方图。

考虑由 SDE 驱动的参考扩散 $\{\mathbf{x}_t\}_{t\in[0,T]}$：
$$
\mathrm{d} \mathbf{x}_t  =  \mathbf{f}(\mathbf{x}_t,t)\,\mathrm{d} t  +  g(t)\,\mathrm{d} \mathbf{w}_t,
$$
其中 $\mathbf{f}\colon \mathbb{R}^D\times[0,T]\!\to\!\mathbb{R}^D$，$g\colon[0,T]\!\to\!\mathbb{R}$，$\{\mathbf{w}_t\}_{t\in[0,T]}$ 为标准布朗运动。记 $R$ 为完整轨迹 $\mathbf{x}_{0:T}:=\{\mathbf{x}_t\}_{t\in[0,T]}$ 的**路径律**（联合分布）；该 $R$ 将作为**参考**轨迹分布。

在此记号下，Schrödinger 桥（SB）问题寻求在 KL 散度意义下最接近 $R$、同时匹配给定端点边缘的轨迹律 $P$：
$$
\mathrm{SB}(p_{\mathrm{src}}, p_{\mathrm{tgt}})
:= \min_{P}\, \mathcal{D}_{\mathrm{KL}}(P\|R)
\quad \text{s.t.} \quad P_0 = p_{\mathrm{src}},\;\; P_T = p_{\mathrm{tgt}}.
$$
最优解 $P^*$ 依赖于所选的参考过程 $R$。

**SB 的随机控制视角.**

![SB 随机控制视角示意图](../arXiv-2510.21890v1/Images/PartB/sb_illustration.pdf)

**图 4.2：SB 随机控制视角示意图。** 桥寻求在连接 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 的同时与参考偏离最小的随机路径。

与在式 (eq:sb-epsilon-general) 中对任意路径分布 $P$ 优化不同，一种更易处理的方式是以参考动力学为锚并允许其漂移。这通过引入随时间变化的漂移 $\mathbf{v}_t(\mathbf{x}_t)$ 实现，该漂移扰动参考过程并生成一族候选轨迹分布。得到的动力学呈**受控扩散**形式：
$$
\mathrm{d} \mathbf{x}_t
= \bigl[\mathbf{f}(\mathbf{x}_t,t) + \mathbf{v}_t(\mathbf{x}_t)\bigr]\mathrm{d} t + g(t) \mathrm{d} \mathbf{w}_t,
$$
其中 $\mathbf{v}_t:\mathbb{R}^D\to\mathbb{R}^D$ 为待优化的漂移（见式 (eq:sb-kinetic)）。在标准可积性条件（如 Novikov 条件）下，由 Girsanov 定理，受控律 $P$ 与参考 $R$ 之间的 KL 散度具有动态（动能）形式
$$
\mathcal{D}_{\mathrm{KL}}(P\|R)
= \mathbb{E}_{P}\!\left[\frac{1}{2}\int_0^T \frac{\|\mathbf{v}_t(\mathbf{x}_t)\|^2}{g^2(t)} \mathrm{d} t\right]
= \frac{1}{2}\int_0^T\!\int_{\mathbb{R}^D}\frac{\|\mathbf{v}_t(\mathbf{x})\|^2}{g^2(t)} p_t(\mathbf{x}) \mathrm{d} \mathbf{x} \mathrm{d} t,
$$
其中 $p_t$ 为受控过程下 $\mathbf{x}_t$ 的时刻-$t$ 边缘。第二个等号由全期望律得到。

因此，SB 问题可重新表述为：在所有将过程在 $t=0$ 从 $p_{\mathrm{src}}$、在 $t=T$ 引导到 $p_{\mathrm{tgt}}$ 的可行漂移 $\mathbf{v}_t$ 上，最小化期望控制能量。这得到**随机控制表述**：
$$
\mathrm{SB}_\varepsilon(p_{\mathrm{src}}, p_{\mathrm{tgt}})
= \min_{\substack{ \mathbf{v}_t \text{ s.t. } \mathrm{d} \mathbf{x}_t = \left[\mathbf{f}(\mathbf{x}_t, t) + \mathbf{v}_t(\mathbf{x}_t) \right]\mathrm{d} t + g(t) \mathrm{d} \mathbf{w}_t, \\ \mathbf{x}_0 \sim p_{\mathrm{src}}, \,\, \mathbf{x}_T \sim p_{\mathrm{tgt}} }} \frac{1}{2} \int_0^T\int_{\mathbb{R}^D} \frac{\|\mathbf{v}_t(\mathbf{x})\|^2}{g^2(t)} p_t(\mathbf{x})\mathrm{d} \mathbf{x} \mathrm{d} t.
$$

重要的是，端点分布 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 是任意的；控制 $\mathbf{v}_t$ 的选取正是为了在这些边缘之间“桥接”参考动力学，同时尽可能（在 KL 散度意义下）接近参考过程 $R$。

**特例：布朗参考.**

式 (eq:sb-kinetic) 与式 (eq:bb-ot) 中 OT 的 Benamou–Brenier 形式相似，尤其当参考过程 $R^\varepsilon$（$\varepsilon>0$）取为布朗运动时：
$$
\mathrm{d} \mathbf{x}_t = \sqrt{\varepsilon} \mathrm{d}\mathbf{w}_t,
$$
即 $\mathbf{f} \equiv \mathbf{0}$，$g(t) \equiv \sqrt{\varepsilon}$。

在此设定下，SB 问题寻求在（KL 散度意义下）最接近布朗参考 $R^\varepsilon$、同时匹配端点边缘的路径分布 $P$：
$$
\mathrm{SB}_\varepsilon(p_{\mathrm{src}}, p_{\mathrm{tgt}})
:= \min_{P} \mathcal{D}_{\mathrm{KL}}(P \| R^\varepsilon)
\quad \text{s.t.} \quad P_0 = p_{\mathrm{src}},   P_T = p_{\mathrm{tgt}}.
$$
等价的随机控制表述则为
$$
\mathrm{SB}_\varepsilon(p_{\mathrm{src}}, p_{\mathrm{tgt}})
= \min_{\substack{
    \mathbf{v}_t \text{ s.t. } \mathrm{d} \mathbf{x}_t = \sqrt{\varepsilon}\,\mathrm{d} \mathbf{w}_t, \\
    \mathbf{x}_0 \sim p_{\mathrm{src}},\;\; \mathbf{x}_T \sim p_{\mathrm{tgt}}
}}
\frac{1}{2\varepsilon}\int_0^T \!\!\int_{\mathbb{R}^D}
    \|\mathbf{v}_t(\mathbf{x})\|^2 \, p_t(\mathbf{x})\,\mathrm{d} \mathbf{x}\,\mathrm{d} t.
$$

**为何需要指定参考分布？**

与经典 OT 不同，SB 问题因其随机性需要参考分布。在 OT 中，成本函数（如 $c(\mathbf{x}, \mathbf{y}) \propto \|\mathbf{x} - \mathbf{y}\|^2$）隐式定义了唯一的确定性测地路径，故不需要参考。相比之下，SB 设定允许无穷多种连接边缘的随机过程，没有内在的“自然”路径概念。参考测度 $R$ 编码系统的底层物理或几何结构（如布朗运动），并定义基于 KL 的优化目标 $\mathcal{D}_{\mathrm{KL}}(P \| R)$，没有它最优性概念就无法定义。

**耦合 PDE 刻画.**

描述 SB 解的一种方便方式是通过两个时空势 $\Psi(x,t)$ 和 $\widehat{\Psi}(x,t)$。记 $p_t^{\mathrm{SB}}$ 为式 (eq:sb-epsilon-general) 中最优轨迹律 $P^*$ 在时刻 $t\in[0,T]$ 的边缘。则有对称分解
$$
p_t^{\mathrm{SB}}(x) = \Psi(x,t) \widehat{\Psi}(x,t),
$$
其中 $\Psi$ 与 $\widehat{\Psi}$ 满足（线性）**Schrödinger 方程组**：
$$
\frac{\partial \Psi}{\partial t}(\mathbf{x},t)
  = -\nabla_{\mathbf{x}} \Psi(\mathbf{x},t) \cdot \mathbf{f}(\mathbf{x},t)
     - \frac{g^2(t)}{2}\,\Delta_\mathbf{x} \Psi(\mathbf{x},t),  
$$
$$
\frac{\partial \widehat{\Psi}}{\partial t}(\mathbf{x},t)
  = -\nabla_{\mathbf{x}}\!\cdot\!\bigl(\widehat{\Psi}(\mathbf{x},t)\,\mathbf{f}(\mathbf{x},t)\bigr)
     + \frac{g^2(t)}{2}\,\Delta_\mathbf{x}\widehat{\Psi}(\mathbf{x},t) 
$$
在边界条件
$$
\Psi(\mathbf{x},0)\,\widehat{\Psi}(\mathbf{x},0) = p_{\mathrm{src}}(\mathbf{x}),
\quad
\Psi(\mathbf{x},T)\,\widehat{\Psi}(\mathbf{x},T) = p_{\mathrm{tgt}}(\mathbf{x}). 
$$

**前向时间 Schrödinger 桥 SDE.**

一旦已知 $\Psi$，最优动力学是由时空因子 $\Psi$ 倾斜的参考扩散：
$$
\mathrm{d} \mathbf{x}_t = \bigl[\mathbf{f}(\mathbf{x}_t, t) + g^2(t)\,\nabla_{\mathbf{x}} \log \Psi(\mathbf{x}_t, t)\bigr]\mathrm{d} t
+ g(t)\mathrm{d} \mathbf{w}_t, \quad \mathbf{x}_0 \sim p_{\mathrm{src}}.
$$
记 $Q$ 为式 (eq:sb-forward-sde) 的轨迹律（故由式 (eq:sb-symmetry)、(eq:pde-schrodinger) 有 $Q_0=p_{\mathrm{src}}$，$Q_T=p_{\mathrm{tgt}}$）。则 **$Q=P^*$**，且式 (eq:sb-kinetic) 的最小化元 $\mathbf{v}^*$ 为
$$
\mathbf{v}_t^{*}(\mathbf{x}) = g^2(t) \nabla_\mathbf{x} \log \Psi(\mathbf{x},t).
$$
即漂移修正 $g^2\nabla_\mathbf{x}\log\Psi$ 正是为匹配端点边缘所需的对参考的最小 KL 扰动。

**逆向时间 Schrödinger 桥 SDE.**

同一最优路径律也可逆向时间生成。得到逆向时间漂移形式的一种简便方式是概念上使用扩散的标准时间反转恒等式：
$$
\mathbf{b}^{-}(\mathbf{x},t)=\mathbf{b}^{+}(\mathbf{x},t)-g^2(t)\nabla_\mathbf{x}\log p_t^{\mathrm{SB}}(\mathbf{x}),
$$
其中 $\mathbf{b}^{+}=\mathbf{f}+g^2\nabla\log\Psi$，$p_t=\Psi\,\widehat{\Psi}$。于是
$$
\mathbf{b}^{-}(\mathbf{x},t)=\mathbf{f}(\mathbf{x},t)-g^2(t) \nabla_\mathbf{x}\log \widehat{\Psi}(\mathbf{x},t).
$$
因此逆向时间 SDE 为
$$
\mathrm{d} \mathbf{x}_t
= \Big[\mathbf{f}(\mathbf{x}_t,t) - g^2(t) \nabla_{\mathbf{x}}\log \widehat{\Psi}(\mathbf{x}_t,t)\Big]\mathrm{d} t
+ g(t)\,\mathrm{d} \bar{\rvw}_t, \quad \mathbf{x}_T \sim p_{\mathrm{tgt}} .
$$
等价地，用 $\mathbf{y}_\tau := \mathbf{x}_{T-\tau}$ 重参数化时间，使 $\tau$ 从 $0$ 增至 $T$。则 $\mathbf{y}_\tau$ 在 $\tau$ 上从 $\mathbf{y}_0 \sim p_{\mathrm{tgt}}$ 前向演化：
$$
\mathrm{d} \mathbf{y}_\tau
= \big[-\mathbf{f}(\mathbf{y}_\tau,T-\tau) + g^2(T-\tau) \nabla_{\mathbf{y}}\log \widehat{\Psi}(\mathbf{y}_\tau,T-\tau)\big]\mathrm{d} \tau
+ g(T-\tau) \mathrm{d} \mathbf{w}_\tau.
$$
在式 (eq:sb-kinetic) 的逆向时间随机控制表述中（相同二次能量、反转时钟），最优控制为
$$
\mathbf{u}_t^{*}(\mathbf{x})=-g^2(t)\nabla_{\mathbf{x}}\log\widehat\Psi(\mathbf{x},t). 
$$

前向与逆向描述都给出同一最优路径律 $P^*$，它们由
$$
\nabla\log p_t^{\mathrm{SB}}=\nabla\log\Psi+\nabla\log\widehat\Psi,
\qquad
\mathbf{b}^{-}=\mathbf{b}^{+}-g^2\,\nabla\log p_t^{\mathrm{SB}},
$$
联系，故其边缘在每一时刻都与 $p_t^{\mathrm{SB}}$ 一致。附加漂移项 $g^2\nabla\log\Psi$（前向）与 $-\,g^2\nabla\log\widehat\Psi$（逆向）作为控制力，在相对熵意义下使参考扩散尽可能接近参考的同时引导其匹配端点边缘。

**耦合 PDE 方法的实际障碍.**

要基于式 (eq:sb-backward-sde) 构造生成过程，必须求解式 (eq:pde-schrodinger) 中的耦合 PDE 以得到逆向 Schrödinger 势 $\widehat{\Psi}$。然而这些 PDE 即便在低维情形下也以难以求解而著称，使其在生成建模中的直接应用具有挑战性。为绕过这一点，若干工作提出了替代策略：利用 Score SDE 技术迭代求解每个半桥问题（$p_{\mathrm{tgt}} \leftarrow p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}} \rightarrow p_{\mathrm{src}}$）；优化代理似然界；或基于样本对 $(\mathbf{x}_0, \mathbf{x}_T) \sim p_{\mathrm{src}} \otimes p_{\mathrm{tgt}}$ 的后验 $\mathbf{x}_t \vert \mathbf{x}_0, \mathbf{x}_T$ 的解析解设计无仿真训练。此处我们不深入技术细节，仅在 4.4 节简要讨论扩散模型与 SB 的联系。

### 4.2.4 全局前推与局部动力学：深度生成模型的 OT 类比

从最优传输视角（式 (eq:ot)），可以利用深度生成模型学习从简单先验到数据的传输（前推）映射，即 $\mathbf{G}_{\bm{\phi}\#}p_{\mathrm{prior}} \approx p_{\mathrm{data}}$。尽管 $\mathbf{G}_{\bm{\phi}}$ 一般不与最优传输映射重合（除非在适当条件下施加 OT 目标的工作），Benamou–Brenier 形式（式 (eq:bb-ot)）提供了互补的动态视角。它不直接学习单一的全局映射，而是将传输描述为由随时间变化的局部向量场生成的连续流，在 $p_{\mathrm{prior}}$ 与 $p_{\mathrm{data}}$ 之间描出一条光滑路径。该动态表述与静态 Schrödinger 桥问题（式 (eq:sb-epsilon-general)）与其随机控制对应（式 (eq:sb-kinetic)）之间的关系平行，其中最优耦合实现为受控扩散过程。在生成建模中也有类似类比：标准 DGM（如 GAN 或 VAE）学习全局前推映射，而扩散模型学习驱动生成动力学的随时间变化的局部向量场。

---

## 4.3 各类最优传输表述之间的关系

**图 4.3：** 在 $c(\mathbf{x},\mathbf{y})=\|\mathbf{x}-\mathbf{y}\|_2^2$ 且 SB 中参考为 $R^\varepsilon$ 时，各类最优传输变体之间的关系。我们总结等价关系：(i) $\mathrm{SB}_\varepsilon$（随机控制）$\Leftrightarrow$ $\mathrm{SB}_\varepsilon$（静态表述），其中 $p_t$ 为路径测度 $P$ 的时刻-$t$ 边缘；(ii) $\mathrm{SB}_\varepsilon$（静态表述）$\Leftrightarrow$ $\mathrm{EOT}_\varepsilon$（见 4.3.1 节）；(iii) $\mathrm{EOT}_\varepsilon$（静态表述）$\Leftrightarrow$ $\mathrm{OT}_\varepsilon$（静态）（见 4.3.2 节）；(iv) $\mathrm{SB}_\varepsilon$（随机控制）$\Leftrightarrow$ $\mathrm{OT}$（动态）（见 4.3.3 节）。

*（原图为 TikZ 框图，此处仅翻译标题与说明；图示结构见图 4.3。）*

在深入技术细节之前，先澄清最优传输及其熵正则化的不同表述如何联系是有益的。高层来看，这些问题可视为相关的（其联系图见图 4.3）：

(i) 以布朗运动为给定参考 $R^\varepsilon$ 的 SB 问题 $\mathrm{SB}_\varepsilon$
$$
\mathrm{d} \mathbf{x}_t = \sqrt{\varepsilon}\,\mathrm{d} \mathbf{w}_t
$$
与其静态表述等价：演化的边缘 $p_t$ 正是最优路径测度 $P$ 的时刻-$t$ 切片。

(ii) $\mathrm{SB}_\varepsilon$ 的静态表述直接连接到熵正则 OT 问题 $\mathrm{EOT}_\varepsilon$。

(iii) $\mathrm{EOT}_\varepsilon$ 又可与熵正则 OT 的静态表述 $\mathrm{OT}_\varepsilon$ 联系起来。

(iv) $\mathrm{SB}_\varepsilon$ 的随机控制视角也可与经典 OT 的动态表述联系起来。

这些非平凡关系共同提供了跨越随机控制、熵正则与经典 OT 框架的紧凑视图。

### 4.3.1 SB 与 EOT（对偶）等价

本节从两个互补视角说明 SB 与 EOT 本质等价。与产生单一确定性映射的经典最优传输不同，SB 给出粒子的**随机**流：质量以概率方式输运，边缘在类扩散动力学下演化。

从静态视角，SB 与 EOT 一致，目标是找到两个端点分布之间的耦合，在传输成本与熵之间取得平衡。从动态视角，SB 描述一个受控扩散过程，在仍匹配期望端点的同时尽可能接近简单参考（如布朗运动）。两种视角各自建立等价性，提供了理解 SB/EOT 作为分布到分布变换典范表述的两种一致方式。

**静态 Schrödinger 桥.**

令
$$
  \widetilde R^\varepsilon(\mathbf{x},\mathbf{y})
  := \frac{1}{Z_\varepsilon}\,e^{-c(\mathbf{x},\mathbf{y})/\varepsilon}\,p_{\mathrm{src}}(\mathbf{x})\,p_{\mathrm{tgt}}(\mathbf{y}),
$$
归一化常数为
$$
Z_\varepsilon := \iint e^{-c(\mathbf{x}, \mathbf{y})/\varepsilon} p_{\mathrm{src}}(\mathbf{x}) p_{\mathrm{tgt}}(\mathbf{y})\mathrm{d}\mathbf{x}\mathrm{d}\mathbf{y}.
$$
则熵正则 OT 目标
$$
\min_{\gamma\in\Gamma(p_{\mathrm{src}},p_{\mathrm{tgt}})}
\Big\{\int c \mathrm{d}\gamma
+ \varepsilon \mathcal{D}_\mathrm{KL}\!\big(\gamma \Vert p_{\mathrm{src}}\!\otimes\!p_{\mathrm{tgt}}\big)\Big\}
= \varepsilon \min_{\gamma\in\Gamma(p_{\mathrm{src}},p_{\mathrm{tgt}})}
\mathcal{D}_\mathrm{KL}\!\big(\gamma \Vert \widetilde R^\varepsilon\big)
-\varepsilon\log Z_\varepsilon,
$$
因此（相差常数）与静态 Schrödinger 桥（式 (eq:sb-epsilon)）等价：
$$
\min_{\gamma\in\Gamma}\mathcal{D}_\mathrm{KL}(\gamma\Vert \widetilde R^\varepsilon).
$$

**动态等价（布朗参考）.** 也可从动态等价看待：经典结果表明，二次成本
$$
c(\mathbf{x},\mathbf{y})=\frac{\|\mathbf{y}-\mathbf{x}\|^2}{2T}
$$
下的熵正则 OT 与参考路径律 $R^\varepsilon$ 为 $[0,T]$ 上布朗运动的 SB 问题仿射等价，
$$
\mathrm{d}\mathbf{x}_t=\sqrt{\varepsilon}\,\mathrm{d}\mathbf{w}_t.
$$
这里“仿射等价”指最优值相差一个正常数缩放和一个与决策变量无关的加性常数，故最小化元一致。特别地，设 $P^*$ 为 SB 的最优路径分布，$\gamma^*$ 为 EOT 的最优传输方案。则若 $\mathbf{x}_{[0:T]}\sim P^*$，端点对 $(\mathbf{x}_0,\mathbf{x}_T)$ 的分布为 $\gamma^*$：
$$
P^* \text{ 解 SB } \iff \gamma^* \text{ 解 EOT 且 } (\mathbf{x}_0,\mathbf{x}_T)\sim\gamma^*.
$$

换言之：动态（SB）问题的最优过程诱导静态（EOT）问题的最优耦合。反之，（在热核的温和条件下）任何最优静态耦合可实现为某个最优 SB 过程的端点。

推导这一事实的关键是：路径上的 KL 散度可按端点分解，这意味着 Schrödinger 桥问题化为仅关于 $(\mathbf{x}_0,\mathbf{x}_T)$ 联合分布的 KL 散度。对布朗运动，$\mathbf{x}$ 到 $\mathbf{y}$ 的转移密度呈高斯形式，故其负对数为二次：
$$
-\varepsilon\log p_T(\mathbf{y}\mid\mathbf{x})=\frac{\|\mathbf{y}-\mathbf{x}\|^2}{2T}+\text{常数}.
$$
这说明端点 KL 与二次成本下的熵正则 OT 目标一致，至多无关常数。

**一般参考下的 SB 决定 EOT 成本.**

如式 (eq:sb-epsilon-general) 所述，SB 问题不限于布朗运动；可用任意（适定）参考过程定义。这一选择唯一决定对应 EOT 问题中的成本函数。关键联系是：SB 的**参考动力学**诱导 EOT 的**成本函数**。

设参考过程由 $[0,T]$ 上的 SDE 驱动，得到转移密度 $p_T(\mathbf{y}|\mathbf{x})$（从时刻 $0$ 的 $\mathbf{x}$ 在时刻 $T$ 到达 $\mathbf{y}$ 的似然）。则 EOT 成本函数（至多常数缩放）由
$$
c(\mathbf{x}, \mathbf{y}) \propto -\log p_T(\mathbf{y}|\mathbf{x}).
$$
给出。在此成本下，求解 SB 问题等价于求解 EOT 问题。简言之，在 SB 中选择参考动力学在数学上等价于在 EOT 中指定传输成本。由式 (eq:eot-sb-static)，熵正则 OT 目标与静态 SB 目标相差；故两问题等价且具有相同最小化元。

### 4.3.2 当 $\varepsilon\to 0$ 时 EOT$_\varepsilon$ 归约为 OT

记 $\gamma^*_\varepsilon$ 为 EOT$_\varepsilon$ 的最优方案，$\gamma^*$ 为式 (eq:ot) 中无正则 OT 问题的最优方案。下列结果表明，当 $\varepsilon \to 0$ 时，熵正则最优方案 $\gamma^*_\varepsilon$ 在适当意义下收敛到 OT 方案 $\gamma^*$，且 EOT 成本收敛到 OT 成本。

这一收敛结果既基本又具有实际重要性。原因之一是熵正则 OT 问题 $\mathrm{EOT}_\varepsilon$ 可通过 Sinkhorn 等算法高效数值求解。因此该结果为使用小 $\varepsilon$ 的 $\mathrm{EOT}_\varepsilon$ 作为式 (eq:ot) 中经典 OT 问题的计算可行代理提供了理论依据，即便成本函数 $c(\mathbf{x}, \mathbf{y})$ 比二次情形更一般。

**定理（非正式）EOT$_\varepsilon$ 收敛到 OT.** 当 $\varepsilon \to 0$ 时，最优值收敛：
$$
\lim_{\varepsilon \to 0}  \mathrm{EOT}_\varepsilon(p_{\mathrm{src}}, p_{\mathrm{tgt}}) = \mathrm{OT}(p_{\mathrm{src}}, p_{\mathrm{tgt}}). 
$$
此外，最优方案 $\gamma^*_\varepsilon$ **弱收敛**到 $\gamma^*$。即对任意有界连续（检验）函数 $g : \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}$，
$$
\mathbb{E}_{(\mathbf{x}, \mathbf{y})\sim\gamma^*_\varepsilon}[g(\mathbf{x}, \mathbf{y})] \to \mathbb{E}_{(\mathbf{x}, \mathbf{y})\sim\gamma^*}[g(\mathbf{x}, \mathbf{y})].
$$

### 4.3.3 当 $\varepsilon\to 0$ 时 SB$_\varepsilon$ 归约为 OT

对每个 $\varepsilon > 0$，设 $\mathbf{v}_t^\varepsilon$ 为式 (eq:sb-epsilon-kinetic) 中 SB 问题的极小元，$p_t^\varepsilon$ 为由 $\mathbf{v}_t^\varepsilon$ 诱导的受控 SDE $\mathbf{x}_t$ 的边缘分布。则 $p_t^\varepsilon$ 满足相应的 Fokker–Planck 方程。与之对照，记 $(p_t^0, \mathbf{v}_t^0)$ 为最优传输的 Benamou–Brenier 表述（见式 (eq:bb-ot)）的极小元。

下列定理表明，当 $\varepsilon \to 0$ 时，SB 问题收敛到 OT 问题。该结果的实际重要性原因与定理 EOT-OT 类似。目标 $\mathrm{SB}_\varepsilon$ 可用 Sinkhorn 型算法高效求解，得到最优传输的数值可处理且可微的代理，在高维或大规模设定下尤其有价值，此时基于 Benamou–Brenier 形式的直接求解器计算昂贵。

**定理（非正式）SB$_\varepsilon$ 收敛到 OT.** 当 $\varepsilon \to 0$ 时：
$$
\lim_{\varepsilon \to 0} \text{SB}_\varepsilon(p_{\mathrm{src}}, p_{\mathrm{tgt}}) = \text{OT}(p_{\mathrm{src}}, p_{\mathrm{tgt}}),
$$
其中 OT 取式 (eq:bb-ot) 的 Benamou–Brenier 形式。此外，$p_t^\varepsilon$ 弱收敛到 $p_t^0$，$\mathbf{v}_t^\varepsilon$ 在适当函数空间中弱收敛到 $\mathbf{v}_t^0$。

至此我们已在各自假设下给出了 EOT 与 SB 之间的基本等价，以及它们通过极限过程与 OT 的重要联系（见图 4.3）。接下来我们探讨扩散模型如何与这些概念相联系。

---

## 4.4 扩散模型的 SDE 是 SB 问题的最优解吗？

### 4.4.1 扩散模型作为 Schrödinger 桥的特例

SB 框架通过允许在任意源与目标分布之间进行非线性插值，扩展了（基于分数的）扩散模型。这是通过添加由标量势 $\Psi(\mathbf{x}, t)$ 和 $\widehat{\Psi}(\mathbf{x}, t)$ 导出的控制漂移项实现的，这些项引导参考扩散过程以匹配给定的端点边缘（见式 (eq:pde-schrodinger)），并满足分解：
$$
\nabla\log{\Psi(x,t)}+\nabla\log{\hat{\Psi}(x,t)}=\nabla \log p_t^{\mathrm{SB}}(\mathbf{x}).
$$
这一推广使模型能够超越标准高斯先验，从更广泛的分布中生成样本。

**与扩散模型的联系.**

扩散模型作为 SB 框架的特例出现。设势为常数 $\Psi(\mathbf{x}, t) \equiv 1$。在此假设下，式 (eq:pde-schrodinger) 中第二个 PDE 化为标准 Fokker–Planck 方程，其解为参考过程的边缘密度：
$$
\widehat{\Psi}(\mathbf{x}, t) = p_t^{\mathrm{SB}}(\mathbf{x}).
$$
相应的 SB 前向 SDE 因而变为无控参考过程：
$$
\mathrm{d}\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)\mathrm{d} t + g(t)\mathrm{d}\mathbf{w}_t,
$$
SB 逆向 SDE 简化为：
$$
\mathrm{d}\mathbf{x}_t = \left[\mathbf{f}(\mathbf{x}_t, t) - g^2(t) \nabla \log p_t^{\mathrm{SB}}(\mathbf{x}_t) \right]\mathrm{d} t + g(t)\mathrm{d}\bar{\mathbf{w}}_t,
$$
与扩散模型中使用的 Anderson 逆向时间 SDE 一致。这一对应表明扩散模型可解释为 SB 的零控制极限，即势不引入附加漂移。

**边界条件与一般性.**

上述约化在边界约束相容时才是形式的。对任意源/目标 $(p_{\mathrm{src}},p_{\mathrm{tgt}})$，式 (eq:pde-schrodinger) 中的 PDE 边界条件一般无法由 $\Psi\equiv 1$ 满足。完整 SB 通过学习非平凡势来解决这一点，这些势诱导非线性控制漂移，弯曲参考动力学以匹配任意给定端点。相比之下，扩散模型将一个端点固定为简单先验（通常为高斯），仅学习逆向时间分数以到达数据。由此视角，SB 是更灵活的总括框架：非平凡势时桥接任意端点；$\Psi\equiv 1$ 时则退化为上述扩散模型情形。我们另指出，在标准线性扩散模型中，$p_T \approx p_{\mathrm{prior}}$ 仅在 $T \to \infty$ 时成立，故与先验的匹配仅是近似的。

### 4.4.2 扩散模型作为 Schrödinger 半桥

本节说明扩散模型不是完整 Schrödinger 桥，但可通过**Schrödinger 半桥**的松弛概念理解。半桥只施加一个端点约束（$p_{\mathrm{prior}}$ 或 $p_{\mathrm{data}}$），而不是两个，因此是完整桥的单侧变体。在形式化这一联系之前，我们在式 (eq:sb-epsilon-general) 的一般表述基础上介绍 Schrödinger 半桥的定义，其中 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 任意。然后回到扩散模型，说明当端点由 $p_{\mathrm{prior}}$ 与 $p_{\mathrm{data}}$ 给定时，半桥视角如何自然适用。

**Schrödinger 半桥.**

SB 问题寻求一个随机过程，其律在 KL 散度意义下最接近简单参考过程，同时**匹配两个端点分布** $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$。求解完整桥需要同时施加两个边界条件，这通常计算困难。一种有用的松弛是**半桥**问题：不匹配两个端点，只匹配其中一个。

形式地，设 $R$ 为参考路径分布。**前向半桥**寻求最小化
$$
\min_{P: P_0 = p_{\mathrm{src}}}
\mathcal{D}_{\mathrm{KL}}(P \,\|\, R),
$$
的路径分布 $P$，约束仅为 $P_0 = p_{\mathrm{src}}$。类似地，**逆向半桥**只约束终端分布：
$$
\min_{P: P_T = p_{\mathrm{tgt}}}
\mathcal{D}_{\mathrm{KL}}(P \,\|\, R).
$$
换言之，前向半桥问：在所有从期望初始分布出发的过程中，哪一个最像参考？逆向半桥对在期望终端分布结束的过程问同样的问题。通过迭代地结合这两种松弛，可以近似完整 SB。

**扩散模型无法精确匹配端点.**

扩散模型与 SB 框架的一个关键区别在于终端分布 $p_T$ 的处理。在标准扩散模型中，前向 SDE 通常对 $\mathbf{x}_t$ 线性（见式 (eq:forward-linear-sde)），且设计为使 $p_T$ 仅在 $T \to \infty$ 时近似先验：
$$
p_T \approx p_{\mathrm{prior}}.
$$
然而在有限时间，$p_T$ 是参数依赖于 $p_{\text{data}}$ 的高斯。因此若不仔细调参，它一般不会匹配期望的先验。

相比之下，SB 框架通过引入形如 $g^2(t)\nabla_\mathbf{x}\log \Psi(\mathbf{x}, t)$ 的附加控制漂移，在有限时间 $T$ 强制精确边缘匹配。这确保终端分布精确满足 $p_T = p_{\mathrm{prior}}$，无论初始数据分布 $p_0 = p_{\mathrm{data}}$ 如何。总结：

- **扩散模型：** $p_T \approx p_{\mathrm{prior}}$，渐近地当 $T \to \infty$。
- **Schrödinger 桥：** 在有限 $T$ 处精确有 $p_T = p_{\mathrm{prior}}$，通过求解控制势 $\Psi$ 与 $\widehat{\Psi}$ 实现。

**扩散 Schrödinger 桥.**

标准扩散模型不强制 $P_T = p_{\mathrm{prior}}$，因此只求解从 $p_{\mathrm{data}}$ 到 $p_{\mathrm{prior}}$ 的 Schrödinger **半桥**。

为解决这一点，扩散 Schrödinger 桥（DSB）按照迭代比例拟合（IPF）算法（一种交替投影方法）的思想，在匹配两个端点边缘之间交替。这将扩散模型推广为如下求解完整 SB：

- **步骤 0：参考过程。** 用 $P^{(0)} := R_{\text{fwd}}$ 初始化，即参考前向 SDE：
  $$
  \mathrm{d} \mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)   \mathrm{d} t + g(t)   \mathrm{d} \mathbf{w}_t, \quad \mathbf{x}_0 \sim p_{\text{data}}.
  $$
  这保证 $P_0^{(0)} = p_{\text{data}}$，但通常 $P_T^{(0)} \neq p_{\text{prior}}$。

- **步骤 1：逆向过程。** 计算在时刻 $T$ 匹配 $p_{\text{prior}}$、同时尽量接近 $P^{(0)}$ 的过程 $P^{(1)}$：
  $$
  P^{(1)} = \argmin_{P : P_T = p_{\text{prior}}} \mathcal{D}_{\text{KL}}(P  \|  P^{(0)}).
  $$
  这通过用神经网络 $\mathbf{s}_{\bm{\phi}^\times}$ 近似 oracle 分数函数实现，得到逆向时间 SDE：
  $$
  \mathrm{d} \mathbf{x}_t = \left[ \mathbf{f}(\mathbf{x}_t, t) - g^2(t) \mathbf{s}_{\bm{\phi}^\times}(\mathbf{x}_t, t) \right] \mathrm{d} t + g(t)  \mathrm{d} \bar{\mathbf{w}}_t,
  $$
  从 $\mathbf{x}_T \sim p_{\text{prior}}$ 逆向模拟。

- **迭代。** 过程 $P^{(1)}$ 满足 $P_T^{(1)} = p_{\text{prior}}$，但其初始边缘 $P_0^{(1)}$ 通常偏离 $p_{\text{data}}$。IPF 通过学习前向 SDE 将 $P_0^{(1)}$ 调回 $p_{\text{data}}$，再经一次逆向过程强制 $p_{\text{prior}}$ 来应对。这一交替持续进行，直至收敛到最优桥 $P^*$，满足 $P_0^* = p_{\text{data}}$ 且 $P_T^* = p_{\text{prior}}$。在温和条件下可证明收敛。

---

## 4.5 扩散模型的 ODE 是 OT 问题的最优映射吗？

本节我们聚焦二次成本最优传输问题。

### 4.5.1 PF-ODE 流一般不是最优传输

本节给出结果：PF-ODE 的解映射在二次成本下一般不给出最优传输映射。

**设定.** 我们考虑 VP SDE，具体为 Ornstein–Uhlenbeck 过程，将光滑初始密度 $p_0$ 演化向标准高斯 $\mathcal{N}(\bm{0}, \mathbf{I})$：
$$
\mathrm{d} \mathbf{x}(t) = -\mathbf{x}(t) \mathrm{d} t + \sqrt{2} \mathrm{d} \rvw(t), \quad \mathbf{x}(0) \sim p_0.
$$
相应的 PF-ODE 为
$$
\frac{\mathrm{d} \mathbf{S}_t(\mathbf{x})}{\mathrm{d} t} = -\mathbf{S}_t(\mathbf{x}) - \nabla \log p_t(\mathbf{S}_t(\mathbf{x})), \quad \mathbf{S}_0(\mathbf{x}) = \mathbf{x}.
$$
这里 $\mathbf{S}_t$ 表示将 $p_0$ 前推到边缘 $p_t$ 的流映射：
$$
\left(\mathbf{S}_t\right)_\# p_0 = p_t,
\quad\text{即}\quad
p_t(\mathbf{y}) = \int_{\mathbb{R}^D} \delta(\mathbf{y} - \mathbf{S}_t(\mathbf{x}))   p_0(\mathbf{x}) \mathrm{d} \mathbf{x}.
$$
这些密度 $p_t$ 由 Fokker-Planck 方程演化：
$$
\frac{\partial p_t}{\partial t} = \nabla \cdot (\mathbf{x} p_t) + \Delta p_t.
$$
这等价于具有速度场的连续性方程：
$$
\mathbf{v}_t(\mathbf{x}) = -\mathbf{x} - \nabla \log p_t(\mathbf{x}),
$$
其流由 $\mathbf{S}_t(\mathbf{x})$ 给出。换言之，PF-ODE 可写为：
$$
\frac{\mathrm{d} \mathbf{S}_t(\mathbf{x})}{\mathrm{d} t} = \mathbf{v}_t\left(\mathbf{S}_t(\mathbf{x})\right).
$$

当 $t \to \infty$ 时，该映射将初始分布输运到先验：
$$
\mathbf{S}_\infty \# p_0 = \mathcal{N}(\bm{0}, \mathbf{I}) =: p_{\text{prior}}.
$$

**论证目标.** 作者并不直接判断从 $p_0$ 到高斯的终端映射 $\mathbf{S}_\infty$ 是否最优，而是构造一个特定的初始分布 $p_0$ 并考察整条 PF-ODE 轨迹。其关键观察是：最优性可能在流的某处失效。

他们考虑中间边缘 $p_t = \mathbf{S}_t \# p_0$，并定义从 $p_{t_0}$ 到高斯的剩余传输映射
$$
\mathbf{T}_{t \to \infty} := \mathbf{S}_\infty \circ \mathbf{S}_{t}^{-1}, \quad \text{对所有 } t\geq 0.
$$
论证的核心表明，对精心选取的 $p_0$，存在时刻 $t_0 \geq 0$，使得 $\mathbf{T}_{t_0 \to \infty}$ 不是从新起点分布 $p_{t_0}$ 到 $\mathcal{N}(\bm{0},\mathbf{I})$ 的二次成本最优传输映射。

该结果说明 PF-ODE 流一般不给出最优传输映射，且对某些初始分布最优性会丧失。

**若干工具.** 论证关键依赖下列结果，即 **Brenier 定理**：

**定理（非正式 Brenier 定理）.** 设 $\nu_1, \nu_2$ 为 $\mathbb{R}^D$ 上具有光滑密度的两个概率分布。光滑映射 $\mathbf{T} : \mathbb{R}^D \to \mathbb{R}^D$ 是从 $\nu_1$ 到 $\nu_2$ 的（二次成本下）最优传输当且仅当对某个凸函数 $u$ 有 $\mathbf{T} = \nabla u$。此时 $\mathrm{D} \mathbf{T}$ 对称且半正定，$u$ 满足 Monge–Ampère 方程：
$$
\det \mathrm{D}^2 u(\mathbf{x}) = \frac{\nu_1(\mathbf{x})}{\nu_2(\nabla u(\mathbf{x}))}.
$$

证明还隐式使用下列事实（我们不再每次重复）：**一个映射是两个分布之间的最优传输当且仅当其逆是反向的最优传输。**

**证明概要：PF-ODE 一般不是 OT 映射.** 作者采用反证法：假设对每个 $t\ge0$，映射
$$
\mathbf{T}_t  =  \mathbf{S}_t \circ \mathbf{S}_\infty^{-1}
$$
是从 $\mathcal{N}(\bm{0},\mathbf{I})$ 到 $p_t$ 的二次成本最优传输映射。

**步骤 1：Brenier 定理.** 由 Brenier 定理，从高斯出发的任意最优传输映射的 Jacobian 对称且半正定。故
$$
\mathrm{D} \mathbf{T}_t(\mathbf{x}) = \mathrm{D} \mathbf{S}_t(\mathbf{S}_\infty^{-1}(\mathbf{x})) \mathrm{D} (\mathbf{S}_\infty^{-1})(\mathbf{x})
$$
对所有 $t$ 和 $\mathbf{x}$ 对称。这里 $\mathrm{D} \mathbf{T}_t(\mathbf{x})$ 表示对 $\mathbf{x}$ 的全微分。

**步骤 2：对称条件对时间求导.** 对时间求导：
$$
\frac{\partial}{\partial t} \mathrm{D}\mathbf{T}_t(\mathbf{x})
= \left( \frac{\partial}{\partial t} \mathrm{D} \mathbf{S}_t \right)(\mathbf{S}_\infty^{-1}(\mathbf{x})) \mathrm{D} (\mathbf{S}_\infty^{-1})(\mathbf{x}).
$$
若对称性对所有 $t$ 成立，则该导数保持对称。

由流 ODE（对 $\mathbf{x}$ 求导）得：
$$
\frac{\partial (\mathrm{D}\mathbf{S}_t)}{\partial t}
= \mathrm{D}\mathbf{v}_t(\mathbf{S}_t) \cdot \mathrm{D}\mathbf{S}_t
= \left( -\mathbf{I} - \mathrm{D}^2 \log p_t(\mathbf{S}_t) \right) \cdot \mathrm{D}\mathbf{S}_t.
$$
结合以上可知
$$
\left( -\mathbf{I} - \mathrm{D}^2 \log p_t(\mathbf{S}_t) \right) \cdot\mathrm{D}\mathbf{S}_t \cdot \mathrm{D} (\mathbf{S}_\infty^{-1})
$$
对所有 $t\geq 0$ 对称。

在 $t = 0$，$\mathbf{S}_0 = \mathbf{I}$，$\mathrm{D}\mathbf{S}_0 = \mathbf{I}$，故
$$
\left( -\mathbf{I} - \mathrm{D}^2 \log p_0(\mathbf{S}_\infty^{-1}(\mathbf{x})) \right) \cdot \mathrm{D} (\mathbf{S}_\infty^{-1})(\mathbf{x}) \quad \text{对称}.
$$

**步骤 3：交换条件.** 由于假定 $\mathbf{T}_0 = \mathbf{S}_\infty^{-1}$ 最优，其 Jacobian $D\mathbf{T}_0 = \mathrm{D}(\mathbf{S}_\infty^{-1})$ 对称。又 $\mathrm{D}^2\log p_0$ 的 Hessian 对称。两个对称矩阵相乘得到对称矩阵当且仅当它们可交换。故对所有 $\mathbf{x}\in\mathbb{R}^D$，
$$
\mathrm{D}^2\log p_0\bigl(\mathbf{S}_\infty^{-1}(\mathbf{x})\bigr)
\quad\text{必须与}\quad
\mathrm{D}(\mathbf{S}_\infty^{-1})(\mathbf{x})
\quad\text{可交换}.
$$
令 $\mathbf{y}=\mathbf{S}_\infty^{-1}(\mathbf{x})$ 得等价条件：对所有 $\mathbf{y}\in\mathbb{R}^D$，
$$
\mathrm{D}^2\log p_0(\mathbf{y})
\quad\text{必须与}\quad
\mathrm{D}\mathbf{S}_\infty(\mathbf{y})
\quad\text{可交换}.
$$

我们将该条件化为更可计算的形式。由于 $\mathbf{S}_\infty$ 是 $p_0$ 与 $\mathcal{N}(\bm{0},\mathbf{I})$ 之间的最优映射，Brenier 定理保证对某个凸函数 $u$ 有 $\mathbf{S}_\infty = \nabla u$。由 Monge–Ampère 方程得：
$$
\log p_0(\mathbf{y}) = \log \det(\mathrm{D}^2 u(\mathbf{y})) - \frac{1}{2} \|\nabla u(\mathbf{y})\|^2 + \text{常数}.
$$
条件变为（其中 $\mathrm{D}\mathbf{S}_\infty = \mathrm{D}^2 u$）：
$$
\mathrm{D}^2 \left(\log \det \mathrm{D}^2 u - \frac{1}{2} \|\nabla u\|^2\right)
\quad\text{必须与}\quad
\mathrm{D}^2 u
\quad\text{可交换}.
$$
这给出 $\mathbf{T}_t$ 最优的一个必要条件。

**步骤 4：构造反例.** 我们说明如何利用该必要条件得到矛盾。

假设能构造凸函数 $u$，使得
$$
\mathrm{D}^2 \left( \log \det \mathrm{D}^2 u(\mathbf{x}) - \frac{1}{2} \lvert \nabla u(\mathbf{x}) \rvert^2 \right)
$$
对某 $\mathbf{x} \in \mathbb{R}^D$ 与 $\mathrm{D}^2 u(\mathbf{x})$ 不可交换。定义 $p_0 = (\nabla u)^{-1}\# \mathcal{N}(\bm{0}, \mathbf{I})$，由 Brenier 定理 $\nabla u$ 是从 $p_0$ 到 $\mathcal{N}(\bm{0}, \mathbf{I})$ 的最优传输。但必要条件不成立，导致矛盾。因此目标是构造这样的函数。考虑
$$
u(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2 + \varepsilon \phi(\mathbf{x}), \quad \text{对小的 } \varepsilon.
$$
则 $\mathrm{D}^2 u(\mathbf{0}) = \mathbf{I} + \varepsilon \mathrm{D}^2 \phi(\mathbf{0})$，在 $\mathbf{x} = \mathbf{0}$ 处的交换条件要求 $\mathrm{D}^2 \phi(\mathbf{0})$ 与 $\mathrm{D}^2 (\Delta \phi)(\mathbf{0})$ 可交换。

例如在 $\mathbb{R}^2$ 中，取
$$
\phi(x_1, x_2) = x_1 x_2 + x_1^4
$$
即得一反例，其中 Hessian $\mathrm{D}^2 \log p_0$ 与 Jacobian $\mathrm{D}^2 u$ 不可交换。

该矛盾表明 $\mathbf{T}_t$ 不能对所有 $t \geq 0$ 最优。因此存在某 $t_0 \geq 0$，使得映射 $\mathbf{T}_{t_0 \to \infty}$ 不是最优的。

### 4.5.2 典范线性流与 Reflow 能否得到 OT 映射？

我们已经看到 PF-ODE（尤其是 VP 型前向核）一般不是 OT 映射。一个自然问题是：

**问题.** 线性插值流 $(1-t)\mathbf{x}_0 + t \mathbf{x}_1$（$\mathbf{x}_0\sim p_{\mathrm{src}}$，$\mathbf{x}_1\sim p_{\mathrm{tgt}}$）在应用于独立耦合 $\pi(\mathbf{x}_0, \mathbf{x}_1) = p_{\mathrm{src}}(\mathbf{x}_0)p_{\mathrm{tgt}}(\mathbf{x}_1)$ 时，是否恢复 OT 映射？

答案是否定的。

尽管如此，将线性路径与给定耦合结合，可得到真实 OT 成本的一个实用上界。在所有可能路径中，线性插值提供最紧的此类上界，如下所述。

**典范线性流与最优传输.**

聚焦二次成本最优传输，我们考虑式 (eq:ot) 的等价形式，即式 (eq:bb-ot) 的 Benamou–Brenier 形式：
$$
\mathcal{K}\left(p_{\mathrm{src}}, p_{\mathrm{tgt}}\right) :=  \min_{\substack{ (p_t, \mathbf{v}_t) \text{ s.t. }
    \partial_t p_t + \nabla \cdot (p_t \mathbf{v}_t) = 0, \\
    p_0 = p_{\mathrm{src}},\,\, p_1 = p_{\mathrm{tgt}}
}}
\int_0^1 \int_{\mathbb{R}^D} \|\mathbf{v}_t(\mathbf{x})\|^2 p_t(\mathbf{x}) \mathrm{d}\mathbf{x} \mathrm{d} t.
$$
然而直接求解该最小化问题通常不可行，因为需要求解高度非线性的偏微分方程，即 Monge–Ampère 方程。

虽然 Benamou–Brenier 形式的求解一般不可行，但文献表明其动能有一个实用上界。这是通过将搜索限制在更简单的**条件流**族上实现的，其中每条路径由其从源与目标分布的耦合 $\pi_{0,1}$ 抽取的固定端点 $(\mathbf{x}_0, \mathbf{x}_1)$ 定义。在该**条件流**族内，典范线性插值是最优选择，形式如下。

**命题（通过条件流得到 OT 动能的上界）.** 设 $\pi_{0,1}$ 为 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 的任意耦合。

(1) 动能由上界：任意连接端点的条件流 $\bm{\Psi}_t(\mathbf{x}_0, \mathbf{x}_1)$ 的路径能量的期望：
$$
\mathcal{K}\left(p_{\mathrm{src}}, p_{\mathrm{tgt}}\right) \le \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1) \sim \pi_{0,1}} \left[ \int_0^1 \|\bm{\Psi}'_t(\mathbf{x}_0, \mathbf{x}_1)\|^2 \mathrm{d} t \right].
$$

(2) 使右侧上界最小的唯一条件流 $\bm{\Psi}^*_t$ 为线性插值路径：
$$
\bm{\Psi}^*_t(\mathbf{x}_0, \mathbf{x}_1) = (1 - t) \mathbf{x}_0 + t \mathbf{x}_1.
$$
代入该最优路径得到最紧形式的上界：
$$
\mathcal{K}\left(p_{\mathrm{src}}, p_{\mathrm{tgt}}\right) \le \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1) \sim \pi_{0,1}} \|\mathbf{x}_1 - \mathbf{x}_0\|^2.
$$

换言之，线性插值 $\bm{\Psi}^*_t$（即 Flow Matching 与 Rectified Flow 使用的前向核）对任意选取的耦合 $\pi_{0,1}$ 最小化真实动能的上界。

我们强调，在该条件流类内的最优性并不保证在边缘分布上的全局最优性。

**Reflow 与最优传输.**

两种分布之间最朴素的传输方案是用简单独立耦合以直线连接其样本。然而该方法显然不是最优的：失败之处不在直线路径本身，而在初始点配对效率低。

Reflow 程序可以提供一个构造性的回应。它是专门为修正这种配对而设计的迭代算法，且每一步都保证成本不增。该性质表明 Reflow 系统地将传输方案推向更优配置，自然引出其收敛性的核心问题：

**问题.** 若迭代地应用 `Rectify` 算子会怎样？得到的传输方案序列能否收敛到最优方案，或者说 Reflow 过程的不动点是否给出 OT 映射？

简短答案是一般不能。下面我们说明可能出问题的地方。回忆 Reflow 程序通过更新
$$
\pi^{(k+1)} = \mathrm{Rectify}(\pi^{(k)}),
$$
迭代地细化 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 之间的耦合，初始化为乘积耦合 $\pi^{(0)} := p_{\mathrm{src}}(\mathbf{x}_0) p_{\mathrm{tgt}}(\mathbf{x}_1)$。更精确地，`Rectify` 通过以下方式输出更新后的耦合 $\pi^{(k+1)}$：在每次迭代 $k = 0, 1, 2, \ldots$，通过
$$
\mathbf{v}_t^{(k)} \in \argmin_{\mathbf{u}_t} \, \mathcal{L}(\mathbf{u}_t\big\vert\pi^{(k)}),
$$
学习速度场 $\mathbf{v}_t^{(k)}$，其中 $\mathcal{L}(\mathbf{u}_t\big\vert\pi^{(k)})$ 为式 (eq:reflow-loss) 中定义的损失（如 RF 或 FM 损失）。此处为记号简便，我们采用速度场的非参数表述，而非其他地方的参数形式 $\bm{\phi}$。更新后的耦合由
$$
\pi^{(k+1)}(\mathbf{x}_0, \mathbf{x}_1) := p_{\mathrm{src}}(\mathbf{x}_0)\, \delta\big(\mathbf{x}_1 - \bm{\Psi}_1^{(k)}(\mathbf{x}_0)\big),
$$
给出，其中 $\bm{\Psi}_1^{(k)}$ 表示从初始条件 $\mathbf{x}_0$ 积分 $\mathbf{v}_t^{(k)}$ 得到的 $t=1$ 时刻的解映射。

文献中观察到，对 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 之间的耦合 $\pi$，存在使 Reflow 损失最小化的速度场 $\mathbf{v}_t$（即满足 $\mathcal{L}(\mathbf{v}_t |\pi) = 0$）并不必然意味着传输是最优的。

受 Benamou–Brenier 框架启发（其中最优传输速度已知为势函数的梯度），文献提出附加约束：速度场 $\mathbf{v}_t$ 应为势场。因此式 (eq:reflow-loss) 中的目标被修改为将 $\mathbf{v}_t$ 限制在梯度向量场空间（也称势流）：
$$
\mathbf{w}_t^{(k)} \in \argmin_{\substack{\mathbf{u}_t:\, \mathbf{u}_t=\nabla\varphi \\ \text{对某 }\varphi\colon\mathbb{R}^D\to\mathbb{R}}} \, \mathcal{L}(\mathbf{u}_t\big\vert\pi^{(k)}),
$$
程序其余部分与 `Rectify` 相同。我们将该关联算子记为 $\mathrm{Rectify}_{\perp}$，强调在无旋向量场上的投影。

设 $\pi$ 为 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 之间的耦合。文献猜想下列刻画最优性的等价关系：

(i) $\pi$ 是最优传输耦合。

(ii) $\pi$ 是势整流算子的不动点：$\pi = \mathrm{Rectify}_{\perp}(\pi)$。

(iii) 存在梯度速度场 $\mathbf{v}_t = \nabla \varphi_t$ 使得整流损失为零：$\mathcal{L}(\mathbf{v}_t |\pi) = 0$。

然而，文献给出两类反例：

1. 当中间分布 $p_t$ 的支撑不连通时，可以找到 $\mathrm{Rectify}_{\perp}$ 的不动点，其 Reflow 损失为零且具有梯度速度场，但仍无法产生最优耦合。
2. 即使两个端点分布都是高斯，也存在耦合，其损失任意小但与最优耦合的偏差任意大。

因此，尽管整流流可以产生强大的生成模型，其作为最优传输求解器的可靠性仍然有限。这凸显了生成建模与有原则的最优传输理论之间的重要差距，邀请在二者交叉处进一步研究。

最后我们指出，传输成本并不总与下游性能相关；因此计算精确最优传输映射未必带来更好的实际效果。尽管如此，最优传输的变体仍是科学与工程中许多问题的基石，扩散模型为探索这些挑战提供了有力框架。
