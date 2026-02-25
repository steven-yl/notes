# 引导与可控生成

扩散模型是强大的生成框架。在无条件设定下，目标是学习 $p_{\text{data}}(\mathbf{x})$ 并在无外部输入的情况下生成样本。

然而，许多应用需要*条件生成*，即输出满足用户指定的准则。这可以通过引导一个无条件模型来实现，或直接学习条件分布 $p_0(\mathbf{x} | \mathbf{c})$，其中条件 $\mathbf{c}$（如标签、文本描述或草图）引导生成过程。

本章建立在条件分数的原则性观点之上：条件分数可分解为*无条件方向*与*引导方向*，后者在保持真实感的同时将样本推向条件。我们说明为何引导不可或缺，展示条件分数如何作为控制的统一接口，并综述近似引导项的各种方式。随后区分*控制*（满足条件）与*对齐*（在条件下满足人类偏好），并描述如何将偏好纳入同一框架。最后，我们讨论在无需额外奖励模型（即对更符合人类偏好的输出赋予更高分数的学习打分器）的情况下直接优化偏好的方法。

---

## 序言

![引导扩散采样示意图。](../arXiv-2510.21890v1/Images/PartC/guidance.pdf)

**图：** 引导扩散采样示意图。反向时间 PF-ODE 采样从右侧的纯噪声（$t = T$）开始，逐渐向左演化至左侧的干净样本（$t = 0$）。在此过程中，由 $w_t$ 加权的引导方向 $\nabla_{\mathbf{x}_t}\log \tilde{p}_t(\mathbf{c} |\mathbf{x}_t)$ 根据 $\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t) + w_t\,\nabla_{\mathbf{x}_t}\log \tilde{p}_t(\mathbf{c} |\mathbf{x}_t)$ 修改速度场。这些附加方向将轨迹导向期望属性（日本画风格），同时样本从粗到细逐步细化。

扩散模型的生成过程以由粗到细的方式进行，为可控生成提供了灵活框架。每一步去除少量噪声、样本变得更清晰，逐渐显现更多结构与细节。这一性质使得可以对生成过程施加控制：通过在所学的时间依赖速度场上加入引导项，我们可以将生成轨迹引导至反映用户意图。

引导式采样在扩散模型中的原则性基础是条件分数的贝叶斯分解。对每个噪声水平 $t$：

$$
\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t |\mathbf{c})
=
\underbrace{\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)}_{\text{unconditional direction}}
+
\underbrace{\nabla_{\mathbf{x}_t}\log p_t(\mathbf{c} |\mathbf{x}_t)}_{\text{guidance direction}}.
$$

该恒等式表明，条件采样可通过在无条件分数之上加入引导项 $\nabla_{\mathbf{x}_t}\log p_t(\mathbf{c} |\mathbf{x}_t)$ 实现。多种可控生成方法（如分类器引导、通用无训练引导等）可理解为对该引导项的不同近似，因为 $p_t(\mathbf{c} |\mathbf{x}_t)$ 通常因对 $\mathbf{x}_0$ 的边际化而难以处理。

一旦得到此类近似，采样只需用条件分数替换无条件分数。利用上式，PF-ODE 变为

$$
\frac{\mathrm{d} \mathbf{x}(t)}{\mathrm{d} t} = f(t)\mathbf{x}(t) - \frac{1}{2} g^2(t) \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}(t) |\mathbf{c}) = f(t)\mathbf{x}(t) - \frac{1}{2} g^2(t) \Big[ \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}(t)) + \nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} |\mathbf{x}(t)) \Big].
$$

需要强调的是，对这些时间依赖向量场的引导本质上依赖于其线性性，因此下文的讨论在分数预测形式下给出，并可通过其线性关系自然推广到 $\mathbf{x}$-、$\bm{\epsilon}$- 与 $\mathbf{v}$-预测。

**引导方向的具体形式。**

1. **分类器引导（CG）。** 分类器引导在加噪数据 $\mathbf{x}_t$ 上训练时间条件分类器 $p_{\bm{\psi}}(\mathbf{c} | \mathbf{x}_t, t)$（由在水平 $t$ 上腐蚀的带标签干净样本得到）。采样时，其输入梯度提供引导项：$\nabla_{\mathbf{x}_t}\log p_{\bm{\psi}}(\mathbf{c} | \mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t}\log p_t(\mathbf{c} | \mathbf{x}_t)$，然后加到无条件分数上。

2. **无分类器引导（CFG）。** CFG 直接训练单一条件模型 $\mathbf{s}_{\bm{\phi}}(\mathbf{x}_t,t,\mathbf{c}) \approx \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t|\mathbf{c})$，其中通过在部分训练步中随机将条件替换为特殊空标记来联合学习无条件模型。

3. **无训练（替代）引导。** 条件 $p_t(\mathbf{c}| \mathbf{x}_t)$ 通常难以处理，因为需要对干净潜变量 $\mathbf{x}_0$ 边际化：$p_t(\mathbf{c}| \mathbf{x}_t) = \int p(\mathbf{c}| \mathbf{x}_0)  p(\mathbf{x}_0| \mathbf{x}_t)  \mathrm{d}\mathbf{x}_0$，且在典型应用中至少有一个因子未知，导致积分不可处理。无训练（基于损失的）引导避免直接计算条件似然 $p_t(\mathbf{c}| \mathbf{x}_t)$，而是引入现成的损失 $\ell(\mathbf{x}_t,\mathbf{c};t)$ 并定义替代条件分布 $\widetilde p_t(\mathbf{c}|\mathbf{x}_t) \propto \exp\!\big(-\tau\,\ell(\mathbf{x}_t,\mathbf{c};t)\big),\ \tau>0$，作为伪似然。该形式绕过真实条件似然的计算困难，仍可通过所选损失的梯度实现引导。其条件分数仅由损失关于 $\tau$ 的梯度给出：$\nabla_{\mathbf{x}_t}\log \widetilde p_t(\mathbf{c}|\mathbf{x}_t) = -\tau \nabla_{\mathbf{x}_t}\ell(\mathbf{x}_t,\mathbf{c}; t)$。该项与引导权重 $w_t$ 一起加到无条件分数上：$\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t) + w_t \bigl[-\tau \nabla_{\mathbf{x}_t}\ell(\mathbf{x}_t,\mathbf{c}; t)\bigr]$，这正是*倾斜*密度 $\widetilde p_t^{\mathrm{tilt}}(\mathbf{x}_t| \mathbf{c}) \propto  p_t(\mathbf{x}_t) \widetilde p_t(\mathbf{c}| \mathbf{x}_t)^{w_t} \propto  p_t(\mathbf{x}_t) \exp\!\big(-w_t\tau \ell(\mathbf{x}_t,\mathbf{c}; t)\big)$ 的分数。实践中，我们用该倾斜分数替换采样中的条件分数，并求解所得 ODE 以抽取样本。由此观之，分类器引导无非是以学习到的分类器 $\widetilde p_t(\mathbf{c}|\mathbf{x}_t):=p_{\bm{\psi}^*}(\mathbf{c}| \mathbf{x}_t,t)$ 为替代的引导，即 $\ell(\mathbf{x}_t,\mathbf{c}; t) = -\log p_{\bm{\psi}^*}(\mathbf{c}| \mathbf{x}_t,t), \quad \tau=1$。引导对采样轨迹的效果见上图。

以上技术均可同样应用于条件模型之上，从而在生成时注入额外控制信号。

**备注：** 带引导的 PF-ODE *并不*（一般而言）从倾斜族中采样。即使分数精确且 ODE 精确积分，用*倾斜*分数替换分数也不会使时间–$t$ 边际等于 $\{\widetilde p_t^{\mathrm{tilt}}(\cdot|\mathbf{c})\}_{t\in[0,1]}$，也不会使终端分布等于 $\widetilde p_0^{\mathrm{tilt}}(\cdot|\mathbf{c})$。（相关推导见原文。）

**从控制到通过直接偏好优化实现更好对齐。**  
强控制可以“满足条件”却“偏离偏好”：样本可能满足条件信号（如提示）却偏离人类实际偏好。我们通过用偏好评分对条件目标进行*倾斜*来形式化这一点：$\widetilde p_0^{\mathrm{tilt}}(\mathbf{x}_0| \mathbf{c}) \propto p_0(\mathbf{x}_0|\mathbf{c}) \exp \big(\beta r(\mathbf{x}_0,\mathbf{c})\big)$，其中 $r(\mathbf{x}_0,\mathbf{c})$ 是干净样本 $\mathbf{x}_0$ 与条件 $\mathbf{c}$ 的标量对齐评分（奖励）（$r$ 越大表示对齐越好）。实践中，$r$ 可以是：(i) 外部奖励/分类器的 logit 或对数概率；(ii) 相似度度量（如 CLIP/感知）；或 (iii) 学习到的偏好模型。

实现此类可引导性的现有方法通常收集模型生成结果的相对质量的人类标签，并通过对齐人类偏好的微调来微调条件扩散模型，常通过人类反馈的强化学习（RLHF）。然而 RLHF 复杂且往往不稳定：先拟合奖励模型以刻画人类偏好，再用强化学习微调条件扩散模型以最大化该估计奖励并约束相对原模型的策略漂移。

这自然引出问题：*能否完全去掉奖励模型训练阶段？* 我们通过 Diffusion-DPO 来回答：它是为扩散模型改编的直接偏好优化（原为大型语言模型提出）。如后文所述，Diffusion-DPO 直接从成对选择学习偏好倾斜，从而在无需单独奖励模型的情况下微调条件扩散模型以对齐偏好。

---

## 分类器引导

### 分类器引导的基础

记 $\mathbf{c}$ 为从分布 $p(\mathbf{c})$ 抽取的条件变量，如类别标签、标题或其他辅助信息。目标是抽取 $p_0(\mathbf{x} | \mathbf{c})$ 的样本。在基于扩散的条件生成中，通过运行时间边际为 $p_t(\cdot | \mathbf{c})$ 的反向时间动力学来实现。该动力学的漂移依赖于条件分数 $\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t | \mathbf{c}),\ t \in [0,T]$。因此标准且有效的途径是估计该量。

基于贝叶斯法则的一个基本结论是，条件分数可分解为：

$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{c}) = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} | \mathbf{x}_t),
$$

其中 $p_t(\mathbf{c} | \mathbf{x}_t)$ 表示在时间 $t$ 由加噪输入 $\mathbf{x}_t$ 预测条件 $\mathbf{c}$ 时 $\mathbf{c}$ 在 $\mathbf{x}_{t}$ 条件下的概率。该分解催生了分类器引导（CG）方法：利用预训练的时间依赖分类器 $p_t(\mathbf{c} |\mathbf{x}_t)$ 引导生成。具体地，定义带引导尺度 $\omega \ge 0$ 的单参数*引导密度*族（倾斜条件）：

$$
p_t(\mathbf{x}_t |\mathbf{c}, \omega) \propto p_t(\mathbf{x}_t)  p_t(\mathbf{c} |\mathbf{x}_t)^{\omega},
$$

得到分数函数：

$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t |\mathbf{c}, \omega)
= \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) + \omega  \nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} |\mathbf{x}_t).
$$

几何上，这沿增大类别似然的方向倾斜无条件流。当 $\omega=1$ 时，$p_t(\mathbf{x}_t |\mathbf{c}, \omega)$ 与真实条件 $p_t(\mathbf{x}_t |\mathbf{c})$ 一致；$\omega\neq 1$ 时则为引导（调温）重加权而非字面条件。

标量 $\omega \ge 0$ 调节分类器的影响：$\omega = 1$ 时恢复真实条件分数；$\omega > 1$ 时放大分类器信号，通常提高条件保真度（常以多样性为代价）；$0 \le \omega < 1$ 时削弱分类器信号，通常增加样本多样性但减弱条件作用。

**CG 中的实用近似。**  
实践中，CG 是一种（相对于扩散模型）无训练方法，用于引导预训练的无条件扩散模型 $\mathbf{s}_{\bm{\phi}^*}(\mathbf{x}_t ,t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$。CG 仅在采样时使用，不修改扩散模型本身。为此，单独训练时间依赖分类器 $p_{\bm{\psi}}(\mathbf{c} |\mathbf{x}_t,t)$，在不同噪声水平 $t$ 从加噪输入 $\mathbf{x}_t$ 预测条件 $\mathbf{c}$。分类器通过最小化交叉熵损失的标准方式训练，其中 $(\mathbf{x},\mathbf{c})\sim p_{\text{data}}$ 表示成对标注数据，$\mathbf{x}_t = \alpha_t \mathbf{x} + \sigma_t \bm{\epsilon}$ 为时间 $t$ 的加噪输入。分类器必须显式以 $t$ 为条件（如通过时间嵌入），因其需在所有噪声水平上可靠工作。训练后，分类器提供的分数作为真实似然梯度的替代：$\nabla_{\mathbf{x}_t} \log p_{\bm{\psi}^*}(\mathbf{c} |\mathbf{x}_t,t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} |\mathbf{x}_t)$。

### 使用 CG 的推理

推理时，将分类器梯度 $\nabla_{\mathbf{x}_t} \log p_{\bm{\psi}^*}(\mathbf{c} | \mathbf{x}_t,t)$ 加到无条件分数上并用引导权重 $\omega$ 缩放，得到引导分数 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{c}, \omega)$ 的近似：

$$
\mathbf{s}^{\text{CG}}(\mathbf{x}_t ,t, \mathbf{c}; \omega) := \mathbf{s}_{\bm{\phi}^*}(\mathbf{x}_t ,t) + \omega \nabla_{\mathbf{x}_t} \log p_{\bm{\psi}^*}(\mathbf{c} | \mathbf{x}_t,t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{c}, \omega).
$$

因此，只需在反向时间 SDE 或 PF-ODE 中用引导分数 $\mathbf{s}^{\text{CG}}(\mathbf{x}_t ,t, \mathbf{c}; \omega)$ 替换无条件分数函数 $\mathbf{s}_{\bm{\phi}^*}(\mathbf{x}_t ,t)$（对给定的 $\omega$），从而将生成轨迹导向与条件 $\mathbf{c}$ 一致的样本。

### 优点与局限

CG 为条件生成提供了简单灵活的机制，可通过 $\omega$ 显式控制条件强度。它可与任意预训练无条件扩散模型配合使用，仅需额外的分类器用于条件化。

但该方法有明显局限：**训练成本**：分类器必须在所有噪声水平上训练，计算昂贵。**鲁棒性**：分类器需对严重腐蚀的输入 $\mathbf{x}_t$ 泛化良好，尤其对大 $t$，这颇具挑战。**独立训练**：分类器与扩散模型独立训练，可能与所学数据分布不完全一致。

---

## 无分类器引导

### 无分类器引导的基础

![CFG 示意图。](../arXiv-2510.21890v1/Images/PartB/classifier_guidance_v2.pdf)

**图：** CFG 示意图。调整后的分数 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t| \mathbf{c}, \omega)$ 由无条件分数 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$ 与条件分数 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t| \mathbf{c})$ 按 $\omega$ 的线性插值得到。所得方向将样本从先验导向与目标条件一致的数据分布模态。

*无分类器引导*（CFG）是一种简化方法，省去了单独的分类器。核心思想是以某种方式修改分数函数的梯度，从而在没有显式分类器的情况下实现有效条件化。具体地，条件分布的对数概率梯度按下式调整：

$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{c} | \mathbf{x}_t) = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{c}) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t).
$$

代入 CG 的引导分数形式可得条件分数的如下形式：

$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{c}, \omega)
= \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) + \omega \left( \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{c}) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \right)
= \omega \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{c}) + (1 - \omega) \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t).
$$

超参数 $\omega$ 再次在控制条件信息的影响上起关键作用（取 $\omega \ge 0$）：$\omega = 0$ 时模型表现为无条件扩散模型，完全忽略条件；$\omega = 1$ 时使用条件分数且无额外引导；$\omega > 1$ 时更强调条件分数、弱化无条件分数，通常加强对 $\mathbf{c}$ 的对齐但降低多样性。

### CFG 的训练与采样

**通过 CFG 联合训练无条件与条件扩散模型。** 与 CG 不同，CFG 需要重新训练显式考虑条件变量 $\mathbf{c}$ 的扩散模型。但为条件和无条件分数各训一个模型往往计算上不可行。为此，CFG 采用单一模型 $\mathbf{s}_{\bm{\phi}}(\mathbf{x}_t, t; \mathbf{c})$，将 $\mathbf{c}$ 作为额外输入，在一个模型内同时学习两个分数。训练过程如下：无条件训练时传入空标记 $\emptyset$ 代替条件输入，得到 $\mathbf{s}_{\bm{\phi}}(\mathbf{x}_t, t, \emptyset)$；条件训练时提供真实条件 $\mathbf{c}$，得到 $\mathbf{s}_{\bm{\phi}}(\mathbf{x}_t, t, \mathbf{c})$。两者通过以概率 $p_\mathrm{uncond}$（用户设定超参数，通常取 $0.1$）随机将 $\mathbf{c}$ 替换为空输入 $\emptyset$ 而统一。该联合训练使模型同时学习条件和无条件分数。完整训练算法见下文两个算法（无条件 DM 与 CFG 条件 DM）；训练时不使用 CFG 权重 $\omega$。

**算法（无条件 DM）：** 重复：从 $p_{\text{data}}(\mathbf{x})$ 采样 $\mathbf{x}$；$t \sim \mathcal{U}[0, T]$；$\bm{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$；$\mathbf{x}_t = \alpha_t \mathbf{x} + \sigma_t \bm{\epsilon}$；对 $\nabla_{\bm{\phi}} \left\| \mathbf{s}_{\bm{\phi}}(\mathbf{x}_t, t) - \mathbf{s} \right\|^2$ 做梯度步；直到收敛。

**算法（CFG 条件 DM）：** 输入 $p_\mathrm{uncond}$（无条件 dropout 概率）。重复：$(\mathbf{x}, \mathbf{c}) \sim p_{\text{data}}(\mathbf{x}, \mathbf{c})$；以概率 $p_\mathrm{uncond}$ 将 $\mathbf{c} \gets \emptyset$；$t \sim \mathcal{U}[0, T]$；$\bm{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$；$\mathbf{x}_t = \alpha_t \mathbf{x} + \sigma_t \bm{\epsilon}$；对 $\nabla_{\bm{\phi}} \left\| \mathbf{s}_{\bm{\phi}}(\mathbf{x}_t, t, \mathbf{c}) - \mathbf{s} \right\|^2$ 做梯度步；直到收敛。

**使用 CFG 的条件采样。**  
一旦用上述 CFG 算法训练好模型 $\mathbf{s}_{\bm{\phi}^*}(\mathbf{x}_t, t, \mathbf{c})$，采样时可应用 CFG。对数概率的梯度为：

$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{c}, \omega)
\approx \omega \mathbf{s}_{\bm{\phi}^*}(\mathbf{x}_t, t, \mathbf{c}) + (1 - \omega) \mathbf{s}_{\bm{\phi}^*}(\mathbf{x}_t, t, \emptyset)
=: \mathbf{s}_{\bm{\phi}^*}^{\text{CFG}}(\mathbf{x}_t ,t, \mathbf{c}; \omega).
$$

采样时施加固定的（或可选的时间依赖）无分类器引导权重 $\omega$，在反向时间 SDE 或 PF-ODE 中用引导分数 $\mathbf{s}_{\bm{\phi}^*}^{\mathrm{CFG}}(\mathbf{x}_t ,t, \mathbf{c}; \omega)$ 替换无条件分数。该形式通过调节 $\omega$ 实现可控生成，使样本在保持多样性的同时被导向条件信号 $\mathbf{c}$。CFG 因此只需训练一个扩散模型即可实现精确条件生成，高效且实用。

---

## （可选）无训练引导

本节概述多种无训练引导方法背后的高层思想。尽管实现与应用各异，它们都由条件分数的贝叶斯分解所表达的核心原则统一。我们先在概念框架中介绍无训练引导的高层思路，再将此思想推广到无训练逆问题求解，并给出简要概览。

**设定与记号。** 记 $\mathbf{c}$ 为条件变量。假设有以分数预测形式表达的预训练扩散模型 $\mathbf{s}_{\bm{\phi}^*}(\mathbf{x}_t, t)$，以及非负函数 $\ell(\cdot, \mathbf{c}) \colon \mathbb{R}^D \to \mathbb{R}_{\geq 0}$，用于度量样本 $\mathbf{x}$ 与条件 $\mathbf{c}$ 的对齐程度（$\ell(\mathbf{x}, \mathbf{c})$ 越小对齐越强）。具体例子：(i) $\mathbf{c}$ 为参考图像，$\ell(\cdot, \mathbf{c})$ 为度量感知接近度的相似度；(ii) $\ell(\cdot, \mathbf{c})$ 为通过 CLIP 等预训练模型计算的特征相似度。考虑标准线性–高斯前向加噪核 $p_{t}(\cdot| \mathbf{x}_0):=\mathcal{N}\big(\cdot; \alpha_t\mathbf{x}_0, \sigma^2_t\mathbf{I}\big)$。回顾 DDIM 更新（以之为例）：$\mathbf{x}_{t \rightarrow t-1} =\alpha_{t-1} \hat{\mathbf{x}}_0(\mathbf{x}_t) - \sigma_{t-1} \sigma_t \hat{\mathbf{s}}(\mathbf{x}_t)$，其中 $\hat{\mathbf{x}}_0(\mathbf{x}_t):=\mathbf{x}_{\bm{\phi}^*}(\mathbf{x}_t, t)$ 为（干净）$\mathbf{x}$-预测，$\hat{\mathbf{s}}(\mathbf{x}_t):=\mathbf{s}_{\bm{\phi}^*}(\mathbf{x}_t, t)$ 为从 $\mathbf{x}_t$ 在时间层 $t$ 的分数预测。

### 无训练引导的概念框架

多数无训练引导方法在*数据空间*或*噪声空间*中引入修正，以将 DDIM 更新导向满足条件 $\mathbf{c}$：

$$
\mathbf{x}_{t \rightarrow t-1} =\alpha_{t-1} \left(\hat{\mathbf{x}}_0(\mathbf{x}_t) + \eta_t^{\text{data}}\mathcal{G}_0 \right) - \sigma_{t-1} \sigma_t \left(\hat{\mathbf{s}}(\mathbf{x}_t) + \eta_t^{\text{latent}}\mathcal{G}_t \right),
$$

其中 $\eta_t^{\text{data}}, \eta_t^{\text{latent}} \geq 0$ 为时间依赖的引导强度，$\mathcal{G}_0$、$\mathcal{G}_t$ 为下面定义的修正项。

**A. 数据空间引导。** 沿负梯度方向 $\mathcal{G}_0 := -\nabla_{\mathbf{x}_0} \ell(\mathbf{x}_0, \mathbf{c})$ 下降，数据空间中修正后的干净估计 $\hat{\mathbf{x}}_0(\mathbf{x}_t) + \eta_t^{\text{data}} \mathcal{G}_0$ 可逐渐被导向更满足条件 $\mathbf{c}$ 的样本。该梯度下降方案可迭代应用以逐步改善对齐。代表工作包括 MGPD、UGD。

**B. 噪声空间引导。** 如前所述，条件分数 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{c}|\mathbf{x}_t)$ 通常难以处理。实用近似是引入替代似然 $\widetilde p_t(\mathbf{c}|\mathbf{x}_t) \propto \exp \big(-\eta\ell \left(\hat{\mathbf{x}}_0(\mathbf{x}_t), \mathbf{c}\right)\big)$，$\eta>0$，从而 $\nabla_{\mathbf{x}_t}\log \widetilde p_t(\mathbf{c}|\mathbf{x}_t) = -\eta\nabla_{\mathbf{x}_t}\ell \left(\hat{\mathbf{x}}_0(\mathbf{x}_t), \mathbf{c}\right) =: \mathcal{G}_t$。代入条件分数的贝叶斯形式得到：$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{c}) \approx \hat{\mathbf{s}}(\mathbf{x}_t) + \eta_t^{\text{latent}} \mathcal{G}_t$，作为噪声空间引导的修正。但计算 $\mathcal{G}_t$ 需要对 $\mathbf{x}$-预测反传，即 $\nabla_{\mathbf{x}_t} \hat{\mathbf{x}}_0(\mathbf{x}_t)^\top \cdot \left.\nabla_{\mathbf{x}_0} \log \ell_{\mathbf{c}}(\mathbf{x}_0)\right|_{\mathbf{x}_0 = \hat{\mathbf{x}}_0(\mathbf{x}_t)}$，在实践中可能带来较大计算成本。代表工作包括 Freedom、Chung 等、Bansal 等。

### 无训练逆问题方法示例

上述概念框架在逆问题中有重要应用。下面先简述背景，再给出若干具体例子，说明如何利用预训练扩散模型在推理时求解逆问题。

**逆问题背景。** 设 $\mathcal{A}$ 为腐蚀算子（可线性或非线性、已知或未知），如模糊核或修复，$\mathbf{y}$ 由腐蚀模型 $\mathbf{y} = \mathcal{A}(\mathbf{x}_0) + \sigma_{\mathbf{y}} \mathbf{z},\ \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 生成。逆问题的目标是从后验 $p_0(\mathbf{x}_0 | \mathbf{y})$ 采样，对给定观测 $\mathbf{y}$ 可能存在无穷多可能重建 $\mathbf{x}_0$，目标是恢复一个去除 $\mathbf{y}$ 中腐蚀并保持其忠实与语义特征的 $\mathbf{x}_0$。传统方法多为监督框架，需要收集腐蚀与恢复样本的成对数据 $(\mathbf{y}, \mathbf{x})$，依赖优化或监督训练神经网络，数据准备成本高且对未见数据泛化可能不足。

**预训练扩散模型作为逆问题求解器。** 条件分数可通过贝叶斯法则分解为数据分数与观测对齐项：$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{y}) = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) + \nabla_{\mathbf{x}_t}\log p_t(\mathbf{y} | \mathbf{x}_t)$。该分解将数据分数与逆问题特有的 $\mathbf{y}$ 对齐项分离，从而能以无监督方式求解逆问题：用预训练扩散模型近似数据分数，在反演时应用。数据分数 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$ 用干净数据上训练的 $\mathbf{s}_{\bm{\phi}^*}(\mathbf{x}_t, t)$ 近似；测量对齐 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} | \mathbf{x}_t)$ 闭式不可处理，因涉及对潜变量的边际化。因此，多数基于预训练扩散模型的无训练方法专注于近似 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} | \mathbf{x}_t)$，常用元形式为 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} | \mathbf{x}_t) \approx - \frac{\mathcal{P}_t \, \mathcal{M}_t}{\gamma_t}$，其中 $\mathcal{M}_t$ 为观测与估计信号之间失配的误差向量，$\mathcal{P}_t$ 将 $\mathcal{M}_t$ 映射回 $\mathbf{x}_t$ 的 ambient 空间，$\gamma_t$ 为控制引导强度的标量。不同方法对 $\mathcal{M}_t$、$\mathcal{P}_t$、$\gamma_t$ 有不同的具体形式。

**基于扩散的逆问题求解器实例。**  
**Score SDE**：考虑已知线性腐蚀 $A$、$\sigma_{\mathbf{y}} = 0$，可构造噪声水平匹配的观测 $\mathbf{y}_t := \alpha_t \mathbf{y} + \sigma_t \bm{\epsilon}$，用残差 $\mathbf{y}_t - A \mathbf{x}_t$ 驱动似然式修正，常见近似为 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y}|\mathbf{x}_t) \approx -A^\top (\mathbf{y}_t - A \mathbf{x}_t)$。  
**ILVR**：在相同设定下估计 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y}| \mathbf{x}_t) \approx -A^\dagger(\mathbf{y}_t - A\mathbf{x}_t)$，其中 $A^\dagger$ 为 Moore–Penrose 伪逆。  
**DPS（扩散后验采样）**：对已知非线性前向算子 $\mathcal{A}$ 与加性高斯噪声 $\sigma_{\mathbf{y}} \ge 0$，近似 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y}|\mathbf{x}_t) \approx \nabla_{\mathbf{x}_t} \log p_t\bigl(\mathbf{y}|X_0 = \hat{\mathbf{x}}_0(\mathbf{x}_t)\bigr)$，其中 $\hat{\mathbf{x}}_0(\mathbf{x}_t)$ 通常由预训练扩散模型的 Tweedie 公式估计。在线性逆问题下可进一步简化为含 $\frac{1}{\sigma_{\mathbf{y}}^2}$、投影项与残差 $\mathbf{y} - A(\hat{\mathbf{x}}_0(\mathbf{x}_t))$ 的形式。更多工作通过提出对 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} | \mathbf{x}_t)$ 的各种近似来探索基于扩散的逆问题求解器，完整综述可参考 Daras 等。

---

## 从强化学习到直接偏好优化以实现模型对齐

在使生成模型与人类意图对齐的实践中，主流范式一直是人类反馈的强化学习（RLHF）。RLHF 有效但复杂、多阶段且可能不稳定。本节介绍*直接偏好优化（DPO）*：一种更简洁、稳定的方法，无需显式奖励建模或强化学习即可达到相同目标；并概述其通过 *Diffusion-DPO* 在扩散模型上的推广。

### 动机：规避 RLHF 的缺陷

对齐的目标是使一个基础预训练模型（如 SFT 模型）趋向人类更偏好的输出。RLHF 分三阶段进行。第一，*监督微调（SFT）* 在提示–响应对上训练基础模型。第二，*奖励建模（RM）* 在偏好数据上拟合模型：数据包含提示 $\mathbf{c}$ 与成对响应（偏好的“赢家”$\mathbf{x}_w$ 与不偏好的“输家”$\mathbf{x}_l$），学习标量 $r(\mathbf{c},\mathbf{x})$ 且 $r(\mathbf{c},\mathbf{x}_w)>r(\mathbf{c},\mathbf{x}_l)$。第三，*RL 微调* 用 PPO 等算法优化 SFT 模型（策略 $\pi$），最大化来自 $r$ 的期望奖励，并用 KL 惩罚正则化以保持 $\pi$ 接近参考/SFT 分布。尽管影响巨大，该流程存在缺陷：RL 阶段不稳定且计算昂贵（因为是在策略的——每次更新需要从当前模型重新采样）；需要训练和部署多个大模型（SFT、奖励，有时还有价值模型）；且只优化人类偏好的代理，奖励模型的缺陷可能被利用。这引出一个核心问题：能否去掉显式奖励建模和不稳定的 RL 步骤，直接在偏好数据上优化模型？

直接偏好优化（DPO）用单一的、监督式的步骤替代多阶段 RLHF 流程，从而简化对齐。不再训练单独的奖励模型或运行 PPO 等不稳定 RL 算法，DPO 直接用简单的逻辑损失将策略拟合到偏好对上，同时保持与固定参考模型接近。关键洞察是：经 KL 正则的 RLHF 目标可以重写，使得策略与参考之间的对数似然比充当隐式奖励。这保持了对参考策略的相同正则化，但避免了昂贵的 rollout 和显式奖励建模。下文中我们简要回顾 RLHF 流程及其对大型奖励模型和 RL 微调的依赖；然后介绍最初为语言模型提出的 DPO，它绕过奖励模型训练并简化对齐微调；最后将这一思想推广到扩散模型，介绍 Diffusion-DPO 作为生成建模设定下实用且稳定的对齐方法。

### RLHF：Bradley–Terry 视角

**RLHF 简介。** RLHF 从一个学习到的“裁判”开始：奖励模型 $r_{\bm\psi}$ 对同一提示 $\mathbf c$ 的候选响应赋予标量偏好分数。数据集 $\mathcal D$ 由带标签 $y$ 的成对 $(\tilde{\mathbf x},\mathbf x)$ 组成，$y$ 表示 $\tilde{\mathbf x}$ 是否优于 $\mathbf x$；$y$ 可为二值 $y\in\{0,1\}$ 或通过多评分者聚合得到的软值 $y\in[0,1]$。训练目标是简单的逻辑损失（见原文公式）；实践中偏好对可来自各种来源。常见约定是以有序格式存储（赢家, 输家），此时设 $y=1$，损失简化为 Bradley–Terry 形式。

**Bradley–Terry 视角与 KL 联系。** 通常通过 Bradley–Terry（BT）模型将分数差解释为胜率概率 $p_{r_{\bm\psi}}(\tilde{\mathbf x}\succ \mathbf x|\mathbf c) := \sigma \big(r_{\bm\psi}(\mathbf c,\tilde{\mathbf x})-r_{\bm\psi}(\mathbf c,\mathbf x)\big)$。该形式有两个要点：(i) 只有分数*差*起作用（故 $r_{\bm\psi}(\mathbf c,\cdot)$ 平移不变）；(ii) 损失推动预测赢家的分数高于输家。最小化该逻辑损失等价于最小化人类标签的经验伯努利分布与模型预测的伯努利分布之间的 KL 散度（在二值情况下精确；软标签时差一个熵常数）。下文采用最常见约定：$\mathcal D$ 以有序格式存储 $(\mathbf{x}^w, \mathbf{x}^l, \mathbf{c})$，标签恒为 $y=1$。

**带固定奖励的 KL 正则策略优化。** 用上述损失拟合奖励 $r:=r_{\bm\psi^\times}$ 后，RLHF 调整可学习策略 $\pi_{\bm\theta}(\mathbf x|\mathbf c)$（通常在 $p_{\bm{\phi}^\times}(\mathbf x|\mathbf c)$ 上微调），使其产生更高奖励的响应，同时用 $\mathcal{D}_{\mathrm{KL}}$ 惩罚使其接近参考 $\pi_{\mathrm{ref}}(\mathbf x|\mathbf c):=p_{\bm{\phi}^\times}(\mathbf x|\mathbf c)$。目标为 $\max_{\bm\theta} \mathbb E_{\mathbf c\sim p(\mathbf c)}\Big[\mathbb E_{\mathbf x\sim \pi_{\bm\theta}(\cdot|\mathbf c)}\big[r_{\bm\psi}(\mathbf c,\mathbf x)\big] -\beta \mathcal{D}_{\mathrm{KL}} \big(\pi_{\bm\theta}(\cdot|\mathbf c) \big\| \pi_{\mathrm{ref}}(\cdot|\mathbf c)\big)\Big]$。奖励目标只用到标注对，不要求 $\mathcal D$ 由参考模型生成；但若从接近目标策略的模型收集对，可减少分布偏移并提高所学奖励在使用区域的可靠性。综上，RLHF 先通过最小化 BT 损失拟合奖励 $r^*$，再通过求解上述策略目标优化策略 $\pi^*$。

### DPO 框架

**从 RLHF 的桥梁。** 在给定拟合奖励 $r := r_{\bm\psi^\times}$ 时，KL 正则策略目标对每个提示 $\mathbf c$ 有简单的闭式解，可写成能量形式：$\pi^*(\mathbf x|\mathbf c) = \frac{1}{Z(\mathbf c)} \pi_{\mathrm{ref}}(\mathbf x|\mathbf c)\exp(r(\mathbf c,\mathbf x)/\beta)$，其中 $Z(\mathbf c)$ 为配分函数。$\beta$ 较小时 $\exp(r/\beta)$ 更尖锐，$\pi^*$ 集中在高奖励区域；$\beta$ 较大时更平坦，$\pi^*$ 更接近 $\pi_{\mathrm{ref}}$。

**通过反转该闭式解定义隐式奖励。** 对任意策略 $\pi$（支撑含于 $\pi_{\mathrm{ref}}$），定义 $r_{\pi}(\mathbf c,\mathbf x) = \beta \log\frac{\pi(\mathbf x|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x|\mathbf c)} + \beta \log Z(\mathbf c)$。则用 $r_{\pi}$ 时闭式解就是 $\pi$；在此意义上 $r_{\pi}$ 是隐式（策略诱导的）奖励，可辨识到与提示相关的常数 $\beta\log Z(\mathbf c)$，该常数在 BT 的成对比较中相消，从而直接得到基于对数概率差的 DPO 损失。

**DPO 的训练损失。** 将隐式奖励代入 BT 模型，对同一提示 $\mathbf c$ 下的标注对 $(\mathbf x_w,\mathbf x_l)$，$Z(\mathbf c)$ 在赢家与输家间相消，得到关于对数概率差的单一逻辑损失目标（见原文）。换言之，DPO 抬高赢家相对输家的（温度缩放）优势，该优势由相对参考的对数似然改进差度量。从而在一个稳定的、类极大似然的阶段达到 RLHF 的目标，且无需训练显式奖励模型。

### Diffusion-DPO

**为何朴素 DPO 对扩散模型失效？** 在扩散模型中计算样本似然 $\pi_{\bm\theta}(\mathbf x|\mathbf c)$ 需要 ODE 求解的瞬时变量替换公式（漂移的散度），计算量大；且对整个采样轨迹求导易出现梯度消失或爆炸。为避免这些问题，Diffusion-DPO 在*路径*层面工作（以离散时间扩散为例，连续时间类似）。

**定义路径式隐式奖励。** 令轨迹为反向时间马尔可夫链下的 $\mathbf x_{0:T}:=(\mathbf x_T,\ldots,\mathbf x_0)$，条件为 $\pi(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)$。将样本级 KL 替换为路径 KL，目标为最大化路径奖励并保持与参考路径分布的 KL 接近。对每个提示 $\mathbf c$，最优策略有能量形式 $\pi^*(\mathbf x_{0:T}|\mathbf c) =\frac{1}{Z(\mathbf c)} \pi_{\mathrm{ref}}(\mathbf x_{0:T}|\mathbf c) \exp \big(R(\mathbf c,\mathbf x_{0:T})/\beta\big)$。反转该式可定义任意策略 $\pi$ 的*隐式路径奖励* $R_{\pi}$，其中常数 $\beta\log Z(\mathbf c)$ 在成对比较中无关。

**从路径隐式奖励到 DPO。** 对同一提示 $\mathbf c$ 下的标注对 $(\mathbf x^w_0,\mathbf x^l_0)$ 在*路径*上应用 Bradley–Terry 模型，使用标准逻辑对数损失，得到 Diffusion-DPO 损失（见原文公式），其中 $\Delta R$ 为赢家路径期望与输家路径期望之差。但该形式在实践中有三点不可行：(1) 端点条件导致难以处理的路径后验；(2) 嵌套且与 $\bm\theta$ 耦合的期望，求 $\nabla_{\bm\theta}$ 需对采样分布求导，方差大；(3) 长链、大和与昂贵的反传。因此需要可处理的替代。

**得到可处理的替代。** 利用 $-\log\sigma(\cdot)$ 的凸性与 Jensen 不等式，可将损失用内层期望外移得到上界；再利用隐式奖励恒等式与常数相消，得到路径对数比的上界。**逐步可处理替代：** 利用反向过程的马尔可夫性，将路径概率比分解为单步转移的乘积，取对数后变为对 $t$ 的求和；若先验在 $T$ 处两者相同，则先验比值为 1。为得到可处理估计，对时间步做单次 Jensen 上界：对每个训练对采样 $t\sim\mathcal U\{1,\dots,T\}$ 并乘以 $T$，得到单步替代目标。对扩散模型中常用的高斯反向条件（以 $\boldsymbol\epsilon$-预测为例），每步对数比与该步的 MSE 差（策略相对参考）成正比。定义 $\Delta\mathrm{MSE}(\mathbf x_t)$ 后，得到 $\tilde{\mathcal L}_{\mathrm{Diff\text{-}DPO}}$ 的实用替代（见原文框式公式），其中 $\mathbf x_t^w$ 与 $\mathbf x_t^l$ 共享同一噪声 $\boldsymbol\epsilon$ 以降低方差，$w(t)>0$ 为时间权重。直观上，最小化 $\tilde{\mathcal L}_{\mathrm{Diff\text{-}DPO}}$ 会提高模型在赢家上相对参考的预测准确度、在输家上降低，从而在保持锚定于参考的同时将策略推向赢家式的去噪轨迹、远离输家式轨迹。

---

## 小结

本章将重点从基础原理转向可控生成这一实际问题。我们建立了基于条件分数贝叶斯分解的引导统一框架，将生成过程优雅地分为无条件方向与引导项。

我们看到该原则在多种有力技术中体现：需要专门训练的方法，如使用外部分类器的分类器引导（CG），以及更高效、在单一模型内学习条件与无条件分数的无分类器引导（CFG）；还有灵活的无训练引导方法，可在推理时通过任意损失定义替代似然来引导预训练模型，从而在从艺术控制到逆问题求解等应用中无需任何再训练。

在简单条件化之外，我们深入探讨了使模型输出与人类偏好对齐这一更细微的任务。在回顾标准但复杂的 RLHF 流程后，我们介绍了直接偏好优化（DPO）及其新变体 Diffusion-DPO，作为更直接、更稳定的替代。该方法通过从偏好数据直接推导损失，优雅地省去了显式奖励模型与强化学习。

通过这些技术，我们组装了一套引导生成过程的强大工具。然而，一个重要的实际障碍尚未触及：迭代采样过程本身的高计算成本与延迟。在讨论了“生成什么”之后，我们接下来将转向同样重要的问题——“多快能生成”。下一章将直接应对这一挑战：(1) 利用采样等价于求解 ODE 的洞察，探索旨在大幅减少所需步数的数值求解器；(2) 考察一系列有影响力的方法（包括 DDIM、DEIS 与 DPM-Solver 族），它们通过将采样速度提升数个数量级使扩散模型更加实用。
