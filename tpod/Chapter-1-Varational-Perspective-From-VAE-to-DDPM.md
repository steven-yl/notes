# 第 1 章 变分视角：从 VAE 到 DDPM

本章从变分视角看待扩散模型。我们从**变分自编码器（VAE）**出发：它用潜变量表示数据，并通过最大化对数似然的一个可处理下界进行训练。在此设定下，学到的编码器将观测映到潜变量，学到的解码器将潜变量映回观测，形成建模闭环。

在这一模式之上，层次化变体（**层次化 VAE，HVAE**）堆叠多个潜层以在多尺度上刻画结构。**去噪扩散概率模型（DDPM）**沿用同一模板：编码器被固定为逐渐将数据映为噪声的前向加噪过程，训练只学习将该路径逐步逆转的解码器。因此，VAE、层次化 VAE 与扩散模型都是在由变分界定义的似然替代上做优化，为这里介绍的方法提供了共同基础。

---

## 1.1 变分自编码器

神经网络如何学会生成逼真数据？一个自然的起点是**自编码器**：它由两个网络组成——将输入压缩到低维潜码的确定性**编码器**，以及从该码重建输入的确定性**解码器**。训练通过最小化原始输入与重建之间的重建误差进行。这种设定能实现准确重建，但潜空间没有结构：随机采样潜码通常产生无意义的输出，限制了模型用于生成的能力。

**变分自编码器（VAE）**通过为潜空间赋予概率结构解决了这一问题，从而将模型从单纯的重建工具转变为真正的生成模型，能够产生新颖且逼真的数据。

### 1.1.1 概率编码器与解码器

![图 1：VAE 示意图](../arXiv-2510.21890v1/Images/PartB/vae-graph.pdf)

**图 1：VAE 示意图。** 由随机编码器 $q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})$ 将数据 $\mathbf{x}$ 映到潜变量 $\mathbf{z}$，解码器 $p_{\boldsymbol{\phi}}(\mathbf{x}|\mathbf{z})$ 从潜变量重建数据。

**解码器（生成器）的构造。** 在 VAE 中我们区分两类变量：**观测变量** $\mathbf{x}$（即我们看到的数据，如图像）与**潜变量** $\mathbf{z}$（刻画隐藏的变化因素，如物体形状、颜色或风格）。模型假设每个观测 $\mathbf{x}$ 由从简单**先验分布**（通常为标准高斯 $\mathbf{z} \sim p_{\text{prior}} := \mathcal{N}(\mathbf{0}, \mathbf{I})$）中采样的潜变量生成。

为将 $\mathbf{z}$ 映回数据空间，我们定义**解码器（生成器）**分布 $p_{\boldsymbol{\phi}}(\mathbf{x}|\mathbf{z})$。实践中该解码器保持简单，常为因子化高斯或类似分布，以便学习专注于提取有用的潜特征而非记忆数据。直观上，逐像素直接生成极其困难；潜变量提供紧凑表示，从中解码出具体像素排布会容易得多。新样本的生成方式为：先采样 $\mathbf{z} \sim p_{\text{prior}}$，再通过 $\mathbf{x} \sim p_{\boldsymbol{\phi}}(\mathbf{x}|\mathbf{z})$ 解码。

VAE 由此通过边缘似然定义潜变量生成模型：
$$
p_{\boldsymbol{\phi}}(\mathbf{x}) = \int p_{\boldsymbol{\phi}}(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) \,\mathrm{d}\mathbf{z}.
$$
理想情况下，解码器参数 $\boldsymbol{\phi}$ 通过最大化该边缘似然（即最大似然估计）学习。但由于对 $\mathbf{z}$ 的积分在富有表达力的非线性解码器下难以处理，直接 MLE 在计算上不可行，这促使 VAE 采用变分方法。

**编码器（推断网络）的构造。** 为处理难以处理的生成器，考虑反向问题：给定观测 $\mathbf{x}$，哪些潜码 $\mathbf{z}$ 可能生成了它？由 Bayes 规则，后验分布为
$$
p_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \;=\; \frac{p_{\boldsymbol{\phi}}(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{p_{\boldsymbol{\phi}}(\mathbf{x})}.
$$
困难在于分母包含边缘似然 $p_{\boldsymbol{\phi}}(\mathbf{x})$，需要对所有潜变量积分，在非线性解码器下不可处理。因此，从 $\mathbf{x}$ 对 $\mathbf{z}$ 做精确推断在计算上不可行。

VAE 中的「变分」步骤用可处理的近似替代难以处理的后验。我们引入由神经网络参数化的**编码器**（推断网络）$q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})$，其作用是作为可学习的代理：
$$
q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x}) \;\approx\; p_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}).
$$
实践中，编码器将每个观测 $\mathbf{x}$ 映到潜码上的分布，提供从 $\mathbf{x}$ 回到 $\mathbf{z}$ 的可行且可训练的路径，从而支持学习。

### 1.1.2 通过证据下界（ELBO）的训练

我们给出可计算的训练目标。虽然无法直接优化 $\log p_{\boldsymbol{\phi}}(\mathbf{x})$，但可以最大化其下界——**证据下界（ELBO）**。

**定理（证据下界 ELBO）** 对任意数据点 $\mathbf{x}$，对数似然满足
$$
\log p_{\boldsymbol{\phi}}(\mathbf{x}) \geq \mathcal{L}_{\text{ELBO}}(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x}),
$$
其中 ELBO 为
$$
\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{\mathbf{z} \sim q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})} \left[ \log p_{\boldsymbol{\phi}}(\mathbf{x}|\mathbf{z}) \right]}_{\text{重建项}} - \underbrace{\mathcal{D}_{\mathrm{KL}}\left( q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}) \right)}_{\text{潜空间正则}}.
$$
该 ELBO 由 Jensen 不等式得到（将 $\log p_{\boldsymbol{\phi}}(\mathbf{x})$ 写成期望后应用 Jensen 不等式即可）。

ELBO 目标自然分解为两部分：**重建**——鼓励从潜码 $\mathbf{z}$ 准确恢复 $\mathbf{x}$；在高斯编码器与解码器假定下该项可化为自编码器熟悉的重建损失，但仅优化该项有记忆训练数据的风险。**潜 KL**——鼓励编码器分布 $q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})$ 接近简单高斯先验 $p_{\mathrm{prior}}(\mathbf{z})$，该正则将潜空间塑造成光滑连续的结构，从而通过保证从先验抽取的样本能被可靠解码而实现有意义的生成。这一折中同时保证忠实重建与连贯采样。

**表 1：VAE 中 KL 散度与重建之间的折中概览。**

|  | **编码器行为** | **KL 项** | **重建质量** | **生成质量** |
|--|----------------|-----------|--------------|--------------|
| 完美 KL | 后验 = 先验 | $\downarrow$ | 差 | 后验坍塌：潜变量无信息，样本模糊 |
| 完美重建 | 后验远离先验 | $\uparrow$ | 好 | 先验空洞问题：从 $p_{\text{prior}}$ 生成差 |

**信息论视角：ELBO 作为散度界。** 回忆最大似然训练等价于最小化 KL 散度 $\mathcal{D}_{\mathrm{KL}}(p_{\text{data}}(\mathbf{x}) \| p_{\boldsymbol{\phi}}(\mathbf{x}))$，它度量模型分布对数据分布的近似程度。由于该量一般难以处理，变分框架引入联合比较。考虑两个联合分布：**生成联合** $p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z}) = p(\mathbf{z}) p_{\boldsymbol{\phi}}(\mathbf{x}|\mathbf{z})$（描述模型如何生成数据）；**推断联合** $q_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) = p_{\text{data}}(\mathbf{x}) q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})$（将真实数据与其推断的潜变量耦合）。比较它们可得不等式 $\mathcal{D}_{\mathrm{KL}}(p_{\text{data}}(\mathbf{x}) \| p_{\boldsymbol{\phi}}(\mathbf{x})) \leq \mathcal{D}_{\mathrm{KL}}(q_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z}) \| p_{\boldsymbol{\phi}}(\mathbf{x},\mathbf{z}))$（有时称为 KL 的链式法则）。形式上可将联合 KL 展开为「真实建模误差」与「推断误差」之和；推断误差即近似后验与真实后验之间的差距，非负，从而说明上述不等式。并且 $\log p_{\boldsymbol{\phi}}(\mathbf{x}) - \mathcal{L}_{\text{ELBO}} = \mathcal{D}_{\mathrm{KL}}(q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})\|p_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}))$，因此推断误差正是对数似然与 ELBO 之间的差距；最大化 ELBO 即直接减小推断误差。

### 1.1.3 高斯 VAE

VAE 的一种标准形式对编码器与解码器均采用高斯分布。

**编码器部分。** 编码器 $q_{\boldsymbol{\theta}}(\mathbf{z} | \mathbf{x})$ 通常建模为高斯：
$$
q_{\boldsymbol{\theta}}(\mathbf{z} | \mathbf{x}) := \mathcal{N}\left(\mathbf{z}; \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}), \operatorname{diag}(\boldsymbol{\sigma}_{\boldsymbol{\theta}}^2(\mathbf{x}))\right),
$$
其中 $\boldsymbol{\mu}_{\boldsymbol{\theta}}$、$\boldsymbol{\sigma}_{\boldsymbol{\theta}}$ 为编码器网络的确定性输出。

**解码器部分。** 解码器通常建模为固定方差的高斯：
$$
p_{\boldsymbol{\phi}}(\mathbf{x}|\mathbf{z}) := \mathcal{N}\bigl(\mathbf{x}; \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{z}), \sigma^2 \mathbf{I}\bigr),
$$
其中 $\boldsymbol{\mu}_{\boldsymbol{\phi}}$ 为神经网络，$\sigma > 0$ 为控制方差的小常数。在此假定下，ELBO 中的重建项简化为 $-\frac{1}{2\sigma^2}\mathbb{E}_{q_{\boldsymbol{\theta}}(\mathbf{z} | \mathbf{x})}[\|\mathbf{x} - \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{z})\|^2] + C$，KL 项因高斯假定有闭式。因此 VAE 的训练归结为最小化正则化重建损失。

### 1.1.4 标准 VAE 的缺陷

尽管 VAE 框架在理论上吸引人，它有一个关键缺陷：生成的输出往往模糊。

**VAE 中生成模糊的原因。** 考虑固定的高斯编码器 $q_{\text{enc}}(\mathbf{z}|\mathbf{x})$ 与形如 $p_{\text{dec}}(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}(\mathbf{z}), \sigma^2 \mathbf{I})$ 的解码器。在任意编码器下，优化 ELBO 等价于（在常数意义下）最小化期望重建误差 $\mathbb{E}[\| \mathbf{x} - \boldsymbol{\mu}(\mathbf{z}) \|^2]$，其解由条件均值给出：$\boldsymbol{\mu}^*(\mathbf{z}) = \mathbb{E}_{q_{\text{enc}}(\mathbf{x}|\mathbf{z})}[\mathbf{x}]$（其中 $q_{\text{enc}}(\mathbf{x}|\mathbf{z})$ 为由编码器诱导的、给定潜变量下输入的后验）。若两个不同输入 $\mathbf{x} \neq \mathbf{x}'$ 被映到潜空间中重叠的区域（即 $q_{\text{enc}}(\cdot|\mathbf{x})$ 与 $q_{\text{enc}}(\cdot|\mathbf{x}')$ 的支撑相交），则 $\boldsymbol{\mu}^*(\mathbf{z})$ 会对多个可能不相关的输入取平均，导致模糊、不鲜明的输出。这种对冲突模态的平均效应是 VAE 生成样本典型模糊的根本原因之一。

### 1.1.5 （可选）从标准 VAE 到层次化 VAE

为建模复杂数据，**层次化变分自编码器（HVAE）**通过引入潜变量的层次结构增强 VAE。这种深的、分层结构使模型能在多个抽象层次上刻画数据特征，显著提升表达力并反映真实数据的组合性质。

![图 2：HVAE 的计算图](../arXiv-2510.21890v1/Images/PartB/hvae-graph.pdf)

**图 2：HVAE 的计算图。** 具有跨多个潜层堆叠的可训练编码器与解码器的层次结构。

**HVAE 的建模。** 与使用单一潜码 $\mathbf{z}$ 的标准 VAE 不同，HVAE 引入多层潜变量，呈自上而下的层次排列。每层潜变量条件于下一层，形成条件先验链，在越来越细的抽象层次上刻画结构。联合分布的自上而下分解为：
$$
p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z}_{1:L}) = p_{\boldsymbol{\phi}}(\mathbf{x} | \mathbf{z}_1) \prod_{i=2}^{L} p_{\boldsymbol{\phi}}(\mathbf{z}_{i-1}|\mathbf{z}_i)   p(\mathbf{z}_L).
$$
生成过程是逐步的：从顶层潜变量 $\mathbf{z}_L$ 开始，依次解码到 $\mathbf{z}_1$，再生成观测 $\mathbf{x}$。编码部分采用与生成层次镜像的、结构化的可学习变分编码器 $q_{\boldsymbol{\theta}}(\mathbf{z}_{1:L}|\mathbf{x})$，常用选择为自下而上的 Markov 分解：$q_{\boldsymbol{\theta}}(\mathbf{z}_{1:L} | \mathbf{x}) = q_{\boldsymbol{\theta}}(\mathbf{z}_1 | \mathbf{x}) \prod_{i=2}^{L} q_{\boldsymbol{\theta}}(\mathbf{z}_i | \mathbf{z}_{i-1})$。

**HVAE 的 ELBO。** 与 VAE 的 ELBO 类似，由 Jensen 不等式可得 $\log p_{\text{HVAE}}(\mathbf{x}) \geq \mathcal{L}_{\text{ELBO}}(\boldsymbol{\phi})$；代入因子化形式后，该层次 ELBO 可分解为可解释项（包括重建项以及各生成条件与对应变分近似之间的 KL 散度）。**观察** 堆叠层使模型能渐进地生成数据，从粗糙细节开始并在每一步添加更细的细节，从而更容易刻画高维数据的复杂结构。

**为何单纯加深平坦 VAE 不够？** 标准平坦 VAE 有两个根本局限，仅靠加深编码器与解码器无法解决。第一，变分族：标准 VAE 中 $q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})$ 为对角协方差的高斯，加深网络只提高 $\boldsymbol{\mu}_{\boldsymbol{\theta}}$ 和 $\boldsymbol{\sigma}_{\boldsymbol{\theta}}$ 的精度，并不扩大族；当真实后验 $p_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})$ 多峰时，单高斯族无法匹配。第二，若解码器过于富有表达力，可能出现**后验坍塌**：ELBO 可写成重建项减互信息再减 $q_{\boldsymbol{\theta}}(\mathbf{z})$ 与先验的 KL；若解码器类中已有不依赖 $\mathbf{z}$ 就能很好拟合数据的分布，则 ELBO 的最大元会使 $q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})=p(\mathbf{z})$，即互信息为零，学到的码与 $\mathbf{x}$ 无关，可控生成失败。单纯加深网络无法消除这一「忽略 $\mathbf{z}$」的解。

**层次带来了什么？** HVAE 引入多层潜变量，其 ELBO 中每个推断条件都与自上而下的生成对应项对齐（$q_{\boldsymbol{\theta}}(\mathbf{z}_1|\mathbf{x})$ 与 $p_{\boldsymbol{\phi}}(\mathbf{z}_1|\mathbf{z}_2)$，中间层与 $p_{\boldsymbol{\phi}}(\mathbf{z}_i|\mathbf{z}_{i+1})$，顶层与先验 $p(\mathbf{z}_L)$），从而将信息惩罚分散到各层并通过相邻 KL 项局部化学习信号。这些性质来自层次潜图结构，而非单纯加深平坦 VAE。

**后续方向。** HVAE 虽用多层潜变量扩展了 VAE 的表达力，但其训练面临独特挑战：编码器与解码器需联合优化，学习不稳定；深层变量得到的梯度往往间接且弱。有趣的是，这种深的层次思想在变分扩散模型中有更强体现：扩散模型继承 HVAE 的渐进结构，但通过固定编码过程、只学习生成逆转，巧妙地规避了 HVAE 的核心弱点，从而获得更好的稳定性与生成质量。

---

## 1.2 变分视角：DDPM

**去噪扩散概率模型（DDPM）**是扩散建模的基石。概念上它们在变分框架下运作，与 VAE 和 HVAE 类似，但 DDPM 通过一个巧妙改动应对了前者的部分挑战。

核心上，DDPM 包含两个不同的随机过程：**前向过程（固定编码器）**——通过转移核 $p(\mathbf{x}_{i}|\mathbf{x}_{i-1})$ 在多步中逐渐注入高斯噪声使数据损坏，最终演化为各向同性高斯（纯噪声）；编码器被固定、不学习。**逆向去噪过程（可学习解码器）**——神经网络通过参数化分布 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_{i})$ 学习逆转噪声损坏，从纯噪声出发逐步去噪生成逼真样本；每一步去噪是比 VAE 常做的「从零生成完整样本」更易处理的子任务。通过固定编码器并将学习集中在渐进生成轨迹上，DDPM 获得了显著的稳定性与表达力。

![图 3：DDPM 示意图](../arXiv-2510.21890v1/Images/PartB/vdm-graph.pdf)

**图 3：DDPM 示意图。** 由固定的前向过程（灰色）逐渐向数据加高斯噪声，以及学到的逆向过程逐步去噪以生成新样本。

### 1.2.1 前向过程（固定编码器）

在 DDPM 中，前向过程是固定的、不可训练的操作，充当编码器。它通过多步加噪逐渐破坏原始数据，最终将其变为简单先验 $p_{\text{prior}} := \mathcal{N}(\mathbf{0}, \mathbf{I})$。

![图 4：DDPM 前向过程](../arXiv-2510.21890v1/Images/PartB/vdm-forward.pdf)

**图 4：DDPM 前向过程示意图。** 高斯噪声被逐步加入，将数据样本破坏为纯噪声。

**固定高斯转移。** 前向过程的每一步由固定高斯转移核刻画（与原始 DDPM 转移核数学等价）：
$$
p(\mathbf{x}_i|\mathbf{x}_{i-1}) := \mathcal{N}(\mathbf{x}_i; \sqrt{1 - \beta_i^2}   \mathbf{x}_{i-1}, \beta_i^2 \mathbf{I}).
$$
过程从 $\mathbf{x}_0$ 开始（来自真实数据分布 $p_{\text{data}}$）。序列 $\{\beta_i\}_{i=1}^L$ 为预先给定的单调递增噪声调度，$\beta_i \in (0, 1)$ 控制第 $i$ 步注入的高斯噪声的方差。记 $\alpha_i := \sqrt{1 - \beta_i^2}$，则迭代更新为 $\mathbf{x}_i = \alpha_i \mathbf{x}_{i-1} + \beta_i \boldsymbol{\epsilon}_i$，$\boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 独立同分布。

**扰动核与先验分布。** 递归应用转移核可得给定原始数据 $\mathbf{x}_0$ 下第 $i$ 步带噪样本分布的闭式：
$$
p_i(\mathbf{x}_i|\mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_i; \bar{\alpha}_i \mathbf{x}_0, (1 - \bar{\alpha}_i^2) \mathbf{I} \right),\quad \bar{\alpha}_i := \prod_{k=1}^i \alpha_k.
$$
因此可直接从 $\mathbf{x}_0$ 采样 $\mathbf{x}_i$：
$$
\mathbf{x}_i = \bar{\alpha}_i \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_i^2}   \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$
当噪声调度 $\{\beta_i\}_{i=1}^L$ 递增时，前向过程的边缘分布收敛到 $\mathcal{N}(\mathbf{0}, \mathbf{I})$（$L \to \infty$），故取先验 $p_{\text{prior}} := \mathcal{N}(\mathbf{0}, \mathbf{I})$，且不依赖数据 $\mathbf{x}_0$。

### 1.2.2 逆向去噪过程（可学习解码器）

DDPM 的核心在于**逆转**前向扩散施加的受控退化。从纯噪声 $\mathbf{x}_L \sim p_{\text{prior}}$ 出发，目标是逐步**去噪**，直至得到连贯、有意义的数据样本。该逆向生成通过 Markov 链进行。

![图 5：DDPM 逆向（去噪）过程](../arXiv-2510.21890v1/Images/PartB/vdm-backward.pdf)

**图 5：DDPM 逆向（去噪）过程示意图。** 从噪声 $\mathbf{x}_L \sim p_{\mathrm{prior}}$ 出发，模型依次采样 $\mathbf{x}_{i-1} \sim p(\mathbf{x}_{i-1}|\mathbf{x}_i)$（$i=L,\ldots,1$）得到新生成数据 $\mathbf{x}$。真实转移 $p(\mathbf{x}_{i-1}|\mathbf{x}_i)$ 未知，故需近似。

核心问题变为：能否精确计算或有效近似这些逆向转移核 $p(\mathbf{x}_{i-1}|\mathbf{x}_i)$（尤其在 $\mathbf{x}_i \sim p_i(\mathbf{x}_i)$ 的分布复杂时）？

**概述：建模与训练目标。** 为使生成过程可行，目标是近似未知的真实逆向转移核 $p(\mathbf{x}_{i-1}|\mathbf{x}_i)$，通过引入可学习参数模型 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_i)$ 并最小化期望 KL 散度 $\mathbb{E}_{p_i(\mathbf{x}_i)}[\mathcal{D}_{\mathrm{KL}}(p(\mathbf{x}_{i-1}|\mathbf{x}_i) \| p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_i))]$。但目标分布 $p(\mathbf{x}_{i-1}|\mathbf{x}_i)$ 难以直接计算：由 Bayes 定理需计算 $p(\mathbf{x}_{i-1}|\mathbf{x}_i) = p(\mathbf{x}_i|\mathbf{x}_{i-1}) \frac{p_{i-1}(\mathbf{x}_{i-1})}{p_i(\mathbf{x}_i)}$，而边缘 $p_i(\mathbf{x}_i)$、$p_{i-1}(\mathbf{x}_{i-1})$ 是对未知 $p_{\mathrm{data}}$ 的期望，无闭式。

**通过条件化克服不可处理性。** DDPM 的关键思路是：对干净数据样本 $\mathbf{x}$ 做条件化。这样可将不可处理的核变为可处理的：
$$
p(\mathbf{x}_{i-1}|\mathbf{x}_i, \mathbf{x}) = p(\mathbf{x}_i|\mathbf{x}_{i-1}) \frac{p(\mathbf{x}_{i-1}|\mathbf{x})}{p(\mathbf{x}_i|\mathbf{x})}.
$$
可处理性来自前向过程的 Markov 性及所有涉及分布的高斯性，故 $p(\mathbf{x}_{i-1}|\mathbf{x}_i, \mathbf{x})$ 本身为高斯且有闭式。

**定理（边缘 KL 与条件 KL 最小化的等价性）** 以下等式成立：边缘 KL 期望等于条件 KL 期望（对 $p_{\mathrm{data}}(\mathbf{x})$ 与 $p(\mathbf{x}_i|\mathbf{x})$ 取期望）加与 $\boldsymbol{\phi}$ 无关的常数 $C$；且该条件 KL 的最小元满足 $p^*(\mathbf{x}_{i-1}|\mathbf{x}_i) = \mathbb{E}_{p(\mathbf{x}|\mathbf{x}_i)}[p(\mathbf{x}_{i-1}|\mathbf{x}_i, \mathbf{x})] = p(\mathbf{x}_{i-1}|\mathbf{x}_i)$。证明见附录。

**引理（逆向条件转移核）** $p(\mathbf{x}_{i-1} | \mathbf{x}_i, \mathbf{x})$ 为高斯，均值为 $\bm{\mu}(\mathbf{x}_i, \mathbf{x}, i):= \frac{\bar{\alpha}_{i-1}\beta_i^2}{1-\bar{\alpha}_i^2}\mathbf{x} + \frac{(1-\bar{\alpha}_{i-1}^2)\alpha_i }{1-\bar{\alpha}_i^2}\mathbf{x}_i$，方差 $\sigma^2(i) := \frac{1-\bar{\alpha}_{i-1}^2}{1-\bar{\alpha}_i^2}\beta_i^2$。

### 1.2.3 逆向转移核 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_i)$ 的建模

利用上述定理的等价性及逆向条件的高斯形式，DDPM 假定每个逆向转移 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}| \mathbf{x}_i)$ 为高斯，方差固定为 $\sigma^2(i)$，均值为可学习的 $\bm{\mu}_{\boldsymbol{\phi}}(\mathbf{x}_i, i)$。对数据 $\mathbf{x}_0 \sim p_{\text{data}}$ 条件化、对所有时间步 $i$ 求平均的扩散损失定义为
$$
\mathcal{L}_{\text{diffusion}}(\mathbf{x}_0;\boldsymbol{\phi}) := \sum_{i=1}^L \mathbb{E}_{p(\mathbf{x}_i|\mathbf{x}_0)}\left[\mathcal{D}_{\text{KL}}\big(p(\mathbf{x}_{i-1}|\mathbf{x}_i, \mathbf{x}_0)  \Vert  p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_i)\big)\right].
$$
因两者均为高斯且方差固定，该目标有闭式并可简化为均值匹配的加权平方误差；略去与 $\boldsymbol{\phi}$ 无关的常数后，得到 DDPM 训练目标 $\mathcal{L}_{\text{DDPM}}(\boldsymbol{\phi})$（对 $\mathbf{x}_0$ 与 $p(\mathbf{x}_i|\mathbf{x}_0)$ 取期望的加权 $\|\bm{\mu}_{\boldsymbol{\phi}}(\mathbf{x}_i, i) - \bm{\mu}(\mathbf{x}_i, \mathbf{x}_0, i)\|_2^2$）。

### 1.2.4 预测与损失的实际选择

**$\boldsymbol{\epsilon}$-预测。** 常见实现不直接使用基于均值预测的损失，而采用等价的**$\boldsymbol{\epsilon}$-预测**（噪声预测）参数化。前向中 $\mathbf{x}_i = \bar{\alpha}_i \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_i^2} \boldsymbol{\epsilon}$；逆向均值可改写为 $\bm{\mu}(\mathbf{x}_i, \mathbf{x}_0, i) = \frac{1}{\alpha_i}(\mathbf{x}_i - \frac{1 - \alpha_i^2}{\sqrt{1 - \bar{\alpha}_i^2}} \boldsymbol{\epsilon})$。因此用神经网络 $\boldsymbol{\epsilon}_{\boldsymbol{\phi}}(\mathbf{x}_i, i)$ 直接预测噪声，令模型均值为 $\bm{\mu}_{\boldsymbol{\phi}}(\mathbf{x}_i, i) = \frac{1}{\alpha_i}(\mathbf{x}_i - \frac{1 - \alpha_i^2}{\sqrt{1 - \bar{\alpha}_i^2}} \boldsymbol{\epsilon}_{\boldsymbol{\phi}}(\mathbf{x}_i, i))$，代入原损失得到预测噪声与真实噪声的 $\ell_2$ 误差（至多相差与 $i$ 有关的权重）。直观上，模型充当「噪声侦探」，估计前向每步加入的噪声；从带噪样本中减去该估计即更接近干净原图，逐步重复即可从纯噪声重建数据。

**带 $\boldsymbol{\epsilon}$-预测的简化损失。** 实践中常省略权重项，得到广泛使用的 DDPM 训练损失 $\mathcal{L}_{\text{simple}}(\boldsymbol{\phi})$：对 $i$、$\mathbf{x}_0 \sim p_{\text{data}}$、$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 取期望的 $\|\boldsymbol{\epsilon}_{\boldsymbol{\phi}}(\mathbf{x}_i, i) - \boldsymbol{\epsilon}\|_2^2$，其中 $\mathbf{x}_i = \bar{\alpha}_i \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_i^2} \boldsymbol{\epsilon}$。$\mathcal{L}_{\text{DDPM}}$ 与 $\mathcal{L}_{\text{simple}}$ 具有相同的最优解 $\boldsymbol{\epsilon}^*(\mathbf{x}_i, i) = \mathbb{E}[\boldsymbol{\epsilon}|\mathbf{x}_i]$。

**另一等价参数化：$\mathbf{x}$-预测。** 用网络 $\mathbf{x}_{\boldsymbol{\phi}}(\mathbf{x}_i, i)$ 预测干净样本，将逆向均值中的真实 $\mathbf{x}_0$ 替换为该预测可得等价的均值参数化；训练目标化为 $\|\mathbf{x}_{\boldsymbol{\phi}}(\mathbf{x}_i, i) - \mathbf{x}_0\|_2^2$ 的期望（可带权重 $\omega_i$）。最优解为 $\mathbf{x}^*(\mathbf{x}_i, i) = \mathbb{E}[\mathbf{x}_0|\mathbf{x}_i]$。$\mathbf{x}$-预测与 $\boldsymbol{\epsilon}$-预测通过前向关系等价：$\mathbf{x}_i = \bar{\alpha}_i \mathbf{x}_{\boldsymbol{\phi}}(\mathbf{x}_i, i) + \sqrt{1 - \bar{\alpha}_i^2} \boldsymbol{\epsilon}_{\boldsymbol{\phi}}(\mathbf{x}_i, i)$。

### 1.2.5 DDPM 的 ELBO

在逆向转移按上述方式定义时，DDPM 的联合生成分布为 $p_{\boldsymbol{\phi}}(\mathbf{x}_0, \mathbf{x}_{1:L}) := p_{\boldsymbol{\phi}}(\mathbf{x}_0|\mathbf{x}_1)\cdots p_{\boldsymbol{\phi}}(\mathbf{x}_{L-1}|\mathbf{x}_L) p_{\text{prior}}(\mathbf{x}_L)$，数据的边缘生成模型为 $p_{\boldsymbol{\phi}}(\mathbf{x}_0) = \int p_{\boldsymbol{\phi}}(\mathbf{x}_0, \mathbf{x}_{1:L}) \,\mathrm{d}\mathbf{x}_{1:L}$。DDPM 通过扩散损失的训练可严格置于最大似然估计框架下：其目标构成 ELBO，满足 $-\log p_{\boldsymbol{\phi}}(\mathbf{x}_0) \leq -\mathcal{L}_{\text{ELBO}}(\mathbf{x}_0; \boldsymbol{\phi})$，其中 $\mathcal{L}_{\text{ELBO}} = \mathcal{L}_{\text{prior}} + \mathcal{L}_{\text{recon.}} + \mathcal{L}_{\text{diffusion}}$。$\mathcal{L}_{\text{prior}}$ 可通过噪声调度使 $p(\cdot|\mathbf{x}_0) \approx p_{\text{prior}}(\cdot)$ 而变得可忽略；$\mathcal{L}_{\text{recon.}}$ 可用蒙特卡洛估计近似优化；$\mathcal{L}_{\text{diffusion}}$ 在各步将 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_i)$ 匹配到真实条件。从数据处理不等式角度，边缘 KL 不超过前向联合与生成联合之间的联合 KL。

**注** 扩散的变分视角符合 HVAE 模板：「编码器」是固定的前向加噪链，潜变量 $\mathbf{x}_{1:L}$ 与数据同维；训练仍是最大化同一 ELBO；没有学习的编码器，也没有逐层 KL，目标分解为从大到小噪声（由粗到细）的良条件去噪子问题，从而得到稳定优化与高样本质量。

### 1.2.6 采样

训练好 $\boldsymbol{\epsilon}$-预测模型 $\boldsymbol{\epsilon}_{\boldsymbol{\phi}^*}(\mathbf{x}_i, i)$ 后（$\boldsymbol{\phi}^*$ 表示已训练并冻结），采样按逆向链进行：从 $\mathbf{x}_L \sim p_{\text{prior}} = \mathcal{N}(\mathbf{0},\mathbf{I})$ 出发，对 $i = L, L-1, \dots, 1$ 按下列更新规则从 $p_{\boldsymbol{\phi}^*}(\mathbf{x}_{i-1} | \mathbf{x}_i)$ 递归采样：
$$
\mathbf{x}_{i-1} \leftarrow \frac{1}{\alpha_i}\left( \mathbf{x}_i  - \frac{1-\alpha_i^2}{\sqrt{1-\bar{\alpha}_i^2}} \boldsymbol{\epsilon}_{\boldsymbol{\phi}^*}(\mathbf{x}_i, i)\right) +\sigma(i) \boldsymbol{\epsilon}_i,\quad \boldsymbol{\epsilon}_i\sim  \mathcal{N}(\mathbf{0},\mathbf{I}).
$$
该「去噪」过程持续到得到 $\mathbf{x}_0$ 作为最终生成样本。

**DDPM 采样的另一解释。** 由前向关系，与噪声估计 $\boldsymbol{\epsilon}_{\boldsymbol{\phi}^*}(\mathbf{x}_i, i)$ 对应的干净样本预测为 $\mathbf{x}_{\boldsymbol{\phi}^*}(\mathbf{x}_i, i) = (\mathbf{x}_i - \sqrt{1 - \bar{\alpha}_i^2} \boldsymbol{\epsilon}_{\boldsymbol{\phi}^*}(\mathbf{x}_i, i))/\bar{\alpha}_i$。代入采样规则可知每步等价于「在 $\mathbf{x}_i$ 与干净预测 $\mathbf{x}_{\boldsymbol{\phi}^*}$ 之间的插值」加上尺度为 $\sigma(i)$ 的高斯噪声。因此 DDPM 采样可视为迭代去噪：先从当前带噪 $\mathbf{x}_i$ 估计干净数据 $\mathbf{x}_{\boldsymbol{\phi}^*}(\mathbf{x}_i, i)$，再用该估计通过更新规则采样更少噪声的 $\mathbf{x}_{i-1}$。

![图 6：DDPM 采样与干净预测](../arXiv-2510.21890v1/Images/PartB/diffusion-clean.pdf)

**图 6：DDPM 采样与干净预测示意图。** 从 $\mathbf{x}_i$ 估计 $\mathbf{x}_{\boldsymbol{\phi}^*}(\mathbf{x}_i, i)$，再更新到 $\mathbf{x}_{i-1}$。

即使 $\mathbf{x}_{\boldsymbol{\phi}^*}$ 作为最优去噪器（条件期望最小元）训练，它也只能预测给定 $\mathbf{x}_i$ 的**平均**干净样本，导致预测模糊，尤其在噪声水平高时。从该视角看，扩散采样通常从高噪声到低噪声，逐步细化对干净信号的估计：前期确定整体结构，后期添加细节，随噪声去除样本更逼真。

**DDPM 采样速度慢。** DDPM（即扩散模型）采样本质上较慢（通常需约 1,000 步去噪），因其逆向过程是序贯的，且受以下因素制约：理论上富有表达力的 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1} | \mathbf{x}_i)$ 可匹配真实逆向分布，但实践中常用单高斯近似，限制表达力；当前向噪声尺度 $\beta_i$ 较小时真实逆向近似高斯，反之大 $\beta_i$ 会带来多峰或强非高斯性，单高斯无法刻画；为保持精度 DDPM 采用大量小 $\beta_i$ 步，形成每步依赖前一步并需一次网络求值 $\boldsymbol{\epsilon}_{\boldsymbol{\phi}^*}(\mathbf{x}_i, i)$ 的链，导致 $\mathcal{O}(L)$ 次串行前向，无法并行。后续章节将从微分方程角度给出该采样瓶颈的更原则性解释，并引出连续时间数值策略以加速生成。

---

## 1.3 结语

本章从变分视角追溯了扩散模型的起源。我们从**变分自编码器（VAE）**出发——一种通过证据下界（ELBO）学习数据与结构化潜空间之间概率映射的基础生成模型；并看到**层次化 VAE（HVAE）**如何通过堆叠潜层扩展这一思想，引入由粗到细的渐进生成概念。然而这些模型在训练稳定性与样本质量上面临挑战。

随后我们将**去噪扩散概率模型（DDPM）**置于同一变分框架下，视为关键演进：通过将编码器固定为渐进加噪过程、仅学习逆向去噪步，DDPM 巧妙地规避了 HVAE 的训练不稳定。重要的是，我们说明了 DDPM 的训练同样是在最大化对数似然的变分界，其训练目标可分解为一系列简单去噪任务；这一可处理性得益于将难以处理的边缘目标转化为可处理条件目标的条件化策略，这也是扩散模型中反复出现的主题。

在提供 DDPM 的完整变分基础之外，下一章我们将从**基于能量的建模**出发，探索**基于分数的视角**：（1）从学习去噪转移概率 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_{i})$ 转向直接学习数据对数密度的梯度，即分数函数；（2）该思路源于 EBM，将引出噪声条件分数网络（NCSN），并揭示 DDPM 中学习的噪声预测（$\boldsymbol{\epsilon}$-预测）与分数函数之间的深刻数学等价。这一替代视角将为后续发展的扩散模型统一连续时间框架提供另一块基石。

---