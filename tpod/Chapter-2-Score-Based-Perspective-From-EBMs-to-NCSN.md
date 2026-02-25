# 第 2 章 基于分数的视角：从 EBM 到 NCSN

前几章我们将扩散模型追溯到其变分根源，并说明它们如何在 VAE 框架下产生。现在我们转向第二个同样根本的视角：**基于能量的模型（EBM）**。EBM 通过能量函数描述分布，能量在数据处低、在其他地方高。采样通常依赖 Langevin 动力学，即沿该能量景观的梯度将样本推向高密度区域。该梯度场称为**分数**，指向概率更高的方向。

核心观察是：已知分数就足以生成——它无需计算难以处理的归一化常数即可将样本移向高概率区域。**基于分数的**扩散模型直接建立在这一想法之上：它们不仅关注干净数据分布，还考虑一系列经高斯噪声扰动、其分数更易逼近的分布；学习这些分数得到一族向量场，逐步将带噪样本引导回数据，从而将生成转化为渐进去噪。

---

## 2.1 基于能量的模型

对已熟悉 EBM 的读者，本节作为简明复习以及通向扩散模型基于分数视角的桥梁。

### 2.1.1 用能量函数建模概率分布

记 $\mathbf{x} \in \mathbb{R}^D$ 为数据点。EBM 通过能量函数 $E_{\boldsymbol{\phi}}(\mathbf{x})$（由 ${\boldsymbol{\phi}}$ 参数化）定义概率密度，对更可能的结构赋予较低能量。所得分布为
$$
p_{{\boldsymbol{\phi}}}(\mathbf{x}) := \frac{\exp(-E_{\boldsymbol{\phi}}(\mathbf{x}))}{Z_\phi}, \quad 
Z_\phi := \int_{\mathbb{R}^D} \exp(-E_{\boldsymbol{\phi}}(\mathbf{x})) \,\mathrm{d}\mathbf{x},
$$
其中 $Z_\phi$ 称为**配分函数**，保证归一化：$\int_{\mathbb{R}^D} p_{{\boldsymbol{\phi}}}(\mathbf{x})  \,\mathrm{d}\mathbf{x} = 1$。

![图 1：EBM 训练示意图](../arXiv-2510.21890v1/Images/PartB/ebm-graph.pdf)

**图 1：EBM 训练示意图。** 模型在「坏」数据点处降低密度（提高能量）（红箭头），在「好」数据点处提高密度（降低能量）（绿箭头）。

在这一视角下，能量低的点对应概率高，如同球滚入谷底。配分函数 $Z_{\phi}$ 保证所有概率之和为 1，因此只有能量的**相对**值重要；例如对所有能量加常数会使分子分母同乘一因子，分布不变。此外，由于 $Z_{\phi}$ 强制概率和为 1，数学上可知：降低某区域的能量会提高该区域概率，而其补集的概率相应下降。因此 EBM 服从严格的全局折中：使一个谷更深必然使其他谷更浅，概率质量在整个空间上重新分配，而非独立分配给各区域。

**EBM 中最大似然训练的难点。** 原则上 EBM 可用最大似然训练，自然地在拟合数据与全局正则之间取得平衡（见前文 MLE 目标）：
$$
\mathcal{L}_{\text{MLE}}(\phi) 
= \mathbb{E}_{p_{\text{data}}(\mathbf{x})}  \left[ \log \frac{\exp(-E_{\phi}(\mathbf{x}))}{Z_\phi} \right] 
= -  \underbrace{\mathbb{E}_{p_{\text{data}}}[E_{\phi}(\mathbf{x})]}_{\text{降低数据能量}}
   -  \underbrace{\log \int \exp(-E_{\phi}(\mathbf{x}))\mathrm{d}\mathbf{x}}_{\text{全局正则}},
$$
其中 $Z_\phi = \int \exp(-E_\phi(\mathbf{x}))\mathrm{d}\mathbf{x}$。第一项降低真实数据的能量，第二项通过配分函数施加归一化。然而在高维下计算 $\log Z_{\phi}$ 及其梯度难以处理，因为需要模型分布下的期望。这促使采用替代目标：或近似该项（如对比散度），或通过**分数匹配**完全避开它。下面我们先在 2.1.2 节引入分数函数的概念，在 2.1.3 节给出分数匹配作为可处理的训练目标（绕过配分函数），然后在 2.1.4 节讨论用分数函数进行 Langevin 采样的实用方法。

### 2.1.2 动机：什么是分数？

对 $\mathbb{R}^D$ 上的密度 $p(\mathbf{x})$，**分数函数**定义为对数密度的梯度：
$$
\mathbf{s}(\mathbf{x}) := \nabla_{\mathbf{x}} \log p(\mathbf{x}), \qquad \mathbf{s} \colon \mathbb{R}^D \to \mathbb{R}^D.
$$
直观上，分数构成一个向量场，指向概率更高的区域，为数据最可能出现的位置提供局部指引（见图 2）。

![图 2：分数向量场示意图](../arXiv-2510.21890v1/Images/PartB/score_function_plot.pdf)

**图 2：分数向量场示意图。** 分数向量场 $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ 指示密度增大的方向。

**为何建模分数而非密度？** 建模分数在理论与实用上都有好处。

**1. 与归一化常数无关。** 许多分布仅由未归一化密度 $\tilde{p}(\mathbf{x})$ 定义，例如 EBM 中的 $\exp(-E_{\boldsymbol{\phi}}(\mathbf{x}))$：
$$
p(\mathbf{x}) = \frac{\tilde{p}(\mathbf{x})}{Z}, 
\qquad 
Z = \int \tilde{p}(\mathbf{x})\,\mathrm{d}\mathbf{x}.
$$
计算 $Z$ 难以处理，但分数只依赖 $\tilde{p}$：
$$
\nabla_{\mathbf{x}}\log p(\mathbf{x}) = \nabla_{\mathbf{x}}\log \tilde{p}(\mathbf{x}) - \underbrace{\nabla_{\mathbf{x}}\log Z}_{=0} = \nabla_{\mathbf{x}}\log \tilde{p}(\mathbf{x}),
$$
因为 $Z$ 与 $\mathbf{x}$ 无关。从而完全绕开配分函数。

**2. 完整表示。** 分数函数完全刻画了底层分布。由于它是对数密度的梯度，密度可由分数（差一常数）恢复：
$$
\log p(\mathbf{x}) 
= \log p(\mathbf{x}_0) + \int_0^1 \mathbf{s}(\mathbf{x}_0 + t(\mathbf{x} - \mathbf{x}_0))^\top (\mathbf{x} - \mathbf{x}_0)  \,\mathrm{d} t,
$$
其中 $\mathbf{x}_0$ 为参考点，$\log p(\mathbf{x}_0)$ 由归一化确定。因此建模分数与建模 $p(\mathbf{x})$ 本身同样富有表达力，且在生成建模中往往更易处理。

### 2.1.3 通过分数匹配训练 EBM

在 EBM 中，密度定义为 $p_{{\boldsymbol{\phi}}}(\mathbf{x}) = \tfrac{\exp(-E_{\boldsymbol{\phi}}(\mathbf{x}))}{Z_\phi}$。最大似然训练需要计算 $Z_{\boldsymbol{\phi}}$，通常难以处理。关键观察是：模型 $p_{{\boldsymbol{\phi}}}$ 的分数可简化为 $-\nabla_{\mathbf{x}}E_{\boldsymbol{\phi}}(\mathbf{x})$，与 $Z_{\boldsymbol{\phi}}$ 无关（见上式）。

**分数匹配**利用分数只依赖能量函数这一事实：不拟合归一化概率，而是通过将模型分数与（未知的）数据分数对齐来训练 EBM：
$$
\mathcal{L}_{\mathrm{SM}}(\boldsymbol{\phi}) 
= \tfrac{1}{2}  \mathbb{E}_{p_{\mathrm{data}}(\mathbf{x})} 
\big\| \nabla_{\mathbf{x}} \log p_{\boldsymbol{\phi}}(\mathbf{x}) 
- \nabla_{\mathbf{x}} \log p_{\mathrm{data}}(\mathbf{x}) \big\|_2^2.
$$

尽管数据分数不可得，分部积分可得到仅含能量及其导数的等价形式（详见附录命题）：
$$
\mathcal{L}_{\mathrm{SM}}(\boldsymbol{\phi}) =
\mathbb{E}_{p_{\mathrm{data}}(\mathbf{x})} \left[
\operatorname{Tr} \big(\nabla_{\mathbf{x}}^2 E_{\boldsymbol{\phi}}(\mathbf{x})\big)
+ \tfrac{1}{2}  \|\nabla_{\mathbf{x}} E_{\boldsymbol{\phi}}(\mathbf{x})\|_2^2
\right] + C,
$$
其中 $\nabla_{\mathbf{x}}^2 E_{\boldsymbol{\phi}}(\mathbf{x})$ 为 $E_{\boldsymbol{\phi}}$ 的 Hessian，$C$ 为与 $\boldsymbol{\phi}$ 无关的常数。

该形式有吸引力：消除了配分函数，且训练时无需从模型采样。主要缺点是需要二阶导数，在高维下计算代价可能很高。本章后面会重访缓解这一限制的方法。

### 2.1.4 用分数函数的 Langevin 采样

从由能量函数 $E_{\boldsymbol{\phi}}(\mathbf{x})$ 定义的 EBM 采样可使用 **Langevin 动力学**。下面先给出离散时间 Langevin 更新，再给出其连续时间极限（随机微分方程，SDE），最后讨论 Langevin 如何有效探索复杂能量景观的直观。

![图 3：Langevin 采样示意图](../arXiv-2510.21890v1/Images/PartB/langevin_sampling.png)

**图 3：Langevin 采样示意图。** 使用分数函数 $\nabla_{\mathbf{x}} \log p_{\boldsymbol{\phi}}(\mathbf{x})$ 通过上文的离散 Langevin 更新将轨迹引向高密度区域（箭头所示）。

**离散时间 Langevin 动力学。** 离散时间 Langevin 更新为
$$
\mathbf{x}_{n+1} = \mathbf{x}_n - \eta \nabla_{\mathbf{x}} E_{\boldsymbol{\phi}}(\mathbf{x}_n) +  \sqrt{2 \eta} \boldsymbol{\epsilon}_n, \quad n=0,1,2,\ldots,
$$
其中 $\mathbf{x}_0$ 从某分布（常取高斯）初始化，$\eta > 0$ 为步长，$\boldsymbol{\epsilon}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 为高斯噪声。噪声通过随机性使探索不局限于局部极小。由于分数可写为 $\nabla_{\mathbf{x}} \log p_{\boldsymbol{\phi}}(\mathbf{x}) = -\nabla_{\mathbf{x}} E_{\boldsymbol{\phi}}(\mathbf{x})$，更新可等价地写为
$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \eta \nabla_{\mathbf{x}} \log p_{\boldsymbol{\phi}}(\mathbf{x}_n) + \sqrt{2 \eta}\boldsymbol{\epsilon}_n,
$$
分数函数将样本引向高密度区域。该形式对扩散模型至关重要，后文会详述。

**连续时间 Langevin 动力学。** 当步长 $\eta \to 0$ 时，离散 Langevin 更新自然收敛到由 **Langevin 随机微分方程（SDE）** 描述的连续时间过程：
$$
\mathrm{d} \mathbf{x}(t) = \nabla_{\mathbf{x}} \log p_{\boldsymbol{\phi}}(\mathbf{x}(t)) \,\mathrm{d} t + \sqrt{2} \,\mathrm{d} \mathbf{w}(t),
$$
其中 $\mathbf{w}(t)$ 为标准布朗运动（又称 Wiener 过程）。离散更新规则（上式） 即该连续 SDE 的 Euler–Maruyama 离散化。

在标准正则性假设下（例如 $p_{\boldsymbol{\phi}}\propto e^{-E_{\boldsymbol{\phi}}}$ 且 $E_{\boldsymbol{\phi}}$ 具有约束性、足够光滑），$\mathbf{x}(t)$ 的分布随 $t\to\infty$（指数）收敛到 $p_{\boldsymbol{\phi}}$；因此可通过模拟（求解）该 SDE 进行采样。

**为何用 Langevin 采样？** 从物理角度理解：能量函数 $E_{\boldsymbol{\phi}}(\mathbf{x})$ 定义势能景观，决定粒子的行为。根据牛顿力学，粒子在该能量导出的力场下的运动由常微分方程（ODE）
$$
\mathrm{d} \mathbf{x}(t) = -\nabla_{\mathbf{x}} E_{\boldsymbol{\phi}}\big(\mathbf{x}(t)\big)  \,\mathrm{d} t
$$
描述，确定性地下坡走向能量函数的局部极小。但这种确定性动力学会陷入局部极小，无法探索完整数据分布。Langevin 动力学引入随机扰动，得到 SDE
$$
\mathrm{d} \mathbf{x}(t) = -\nabla_{\mathbf{x}} E_{\boldsymbol{\phi}}\big(\mathbf{x}(t)\big)  \,\mathrm{d} t + \underbrace{\sqrt{2} \,\mathrm{d} \mathbf{w}(t)}_{\text{注入的噪声}},
$$
其中 $\mathbf{w}(t)$ 为标准布朗运动。噪声项使粒子能够越过能垒逃离局部极小，轨迹成为平稳分布收敛于 Boltzmann 分布 $p_{\boldsymbol{\phi}}(\mathbf{x}) \propto e^{-E_{\boldsymbol{\phi}}(\mathbf{x})}$ 的随机过程。因此 EBM 可视为学习一个将样本推向高概率区域的力场；Langevin 采样对 EBM 特别有用，因为它提供了从模型分布 $p_{\boldsymbol{\phi}}(\mathbf{x})$ 生成样本的实用方法，而无需显式计算配分函数。

**Langevin 采样的固有难点。** Langevin 动力学作为一种广泛使用的基于 MCMC 的采样器，在高维空间中面临严重局限。其效率对步长 $\eta$、噪声尺度以及逼近目标分布所需的迭代次数高度敏感。低效的核心在于「混合时间」差：在具有多个孤立模态的复杂数据分布上，Langevin 采样往往需要极长时间才能在高概率区域之间转移；该问题随维度增加而显著恶化，导致收敛过慢。可将采样想象成在广阔崎岖、多谷的景观中探索，每个谷对应不同数据模态；Langevin 依赖局部随机更新，难以在谷间高效穿越，因此常无法捕捉分布的完整多样性。这种低效暗示需要更结构化、有引导的采样方法，以比纯随机探索更有效地穿越复杂数据流形。

---

## 2.2 从基于能量到基于分数的生成模型

EBM 表明生成只依赖分数（指向更高概率区域），而不依赖完整的归一化密度。分数匹配虽绕开了配分函数，但通过能量训练仍需要昂贵的二阶导数。关键想法是：既然用 Langevin 动力学采样只需要分数，我们可以用神经网络直接学习它。从建模能量转向建模分数，构成了基于分数生成模型的基础。

![图 4：分数匹配示意图](../arXiv-2510.21890v1/Images/PartB/score_function_approximation.pdf)

**图 4：分数匹配示意图。** 神经网络分数 $\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})$ 通过 MSE 损失训练以匹配真实分数 $\mathbf{s}(\mathbf{x})$，二者均表示为向量场。

### 2.2.1 用分数匹配训练

**分数匹配。** 为从 $p_{\mathrm{data}}$ 的样本逼近分数函数 $\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p_{\mathrm{data}}(\mathbf{x})$，我们直接用神经网络参数化的向量场 $\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})$ 来逼近（见图 4）：
$$
\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}) \approx \mathbf{s}(\mathbf{x}).
$$
**分数匹配**通过最小化真实分数与估计分数之间的均方误差（MSE）来拟合该向量场：
$$
\mathcal{L}_{\mathrm{SM}}(\boldsymbol{\phi}) 
:= \frac{1}{2}  \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}}
\Big[\| \mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}) - \mathbf{s}(\mathbf{x})\|_2^2\Big].
$$

**可处理的分数匹配。** 乍看该目标不可行，因为作为回归目标的真实分数 $\mathbf{s}(\mathbf{x})$ 未知。Hyvärinen 表明，分部积分可得到仅依赖模型 $\mathbf{s}_{\boldsymbol{\phi}}$ 与数据样本的等价目标，无需真实分数。该结果可表述为：

**命题（Hyvärinen 的分数匹配可处理形式）** 有
$$
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}) = \widetilde{\mathcal{L}}_{\text{SM}}(\boldsymbol{\phi}) + C,
$$
其中
$$
\widetilde{\mathcal{L}}_{\text{SM}}(\boldsymbol{\phi}) := \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})} \left[\operatorname{Tr}\left(\nabla_{\mathbf{x}}\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})\right) + \frac{1}{2} \| \mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})\|_2^2 \right],
$$
$C$ 为与 $\boldsymbol{\phi}$ 无关的常数。最小元 $\mathbf{s}^*$ 满足 $\mathbf{s}^*(\cdot) = \nabla_{\mathbf{x}} \log p(\cdot)$。证明见附录。

使用上述等价目标，我们仅从 $p_{\mathrm{data}}$ 的观测样本训练分数模型 $\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})$，无需真实分数函数。

**式 $\widetilde{\mathcal{L}}_{\text{SM}}$ 的直观。** 替代分数匹配目标 $\widetilde{\mathcal{L}}_{\text{SM}}(\boldsymbol{\phi})$ 的两项可直接理解：范数项 $\tfrac12\|\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})\|^2$ 在 $p_{\mathrm{data}}$ 大的区域抑制分数，使其成为驻点；散度项 $\operatorname{Tr}(\nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}))$ 倾向于负值，故这些驻点成为吸引子。两者共同将高密度区域塑成分数场的稳定收缩点。

**由模长项得到的驻点性。** 因 $\widetilde{\mathcal{L}}_{\mathrm{SM}}(\boldsymbol{\phi})$ 的期望在 $p_{\mathrm{data}}$ 下取，$p_{\mathrm{data}}(\mathbf{x})$ 大的区域对损失贡献最大。模长项 $\tfrac12\|\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})\|^2$ 因此在这些高概率区域驱动 $\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})\to 0$，即这些位置成为**驻点**。

**当场（近似）为梯度时的凹性。** 散度项 $\operatorname{Tr}(\nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}))$ 鼓励向量场在高数据密度区域具有负散度。负散度表示邻近向量收缩而非扩散，故该区域内的驻点成为**汇**：邻近轨迹被向内吸引。精确地说，若 $\mathbf{s}_{\boldsymbol{\phi}}=\nabla_{\mathbf{x}}u$（$u:\mathbb{R}^D\to\mathbb{R}$ 为标量函数，在匹配对数密度时自然），则 $\nabla_{\mathbf{x}}\mathbf{s}_{\boldsymbol{\phi}}=\nabla_{\mathbf{x}}^2 u$（Hessian），$\nabla \cdot \mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})=\operatorname{Tr}(\nabla_{\mathbf{x}}^2 u(\mathbf{x}))$（散度）。在驻点 $\mathbf{x}_*$（$\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}_*)=\nabla_{\mathbf{x}}u(\mathbf{x}_*)=\mathbf{0}$），二阶 Taylor 展开给出 $u(\mathbf{x}) = u(\mathbf{x}_*) + \tfrac12(\mathbf{x}-\mathbf{x}_*)^\top \nabla_{\mathbf{x}}^2 u(\mathbf{x}_*)(\mathbf{x}-\mathbf{x}_*) + o(\|\mathbf{x}-\mathbf{x}_*\|^2)$。若 Hessian $\nabla_{\mathbf{x}}^2 u(\mathbf{x}_*)$ 负定，则 $u$ 在 $\mathbf{x}_*$ 处局部凹，对数密度在该处取严格局部极大；Hessian 所有特征值为负故迹也为负，$\operatorname{Tr}(\nabla_{\mathbf{x}}^2 u(\mathbf{x}_*))<0$，学到的向量场具有负散度，驻点为**汇**：小扰动被收缩回 $\mathbf{x}_*$。

### 2.2.2 用 Langevin 动力学采样

最小化上式训练好后，分数模型 $\mathbf{s}_{\boldsymbol{\phi}^*}(\mathbf{x})$（$\boldsymbol{\phi}^*$ 表示已训练）可在 Langevin 动力学中替代真实分数进行采样：
$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \eta   \mathbf{s}_{\boldsymbol{\phi}^*}(\mathbf{x}_n) + \sqrt{2\eta}    \boldsymbol{\epsilon}_n, \quad \boldsymbol{\epsilon}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
$$
$n=0,1,2,\dots$，从 $\mathbf{x}_0$ 初始化。与 EBM 情形类似，该递推正是连续时间 Langevin SDE 的 Euler–Maruyama 离散化：
$$
\mathrm{d} \mathbf{x}(t) = \mathbf{s}_{\boldsymbol{\phi}^*}(\mathbf{x}(t)) \,\mathrm{d} t + \sqrt{2} \,\mathrm{d} \mathbf{w}(t),
$$
从 $\mathbf{x}(0)$ 初始化。因此在小步长极限下离散与连续形式一致。实践中可运行离散采样器或直接模拟该 SDE。

### 2.2.3 引言：基于分数的生成模型

本章余下部分考察分数函数在现代扩散模型中的基础作用。分数函数最初为 EBM 的高效训练而引入，现已演化为新一代生成模型的核心组件。在此基础上，我们探讨分数函数如何影响**基于分数的扩散模型**的理论表述与实现，为通过随机过程进行数据生成提供原则性框架。

---

## 2.3 去噪分数匹配（DSM）

### 2.3.1 动机

尽管替代目标
$$
\widetilde{\mathcal{L}}_{\text{SM}}(\boldsymbol{\phi}) = \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}} \left[\mathrm{Tr}\big(\nabla_{\mathbf{x}}\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})\big) + \frac{1}{2} \|\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})\|_2^2 \right]
$$
更易处理，但仍需计算 Jacobian 的迹 $\mathrm{Tr}(\nabla_{\mathbf{x}}\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}))$，最坏复杂度为 $\mathcal{O}(D^2)$，限制了在高维数据上的可扩展性。

为解决此问题，切片分数匹配用基于随机投影的随机估计替代迹项。下面简述其思想。

**切片分数匹配与 Hutchinson 估计。** 切片分数匹配用沿随机「切片」方向的方向导数平均替代分数匹配中的迹。设 $\mathbf{u}\in\mathbb{R}^D$ 为**各向同性**随机向量（如 Rademacher 或标准高斯），$\mathbb{E}[\mathbf{u}]=0$，$\mathbb{E}[\mathbf{u}\mathbf{u}^\top]=\mathbf{I}$。由 Hutchinson 恒等式
$$
\operatorname{Tr}(\mathbf{A})=\mathbb{E}_{\mathbf{u}}[\mathbf{u}^\top \mathbf{A} \mathbf{u}],\quad\text{且}\quad\mathbb{E}_{\mathbf{u}}[(\mathbf{u}^\top\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}))^2]=\|\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})\|_2^2,
$$
得到精确形式
$$
\widetilde{\mathcal{L}}_{\mathrm{SM}}(\boldsymbol{\phi})
=\mathbb{E}_{\mathbf{x},\mathbf{u}} \Big[\mathbf{u}^\top\big(\nabla_{\mathbf{x}}\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})\big)\mathbf{u}+\tfrac12(\mathbf{u}^\top\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}))^2\Big].
$$
该目标可用自动微分高效计算，使用 Jacobian-向量积（JVP/VJP）操作而非显式构造大 Jacobian 或 Hessian。对 $K$ 个随机探测取平均得到方差 $\mathcal{O}(1/K)$ 的无偏估计；方向项 $\mathbf{u}^\top(\nabla_{\mathbf{x}}\mathbf{s}_{\boldsymbol{\phi}})\mathbf{u}$ 可用 JVP/VJP 高效计算而无需显式 Jacobian。直观上，我们只沿随机方向检查模型行为：投影分数被推向更高数据密度区域，从而数据点在期望意义下成为驻点。

**从切片到去噪分数匹配。** 切片分数匹配绕开了 Jacobian，但仍依赖原始数据分布。对位于低维流形上的图像数据，分数 $\nabla_{\mathbf{x}}\log p_{\mathrm{data}}(\mathbf{x})$ 可能未定义或不稳定，且方法仅在观测点处约束向量场，在邻域内控制较弱；还受探测引起的方差和重复 JVP/VJP 成本影响。更稳健的替代是**去噪分数匹配（DSM）**，提供原则性且可扩展的解决方案。

### 2.3.2 训练

回到分数匹配损失：
$$
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}) =  \frac{1}{2} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})} \left[ \|\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})\|_2^2 \right],
$$
困难来自难以处理的项 $\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$。

**Vincent 等人通过条件化的解法。** 为克服 $\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$ 的不可处理性，Vincent 等人提出通过已知条件分布 $p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})$（尺度 $\sigma$）向数据 $\mathbf{x} \sim p_{\text{data}}$ 注入噪声。神经网络 $\mathbf{s}_{\boldsymbol{\phi}}(\tilde{\mathbf{x}}; \sigma)$ 训练为逼近扰动边缘分布的分数
$$
p_\sigma(\tilde{\mathbf{x}}) = \int p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) p_{\text{data}}(\mathbf{x}) \,\mathrm{d} \mathbf{x}
$$
，通过最小化损失
$$
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}; \sigma) := \frac{1}{2} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma} \left[ \|\mathbf{s}_{\boldsymbol{\phi}}(\tilde{\mathbf{x}}; \sigma) - \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})\|_2^2 \right].
$$
尽管 $\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$ 一般难以处理，Vincent 等人证明对 $\mathbf{x} \sim p_{\text{data}}$ **条件化**可得到等价的、可处理的目标——**去噪分数匹配（DSM）**损失：
$$
\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma) := \frac{1}{2} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, \tilde{\mathbf{x}} \sim p_{\sigma}(\cdot|\mathbf{x})} \left[ \|\mathbf{s}_{\boldsymbol{\phi}}(\tilde{\mathbf{x}}; \sigma) - \nabla_{\tilde{\mathbf{x}}} \log p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})\|_2^2 \right].
$$

损失 $\mathcal{L}_{\text{DSM}}$ 的最优最小元 $\mathbf{s}^*$ 满足 $\mathbf{s}^*(\tilde{\mathbf{x}}; \sigma) = \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$，同时也是 $\mathcal{L}_{\text{SM}}$ 噪声形式的最优解。

例如当 $p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})$ 为方差 $\sigma^2$ 的高斯噪声时，$p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 \mathbf{I})$，梯度 $\nabla_{\tilde{\mathbf{x}}} \log p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})$ 有闭式，使回归目标显式且可计算。此外当 $\sigma \approx 0$ 时，$p_\sigma(\tilde{\mathbf{x}}) \approx p_{\text{data}}(\mathbf{x})$ 且 $\mathbf{s}^*(\tilde{\mathbf{x}}; \sigma) = \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}}) \approx \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$，表明学到的分数逼近原始数据分数，可用于生成。

**定理（$\mathcal{L}_{\text{SM}}$ 与 $\mathcal{L}_{\text{DSM}}$ 的等价性）** 对任意固定噪声尺度 $\sigma > 0$，有 $\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}; \sigma) = \mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma) + C$，其中 $C$ 与 $\boldsymbol{\phi}$ 无关。且两损失的最小元 $\mathbf{s}^*(\cdot; \sigma)$ 满足 $\mathbf{s}^*(\tilde{\mathbf{x}}; \sigma) = \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$（对几乎处处 $\tilde{\mathbf{x}}$）。等价性由展开 MSE 后直接计算得到；与 DDPM 中的条件化技巧一致：对数据点 $\mathbf{x}$ 条件化将不可处理的损失变为可 Monte Carlo 估计的可处理损失。

![图 5：通过条件化技术的 DSM 示意图](../arXiv-2510.21890v1/Images/PartB/dsm-graph.pdf)

**图 5：通过条件化技术的 DSM 示意图。** 用小的加性高斯噪声 $\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ 扰动数据分布 $p_{\mathrm{data}}$，所得条件分布 $p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 \mathbf{I})$ 具有闭式分数函数。

**特例：加性高斯噪声。** 考虑常见情形：向每个数据点 $\mathbf{x} \sim p_{\text{data}}$ 加方差 $\sigma^2$ 的高斯噪声 $\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$：
$$
\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
$$
故污染数据 $\tilde{\mathbf{x}}$ 服从 $p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 \mathbf{I})$。此时条件分数解析地为
$$
\nabla_{\tilde{\mathbf{x}}} \log p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) = \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2}.
$$
因此 DSM 损失简化为：
$$
\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma) = \frac{1}{2} \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}|\mathbf{x}} \left[ \|\mathbf{s}_{\boldsymbol{\phi}}(\tilde{\mathbf{x}}; \sigma) - \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2}\|_2^2 \right] = \frac{1}{2} \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}} \left[ \|\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x} + \sigma \boldsymbol{\epsilon}; \sigma) + \frac{\boldsymbol{\epsilon}}{\sigma}\|_2^2 \right],
$$
其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。该目标构成（基于分数的）扩散模型的核心。

当噪声水平 $\sigma$ 较小时，高斯平滑边缘 $p_\sigma = p_{\mathrm{data}} * \mathcal{N}(\mathbf{0},\sigma^2 \mathbf{I})$，高密度区域与分数几乎一致；沿带噪分数方向小步移动会将带噪样本移向干净分布的同一高似然区域。反之当 $\sigma$ 较大时，平滑会「过度简化」景观：$p_\sigma$ 抹平局部模态，其分数主要拉向全局质量（可理解为向均值收缩），导致粗糙去噪甚至过度平滑。实践中 DSM 通常假设注入噪声小而温和。

### 2.3.3 采样

在噪声水平 $\sigma$ 下得到训练好的分数模型 $\mathbf{s}_{\boldsymbol{\phi}^*}(\tilde{\mathbf{x}}; \sigma)$ 后，用学到的模型替代真实分数、通过 Langevin 动力学生成样本。更新规则为：
$$
\tilde{\mathbf{x}}_{n+1} = \tilde{\mathbf{x}}_n + \eta \,\mathbf{s}_{\boldsymbol{\phi}^*}(\tilde{\mathbf{x}}_n; \sigma) + \sqrt{2\eta}    \boldsymbol{\epsilon}_n, \quad \boldsymbol{\epsilon}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
$$
$n = 0, 1, 2, \dots$，从初值 $\tilde{\mathbf{x}}_0$ 开始。若 $\sigma$ 足够小且迭代足够多，$\tilde{\mathbf{x}}_n$ 近似 $p_{\text{data}}$ 的样本。

**注入噪声的好处。** 与原始分数匹配相比，注入高斯噪声形成 $p_\sigma$（如上式）带来两点好处：**梯度定义良好**——噪声将数据扰动到低维流形之外，得到支撑为全 $\mathbb{R}^D$ 的 $p_\sigma$，故分数 $\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$ 处处有定义；**覆盖改善**——噪声平滑模态间的稀疏区域，增强训练信号质量，便于 Langevin 更有效穿越低密度区域。

### 2.3.4 为何 DSM 是去噪：Tweedie 公式

我们从 **Tweedie 公式**出发，它仅从带噪观测出发为去噪提供了原则性基础。具体地：给定来自未知 $\mathbf{x} \sim p_{\mathrm{data}}$ 的单一高斯污染观测 $\tilde{\mathbf{x}}\sim\mathcal N(\,\cdot\,;\alpha\mathbf{x},\sigma^2\mathrm{I})$，去噪估计（给定 $\tilde{\mathbf{x}}$ 下所有可能干净信号的平均）可通过在带噪边缘的分数 $\nabla_{\tilde{\mathbf{x}}}\log p_\sigma(\tilde{\mathbf{x}})$ 方向上、以步长 $\sigma^2$ 推动 $\tilde{\mathbf{x}}$ 得到，其中
$$
p_\sigma(\tilde{\mathbf{x}}):= \int \mathcal{N}(\tilde{\mathbf{x}};\alpha \mathbf{x}_0,\sigma^2\mathrm{I})p_{\mathrm{data}}(\mathbf{x})\mathrm{d}\mathbf{x}.
$$

**引理（Tweedie 公式）** 设 $\mathbf{x}\sim p_{\mathrm{data}}$，且在 $\mathbf{x}$ 条件下 $\tilde{\mathbf{x}}\sim\mathcal N(\,\cdot\,;\alpha\mathbf{x},\sigma^2\mathrm{I})$，$\alpha\neq0$。则 Tweedie 公式为
$$
\alpha   \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x} |\tilde{\mathbf{x}})}\big[\mathbf{x}  \big|  \tilde{\mathbf{x}}\big] = \tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}}),
$$
其中期望对给定 $\tilde{\mathbf{x}}$ 的后验 $p(\mathbf{x} |\tilde{\mathbf{x}})$ 取。证明见附录。

Tweedie 公式在扩散模型中起核心作用（如 DDPM 中引入多层噪声）。它使我们可以通过分数函数从带噪观测估计干净样本，从而建立分数预测与去噪器之间的基本联系：
$$
\underbrace{\mathbb{E} \left[\mathbf{x}|\tilde{\mathbf{x}}\right]}_{\text{从 }\tilde{\mathbf{x}}\text{ 估计的去噪器}}
= \frac{1}{\alpha}\left(\tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})\right).
$$
特别地，以特定步长 $\sigma^2$ 在带噪对数似然上做单步梯度上升，得到的更新即为去噪估计（条件平均干净信号）。因此 DSM 训练与去噪紧密相关：若 $\mathbf{s}_{\boldsymbol{\phi}}(\tilde{\mathbf{x}})\approx \nabla_{\tilde{\mathbf{x}}}\log p_\sigma(\tilde{\mathbf{x}})$ 由 DSM 训练得到，则 $\frac{1}{\alpha}\left(\tilde{\mathbf{x}}+\sigma^2 \mathbf{s}_{\boldsymbol{\phi}}(\tilde{\mathbf{x}})\right)$ 即为去噪器。

**（可选）高阶 Tweedie 公式。** 经典 Tweedie 公式通过梯度 $\nabla_{\tilde{\mathbf{x}}}\log p(\tilde{\mathbf{x}})$ 表示后验均值 $\mathbb{E}[\mathbf{x}_0|\tilde{\mathbf{x}}]$。高阶推广通过 $\log p(\tilde{\mathbf{x}})$ 的高阶导数表示后验协方差及更高阶 cumulant。

### 2.3.5 （可选）为何 DSM 是去噪：SURE

**SURE（Stein 无偏风险估计）。** Stein 无偏风险估计（SURE）是一种在**不知道干净信号**的情况下估计去噪器 $\mathbf{D}$ 的均方误差（MSE）的技术；即 SURE 在仅有带噪数据时为选择或训练去噪器提供途径。

考虑加性高斯噪声设定：$\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\epsilon}$，$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$，$\mathbf{x}$ 为未知干净信号，$\tilde{\mathbf{x}}$ 为观测到的带噪版本。去噪器为任意（弱可微）映射 $\mathbf{D}:\mathbb{R}^D \to \mathbb{R}^D$，产生 $\mathbf{x}$ 的估计 $\mathbf{D}(\tilde{\mathbf{x}})$。自然的质量度量是条件 MSE $R(\mathbf{D};\mathbf{x}) := \mathbb{E}_{\tilde{\mathbf{x}}|\mathbf{x}}\left[\|\mathbf{D}(\tilde{\mathbf{x}})-\mathbf{x}\|_2^2 \,\big|\, \mathbf{x}\right]$。该量依赖未知真值 $\mathbf{x}$，无法直接计算。由 Stein 恒等式可得以下**可观测**替代：
$$
\mathrm{SURE}(\mathbf{D};\tilde{\mathbf{x}})
= \|\mathbf{D}(\tilde{\mathbf{x}})-\tilde{\mathbf{x}}\|_2^2
+ 2\sigma^2\,\nabla_{\tilde{\mathbf{x}}}\cdot \mathbf{D}(\tilde{\mathbf{x}})
- D\sigma^2,
$$
其中 $\nabla_{\tilde{\mathbf{x}}}\cdot \mathbf{D}(\tilde{\mathbf{x}})$ 为 $\mathbf{D}$ 的散度。$\mathrm{SURE}(\mathbf{D};\tilde{\mathbf{x}})$ 只需带噪观测 $\tilde{\mathbf{x}}$，不需干净 $\mathbf{x}$。直观上 SURE 由两项组成：$\|\mathbf{D}(\tilde{\mathbf{x}})-\tilde{\mathbf{x}}\|^2$ 度量去噪输出与带噪输入的距离，单独会低估真实误差；散度项作为校正，刻画去噪器对输入小扰动的敏感度，有效计入噪声引入的方差。对任意固定未知 $\mathbf{x}$，$\mathbb{E}_{\tilde{\mathbf{x}}|\mathbf{x}}\left[\mathrm{SURE}(\mathbf{D};\mathbf{x}+\sigma\boldsymbol{\epsilon}) \,\big|\, \mathbf{x}\right] = R(\mathbf{D};\mathbf{x})$，故最小化 SURE（在期望或经验上）等价于最小化真实 MSE，且仅依赖带噪数据。

**与 Tweedie 公式及 Bayes 最优性的联系。** SURE 是给定 $\mathbf{x}$ 下、对噪声的条件 MSE 的无偏估计；最小化期望 SURE 等于最小化 Bayes 风险；逐点最优去噪器为 $\mathbf{D}^*(\tilde{\mathbf{x}})=\mathbb{E}[\mathbf{x}| \tilde{\mathbf{x}}]$。由 Tweedie 恒等式，$\mathbf{D}^*(\tilde{\mathbf{x}})=\tilde{\mathbf{x}}+\sigma^2\nabla_{\tilde{\mathbf{x}}}\log p_\sigma(\tilde{\mathbf{x}})$。若将去噪器参数化为 $\mathbf{D}(\tilde{\mathbf{x}})=\tilde{\mathbf{x}}+\sigma^2\mathbf{s}_{\boldsymbol{\phi}}(\tilde{\mathbf{x}};\sigma)$，代入 SURE 可知最小化 SURE 与在噪声水平 $\sigma$ 下最小化 Hyvärinen 的替代分数匹配目标（期望在 $p_\sigma$ 下取）等价（至多差常数），故两者共享同一最小元，即上述去噪器。

### 2.3.6 （可选）广义分数匹配

经典分数匹配、去噪分数匹配及其高阶变体都针对 $\frac{\mathcal L p(\mathbf{x})}{p(\mathbf{x})}$（$\mathcal L$ 为作用于密度的线性算子）。经典情形 $\mathcal L=\nabla_{\mathbf{x}}$ 给出 $\nabla_{\mathbf{x}}\log p(\mathbf{x})$。$\frac{\mathcal L p}{p}$ 结构允许分部积分消去归一化常数，得到仅依赖 $p$ 的样本与所学场 $\mathbf{s}_{\boldsymbol{\phi}}$ 的可处理目标，由此可推广为广义分数匹配框架。**广义 Fisher 散度**：对线性算子 $\mathcal L$，定义广义 Fisher 散度 $\mathcal D_{\mathcal L}(p \,\|\, q)$；若 $\mathcal L$ **完备**（即 $\frac{\mathcal L p_1}{p_1}=\frac{\mathcal L p_2}{p_2}$ a.e. 推出 $p_1=p_2$ a.e.），则 $\mathcal D_{\mathcal L}(p \,\|\, q)=0$ 当且仅当 $q=p$。$\mathcal L=\nabla_{\tilde{\mathbf{x}}}$ 时即经典 Fisher 散度。**分数参数化**：实践中不建模归一化密度 $q$，而是直接参数化向量场 $\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x})$ 逼近广义分数 $\frac{\mathcal L p(\mathbf{x})}{p(\mathbf{x})}$；通过「分部积分」使损失仅依赖 $\mathbf{s}_{\boldsymbol{\phi}}$，得到可处理目标 $\mathcal L_{\text{GSM}}(\boldsymbol{\phi})$。$\mathcal L=\nabla$ 时 $\mathcal L^\dagger=-\nabla\cdot$，即 Hyvärinen 的分数匹配目标。**算子例子**：经典分数匹配（$\mathcal L = \nabla_{\mathbf{x}}$）；去噪分数匹配（加性高斯噪声下定义特定算子，$\frac{\mathcal L p_\sigma}{p_\sigma}$ 即为 Tweedie 恒等式中的后验均值）；高阶目标（在 $\mathcal L$ 中堆叠导数可得到 $\nabla^2\log p$ 及更高阶导数，对应后验协方差与高阶 cumulant）。广义分数匹配可推广到离散设定（如语言建模），并启发具有去噪风格目标的训练；该算子视角统一多种目标，允许从数据做经验估计，并为通过选择 $\mathcal L$ 设计损失函数提供一般原则。

---

## 2.4 去噪分数匹配的多噪声水平（NCSN）

### 2.4.1 动机

仅对数据分布施加单一固定方差的高斯噪声会在一定程度上平滑分布，但仅在单一噪声水平下训练基于分数的模型会带来主要局限：在低噪声水平下，Langevin 动力学因低密度区域梯度消失而难以穿越多模态分布；在高噪声水平下采样更容易，但模型只捕捉粗结构，样本模糊、缺乏细节。此外 Langevin 在高维空间可能收敛慢甚至失败；因其依赖对数密度的梯度引导，不良初始化（尤其在平台区或鞍点附近）会阻碍探索或使采样器陷入单模态。为解决这些挑战，Song 等人提出在多个水平向数据分布注入高斯噪声，并联合训练**噪声条件分数网络（NCSN）**以估计一系列噪声尺度上的分数函数。生成时以噪声退火方式应用 Langevin 动力学：从高噪声水平开始以利粗探索，逐渐细化到低噪声水平以恢复细节。

![图 6：SM 不准确性示意图（重访图 4）](../arXiv-2510.21890v1/Images/PartB/score_function_acc.pdf)

**图 6：SM 不准确性示意图。** 红色区域表示低密度区域，因样本覆盖有限可能导致分数估计不准确；高密度区域往往得到更准确的估计。

![图 7：NCSN 示意图](../arXiv-2510.21890v1/Images/PartB/ncsn-graph.pdf)

**图 7：NCSN 示意图。** 前向过程用多水平加性高斯噪声 $p_{\sigma}(\mathbf{x}_{\sigma} | \mathbf{x})$ 扰动数据。生成通过在每一噪声水平上运行 Langevin 采样进行，用当前水平的结果初始化下一更低方差的采样。

### 2.4.2 训练

为克服仅在单一噪声水平下训练的基于分数模型的局限，Song 等人提出在多个水平向数据分布添加高斯噪声。具体地，选取 $L$ 个噪声水平 $\{\sigma_i\}_{i=1}^L$ 满足 $0 < \sigma_1 < \sigma_2 < \cdots < \sigma_L$，其中 $\sigma_1$ 足够小以保留大部分数据细节，$\sigma_L$ 足够大以充分平滑分布、便于训练。每个带噪样本由干净数据 $\mathbf{x} \sim p_{\text{data}}$ 扰动得到：$\mathbf{x}_{\sigma} = \mathbf{x} + \sigma \boldsymbol{\epsilon}$，$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。这定义**扰动核** $p_{\sigma}(\mathbf{x}_{\sigma} | \mathbf{x}) := \mathcal{N}(\mathbf{x}_{\sigma}; \mathbf{x}, \sigma^2 \mathbf{I})$ 及**边缘分布** $p_\sigma(\mathbf{x}_{\sigma}) = \int p_{\sigma} (\mathbf{x}_{\sigma} | \mathbf{x}) p_{\text{data}}(\mathbf{x}) \,\mathrm{d} \mathbf{x}$，即每个噪声水平上的高斯平滑数据分布。

**NCSN 的训练目标。** 目标是训练噪声条件分数网络 $\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}, \sigma)$，使其对所有 $\sigma \in \{\sigma_i\}_{i=1}^L$ 估计分数函数 $\nabla_{\mathbf{x}} \log p_\sigma(\mathbf{x})$。通过对所有噪声水平最小化 DSM 目标实现：
$$
\mathcal{L}_{\text{NCSN}}(\boldsymbol{\phi}) := \sum_{i=1}^{L} \lambda(\sigma_i)   \mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma_i),
$$
其中 $\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma) = \frac{1}{2} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x}), \tilde{\mathbf{x}} \sim p_{\sigma}(\tilde{\mathbf{x}} | \mathbf{x})} \left[ \|\mathbf{s}_{\boldsymbol{\phi}}(\tilde{\mathbf{x}}, \sigma) - \left( \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2} \right)\|_2^2 \right]$，$\lambda(\sigma_i)>0$ 为各尺度的权重函数。最小化该目标得到分数模型 $\mathbf{s}^*(\mathbf{x}, \sigma)$，在各噪声水平恢复真实分数：$\mathbf{s}^*(\cdot, \sigma) = \nabla_{\mathbf{x}} \log p_\sigma(\cdot)$（对所有 $\sigma \in \{\sigma_i\}_{i=1}^L$）。

**与 DDPM 损失的关系。** 令 $\mathbf{x}_\sigma=\mathbf{x}+\sigma\boldsymbol{\epsilon}$，$\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$，$p_\sigma$ 为边缘分布。由 Tweedie 公式，$\nabla_{\mathbf{x}_\sigma}\log p_\sigma(\mathbf{x}_\sigma) = -\frac{1}{\sigma} \mathbb{E} \left[\boldsymbol{\epsilon} \middle|  \mathbf{x}_\sigma\right]$。故 NCSN 最优解为真实分数 $\mathbf{s}^*(\mathbf{x}_\sigma,\sigma)=\nabla_{\mathbf{x}_\sigma}\log p_\sigma(\mathbf{x}_\sigma)$，而 DDPM 损失下的 Bayes 最优噪声预测器为 $\boldsymbol{\epsilon}^*(\mathbf{x}_\sigma,\sigma)=\mathbb{E}[\boldsymbol{\epsilon}|\mathbf{x}_\sigma]$。二者通过 $\mathbf{s}^*(\mathbf{x}_\sigma,\sigma) = - \frac{1}{\sigma} \boldsymbol{\epsilon}^*(\mathbf{x}_\sigma,\sigma)$ 与 $\boldsymbol{\epsilon}^*(\mathbf{x}_\sigma,\sigma) = - \sigma \mathbf{s}^*(\mathbf{x}_\sigma,\sigma)$ 精确等价。在 DDPM 的离散指标 $i$ 的扰动形式下，$\mathbf{x}_i = \bar{\alpha}_i \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_i^2}\boldsymbol{\epsilon}$，同一关系给出 $\mathbf{s}^*(\mathbf{x}_i,i)= -\frac{1}{\sigma_i} \mathbb{E} \left[\boldsymbol{\epsilon} \middle|  \mathbf{x}_i\right]$，故最小化 DDPM 简化损失即学习 $\boldsymbol{\epsilon}$ 的条件去噪器，它是噪声水平 $i$ 处真实分数的缩放重参数化。

### 2.4.3 采样

**算法：退火 Langevin 动力学**

- **输入**：每个噪声水平 $\ell = L, \dots, 2$ 上训练好的分数 $\mathbf{s}_{\boldsymbol{\phi}^*}(\cdot, \sigma_\ell)$、步长 $\eta_\ell$ 与 Langevin 迭代预算 $N_\ell$。
- $\mathbf{x}^{\sigma_L} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。
- **for** $\ell = L, \dots, 2$：
  - $\tilde{\mathbf{x}}_0 \gets \mathbf{x}^{\sigma_\ell}$（用上一噪声水平的输出初始化 Langevin）
  - **for** $n = 0$ **to** $N_\ell - 1$：
    - $\boldsymbol{\epsilon}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
    - $\tilde{\mathbf{x}}_{n+1} \gets \tilde{\mathbf{x}}_n + \eta_\ell \mathbf{s}_{\boldsymbol{\phi}^*}(\tilde{\mathbf{x}}_n, \sigma_\ell) + \sqrt{2\eta_\ell} \boldsymbol{\epsilon}_n$
  - $\mathbf{x}^{\sigma_{\ell-1}} \gets \tilde{\mathbf{x}}_{N_\ell}$（输出作为下一噪声水平的初始化）
- **输出** $\mathbf{x}^{\sigma_1}$。

在多个噪声水平上得到训练好的分数网络后，**退火 Langevin 动力学**从高噪声水平 $\sigma_L$ 逐步去噪到低噪声水平 $\sigma_1 \approx 0$ 生成数据。从高斯噪声 $\mathbf{x}^{\sigma_L} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 开始，算法在每一噪声水平 $\sigma_\ell$ 上应用 Langevin 动力学以近似从扰动分布 $p_{\sigma_\ell}(\mathbf{x})$ 采样；该水平的输出作为下一更低噪声水平 $\sigma_{\ell-1}$ 的**更好初始化**。在每一水平上，Langevin 迭代更新 $\tilde{\mathbf{x}}_{n+1} = \tilde{\mathbf{x}}_n + \eta_\ell \mathbf{s}_{\boldsymbol{\phi}^*}(\tilde{\mathbf{x}}_n, \sigma_\ell) + \sqrt{2\eta_\ell} \boldsymbol{\epsilon}_n$，从 $\tilde{\mathbf{x}}_0 := \mathbf{x}^{\sigma_\ell}$ 开始。步长通常按噪声水平缩放：$\eta_\ell = \delta \cdot \frac{\sigma_\ell^2}{\sigma_1^2}$（$\delta>0$ 固定）。该噪声退火细化直至最低噪声水平 $\sigma_1$，得到最终样本 $\mathbf{x}^{\sigma_1}$。

**NCSN 采样速度慢。** NCSN 在噪声尺度 $\{\sigma_i\}_{i=1}^L$ 上使用退火 MCMC（通常为 Langevin 动力学）生成样本。对每个尺度 $\sigma_i$ 执行 $K$ 次形如「用分数 $\mathbf{s}_{\boldsymbol{\phi}^*}(\tilde{\mathbf{x}}_n,\sigma_i)$ 加小随机扰动更新 $\tilde{\mathbf{x}}_n$」的迭代，每次需一次分数网络前向。两因素导致需要较大的 $L \times K$：（i）**局部精度与稳定性**：学到的分数仅对小扰动可靠，需要小步长与每噪声水平多次迭代以避免偏差或不稳定；（ii）**高维混合慢**：局部 MCMC 移动在探索多模态高维目标时低效，需要大量迭代才能到达典型数据区域。因更新严格串行且每次需昂贵网络求值，总成本为 $\mathcal{O}(L  K)$ 次串行网络前向，采样计算慢。

---

## 2.5 小结：NCSN 与 DDPM 的对比

**对比。** NCSN 与 DDPM 的图模型及主要异同总结如下表。

**表 1：NCSN 与 DDPM 的对比**

|  | **NCSN** | **DDPM** |
|---|----------|----------|
| $\mathbf{x}_{i+1}\|\mathbf{x}_{i}$ | 由 $\mathbf{x}_{i+1} =  \mathbf{x}_i + \sqrt{\sigma^2_{i+1}-\sigma^2_{i}}\boldsymbol{\epsilon}$ 推导 | 定义为 $\mathbf{x}_{i+1} = \sqrt{1-\beta_{i}} \mathbf{x}_{i} + \sqrt{\beta_{i}}\boldsymbol{\epsilon}$ |
| $\mathbf{x}_{i}\|\mathbf{x}$ | 定义为 $\mathbf{x}_{i} =  \mathbf{x} + \sigma^2_{i}\boldsymbol{\epsilon}$ | 由 $\mathbf{x}_i = \bar{\alpha}_i\mathbf{x} + \sqrt{1 - \bar{\alpha}_i^2}\boldsymbol{\epsilon}$ 推导 |
| $p_{\text{prior}}$ | $\mathcal{N}(\mathbf{0}, \sigma_L^2\mathbf{I})$ | $\mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| **损失** | $\mathbb{E}_{i} \mathbb{E}_{p_{\text{data}}(\mathbf{x})} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \|\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}_i, \sigma_i) +   \frac{\boldsymbol{\epsilon}}{\sigma_i}\|_2^2 \right]$ | $\mathbb{E}_{i}\mathbb{E}_{p_{\text{data}}(\mathbf{x})} \mathbb{E}_{\boldsymbol{\epsilon}\sim \mathcal{N}(\mathbf{0},\mathbf{I})}\left[ \|\boldsymbol{\epsilon}_{\boldsymbol{\phi}}(\mathbf{x}_i, i) - \boldsymbol{\epsilon}\|_2^2 \right]$ |
| **采样** | 每层应用 Langevin；用输出初始化下一层 | 用 $p_{\boldsymbol{\phi}^*}(\mathbf{x}_{i-1} | \mathbf{x}_i)$ 沿 Markov 链遍历 |

**共同瓶颈。** 尽管表述不同，NCSN 与 DDPM 都依赖稠密时间离散化，导致关键局限：采样常需数百甚至数千次迭代，生成慢且计算密集。我们将在后续章节中重访如何加速扩散模型采样这一挑战。

---

## 2.6 结语

本章勾勒了通向扩散模型的第二条主要路径，即根植于基于能量的模型（EBM）的基于分数视角。我们首先指出 EBM 的核心难点——难以处理的配分函数，并引入分数函数 $\nabla_{\mathbf{x}}\log{p(\mathbf{x})}$ 作为完全绕过该问题的有力工具。

我们从经典分数匹配走到其更可扩展、更稳健的变体——去噪分数匹配（DSM）。通过 DSM，我们看到用噪声扰动数据如何得到可处理的训练目标，再次利用条件化策略构造简单回归目标。此外，我们通过 Tweedie 公式建立了分数估计与去噪行为之间的深刻联系：分数给出了从带噪观测估计干净信号所需的方向。

该思想随后从单一噪声水平推广到多水平，即**噪声条件分数网络（NCSN）**：在多个噪声尺度上条件化的单一分数模型，通过退火 Langevin 动力学生成样本。探索结束时我们发现，NCSN 与来自变分视角的 DDPM 尽管起源不同，却具有极为相似的结构与共同瓶颈：慢、串行的采样。

这种收敛并非巧合，它暗示更深层的统一数学结构。这些离散时间模型的局限促使我们需要更一般的框架。下一章将迈出关键一步：（1）进入连续时间视角，表明 DDPM 与 NCSN 都可优雅地统一为同一由随机微分方程（SDE）描述的过程的不同离散化；（2）该分数 SDE 框架将形式化地连接变分视角与基于分数的视角，把生成问题重述为求解微分方程。这一统一视角不仅带来理论上的清晰，还将解锁一类新的先进数值方法，以应对采样慢这一根本挑战。

---

