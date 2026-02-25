# 基于蒸馏的快速采样方法

本章介绍基于训练的快速采样方法：通过让新的生成器学会仅用一步或少量几步即可产生样本，从而加速扩散模型采样。核心思想称为*蒸馏*，即让快速的*学生*模型从慢速的、预训练的扩散模型（*教师*）采样器学习。教师可能需要数百步，而学生只需少量几步即可达到相当的质量[^1]。与改进数值积分格式的基于求解器的加速不同，蒸馏直接训练一个生成器以走高效的捷径。我们重点介绍两种主要范式：*分布级蒸馏*——跳过对完整轨迹的模拟，转而让学生输出分布与教师对齐；以及*流映射级蒸馏*——训练学生以更快、更紧凑的方式复现教师的采样路径。

[^1]: 此处蒸馏指减少采样步数，而非缩小模型规模。

---

## 引言

扩散模型的一个核心瓶颈是采样速度慢。

如 Tweedie 公式（见等价参数化小节）所示，扩散模型可解释为「$\mathbf{x}$-预测」模型 $\mathbf{x}_{\boldsymbol{\phi}^\times}(\mathbf{x}_t, t)$，其训练目标是在噪声水平 $t$ 下从带噪输入 $\mathbf{x}_t$ 恢复期望的干净数据：
$$
    \mathbf{x}_{\boldsymbol{\phi}^\times}(\mathbf{x}_t, t) \approx \mathbb{E}[\mathbf{x}_0 |\mathbf{x}_t],
$$
其中期望对 $p(\mathbf{x}_0 |\mathbf{x}_t)$ 求取，表示与 $\mathbf{x}_t$ 对应的所有可能干净数据的分布。一个自然的想法是用 $\mathbf{x}_{\boldsymbol{\phi}^\times}(\mathbf{x}_t, t)$ 做一步生成。但由于该去噪器对多种可能结果取平均，预测会过于平滑，仅用少量去噪步生成会导致模糊、低质量的样本。

另一方面，如 SDE 采样小节所述，扩散采样沿 ODE 或 SDE 轨迹经过一长串迭代步进行。这能产生高保真样本，但所需步数众多，过程本质上是慢的。减少 NFE（即采样步数与模型调用次数的乘积）可以加速生成，但不可避免地会降低保真度。每个求解器步会引入 $\mathcal{O}(h^{n})$ 阶的积分误差，其中 $n$ 为求解器阶数，$h=\max_i |t_i - t_{i-1}|$ 为步长。步数越少意味着时间增量 $h$ 越大，进而增大累积采样误差，导致轨迹精度下降。这构成了扩散采样中质量与效率之间的根本权衡。

为克服该瓶颈，一条重要研究方向是*蒸馏*：假设已有训练好的扩散模型（*教师*），并训练一个生成器（*学生*）通过单次前向或少量几步计算复现其行为。这将教师的众多采样步压缩为快速过程，在保持高样本保真度的同时有效绕过慢速迭代求解器。

下面我们从两种视角介绍蒸馏：*分布级蒸馏*与*流映射级蒸馏*[^2]。

[^2]: 按时间顺序，流映射级蒸馏以 Knowledge Distillation (KD) 和 Progressive Distillation (PD) 为代表，早在 2021 年即被提出，早于约 2023 年出现的分布级蒸馏一族。为叙述连贯并与下一章衔接，此处先介绍分布级蒸馏。

### 分布级蒸馏

基于分布的蒸馏的目标是训练一个一步生成器 $\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{z})$，将噪声 $\mathbf{z} \sim p_{\mathrm{prior}}$ 映射为样本 $\hat{\mathbf{x}} = \mathrm{G}_{\boldsymbol{\theta}}(\mathbf{z})$，使其诱导的分布 $p_{\boldsymbol{\theta}}(\hat{\mathbf{x}})$ 逼近目标数据分布 $p_{\mathrm{data}}(\mathbf{x})$。通常通过最小化某种统计散度实现：
$$
\min_{\boldsymbol{\theta}}  \mathcal{D}\!\left(p_{\boldsymbol{\theta}}(\hat{\mathbf{x}}),  p_{\mathrm{data}}(\hat{\mathbf{x}})\right),
$$
其中 $\mathcal{D}$ 表示合适的散度度量（如 KL 散度）。

实践中，基于分布的方法让生成器的分布与由预训练扩散模型产生的经验分布 $p_{\boldsymbol{\phi}^\times}(\mathbf{x})$ 对齐：
$$
\min_{\boldsymbol{\theta}}  \mathcal{D}\!\left(p_{\boldsymbol{\theta}}(\hat{\mathbf{x}}),  p_{\boldsymbol{\phi}^\times}(\hat{\mathbf{x}})\right),
$$
其中 $p_{\boldsymbol{\phi}^\times}$ 作为 $p_{\mathrm{data}}$ 的代理。这些方法不显式计算该散度，而是近似其梯度，而梯度可直接由预训练教师模型计算得到。因此学生可以在不进行完整散度计算的情况下与教师的分布对齐。

该表述通过分布对齐，将扩散模型的多步生成过程蒸馏为单步模型。我们将在「基于分布的蒸馏」一节详述。

### 流映射级蒸馏

考虑 PF-ODE，其对任意预测模型可写成（见预测等价式）：
$$
    \frac{\mathrm{d} \mathbf{x}(\tau)}{\mathrm{d} \tau} 
    = f(\tau) \mathbf{x}(\tau) - \tfrac{1}{2}g^2(\tau) \nabla_\mathbf{x} \log p_\tau(\mathbf{x}(\tau)) 
    =: \mathbf{v}^*(\mathbf{x}(\tau), \tau).
$$

其解映射为：从时刻 $s$ 的 $\mathbf{x}_s$ 出发，反向演化到 $t \leq s$，记为 $\boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s)$，即
$$
    \boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s) 
    \coloneqq \mathbf{x}_s + \int_s^t \mathbf{v}^*(\mathbf{x}(\tau), \tau) \mathrm{d} \tau,
$$
其中积分对应 PF-ODE 的解。直观上，$\boldsymbol{\Psi}_{s\to t}$ 将时刻 $s$ 的噪声 $\mathbf{x}_s$ 输运到时刻 $t$ 的更低噪声状态（最终在 $t=0$ 为数据）。

从扩散模型采样即对 $\mathbf{x}_T \sim p_{\mathrm{prior}}$ 计算 $\boldsymbol{\Psi}_{T\to 0}(\mathbf{x}_T)$。通常该积分由利用速度场 $\mathbf{v}$ 的迭代数值求解器近似（见求解器一章），但需要很多步（例如即使用 DPM-Solver 也至少约 10 步），使得采样慢于 GAN 等经典一步生成模型。这引出一个自然问题：我们能否直接学习解映射 $\boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s)$？特别地，学习映射 $\boldsymbol{\Psi}_{T\to 0}(\mathbf{x}_T)$（其中 $\mathbf{x}_T \sim p_{\mathrm{prior}}$）即可实现一步生成。

**轨迹蒸馏。**

轨迹蒸馏旨在训练一个在实例层面近似解映射的神经生成器。由于 PF-ODE 的积分一般没有闭式解，训练时必须用数值近似。形式化地，引入一般求解器记号
$$
    \mathtt{Solver}_{s\to t}(\mathbf{x}_s;\,\boldsymbol{\phi}^\times)
    \quad \text{或简写为} \quad
    \mathtt{Solver}_{s\to t}(\mathbf{x}_s),
$$
表示从 $\mathbf{x}_s$ 出发、在教师参数 $\boldsymbol{\phi}^\times$ 下从 $s$ 到 $t$ 的经验 PF-ODE 数值积分（上下文清楚时省略 $\boldsymbol{\phi}^\times$）。

**早期方法：直接知识蒸馏。**

为实现少步甚至一步生成，一种直接做法是训练生成器 $\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_T, T,0)$ 模仿沿完整轨迹的数值求解器输出：
$$
\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_T, T,0)\approx \mathtt{Solver}_{T\to 0}(\mathbf{x}_T), 
\qquad \mathbf{x}_T \sim p_{\mathrm{prior}}.
$$
该思想是早期轨迹蒸馏方法之一 **Knowledge Distillation** 的基础，其使用回归损失
$$
\mathcal{L}_{\mathrm{KD}}(\boldsymbol{\theta})
:= 
\mathbb{E}_{\mathbf{x}_T \sim p_{\mathrm{prior}}}
\left\|\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_T, T,0) - \mathtt{Solver}_{T\to 0}(\mathbf{x}_T)\right\|_2^2.
$$
尽管该方法提供来自预训练教师的直接监督，它无法利用原始训练数据中的强监督。此外，若在训练循环内调用 ODE 积分，计算开销很大，因为每次参数更新都需要求解 ODE 以构造目标。最后，由于生成器只学习从 $T$ 到 $0$ 的全局映射，可能失去从中间状态引导生成过程的可控性，因此引导一章中介绍的大多数可控生成技术无法直接应用。

**渐进式蒸馏前言。**

**Progressive Distillation (PD)** 使用来自 $\mathtt{Teacher}$ 片段的*局部*监督训练时间条件 $\mathtt{Student}$。设 $t_0=T>t_1>\cdots>t_N=0$ 为固定时间网格。$\mathtt{Teacher}$ 提供 $k=0,\ldots,N-1$ 的步进映射 $\mathtt{Teacher}_{t_k\to t_{k+1}}$。

PD 不是只监督一跳 $T\to 0$，而是训练 $\mathtt{Student}$ 的两步跳跃映射以匹配 $\mathtt{Teacher}$ 的连续两步：
$$
\mathtt{Student}_{t_k\to t_{k+2}}
\;\approx\;
\mathtt{Teacher}_{t_{k+1}\to t_{k+2}}
\circ
\mathtt{Teacher}_{t_{k}\to t_{k+1}},
$$
对 $k = 0, 2, 4, \dots$。匹配通过简单回归损失（如均方误差）实现。

在局部配对片段上训练后，$\mathtt{Student}$ 不再沿原网格的每个时间间隔推进，而是在每隔一个时间点上推进，
$$
t_0 \to t_2 \to t_4 \to \cdots \to t_N,
$$
即每个 $\mathtt{Student}$ 步实际覆盖 $\mathtt{Teacher}$ 的连续两步。因此 $\mathtt{Student}$ 仅用 $N/2$ 次转移即可完成同一总时间跨度 $[0,T]$。

该阶段结束后，训练好的 $\mathtt{Student}$ 取代 $\mathtt{Teacher}$ 作为新的参考模型。整个过程随后在更粗的网格上重复，时间步长加倍（$N \to N/2 \to N/4 \to \cdots$），逐步将轨迹蒸馏到越来越少的步数，直至达到期望的推理步数。这种迭代减半在保持全局时间范围的同时，不断压缩生成过程的时间分辨率。

**流映射学习的统一视角。**

包括 KD 与 PD 在内的多种方法可在统一损失框架下写出：
$$
  \mathcal{L}_{\mathrm{oracle}}(\boldsymbol{\theta})
  := \mathbb{E}_{s,t} \mathbb{E}_{\mathbf{x}_s \sim p_s} 
  \left[  w(s,t)  d \big(\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_s, s,t),  \boldsymbol{\Psi}_{s\to t}(\mathbf{x}_s)\big)\right],
$$
其中 $\boldsymbol{\Psi}_{s\to t}$ 为 oracle 流映射，$w(s,t) \geq 0$ 指定不同时间对 $(s,t)$ 的权重，$d(\cdot,\cdot)$ 为差异度量（如 $d(\mathbf{x},\mathbf{y})=\|\mathbf{x}-\mathbf{y}\|_2^2$ 或 $d(\mathbf{x},\mathbf{y})=\|\mathbf{x}-\mathbf{y}\|_1$），$p_s$ 为时刻 $s$ 的前向加噪边缘分布。由于 $\boldsymbol{\Psi}_{s\to t}$ 一般无闭式形式，必须依赖近似，通常通过预训练扩散模型（教师）或其他可处理的代理。

KD 是上述统一损失的一个简单特例。取退化权重 $w(s,t)=\delta(s{-}T)\,\delta(t{-}0)$ 并令先验分布 $p_T=p_{\mathrm{prior}}$，oracle 损失 $\mathcal{L}_{\mathrm{oracle}}(\boldsymbol{\theta})$ 化为：
$$
\mathbb{E}_{\mathbf{x}_T\sim p_T}
\big\|\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{x}_T,T,0)-\boldsymbol{\Psi}_{T\to 0}(\mathbf{x}_T)\big\|_2^2
\approx \mathcal{L}_{\mathrm{KD}}(\boldsymbol{\theta}),
$$
其中 $\mathtt{Solver}_{T\to 0}\approx\boldsymbol{\Psi}_{T\to 0}$。

PD 也符合该模板，但不是仅用单一极端对 $(T,0)$ 监督，而是使用许多*邻近*时间对并施加简单的*局部一致性*规则：一个短步再接一个短步应与直接两步移动一致。我们在后文 PD 局部损失中回到这一点。

实践中主要难点是 oracle 流映射 $\boldsymbol{\Psi}_{s\to t}$ 一般没有闭式表达式，无法直接监督。一系列方法被提出以高效近似该目标，但其成功往往依赖教师模型的质量。我们将在「从零训练的快速采样」一章回到该统一损失，给出一种从学习循环中消除教师的、从零训练方法的原则性框架。

---

## 基于分布的蒸馏

多项工作在不同名称下并行推进了这类基于分布的蒸馏，包括 Distributional Matching Distillation (DMD)、Variational Score Distillation (VSD)、Score Identity Distillation (SiD) 等。尽管技术细节不同，它们共享同一原则：训练一个生成器，使其前向加噪边缘与教师一致。我们以 VSD 为代表表述，其余方法遵循类似原理。

### 以 VSD 为代表的表述

**前向过程。**

设 $\{p_t\}_{t \in [0,T]}$ 表示由
$$
\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathrm{I}),
$$
诱导的前向扩散过程的边缘密度，初始分布为 $p_0 = p_{\mathrm{data}}$。与之相对，设 $p_0^{\boldsymbol{\theta}}$ 表示由确定性一步生成器 $\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{z})$ 从潜变量 $\mathbf{z} \sim p_{\mathrm{prior}}(\mathbf{z})$ 生成的合成样本的分布。定义 $\{p_t^{\boldsymbol{\theta}}\}_{t \in [0,T]}$ 为对 $p_0^{\boldsymbol{\theta}}$ 施加同一前向扩散过程得到的边缘密度，即
$$
    \mathbf{x}_t^{\boldsymbol{\theta}} := \alpha_t \mathrm{G}_{\boldsymbol{\theta}}(\mathbf{z}) + \sigma_t \boldsymbol{\epsilon}, 
$$
其中 $\mathbf{z} \sim p_{\mathrm{prior}}$ 且 $ \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathrm{I})$。因此 $p_t$ 与 $p_t^{\boldsymbol{\theta}}$ 共享同一高斯扩散核 $p_t(\mathbf{x}_t| \mathbf{x}_0)$，但起始分布不同（$p_{\mathrm{data}}$ 与一步合成样本的 $p_0^{\boldsymbol{\theta}}$）。

**训练目标与梯度。**

文献通常采用 KL 散度来匹配 $p_t$ 与 $p_t^{\boldsymbol{\theta}}$，常见做法是最小化
$$
    \mathcal{L}_{\text{VSD}}(\boldsymbol{\theta}) :=\mathbb{E}_t \left[ \omega(t)  \mathcal{D}_{\text{KL}}(p_t^{\boldsymbol{\theta}} \, \| \, p_t) \right]
= \mathbb{E}_{t, \mathbf{z}, \boldsymbol{\epsilon}} \left[ \omega(t) \left( \log p_t^{\boldsymbol{\theta}}(\mathbf{x}_t^{\boldsymbol{\theta}}) - \log p_t(\mathbf{x}_t^{\boldsymbol{\theta}}) \right) \right],
$$
其中 $\omega(t)$ 为时间依赖的权重函数。我们将在 VSD 讨论小节说明为何 KL 散度在分布级蒸馏中具有特殊地位。

如文献所述，当 $ p_0^{\boldsymbol{\theta}^*} = p_{\mathrm{data}} $ 时达到最优，表明生成器分布与数据分布一致，该训练目标可作为学习数据分布的有效损失。

然而，基于密度的目标表述缺乏高效的训练机制。所幸，对 $\boldsymbol{\theta}$ 求梯度可得后文给出的表达式，由下列命题概括。为记号简洁，记 $\hat{\mathbf{x}}_t := \mathbf{x}_t^{\boldsymbol{\theta}}$（由前向式定义）。

**命题（$\boldsymbol{\theta}$-梯度 of $\mathcal{L}_{\text{VSD}}$）** 我们有
$$
        \nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{VSD}}(\boldsymbol{\theta}) = \mathbb{E}_{t, \mathbf{z}, \boldsymbol{\epsilon}} \left[ \omega(t)\alpha_t  \left( 
\nabla_{\mathbf{x}} \log p_t^{\boldsymbol{\theta}}(\hat{\mathbf{x}}_t) - \nabla_{\mathbf{x}} \log p_t(\hat{\mathbf{x}}_t) 
\right) \cdot \partial_{\boldsymbol{\theta}} \mathrm{G}_{\boldsymbol{\theta}}(\mathbf{z}) \right].
$$

推导使用链式法则；第一项由 score-function identity 为零；利用重参数化 $\hat{\mathbf{x}}_t=\alpha_t\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{z})+\sigma_t\boldsymbol{\epsilon}$ 得 $\partial_{\boldsymbol{\theta}}\hat{\mathbf{x}}_t=\alpha_t\partial_{\boldsymbol{\theta}}\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{z})$，即可得上式。细节见附录流映射小节。

我们观察到，对 $\boldsymbol{\theta}$ 求梯度时 score 函数自然出现。因此需要近似一步生成器的 score $\nabla_{\mathbf{x}}\log p_t^{\boldsymbol{\theta}}(\hat{\mathbf{x}}_t)$ 以及数据分布的 score $\nabla_{\mathbf{x}}\log p_t(\hat{\mathbf{x}}_t)$，下一小节将详述。

### VSD 的训练流程

现有工作通常采用双层优化：在 $\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{z})$ 的样本上训练一个新的扩散模型以近似 $\nabla_{\mathbf{x}} \log p_t^{\boldsymbol{\theta}}(\hat{\mathbf{x}}_t)$，并采用预训练扩散模型作为合成样本 $\hat{\mathbf{x}}_t$ 上难以处理的 oracle score $\nabla_{\mathbf{x}} \log p_t(\hat{\mathbf{x}}_t)$（即教师的 score）的代理。更精确地说，训练在两个阶段之间交替进行：

- **Score 估计阶段。** 固定 $\boldsymbol{\theta}$。令 $\hat{\mathbf{x}}_0=\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{z})$，$\hat{\mathbf{x}}_t=\alpha_t \hat{\mathbf{x}}_0+\sigma_t\boldsymbol{\epsilon}$，其中 $\mathbf{z}\sim p_{\mathrm{prior}}$，$\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0},\mathrm{I})$。用已知高斯扩散核 $p_t(\mathbf{x}_t|\mathbf{x}_0)$ 通过 DSM 训练 $s_{\boldsymbol{\zeta}}$：
  $$
  \mathcal{L}_{\text{DSM}}(\boldsymbol{\zeta};\boldsymbol{\theta})
  =\mathbb{E}_{t,\mathbf{z},\boldsymbol{\epsilon}}\Big\|\,\mathbf{s}_{\boldsymbol{\zeta}}(\hat{\mathbf{x}}_t,t)
  - \nabla_{\mathbf{x}_t}\log p_t(\hat{\mathbf{x}}_t|\hat{\mathbf{x}}_0)\,\Big\|^2,
  $$
  在最优时（对固定 $\boldsymbol{\theta}$）有 $\mathbf{s}_{\boldsymbol{\zeta}}(\cdot,t)\approx \nabla_{\mathbf{x}}\log p_t^{\boldsymbol{\theta}}(\cdot)$。

- **生成器更新阶段。** 在 $s_{\boldsymbol{\zeta}}$ 冻结（stop-grad）下，用前述梯度更新 $\boldsymbol{\theta}$，并将两个 score 项分别替换为其代理：
  $\mathbf{s}_{\boldsymbol{\zeta}}(\hat{\mathbf{x}}_t,t)\approx \nabla_{\mathbf{x}}\log p_t^{\boldsymbol{\theta}}(\hat{\mathbf{x}}_t)$，$\mathbf{s}_{\boldsymbol{\phi}^\times}(\hat{\mathbf{x}}_t,t)\approx \nabla_{\mathbf{x}}\log p_t(\hat{\mathbf{x}}_t)$（教师）。则该梯度近似为
  $$
    \nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{VSD}}(\boldsymbol{\theta})
    \approx \mathbb{E}_{t,\mathbf{z},\boldsymbol{\epsilon}}\Big[\omega(t)\alpha_t
      \big(\mathbf{s}_{\boldsymbol{\zeta}}(\hat{\mathbf{x}}_t,t) - \mathbf{s}_{\boldsymbol{\phi}^\times}(\hat{\mathbf{x}}_t,t)\big)^{\top}
      \partial_{\boldsymbol{\theta}}\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{z})
    \Big].
  $$

两阶段重复直至对所有 $t$ 在 $p_t^{\boldsymbol{\theta}}$ 的支撑上有 $\mathbf{s}_{\boldsymbol{\zeta}}(\cdot,t)\approx \mathbf{s}_{\boldsymbol{\phi}^\times}(\cdot,t)$，此时插入的梯度为零。在这种收敛状态下，对所有 $t>0$ 有 $p_t^{\boldsymbol{\theta}}\approx p_t^{\boldsymbol{\phi}^\times}$（教师的边缘）。由于对任意固定 $t>0$ 前向加噪算子（高斯卷积）是单射，故 $p_0^{\boldsymbol{\theta}}\approx p_0^{\boldsymbol{\phi}^\times}$（教师在 $t=0$ 的分布）。因此学到的一步生成器 $\mathrm{G}_{\boldsymbol{\theta}}$ 在 $t=0$ 与教师分布一致；当教师充分逼近 $p_{\mathrm{data}}$ 时，进一步有 $p_0^{\boldsymbol{\theta}}\approx p_{\mathrm{data}}$。

### 补充讨论：散度选择与 VSD 应用

**除 KL 之外：能否使用一般散度？**

原则上，可将 VSD 中的前向 KL 项 $\mathcal{D}_{\mathrm{KL}}(p_t^{\boldsymbol{\theta}}\|p_t)$ 替换为更一般的散度族，例如 $f$-散度：
$$
\mathcal{D}_f(p_t^{\boldsymbol{\theta}}\|p_t)
= \int p_t(\mathbf{x})  f\!\left(\frac{p_t^{\boldsymbol{\theta}}(\mathbf{x})}{p_t(\mathbf{x})}\right)\mathrm{d} \mathbf{x}.
$$
然而，梯度 $\nabla_{\boldsymbol{\theta}}\mathcal{D}_f(p_t^{\boldsymbol{\theta}}\|p_t)$ 通过 $f'(r_t)$ 依赖于*密度比* $r_t(\mathbf{x})=\frac{p_t^{\boldsymbol{\theta}}(\mathbf{x})}{p_t(\mathbf{x})}$，这对*隐式学生生成器*是难以处理的。此处学生称为*隐式*，因其可通过随机映射 $\hat{\mathbf{x}}_t=\alpha_t\mathrm{G}_{\boldsymbol{\theta}}(\mathbf{z})+\sigma_t\boldsymbol{\epsilon}$ 产生样本 $\hat{\mathbf{x}}_t$，但不提供其诱导密度 $p_t^{\boldsymbol{\theta}}(\mathbf{x})$ 的闭式或似然。因此，计算 $\mathcal{D}_f$ 的泛函导数需要逐点得到 $r_t(\mathbf{x})$ 或其对数梯度，二者在此设定下都无法计算。常见变通是引入辅助判别器，通过 $f$-散度的变分形式近似密度比（如 $f$-GAN），但这会引入额外网络和嵌套的极小极大优化。

相比之下，对前向 KL，路径式梯度简化为 score 差形式：$\nabla_{\boldsymbol{\theta}}\mathcal{D}_{\mathrm{KL}}(p_t^{\boldsymbol{\theta}}\|p_t)$ 等于期望 $(\nabla_{\mathbf{x}}\log p_t^{\boldsymbol{\theta}}(\hat{\mathbf{x}}_t) - \nabla_{\mathbf{x}}\log p_t(\hat{\mathbf{x}}_t))^\top \partial_{\boldsymbol{\theta}}\hat{\mathbf{x}}_t$。这一结构允许可处理的、仅依赖 score 的更新。教师预训练扩散模型已提供 $\nabla_{\mathbf{x}}\log p_t(\cdot)$，可直接复用而无需学习辅助密度比估计器。该表述给出非对抗、完全可微且计算高效的目标。

**仅用 2D 预训练扩散模型的 3D 生成中的 VSD。**

VSD 及其更早特例 SDS（其中生成器为由 $\boldsymbol{\theta}$ 参数化的 Dirac）最初针对无 3D–2D 配对监督（即无真实 3D 标签）的 3D 场景。设 $\boldsymbol{\theta}\in\mathbb{R}^d$ 表示 3D 场景参数，$\mathrm{R}(\boldsymbol{\theta})$ 为可微渲染器，输出图像 $\hat{\mathbf{x}}_0:=\mathrm{R}(\boldsymbol{\theta})$。前向加噪过程定义为
$$
\hat{\mathbf{x}}_t=\alpha_t\,\mathrm{R}(\boldsymbol{\theta})+\sigma_t\boldsymbol{\epsilon},\quad \boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0},\mathrm{I}).
$$
预训练 2D（图像）扩散教师提供 score $\mathbf{s}_{\boldsymbol{\phi}^\times}(\hat{\mathbf{x}}_t,t|\mathbf{c})\approx\nabla_{\hat{\mathbf{x}}_t}\log p_t(\hat{\mathbf{x}}_t|\mathbf{c})$，可任选以文本 $\mathbf{c}$ 为条件。目标是在每个 $t$ 让加噪渲染的分布与教师的边缘对齐。一种最小表述是在渲染分布下的 score 对齐（VSD）目标：
$$
\mathcal{L}^{\text{3D}}_{\text{VSD}}(\boldsymbol{\theta})
:=\mathbb{E}_{t,\boldsymbol{\epsilon}}\!\left[\omega(t)\,\big\|
\mathbf{s}_{\boldsymbol{\zeta}}(\hat{\mathbf{x}}_t,t)-\mathbf{s}_{\boldsymbol{\phi}^\times}(\hat{\mathbf{x}}_t,t|\mathbf{c})
\big\|_2^2\right],
\quad \hat{\mathbf{x}}_t=\alpha_t\mathrm{R}(\boldsymbol{\theta})+\sigma_t\boldsymbol{\epsilon},
$$
通过渲染器将图像空间 score 引导传到 3D 参数。在更新 $\boldsymbol{\theta}$ 时对 $\hat{\mathbf{x}}_t$ 将两个 score 都做 stop gradient，得到
$$
\nabla_{\boldsymbol{\theta}}\mathcal{L}^{\text{3D}}_{\text{VSD}}(\boldsymbol{\theta})
=\mathbb{E}_{t,\boldsymbol{\epsilon}}\!\left[\omega(t)\,\alpha_t\,
\big(\mathbf{s}_{\boldsymbol{\zeta}}-\mathbf{s}_{\boldsymbol{\phi}^\times}\big)^\top
\frac{\partial \mathrm{R}}{\partial\boldsymbol{\theta}}(\boldsymbol{\theta})\right].
$$
当学生 score $\mathbf{s}_{\boldsymbol{\zeta}}$ 被压制（Dirac 生成器）时，即退化为 SDS。实践中，优化按 VSD 训练小节所述交替：先更新加噪渲染上的学生 score，再通过两个 score 的 stop gradient 更新 $\boldsymbol{\theta}$。更多数学细节此处从略。

---

## 渐进式蒸馏

Progressive Distillation (PD) 由两个步骤共同实现更高效地学习 PF-ODE 轨迹：在保持对教师轨迹保真度的同时，逐步减少高质量采样所需的积分步数。

- **蒸馏操作：** 将基于预训练教师模型（初始为扩散模型）的确定性采样器（如 DDIM）蒸馏为学生模型，使学生仅用一半的采样步数复现同一轨迹。
- **渐进操作：** 迭代重复该蒸馏过程，每次将步数减半，直至学生能在固定的小预算（通常 1–4 步）内生成高质量样本。

![渐进式蒸馏 (PD) 示意图。在每一轮中，学生模型被训练为单步复现相邻两个教师步的效果。该过程将 N 个教师步蒸馏为 N/2 个学生步，重复该过程逐步将轨迹长度减半直至达到期望步数。箭头表示从数据到噪声的方向上，多步教师转移如何被压缩为更少的学生步。](../arXiv-2510.21890v1/Images/PartD/PD.pdf)

我们先在「PD 中的蒸馏操作」介绍 PD 的蒸馏操作，再在「PD 的完整训练流程与采样」小结整个训练流程。「带引导的 PD」介绍 CFG 引导的扩展。

### PD 中的蒸馏操作

本节固定采用 $\mathbf{x}$-预测参数化下的 DDIM 作为步进规则，仍用 $\mathtt{Solver}_{s \to t}$ 表示将当前教师的 $\mathbf{x}$-去噪器代入 DDIM 得到的确定性映射。

在第一轮 PD（教师 = 预训练扩散模型）中，这与用 DDIM 积分扩散 PF-ODE 一致；在后续轮（教师 = 上一轮学生）中，$\mathtt{Solver}_{s \to t}$ 仅是当前教师诱导的 DDIM 转移，而非原始扩散 PF-ODE。

蒸馏步骤如下：从带噪输入 $\mathbf{x}_s$（干净数据的扰动版本，$\mathbf{x}_s = \alpha_s \mathbf{x}_0 + \sigma_s \boldsymbol{\epsilon}$）出发，训练学生预测目标 $\tilde{\mathbf{x}}$，使得学生的一步 $s\!\to\! t$ 复现教师的两步 $s\!\to\! u\!\to\! t$。记本轮教师的 $\mathbf{x}$-预测去噪器为 $\mathbf{x}_{\boldsymbol{\phi}^\times}(\mathbf{x},\tau)$。应用教师诱导的 DDIM 转移两次得
$$
\tilde{\mathbf{x}}_u := \mathtt{Solver}_{s\to u}\!\left(\mathbf{x}_s;\,\mathbf{x}_{\boldsymbol{\phi}^\times}\right),
\qquad
\tilde{\mathbf{x}}_t := \mathtt{Solver}_{u\to t}\!\left(\tilde{\mathbf{x}}_u;\,\mathbf{x}_{\boldsymbol{\phi}^\times}\right).
$$
此处使用求解器记号表示将 $\mathbf{x}_{\boldsymbol{\phi}^\times}$ 代入 DDIM 得到的从 $s$ 到 $t$（从 $\mathbf{x}_s$ 出发）的确定性转移映射。

**问题：** 时刻 $s$ 的伪干净 $\tilde{\mathbf{x}}$ 是什么，使得求解器直接执行 $s \to t$ 与经 $s \to u \to t$ 得到相同输出 $\tilde{\mathbf{x}}_t$？即求满足 $\tilde{\mathbf{x}}_t = \mathtt{Solver}_{s\to t}\left(\mathbf{x}_s; \tilde{\mathbf{x}}\right)$ 的 $\tilde{\mathbf{x}}$。

得到 $\tilde{\mathbf{x}}$ 的闭式后，训练学生模型 $\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}_s, s)$（此处也为 $\mathbf{x}$-预测模型）通过最小化下式逼近「两步合一」目标 $\tilde{\mathbf{x}}$：
$$
\min_{\boldsymbol{\theta}} \mathbb{E}_s
\mathbb{E}_{\mathbf{x}_s\sim p_s}\!\left[w(\lambda_s)\,\big\|\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}_s, s) - \tilde{\mathbf{x}}\big\|_2^2\right].
$$

下面说明 DDIM 规则通过初等代数给出 $\tilde{\mathbf{x}}$ 的闭式（结果对离散和连续时间均成立）。

**引理（DDIM 的「两步合一」目标 $\tilde{\mathbf{x}}$）** 从初值 $\mathbf{x}_s$ 出发，若求解器取为 DDIM，则「两步合一」目标 $\tilde{\mathbf{x}}$ 可计算为
$$
    \tilde{\mathbf{x}} 
    =\frac{\sigma_s}{\alpha_t\sigma_s - \alpha_s\sigma_t} \tilde{\mathbf{x}}_t - \frac{\sigma_t}{\alpha_t\sigma_s - \alpha_s\sigma_t} \mathbf{x}_s.
$$
其中 $\tilde{\mathbf{x}}_t$ 由对 DDIM 应用两次 $s \to u \to t$ 得到：
$$
    s \to u: \quad \tilde{\mathbf{x}}_u = \frac{\sigma_u}{\sigma_s}\mathbf{x}_s + \alpha_s \left(\frac{\alpha_u}{\alpha_s} - \frac{\sigma_u}{\sigma_s} \right)\mathbf{x}_{\boldsymbol{\phi}^\times}(\mathbf{x}_s, s)  \\\\
    u \to t: \quad \tilde{\mathbf{x}}_t = \frac{\sigma_t}{\sigma_u}\tilde{\mathbf{x}}_u + \alpha_u \left(\frac{\alpha_t}{\alpha_u} - \frac{\sigma_t}{\sigma_u} \right)\mathbf{x}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_u, u).
$$
证明：$\tilde{\mathbf{x}}_t$ 需与从 $s$ 到 $t$ 的一步 DDIM $\tilde{\mathbf{x}}_t'$ 相等，即 $s \to t$：$\tilde{\mathbf{x}}_t' = \frac{\sigma_t}{\sigma_s}\mathbf{x}_s + \alpha_s \left(\frac{\alpha_t}{\alpha_s} - \frac{\sigma_t}{\sigma_s} \right) \tilde{\mathbf{x}}$。令 $\tilde{\mathbf{x}}_t'=\tilde{\mathbf{x}}_t$ 可解出用 $\tilde{\mathbf{x}}_t,s,t$ 表示的 $\tilde{\mathbf{x}}$，即得上述闭式。

利用该公式，PD 计算出时刻 $s$ 的伪干净目标，使得单次 DDIM 步 $s\!\to\!t$ 恰好落在两步输出 $\tilde{\mathbf{x}}_t$。

**实践中的离散时间网格与损失。**

实践中固定递减网格 $t_0=T>t_1>\cdots>t_N=0$，简记 $s:=t_k$，$u:=t_{k+1}$，$t:=t_{k+2}$。教师提供单步映射 $\mathtt{Teacher}_{t_k\to t_{k+1}}$，学生学习与教师复合匹配的两步跳跃映射：
$$
\mathtt{Student}_{t_k\to t_{k+2}}
 \approx 
\mathtt{Teacher}_{t_{k+1}\to t_{k+2}}
\circ
\mathtt{Teacher}_{t_{k}\to t_{k+1}}.
$$

对 $k\in\{0,\ldots,N-2\}$ 采样三元组 $(s,u,t)=(t_k,t_{k+1},t_{k+2})$。目标变为
$$
\min_{\boldsymbol{\theta}} 
\mathbb{E}_{\,k \sim \mathcal{U}[\![0,N{-}2]\!]}\,
\mathbb{E}_{\,\mathbf{x}_{t_k}\sim p_{t_k}}\!
\Big[
  w(\lambda_{t_k})\,
  \big\|
  \mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}_{t_k},t_k) - \tilde{\mathbf{x}}^{(k)}
  \big\|_2^2
\Big],
$$
其中教师两步目标 $\tilde{\mathbf{x}}^{(k)}$ 由上述引理计算。若网格均匀，可写 $t_k = T(1 - k/N)$，即 $s = T(1 - k/N)$，$u = T(1 - (k+1)/N)$，$t = T(1 - (k+2)/N)$，对应步长 $\Delta s =  T/N$ 的等距时间步。

### PD 的完整训练流程与采样

通过上述优化在局部配对片段上训练后，$\mathtt{Student}$ 不再沿原网格的每个区间推进，而是每个学到的步覆盖 $\mathtt{Teacher}$ 的连续两步，因此 $\mathtt{Student}$ 在每隔一个时间点推进：
$$
t_0 \to t_2 \to t_4 \to \cdots \to t_N,
$$
从而仅用 $N/2$ 次转移遍历同一时间范围 $[0,T]$。该阶段后，训练好的 $\mathtt{Student}$ 取代 $\mathtt{Teacher}$ 作为新的去噪模型。随后在更粗的网格上重复该过程（时间步加倍），得到
$$
N \;\to\; N/2 \;\to\; N/4 \;\to\; \cdots,
$$
直至达到期望的推理步数。每轮迭代中，新 $\mathtt{Student}$ 由更新后的 $\mathtt{Teacher}$ 初始化。该迭代减半在保持全局时间范围的同时，逐步压缩生成过程的时间分辨率。

**采样。**

推理时，使用以当前 $\mathtt{Student}$ 为去噪器的（DDIM）求解器，采样器沿训练诱导的更粗网格推进。第一轮后为「跳 2」($t_0 \to t_2 \to \cdots \to t_N$)，下一轮为「跳 4」($t_0 \to t_4 \to \cdots \to t_N$)，依此类推，每轮将采样步数减半而保持起止时间不变。

### 补充讨论：局部半群匹配与广义求解器的可能性

**渐进式蒸馏即局部半群匹配。**

在统一目标中，难以处理的 oracle 目标 $\boldsymbol{\Psi}_{s\to 0}$ 被利用 ODE 流*半群性质*的教师诱导代理替代：从 $s$ 演化到 $t$ 应等价于从 $s$ 到某中间 $u$ 再从 $u$ 到 $t$，即 $\boldsymbol{\Psi}_{s\to t} = \boldsymbol{\Psi}_{u\to t}\circ\boldsymbol{\Psi}_{s\to u}$。PD 通过训练学生的一步映射匹配教师两段相邻一步的复合，在局部强制该性质。最小化该局部损失即在短递减网格上实例化半群等式，因此训练只需短的教师片段，而不需要从 $s$ 一直到 $0$ 的完整 rollout。

与少步去噪器视角的衔接：将学生的少步映射定义为学到的跳跃的复合。该局部损失在概念上为全局回归（学生 $s\to 0$ 对教师全复合）提供了高效的*局部代理*。

**能否使用其他求解器？**

上述 PD 介绍中，我们以 $\mathbf{x}$-预测参数化下的 DDIM 为具体的 PF-ODE 采样器。在确定性状态到状态映射层面，局部半群匹配与网格减半与求解器无关，可在参数化（$\mathbf{x}$、$\boldsymbol{\epsilon}$、$\mathbf{v}$、score）间标准转换后推广到求解器一章的时间步进方法。但此处的闭式伪目标依赖于*单步、显式*更新，其一步映射对回归目标是*仿射*的（如 DDIM 以及应用于 PF-ODE 的指数 Euler 或显式 RK 等单步显式格式）。对需要步历史或内层求解的*多步*或*隐式*求解器，应直接匹配相应的转移映射并提供必要历史或热启动；一般不存在类似的闭式反解。

若采样器是随机的，可对每个样本固定噪声序列得到确定性转移 $\mathtt{Teacher}^{(\omega)}_{s\to t}$（$\omega$ 为固定噪声种子）。此时 PD 回归到固定转移映射；闭式伪目标一般需要单步显式仿射更新，否则采用如局部损失中的直接匹配。

### 带引导的 PD

Meng 等人提出带 classifier-free guidance (CFG) 的扩散模型蒸馏的两阶段流程：(1) 将*引导蒸馏*进一个以引导权重为输入的单网络；(2) 应用*渐进式蒸馏 (PD)* 减少采样步数。他们在像素空间与潜空间（如 Stable Diffusion）中均进行了演示。

**阶段一蒸馏：引导的蒸馏。**

记 $\mathbf{x}_{\boldsymbol{\phi}^\times}(\mathbf{x}_s, s, \mathbf{c})$ 为（预训练）条件扩散模型在「$\mathbf{x}$-预测」参数化下的输出（即干净估计），时刻 $s$、条件 $\mathbf{c}$；条件也可为空 $\mathbf{c}=\emptyset$（无条件分支）。$\omega$-加权 CFG 组合可写为
$$
   \mathbf{x}_{\boldsymbol{\phi}^\times}^{\,\omega}(\mathbf{x}_s, s, \mathbf{c})
   := (1+\omega)\,\mathbf{x}_{\boldsymbol{\phi}^\times}(\mathbf{x}_s, s, \mathbf{c})
      - \omega\,\mathbf{x}_{\boldsymbol{\phi}^\times}(\mathbf{x}_s, s, \emptyset),
$$
其中 $\omega \sim p_\omega(\omega)$，$p_\omega$ 为某 CFG 权重分布，通常 $p_\omega(\omega)=\mathcal{U}[\omega_{\min},\omega_{\max}]$。

阶段一引入新模型 $\mathbf{x}_{\boldsymbol{\theta}_1}(\mathbf{x}_s, s, \mathbf{c}, \omega)$，直接以 $\omega$ 为输入，通过监督回归学习复现 CFG 输出 $\mathbf{x}_{\boldsymbol{\phi}^\times}^{\,\omega}(\mathbf{x}_s, s, \mathbf{c})$：
$$
\min_{\boldsymbol{\theta}_1}\;
\mathbb{E}_{\omega\sim p_\omega, s, \mathbf{x}\sim p_{\mathrm{data}}, \mathbf{x}_s \sim p(\mathbf{x}_s\mid \mathbf{x})}
\lambda(s)
\big\|\mathbf{x}_{\boldsymbol{\theta}_1}(\mathbf{x}_s, s, \mathbf{c}, \omega)-\mathbf{x}_{\boldsymbol{\phi}^\times}^{\omega}(\mathbf{x}_s, s, \mathbf{c})\big\|_2^2.
$$
其中 $\lambda(s)$ 为标准 schedule 依赖权重；每轮采样 $\omega$ 使单网络学会在任意引导强度下模仿 CFG。

**阶段二蒸馏：PD。**

阶段一模型 $\mathbf{x}_{\boldsymbol{\theta}_1}(\mathbf{x}_s, s, \mathbf{c}, \omega)$ 在 PD 中作为教师，按 PD 训练小节逐步蒸馏为步数更少的学生 $\mathbf{x}_{\boldsymbol{\theta}_2}(\mathbf{x}_s, s, \mathbf{c}, \omega)$。每轮迭代步数减半（如 $N \to N/2 \to N/4 \to \cdots$）。

---

## 结语

本章介绍了基于训练的加速的第一大范式。在穷尽基于数值求解器的无训练改进之后，我们将焦点转向新策略：训练一个快速的*学生*生成器，使其学会复现慢速、预训练*教师*扩散模型的行为。

我们探讨了两种主要蒸馏思路。其一，在以 Variational Score Distillation (VSD) 为代表的基于分布的蒸馏中，学生的输出分布被训练为与教师一致，通过在不同噪声水平上对齐两者的 score 函数实现，提供稳定、非对抗的目标。其二，在流映射蒸馏中，我们看到如 Progressive Distillation (PD) 等方法如何训练学生直接近似教师的解轨迹。PD 的迭代方式——每轮将采样步数减半——被证明是将长迭代过程压缩为少量几步的强有力且实用的方法。

这些蒸馏技术成功弥合了迭代扩散模型的高样本质量与一步生成器推理速度之间的鸿沟，为高效、高保真合成提供了令人信服的路径。

然而，对预训练教师模型的依赖引入了两阶段流程：先训练慢而强的教师，再蒸馏为快速学生。这引出了生成模型研究前沿的一个根本问题：我们能否完全绕过教师？能否设计一种独立的训练原理，直接从数据学习这些快速的少步生成器？本专著的最后一章将回答这一问题：(1) 我们将探讨如 Consistency Models 等开创性方法，它们学习从 ODE 轨迹上任意点到终点的一步映射；(2) 我们将深入 Consistency Models 的推广概念，学习在一步内将轨迹上任意点映射到另一点。

从改进求解器或蒸馏解，转向学习解映射本身，标志着向一类既原则化又高效的新生成模型迈出重要一步。
