# 第 3 章 扩散模型的当下：分数 SDE 框架

> *表述定律只有一种精确的方式，那就是微分方程。它们具有根本性，且就我们所知是精确的。*  
> —— Richard P. Feynman

此前我们从两个视角学习了扩散模型：变分视角与基于分数的视角，后者自然地从 EBM 表述中产生。现在我们进一步转向**连续时间框架**。其核心是**分数 SDE（Score SDE）**——将 DDPM 与 NCSN 统一为单一表述的连续极限。这一视角之所以有力，是因为它用基于微分方程（DE）的清晰、有原则的描述扩展了离散更新。

在这一视角下，生成等价于随时间求解一个微分方程。因此可以直接应用数值分析中的工具：例如基本 Euler 方法可以模拟动力学，而更高级的求解器能提高稳定性和效率。在连续时间下工作还带来更丰富的数学结构和统一的基础，用于理解、分析和改进扩散模型。本专著将在后文进一步展开这一视角。

---

## 3.1 分数 SDE：其原理

多噪声尺度在 NCSN 与 DDPM 框架的成功中一直是关键要素。本节介绍 **Score SDE** 的基础，将这一思想提升为考虑噪声水平的连续统。前向与逆向扩散过程的连续时间极限早已被 Sohl-Dickstein 等人指出，但 Song 等人将数据演化表述为随机/常微分方程（噪声水平随时间平滑增加），使这一视角成为中心。这一连续时间表述不仅统一了以往的离散时间模型，还将生成建模归结为求解微分方程的问题，提供了有原则且灵活的基础。

### 3.1.1 动机：从离散到连续时间过程

![离散时间加噪步示意图](../arXiv-2510.21890v1/Images/PartB/sde-forward-noise.pdf)

**图 1：离散时间加噪步的示意图。** 在 $t$ 到 $t+\Delta t$ 间以均值漂移 $\mathbf{f}(\mathbf{x}_t, t)$ 和扩散系数 $g(t)$ 加噪。

我们回顾 NCSN 与 DDPM 的前向加噪方案。NCSN 使用递增噪声水平序列 $\{\sigma_i\}_{i=1}^L$。每个干净样本 $\mathbf{x} \sim p_{\mathrm{data}}$ 被扰动为
$$
\mathbf{x}_{\sigma_i} = \mathbf{x} + \sigma_i \boldsymbol{\epsilon}_i, 
\qquad \boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$
DDPM 则用方差调度 $\{\beta_i\}_{i=1}^L$ 逐步加噪：
$$
\mathbf{x}_i = \sqrt{1 - \beta_i^2} \mathbf{x}_{i-1} + \beta_i \boldsymbol{\epsilon}_i, 
\qquad \boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$

我们在离散时间格点上统一看待它们：从 $\mathbf{x}_t$ 到 $\mathbf{x}_{t+\Delta t}$ 的序贯更新具有形式（为方便起见，我们用 $\mathbf{x}(t)$ 与 $\mathbf{x}_t$ 可互换地表示时刻 $t$ 的样本）：
$$
\text{NCSN:} \quad \mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \sqrt{\sigma_{t+\Delta t}^2 - \sigma_t^2}\boldsymbol{\epsilon}_t \approx \mathbf{x}_t + \sqrt{\frac{\mathrm{d} \sigma_t^2}{\mathrm{d} t} \Delta t} \boldsymbol{\epsilon}_t
$$
$$
\text{DDPM:} \quad \mathbf{x}_{t+\Delta t} = \sqrt{1 - \beta_t} \mathbf{x}_t + \sqrt{\beta_t} \boldsymbol{\epsilon}_t \approx \mathbf{x}_t - \frac{1}{2} \beta_t \mathbf{x}_t \Delta t + \sqrt{\beta_t \Delta t} \boldsymbol{\epsilon}_t,
$$
其中 $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。有趣的是，两种加噪过程都遵循共同的结构形式：
$$
\mathbf{x}_{t+\Delta t} \approx \mathbf{x}_t + \mathbf{f}(\mathbf{x}_t, t) \Delta t + g(t) \sqrt{\Delta t} \boldsymbol{\epsilon}_t,
$$
其中 $\mathbf{f}: \mathbb{R}^D \times \mathbb{R} \to \mathbb{R}^D$ 与 $g: \mathbb{R} \to \mathbb{R}$ 由下式给出：
$$
\text{NCSN:} \quad \mathbf{f}(\mathbf{x}, t) = 0, \qquad g(t) = \sqrt{\frac{\mathrm{d} \sigma^2(t)}{\mathrm{d} t}};\qquad
\text{DDPM:} \quad \mathbf{f}(\mathbf{x}, t) = -\frac{1}{2} \beta(t) \mathbf{x}, \qquad g(t) = \sqrt{\beta(t)}.
$$
这对应如下高斯转移：
$$
p(\mathbf{x}_{t+\Delta t} | \mathbf{x}_t) := \mathcal{N}\left(\mathbf{x}_{t+\Delta t}; \mathbf{x}_t + \mathbf{f}(\mathbf{x}_t, t)\Delta t, g^2(t) \Delta t \mathbf{I}\right),
$$
这里我们略作记号滥用，将 $\mathbf{x}_t$ 视为固定样本，$\mathbf{x}_{t+\Delta t}$ 视为随机变量。

当 $\Delta t \to 0$（可理解为准备**无穷多**层噪声）时，离散时间过程收敛到随时间前向演化的连续时间 SDE（前向核在 $\Delta t\to0$ 时收敛到相应 Itô SDE 的解；完全严格的证明依赖进阶结果，此处从略）：
$$
\mathrm{d} \mathbf{x}(t) = \mathbf{f}(\mathbf{x}(t), t) \mathrm{d} t + g(t) \mathrm{d} \mathbf{w}(t),
$$
其中 $\mathbf{w}(t)$ 为标准 Wiener 过程（或布朗运动）。**注：** *Wiener 过程*是连续时间随机过程 $\mathbf{w}(t)$：从零出发、具有独立增量，且对任意 $s < t$，增量 $\mathbf{w}(t) - \mathbf{w}(s)$ 服从均值为零、方差为 $t - s$ 的正态分布。它表示随时间累积的独立高斯涨落，几乎必然连续但处处不可微。在无穷小时间区间 $[t, t + \mathrm{d} t]$ 上，Wiener 过程的增量定义为 $\mathrm{d} \mathbf{w}(t) := \mathbf{w}(t + \mathrm{d} t) - \mathbf{w}(t)$，可建模为零均值、方差 $\mathrm{d} t$ 的高斯随机变量：$\mathrm{d} \mathbf{w}(t) \sim \mathcal{N}(\mathbf{0}, \mathrm{d} t \mathbf{I})$。

离散与连续表述之间的概念对应关系可理解为：
- $\mathbf{x}(t+\Delta t) - \mathbf{x}(t) \approx \mathrm{d} \mathbf{x}(t)$，
- $\Delta t \approx \mathrm{d} t$，
- $\sqrt{\Delta t} \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \Delta t \mathbf{I}) \approx \mathrm{d} \mathbf{w}(t)$。

一旦漂移 $\mathbf{f}(\mathbf{x},t)$ 和扩散 $g(t)$ 给定，前向时间 SDE 就自动诱导出一个逆向时间 SDE，将终端噪声分布输送回数据分布。逆向动力学中只涉及一个未知项，即**每个连续时间水平上的分数函数**。因此训练目标就是分数匹配；一旦学到分数，采样就是用学到的分数对逆向时间 SDE 进行数值积分。

下面我们先在 3.1.2 与 3.1.3 小节考察前向与逆向过程的理论基础；3.2 节再给出训练与采样的具体实现。

### 3.1.2 前向时间 SDE：从数据到噪声

![扩散模型前向过程（1D）可视化](../arXiv-2510.21890v1/Images/PartB/forward_sde_ode_trajectories.pdf)

**图 2：（1D）扩散模型中前向过程的可视化。** 过程从复杂双峰数据分布 $p_0 = p_{\mathrm{data}}$ 的初始采样点（标为「$\times$」）出发，演化向简单单峰高斯先验 $p_T \approx p_{\mathrm{prior}}$。背景热图表示演化的边缘概率密度 $p_t$，随时间平滑。样本轨迹从 $t=0$ 到 $t=T$，比较随机前向 SDE（蓝色路径）与确定性对应物 PF-ODE（白色路径）。注意 PF-ODE 是密度的确定性输运映射，一般不是从单点出发的样本路径的均值。

在此表述下，基于离散时间的早期方法（如 NCSN、DDPM）可通过在区间 $[0, T]$ 上由前向 SDE 驱动的随机过程 $\mathbf{x}(t)$ 统一到**连续时间**框架下：
$$
\mathrm{d}\mathbf{x}(t) = \mathbf{f}(\mathbf{x}(t), t) \mathrm{d} t + g(t) \mathrm{d}\mathbf{w}(t), \quad \mathbf{x}(0) \sim p_{\mathrm{data}}.
$$
其中 $\mathbf{f}(\cdot, t):\mathbb{R}^D\to\mathbb{R}^D$ 为漂移，$g(t) \in \mathbb{R}$ 为标量扩散系数，$\mathbf{w}(t)$ 为标准 Wiener 过程。我们称其为**前向 SDE**，描述干净数据如何随时间逐渐被扰动为噪声。

一旦漂移 $\mathbf{f}$ 与扩散系数 $g$ 给定，前向过程就完全确定，描述数据变量如何通过注入高斯噪声逐步被破坏。特别地，会诱导出两类随时间变化的密度：

**扰动核。** 条件律 $p_t(\mathbf{x}_t |\mathbf{x}_0)$ 描述干净数据样本 $\mathbf{x}_0 \sim p_{\mathrm{data}}$ 在时刻 $t$ 如何演化为带噪版本 $\mathbf{x}_t$。一般地，前向 SDE 中的漂移项 $\mathbf{f}(\mathbf{x},t)$ 可以是 $\mathbf{x}$ 的任意函数，但常用且便于分析的选择是假定其为仿射：
$$
\mathbf{f}(\mathbf{x},t) = f(t) \mathbf{x},
$$
其中 $f(t)$ 为 $t$ 的标量函数，通常取为非正。在此结构下，过程在每一时刻保持高斯，条件分布可通过求解相应的均值–方差 ODE 得到闭式解。具体地，
$$
p_t(\mathbf{x}_t |\mathbf{x}_0) = \mathcal{N} \big(\mathbf{x}_t; \mathbf{m}(t), P(t)\mathbf I_D\big),
$$
其中
$$
\mathbf{m}(t) = \exp \Big(\int_0^t f(u) \mathrm{d} u\Big) \mathbf{x}_0,\quad P(t) = \int_0^t \exp \Big(2 \int_s^t f(u) \mathrm{d} u\Big) g^2(s) \mathrm{d} s,
$$
初始条件为 $\mathbf{m}(0)=\mathbf{x}_0$，$P(0)=0$。

这一显式形式允许在给定 $\mathbf{x}_0$ 时直接采样 $\mathbf{x}_t$，而无需对 SDE 做数值积分，故称**无需仿真**。NCSN 与 DDPM 都属于这一仿射漂移设定。后文我们对任意漂移 $\mathbf{f}(\mathbf{x},t)$ 发展一般理论，在需要闭式分析时再回到仿射漂移。

**边缘密度。** 时间边缘密度 $p_t(\mathbf{x}_t)$ 由扰动核积分得到：
$$
p_t(\mathbf{x}_t) := \int p_t(\mathbf{x}_t|\mathbf{x}_0) p_{\mathrm{data}}(\mathbf{x}_0) \mathrm{d} \mathbf{x}_0, \quad \text{其中 } p_0 = p_{\mathrm{data}}.
$$

适当选择系数 $f(t)$ 和 $g(t)$ 时，前向过程会逐渐加噪直至初始状态的影响被有效遗忘。当 $T$ 很大时，条件分布 $p_T(\mathbf{x}_T |\mathbf{x}_0)$ 不再依赖 $\mathbf{x}_0$，因为其均值满足 $\mathbf{m}(T) = \exp(\int_0^T f(u)\mathrm{d} u) \mathbf{x}_0 \to \mathbf{0}$（当 $T\to\infty$，只要 $f(u)$ 非正使指数因子衰减）。同时方差增长并稳定到所选先验。因此，最初表示对数据样本复杂混合的边缘 $p_T(\mathbf{x}_T) = \int p_T(\mathbf{x}_T |\mathbf{x}_0) p_{\mathrm{data}}(\mathbf{x}_0)\mathrm{d}\mathbf{x}_0$ 收敛到简单先验 $p_{\mathrm{prior}}$（通常为高斯）。在这一极限下，$p_T(\mathbf{x}_T) \approx p_{\mathrm{prior}}(\mathbf{x}_T)$ 且 $p_T(\mathbf{x}_T |\mathbf{x}_0) \approx p_{\mathrm{prior}}(\mathbf{x}_T)$，故前向过程将任意数据分布映射到易处理先验，为逆转与生成提供了方便的起点。

### 3.1.3 用于生成的逆向时间随机过程

![用于数据生成的逆向时间随机过程](../arXiv-2510.21890v1/Images/PartB/reverse_sde_and_ode_trajectories.pdf)

**图 3：用于数据生成的逆向时间随机过程可视化。** 从 $t=T$ 时简单先验 $p_{\mathrm{prior}}$ 的采样（标为「$\times$」）出发，用逆向 SDE 沿时间向后演化。所得轨迹在 $t=0$ 终止，共同形成目标双峰数据分布 $p_0 = p_{\mathrm{data}}$。背景热图展示概率密度如何从简单高斯逐渐变为复杂目标分布。

直观上，从噪声生成数据可以通过「逆转」前向过程实现：从先验分布中采样的随机点出发，沿时间向后演化得到生成样本。对确定性系统（即 ODE），这一想法自然成立：没有随机性，逆转时间无非是沿前向过程的同一条路径反向追踪点的轨迹（技术上对应做时间翻转 $t \leftrightarrow T - t$ 后求解 ODE）。相比之下，SDE 在每步都含有随机性，单点可沿多条可能的随机轨迹演化，因此逆转这类过程更微妙（简单翻转时间并不能得到正确的逆向过程）。

虽然单条随机轨迹不可逆，但重要结论是：**这些轨迹上的分布是可以逆转的**。Anderson 的奠基性结果将前向 SDE 的逆向时间过程 $\{\bar{\mathbf{x}}(t)\}_{t \in [0, T]}$（我们用上划线 $\bar{\cdot}$ 区分逆向过程与前向 SDE 定义的前向过程）形式化为一个良定义的 SDE。该逆向时间过程从 $T$ 演化到 $0$，动力学由下式给出：
$$
\mathrm{d}\bar{\mathbf{x}}(t) = \left[\mathbf{f}(\bar{\mathbf{x}}(t), t) - g^2(t) \nabla_{\mathbf{x}} \log p_t(\bar{\mathbf{x}}(t)) \right] \mathrm{d} t + g(t) \mathrm{d} \bar{\mathbf{w}}(t),\quad \bar{\mathbf{x}}(T) \sim p_{\mathrm{prior}} \approx p_T.
$$
其中 $\bar{\mathbf{w}}(t)$ 表示逆向时间上的标准 Wiener 过程，定义为 $\bar{\mathbf{w}}(t) := \mathbf{w}(T - t) - \mathbf{w}(T)$。

随机性 ($g\neq 0$) 的存在会引入额外修正项 $-g^2(t) \nabla_{\mathbf{x}} \log p_t(\bar{\mathbf{x}}(t))$，用于刻画扩散效应并保证逆向动力学正确复现前向 SDE 诱导的边缘分布演化（见 3.1.5 节）。

**概念上，逆向过程为何有效？** 逆向时间 SDE 中出现布朗噪声初看可能令人困惑：前向扩散把数据扩散成越来越噪的构型，为何逆转这一过程（尤其还通过 $\bar{\mathbf{w}}(t)$ 引入额外随机性）却能产生集中在数据流形附近的干净、结构化样本？要点在于逆向时间 SDE 并非注入任意随机性。扩散项 $g(t)\mathrm{d}\bar{\mathbf{w}}(t)$ 始终与由分数驱动的漂移 $- g^2(t)\nabla_{\mathbf{x}}\log p_t(\bar{\mathbf{x}}(t))$ 耦合。两者相互平衡：分数将轨迹导向更高密度区域，噪声则引入受控随机性以利探索而不淹没动力学。

当 $f(t)\equiv 0$ 时，逆向 SDE 可写为
$$
\mathrm{d}\bar{\mathbf{x}}(t) = - g^2(t) \nabla_{\mathbf{x}}\log p_t(\bar{\mathbf{x}}(t)) \mathrm{d} t + g(t) \mathrm{d}\bar{\mathbf{w}}(t).
$$
令 $s:=T-t$（故 $\mathrm{d} t=-\mathrm{d} s$），并在分布意义下重命名布朗运动使 $\mathrm{d}\bar{\mathbf{w}}(t)=-\mathrm{d}\mathbf{w}_s$。记 $\bar{\mathbf{x}}_s:=\bar{\mathbf{x}}(T-s)$ 和 $\pi_s:=p_{T-s}$，则得到具有时变温度 $\tau(s):=\tfrac{1}{2}g^2(T-s)$、以演化密度 $\pi_s$ 为目标的 Langevin 形式。由 Tweedie 公式，分数方向 $\nabla\log \pi_s$ 在每个时间切片指向条件干净信号，因此漂移不断「拉回」去噪结构。关键的是，$g(t)$ 沿逆向轨迹是**退火**的：初期 $g(T-s)$ 较大、注入噪声较强、过程大范围探索；随着 $s$ 增大，$g(T-s)$ 减小、随机项变弱、分数项主导，将样本拉入 $\pi_s$ 的高密度区域；到 $s=T$（即 $t=0$）时，轨迹集中在数据流形附近。

**逆向时间 SDE 能力概览。** 随时间变化的分数函数 $\mathbf{s}(\mathbf{x}, t) := \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ 自然出现在逆向 SDE 中。一旦前向系数 $f(t)$ 和 $g(t)$ 给定，分数就是逆向动力学中唯一未知量。这凸显其核心地位：有了分数，逆向过程就确定，采样即是用学到的分数对逆向 SDE 做数值积分。由于真实分数一般没有闭式，我们采用分数匹配用神经网络 $\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}, t)$ 近似它（详见 3.2 节训练部分）。在逆向 SDE 中用 $\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}, t)$ 替代 $\mathbf{s}(\mathbf{x}, t)$ 即完全确定逆向动力学。生成对应于从 $t = T$、$\mathbf{x}_T \sim p_{\mathrm{prior}}$ 出发逆向求解到 $t = 0$。Anderson 证明前向与逆向过程的边缘密度一致，从而当 $p_{\mathrm{prior}} \approx p_T$ 时 $t=0$ 的样本近似服从 $p_{\mathrm{data}}$。

### 3.1.4 用于生成的确定性过程（概率流 ODE）

尽管逆向时间 SDE 引入随机性并可能增加生成样本的多样性，自然的问题是：**是否必须用该 SDE 采样？**

受 Maoutsa 等人启发，Song 等人还引入了一个确定性过程——一个 ODE，其样本演化与前向 SDE 具有相同的边缘分布。该过程 $\{\tilde{\mathbf{x}}(t)\}_{t\in[0,T]}$（我们用波浪号区分与前向/逆向 SDE 相关的过程；后文为简记会省略这一区分）称为**概率流 ODE（PF-ODE）**，由下式给出：
$$
\frac{\mathrm{d}}{\mathrm{d} t}\tilde{\mathbf{x}}(t) = \mathbf{f}(\tilde{\mathbf{x}}(t), t) - \frac{1}{2} g^2(t) \nabla_{\mathbf{x}} \log p_t(\tilde{\mathbf{x}}(t)).
$$

与 SDE 情形类似，可用学到的近似替换真实分数，并从 $t = T$ 到 $t = 0$ 积分逆向时间 ODE 以生成样本。具体地，生成样本（PF-ODE 在 $t=0$ 的解）形如
$$
\tilde{\mathbf{x}}(T) + \int_{T}^{0} \Big[ \mathbf{f}(\tilde{\mathbf{x}}(\tau), \tau) - \frac{1}{2} g^2(\tau)  \nabla_{\mathbf{x}} \log p_\tau(\tilde{\mathbf{x}}(\tau)) \Big]   \mathrm{d} \tau,
$$
初始条件为 $\tilde{\mathbf{x}}(T) \sim p_{\mathrm{prior}}$。该积分无闭式，实际生成依赖数值求解器（如 Euler 方法）。

相比逆向时间 SDE，PF-ODE 有两个明显优点：（1）ODE 可沿两个方向积分（从 $t=0$ 到 $t=T$ 或从 $t=T$ 到 $t=0$），只要在所选端点给定相应初始条件；这与一般只能前向积分的 SDE 不同。（2）可充分利用为 ODE 开发的众多成熟、现成数值求解器。

需要强调的是，PF-ODE 并非简单去掉逆向 SDE 中的扩散项；其漂移中的系数 $\frac{1}{2}$ 有明确来源。高层面上，该 ODE 的漂移被选成使其演化保持与前向 SDE 相同的边缘密度。保证这一对齐的底层原理（即 Fokker–Planck 方程）将在下一小节详述。

### 3.1.5 前向/逆向时间 SDE 与 PF-ODE 中边缘分布的对齐

![边缘密度 p_t 的时空演化（2D）](../arXiv-2510.21890v1/Images/PartB/density_evolution_3d_orange.pdf)

**图 4：（2D）边缘密度 $p_t$ 随时间演化。** 前向 SDE 在 $[0,T]$ 上取 $\mathbf{f} \equiv \mathbf{0}$，$g(t) = \sqrt{2t}$。从双峰高斯混合 $p_0 = p_{\mathrm{data}}$ 出发，在 $p_T \approx p_{\mathrm{prior}} := \mathcal{N}(\mathbf{0}, T^2 \mathbf{I})$ 结束。$p_t$ 的时空演化由 Fokker–Planck 方程刻画。

**用 Fokker–Planck 方程保证边缘密度对齐。** 扩散模型的一个核心概念是：**不同过程可以导致相同的边缘分布序列**。目标是构造一个过程，通过在各时刻（尤其 $t=0$）对齐边缘，将 $p_{\mathrm{prior}}$ 变为 $p_{\mathrm{data}}$。这自然引出一个基本问题：**如何保证不同过程给出相同的边缘分布？**

一旦前向 SDE 给定，它就定义了从 $p_{\mathrm{data}}$ 到 $p_{\mathrm{prior}}$ 的边缘密度演化。逆向时间 SDE 与 PF-ODE 的构造使得它们的轨迹给出的边缘分布与前向过程完全一致。这一对应的关键在于 Fokker–Planck 方程。下列定理建立了这一联系：(i) **PF-ODE** 若从 $\tilde{\mathbf{x}}(0)\sim p_0$ 前向运行或从 $\tilde{\mathbf{x}}(T)\sim p_T$ 逆向运行，则对所有 $t\in[0,T]$ 有边缘 $\tilde{\mathbf{x}}(t)\sim p_t$。(ii) **逆向时间 SDE** 有边缘 $\bar{\mathbf{x}}(t)\sim p_{T-t}$。证明见附录。

**固定边缘下的多种条件分布。** 考虑**流映射** $\Psi_{s \to t}(\mathbf{x}_s) := \mathbf{x}_s + \int_s^t \mathbf{v}(\mathbf{x}_\tau, \tau)\,\mathrm{d}\tau$，速度场为 $\mathbf{v}(\mathbf{x}, \tau) := \mathbf{f}(\mathbf{x}, \tau) - \frac{1}{2} g^2(\tau)\nabla_{\mathbf{x}}\log p_\tau(\mathbf{x})$。定理保证前推密度 $p_t^{\mathrm{fwd}} = p_t$。因此无穷多个条件分布 $Q_t(\mathbf{x}_t | \mathbf{x}_0)$ 可给出同一个 $p_t(\mathbf{x}_t)$，例如：随机（无需仿真）、确定性（需求解 ODE）、或混合。**观察** 多种过程可以产生相同的边缘密度序列；真正重要的是满足 Fokker–Planck 方程。Fokker–Planck 方程位于扩散模型的核心，根植于概率密度的变量替换公式。

### 3.1.6 可计算例子：高斯动力学的演化

当 $p_{\mathrm{data}}$ 为正态分布（或高斯混合）时，分数函数有闭式表达式。可以仅用初等微积分显式推导逆向时间 SDE 与 PF-ODE。**高斯的逆向时间 SDE：** 从前向 SDE 的一小步 Euler 与 Bayes 规则出发，可得到逆向转移的均值和方差；令 $\Delta t\to 0$ 即得逆向 SDE，其漂移可用分数写出（因高斯边缘有 $\partial_x \log p_t(x) = -(x-m_t)/s_t^2$）。**高斯的 PF-ODE：** 设 $x_t\sim\mathcal N(m_t,s_t^2)$，小步长确定性步需保持高斯；可推出 $v_t$ 必须为线性加平移，并与前向 SDE 的均值、方差 ODE 匹配，得到 PF-ODE $x'(t)  =  f(t) x(t)  -  \tfrac12 g^2(t) \partial_x\log p_t(x(t))$。该 ODE 与前向 SDE 在每时刻 $t$ 具有相同的边缘 $p_t=\mathcal N(m_t,s_t^2)$。

---

## 3.2 分数 SDE：训练与采样

### 3.2.1 训练

延续基于分数一章的思想，我们用时间条件的神经网络 $\mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}, t)$ 在所有 $t\in[0,T]$ 上近似真实分数 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$，通过最小化分数匹配目标（其中 $p_{\text{time}}$ 为某时间分布，$\omega(\cdot)$ 为时间权重函数）。为避免依赖难以处理的真实分数，采用 DSM 损失；在给定 $\mathbf{x}_0$ 下使用解析可处理的 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0)$。**命题（DSM 的最小元）** 最小元 $\mathbf{s}^*$ 满足 $\mathbf{s}^*(\mathbf{x}_t, t)  = \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0 | \mathbf{x}_t)} [ \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{x}_0) ]= \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$（几乎处处）。证明见附录。

### 3.2.2 采样与推断

![从分数 SDE 采样的（2D）示意图](../arXiv-2510.21890v1/Images/PartB/sde_ode_trajectories.pdf)

**图 5：（2D）从分数 SDE 采样的示意图。** 采样通过求解逆向时间 SDE（蓝色；Euler–Maruyama）与 PF-ODE（红色；Euler）实现，前向 SDE 设定与图 4 相同。从先验 $\mathbf{x}_T\sim p_{\mathrm{prior}}$（深色「$\times$」）出发，两条轨迹均在 $t=0$ 终止于 $p_{\mathrm{data}}$ 支撑附近。

学到 $\mathbf{s}_{\boldsymbol{\phi}^*}(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ 后，用其替换逆向时间 SDE 与 PF-ODE 中的真实分数即可进行推断。**经验逆向时间 SDE**：用训练好的 $\mathbf{s}_{\boldsymbol{\phi}^*}$ 代入逆向 SDE；从先验采 $\mathbf{x}_T$，从 $t=T$ 到 $t=0$ 用 Euler–Maruyama 数值求解。迭代得 $\mathbf{x}^{\mathrm{SDE}}_{\boldsymbol{\phi}^*}(0)$，其分布近似 $p_{\mathrm{data}}$。DDPM 采样是该离散化的特例。**经验 PF-ODE**：用 $\mathbf{s}_{\boldsymbol{\phi}^*}$ 代入 PF-ODE，同样从 $t=T$ 到 $t=0$ 数值积分得 $\mathbf{x}^{\mathrm{ODE}}_{\boldsymbol{\phi}^*}(0)$。Euler 更新为 $\mathbf{x}_{t - \Delta t} \leftarrow \mathbf{x}_t -[\mathbf{f}(\mathbf{x}_t, t) - \frac{1}{2} g^2(t) \mathbf{s}_{\boldsymbol{\phi}^*}(\mathbf{x}_t, t)] \Delta t$。

**洞察** 从扩散模型采样本质上等价于求解相应的概率流 ODE 或逆向时间 SDE；这也解释了采样慢——数值求解需大量步数（如约 1,000 次函数求值）。PF-ODE 的优点是可用成熟 ODE 求解器加速。**逆映射**：PF-ODE 可正向求解，将数据映为各时刻的潜表示（编码器），用于可控生成。**精确对数似然**：将 PF-ODE 视为仅参数化分数的 Neural ODE，通过变量替换可计算 $\log p^{\text{ODE}}_{\boldsymbol{\phi}^*}(\mathbf{x}_0; 0) = \log p_{\mathrm{prior}}( \mathbf{x}(T) ) + \delta(T)$（积分增广 ODE 得到 $\mathbf{x}(T)$ 与 $\delta(T)$）。

---

## 3.3 SDE 的实例化

前向 SDE 的漂移与扩散可按方差演化行为分为几类；这里聚焦 **VE SDE**（方差爆炸）与 **VP SDE**（方差保持）。表 1 概括这两种 SDE 实例。

**表 1：前向 SDE 小结**

|  | **VE SDE** | **VP SDE** |
|---|---|---|
| $\mathbf{f}(\mathbf{x}, t)$ | $\mathbf{0}$ | $-\frac{1}{2}\beta(t)\mathbf{x}$ |
| $g(t)$ | $\sqrt{\frac{\mathrm{d} \sigma^2(t)}{\mathrm{d} t}}$ | $\sqrt{\beta(t)}$ |
| SDE | $\mathrm{d}\mathbf{x}(t) =  g(t) \mathrm{d}\mathbf{w}(t)$ | $\mathrm{d}\mathbf{x}(t) = -\frac{1}{2}\beta(t) \mathbf{x}(t) \mathrm{d} t + \sqrt{\beta(t)} \mathrm{d}\mathbf{w}(t)$ |
| $p_t(\mathbf{x}_t\mid\mathbf{x}_0)$ | $\mathcal{N}\left(\mathbf{x}_t ; \mathbf{x}_0, \left(\sigma^2(t) - \sigma^2(0)\right)\mathbf{I}\right)$ | $\mathcal{N}\left(\mathbf{x}_t ; \mathbf{x}_0 e^{-\frac{1}{2}\int_{0}^{t} \beta(\tau)\mathrm{d} \tau}, \mathbf{I} - \mathbf{I} e^{-\int_{0}^{t} \beta(\tau)\mathrm{d} \tau}\right)$ |
| $p_{\mathrm{prior}}$ | $\mathcal{N}(\mathbf{0}, \sigma^2(T)\mathbf{I})$ | $\mathcal{N}(\mathbf{0}, \mathbf{I})$ |

**VE SDE**：$\mathbf{f}=\mathbf{0}$，$g(t)=\sqrt{\mathrm{d} \sigma^2(t)/\mathrm{d} t}$；扰动核 $p_t(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t ; \mathbf{x}_0, (\sigma^2(t) - \sigma^2(0))\mathbf{I})$，先验 $\mathcal{N}(\mathbf{0}, \sigma^2(T)\mathbf{I})$。NCSN 对应 $\sigma(t)=\sigma_{\text{min}}(\sigma_{\text{max}}/\sigma_{\text{min}})^t$。**VP SDE**：$\mathbf{f}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x}$，$g(t) = \sqrt{\beta(t)}$；扰动核与先验见上表；DDPM 对应 $\beta(t) := \beta_{\text{min}} + t(\beta_{\text{max}} - \beta_{\text{min}})$。**（可选）扰动核的推导**：线性漂移 $f(t)\mathbf{x}$ 时，均值与方差满足 ODE，可得闭式转移核；VE/VP 的特例见正文。

---

## （可选）基于分数与变分扩散模型的前向核再认识

DDPM 与 Score SDE 通常分别通过离散定义的前向转移核 $p(\mathbf{x}_t|\mathbf{x}_{t-\Delta t})$ 与连续时间 SDE 引入，但实践中（尤其在其损失中）更相关的是从数据出发的累积转移核 $p_t(\mathbf{x}_t|\mathbf{x}_0)$。本节直接从（连续时间）$p_t(\mathbf{x}_t|\mathbf{x}_0)$ 出发，给出更简洁的视角。

![引理（前向扰动核 ⇔ 线性 SDE）示意图](../arXiv-2510.21890v1/Images/PartB/vdm-sde.pdf)

**图 6：引理（前向扰动核 $\Leftrightarrow$ 线性 SDE）的示意图。** 连续时间 SDE 的增量加噪（$\Delta t\to 0$）与直接扰动（前向核）在数学上等价。

**一般仿射前向过程** 定义 $p_t(\mathbf{x}_t| \mathbf{x}_0) := \mathcal{N}(\mathbf{x}_t; \alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$，其中 $\alpha_t,\sigma_t$ 为 $t\in[0,T]$ 的非负标量函数，通常 $\alpha_0=1$，$\sigma_0=0$。采样形式为 $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$。该框架涵盖 VE（NCSN）、VP（DDPM）及 Flow Matching 前向核等。**与 Score SDE 的联系** 给定该前向扰动核，对应前向 SDE 为关于 $\mathbf{x}$ 线性的形式，系数 $f(t)$、$g(t)$ 可由 $\alpha_t$、$\sigma_t$ 解析表出（见引理：前向扰动核 $\Leftrightarrow$ 线性 SDE）。**与变分扩散模型的联系** DDPM 的核心等式在连续时间下可推出逆向条件转移核，进而得到参数化逆向模型与扩散损失（KL 可化为加权回归，含 SNR 项），以及采样公式。

---

## （可选）Fokker–Planck 方程与逆向时间 SDE：基于边际化与 Bayes 规则的视角

本节从概率角度说明 Fokker–Planck 方程与逆向时间 SDE 的结构，利用边际化技巧和 Bayes 规则阐明随机过程的统计表述与相应微分方程之间的联系。此处为启发式论证而非严格证明。**从转移核的边际化得到 Fokker–Planck 方程** 给定前向转移概率与边缘，由 Markov 性与 Chapman–Kolmogorov 方程做变量替换与 Taylor 展开，令 $\Delta t\to 0$ 即得 Fokker–Planck 方程。**逆向时间 SDE 为何是这种形式？** 先考虑离散时间逆向转移 $p(\mathbf{x}_t | \mathbf{x}_{t+\Delta t})$，用 Bayes 写成前向核乘以 $p_t/p_{t+\Delta t}$；对指数项做一阶 Taylor、配方，即得逆向转移为均值含 $\mathbf{f} - g^2\nabla\log p_t$、协方差 $g^2(t)\Delta t\mathbf{I}$ 的高斯。令 $\Delta t \to 0$ 即得连续时间逆向 SDE。

---

## 3.4 结语

本章将变分视角与基于分数视角下的离散时间扩散过程统一到一个简洁的连续时间框架中。我们说明 DDPM 与 NCSN 均可理解为具有不同漂移/扩散系数的 SDE 的离散化。

该框架的基石是相应**逆向时间 SDE** 的存在，它形式地定义了一个逆转噪声破坏的生成过程。关键的是，该逆向过程的漂移仅依赖于一个未知量：每个时刻边缘数据分布的**分数函数** $\nabla_{\mathbf{x}}\log p_{t}(\mathbf{x})$。这一认识巩固了分数函数在生成建模中的核心地位。

此外，我们引入了纯确定性对应物——**概率流 ODE（PF-ODE）**，其解轨迹与 SDE 沿相同的边缘密度族 $\{p_{t}\}$ 演化。这种一致性由底层 Fokker–Planck 方程保证。深刻含义是：复杂的生成任务在本质上等价于求解微分方程。训练归结为学习定义该方程向量场的分数函数，而采样则成为数值积分问题。

PF-ODE 这一纯确定性流的引入，为扩散模型的第三个也是最后一个视角架起了桥梁。由速度场支配的确定性变换的学习，是近年来一大类生成模型的核心原则。下一章我们将：（1）从 Normalizing Flows 与 Neural ODE 的起源出发，探索这一基于流的视角；（2）说明该视角如何引出现代的 Flow Matching 框架——直接学习速度场以在分布间输运样本。最终我们将看到，从随机原理推导出的确定性 PF-ODE，如何从这一完全不同的、基于流的出发点被构造与推广，从而完成我们对扩散建模的统一图景。
