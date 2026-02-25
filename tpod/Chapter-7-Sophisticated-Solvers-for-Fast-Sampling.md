# 第七章：用于快速采样的精密求解器

扩散模型的生成过程将噪声映射到数据样本，在数学上等价于求解一个 SDE 或其对应的 ODE。该过程本质上是缓慢的，因为它依赖数值求解器用许多小积分步来近似解轨迹（简要介绍见附录）。因此，加速推理已成为核心研究目标。现有方法大致分为两类：

- **无训练方法：** 本章重点。这类方法发展先进的数值求解器，在不额外训练的前提下提高扩散采样的效率。
- **基于训练的方法：** 在后续章节讨论。这些技术要么将预训练扩散模型蒸馏为快速生成器，要么直接学习 ODE 流映射（解），从而只需少量采样步。

基于 SDE 的采样器（如 Euler–Maruyama）因随机性可能产生更多样化的样本，但通常需要更多步。此处我们主要关注基于 ODE 的生成，其原理可自然推广到 SDE 情形。

---

## 序言

### 扩散模型的高级求解器

Score SDE 框架将离散时间扩散与 ELBO 形式与生成建模的连续时间 SDE/ODE 视角严格联系起来，奠定了重要基础。这种统一不仅带来理论清晰性，也使得基于数值积分的高效采样算法得以有原则地发展。

具体地，设我们有预训练扩散模型 $\mathbf{s}_{\bm{\phi}^*}(\mathbf{x}, t) \approx \nabla_{\mathbf{x}}\log p_t(\mathbf{x})$（并承认另外三种等价表述）。此时，采样过程可视为求解 PF-ODE，初值为 $\mathbf{x}(T) \sim p_{\text{prior}}$，从 $t = T$ 反向积分到 $t = 0$：

$$
\frac{\mathrm{d} \mathbf{x}(t)}{\mathrm{d} t} 
= \mathbf{f}(\mathbf{x}(t),t) 
- \frac{1}{2} g^2(t) \underbrace{\nabla_{\mathbf{x}}\log p_t(\mathbf{x}(t))}_{\approx\,  \mathbf{s}_{\bm{\phi}^*}(\mathbf{x}(t), t)}.
$$

该 ODE 直接对应于前向随机过程
$$
\mathrm{d}\mathbf{x}(t) = \mathbf{f}(\mathbf{x}(t),t)\mathrm{d} t + g(t)\mathrm{d}\mathbf{w}(t),
$$
体现了生成（反向时间）与加噪（前向时间）动力学之间的连续时间联系。

PF-ODE 的精确解可等价地写成积分形式：
$$
\Psi_{T\to 0} \left(\mathbf{x}(T)\right) 
= \mathbf{x}(T) + \int_T^0  \Big[f(\tau)\mathbf{x}(\tau) - \frac{1}{2}g^2(\tau) \nabla_{\mathbf{x}}\log p_\tau(\mathbf{x}(\tau))\Big]\mathrm{d} \tau
$$
$$
\approx \mathbf{x}(T) + \int_T^0  \Big[f(\tau)\mathbf{x}(\tau) - \frac{1}{2}g^2(\tau) \mathbf{s}_{\bm{\phi}^*} \big(\mathbf{x}(\tau), \tau\big)\Big]\mathrm{d} \tau 
=:\widetilde{\Psi}_{T\to 0} \left(\mathbf{x}(T)\right).
$$

这里 $\Psi_{s\to t}(\mathbf{x})$ 表示**理想** PF-ODE 的流映射，将时刻 $s$ 的状态 $\mathbf{x}$ 映到时刻 $t$ 的演化状态；$\widetilde{\Psi}_{s\to t}(\mathbf{x})$ 表示**经验** PF-ODE 的流映射，由用学习近似 $\mathbf{s}_{\bm{\phi}^*}(\mathbf{x},t)$ 替代真实扩散模型的 $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$ 得到。因此 $\widetilde{\Psi}_{s\to t}  \approx  \Psi_{s\to t}$。

由于 $\widetilde{\Psi}_{s\to t}$ 的积分形式无法闭式求值，采样必须依赖**数值求解器**。这些方法通过对时间离散化并用有限个局部漂移求值的和来替代连续积分，从而追踪近似轨迹。这类基于求解器的积分近似被称为扩散快速采样的**无训练**算法，因为它们直接从冻结的预训练得分模型 $\mathbf{s}_{\bm{\phi}^*}$ 近似 PF-ODE 解，无需任何额外学习。

下面先详述数值求解器的通用概念并引入后文记号。

**连续轨迹的离散近似。** 记 $\mathbf{x}_T$ 为时刻 $T$ 的初始状态，考虑递减划分
$$
T = t_0 > t_1 > \cdots > t_M = 0.
$$
从 $\tilde{\mathbf{x}}_{t_0} = \mathbf{x}_T \sim p_{\mathrm{prior}}$ 出发，求解器产生序列 $\{\tilde{\mathbf{x}}_{t_i}\}_{i=0}^M$，理想情况下近似经验 PF-ODE 流 $\widetilde{\Psi}_{T\to t_i}(\mathbf{x}_T)$，而后者又是理想映射 $\Psi_{T\to t_i}(\mathbf{x}_T)$ 的代理。每一步数值更新都通过该经验速度场推进状态，最终迭代 $\tilde{\mathbf{x}}_{t_M}$ 作为 $t=0$ 时干净样本 $\mathbf{x}_0$ 的估计。

### 文献中设计求解器的通用框架

Zhang 等人强调了为扩散模型相关 PF-ODE 设计数值求解器的三条实用原则。

**I. 半线性结构。** 尽管 Song 等人为一般漂移 $\mathbf{f}(\mathbf{x}(t),t)$ 建立了基础，在大多数调度形式中漂移被实例化为线性形式
$$
\mathbf{f}(\mathbf{x}, t) := f(t)\,\mathbf{x}, \quad f:\mathbb{R}\to\mathbb{R},
$$
从而得到**半线性**结构的 PF-ODE：
$$
\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d} t}
= \underbrace{f(t) \mathbf{x}(t)}_{\text{线性部分}}
- \underbrace{\tfrac{1}{2} g^2(t) \mathbf{s}_{\bm{\phi}^*}(\mathbf{x}(t), t)}_{\text{非线性部分}}.
$$
$\mathbf{x}$ 上的这种线性–非线性分解有利于精度与稳定性，并催生了专门的积分器。

**II. 超越得分的参数化。** 当 $t \to 0$ 时，真实得分 $\nabla_{\mathbf{x}}\log p_t(\cdot)$ 可能变化非常剧烈（例如当 $p_{\text{data}}$ 集中在低维流形附近时）。这使得直接近似得分的神经网络 $\mathbf{s}_{\bm{\phi}^*}$ 难以保持准确。

原因简述：理想关系为
$$
\bm{\epsilon}^*(\mathbf{x}_t,t) = -\sigma_t \nabla_{\mathbf{x}}\log p_t(\mathbf{x}_t),
$$
其中 $\bm{\epsilon}^*(\mathbf{x}_t,t)=\mathbb{E}[\bm{\epsilon} |\mathbf{x}_t]$ 为理想噪声，$(\alpha_t,\sigma_t)$ 为扰动核 $\mathbf{x}_t |\mathbf{x}_0 \sim \mathcal{N}(\alpha_t \mathbf{x}_0, \sigma_t^2 I)$ 的均值和标准差。由 $L^2$ 正交性，
$$
\mathbb{E}\,\lVert\bm{\epsilon}\rVert_2^2
=\mathbb{E}\,\lVert\bm{\epsilon}^*\rVert_2^2+\mathbb{E}\,\lVert\bm{\epsilon}-\bm{\epsilon}^*\rVert_2^2
\;\;\Rightarrow\;\;
\mathbb{E}\,\lVert\bm{\epsilon}^*\rVert_2^2 \leq \mathbb{E}\,\lVert\bm{\epsilon}\rVert_2^2 = D.
$$
故理想噪声预测有界，但得分按
$$
\mathbb{E}\,\lVert\mathbf{s}^*(\mathbf{x}_t,t)\rVert_2^2
= \sigma_t^{-2}\,\mathbb{E}\,\lVert\bm{\epsilon}^*(\mathbf{x}_t,t)\rVert_2^2
 \leq  \frac{D}{\sigma_t^2}
$$
增长。因此当 $t\to 0$ 时，得分可以按 $1/\sigma_t^2$ 发散，而噪声预测保持有界。神经网络只能近似平滑增长函数，得分预测往往数值不稳定、精度较差，进而在依赖预训练模型作为漂移时损害 PF-ODE 数值求解器。

因此，广泛采用的替代是预测噪声 $\bm{\epsilon}_{\bm{\phi}^*}$（或其变体如 $\mathbf{x}$-或 $\mathbf{v}$-预测），它有界且与得分有简单闭式关系：
$$
\mathbf{s}_{\bm{\phi}^*}(\mathbf{x},t) = -\frac{1}{\sigma_t}\,\bm{\epsilon}_{\bm{\phi}^*}(\mathbf{x},t).
$$

代入 PF-ODE 得到
$$
\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d} t}
= \underbrace{f(t) \mathbf{x}(t)}_{\text{线性部分}}
+ \underbrace{\tfrac{1}{2} \tfrac{g^2(t)}{\sigma_t} \bm{\epsilon}_{\bm{\phi}^*}(\mathbf{x}(t), t)}_{\text{非线性部分}}.
$$
现代 PF-ODE 求解器普遍采用该参数化。

**III. 半线性 PF-ODE 的指数积分器。** 对上述半线性结构，**指数积分器公式**（常数变易）给出解的精确等价表示。记 $\mathbf{x}_s$ 为起始时刻 $s$ 的状态，$t\in[0,s]$ 为终端时刻。将非线性部分记为
$$
\mathrm{N}(\mathbf{x}(t), t) := \frac{1}{2}\frac{g^2(t)}{\sigma_t}\,\bm{\epsilon}_{\bm{\phi}^*}(\mathbf{x}(t), t).
$$
则 ODE 可写为
$$
\frac{\mathrm{d} \mathbf{x}(t)}{\mathrm{d} t} - \underbrace{f(t)\mathbf{x}(t)}_{\text{线性部分}} = \underbrace{\mathrm{N}(\mathbf{x}(t), t)}_{\text{非线性部分}}.
$$
引入**指数积分因子**
$$
\mathcal{E}(s \to t) := \exp \Bigl(\int_s^t f(u) \mathrm{d} u\Bigr),
$$
并用其逆 $\mathcal{E}(t \to s)$ 乘 ODE 两边。由乘积法则可得解：
$$
\widetilde{\Psi}_{s\to t}(\mathbf{x}_s)
= \underbrace{\mathcal{E}(s \to t)\mathbf{x}_s}_{\text{线性部分}}
+ \frac{1}{2}\int_s^t \frac{g^2(\tau)}{\sigma_\tau} 
\mathcal{E}(\tau \to t) 
\bm{\epsilon}_{\bm{\phi}^*}(\mathbf{x}_\tau,\tau) \mathrm{d}\tau.
$$

为说明为何在少步采样（大步长 $\Delta s$）下该指数积分形式优于原 ODE 形式，比较其单步更新。在常数变易下，令 $\mathcal{E}(s\to s-\Delta s)=e^{-f(s)\Delta s}$ 并在 $\tau\in[s-\Delta s,s]$ 上冻结 $\mathrm{N}(\mathbf{x}(\tau),\tau)\approx \mathrm{N}(\mathbf{x}_s,s)$，则指数–Euler 更新为
$$
\mathbf{x}^{\mathrm{Exp}\text{-}\mathrm{Euler}}_{s-\Delta s}
= \underbrace{e^{-f(s)\Delta s}\mathbf{x}_s}_{\text{线性部分}}
+\underbrace{\frac{e^{-f(s)\Delta s}-1}{f(s)}\,\mathrm{N}(\mathbf{x}_s,s)}_{\text{非线性部分}},
$$
自然极限为 $f\to 0$ 时 $(e^{-f\Delta s}-1)/f\to -\Delta s$。这里线性因子 $e^{-f(s)\Delta s}$ 被精确计算（无近似）。

相反，在 $\tau\in[s-\Delta s,s]$ 上令 $f(\tau)\mathbf{x}_\tau - \mathrm{N}(\mathbf{x}_\tau,\tau)\approx f(s)\mathbf{x}_s - \mathrm{N}(\mathbf{x}_s,s)$ 得到原 ODE 的**普通 Euler** 步：
$$
\mathbf{x}^{\mathrm{Euler}}_{\,s-\Delta s}
=\mathbf{x}_s-\Delta s\,[\,f(s)\,\mathbf{x}_s+\mathrm{N}(\mathbf{x}_s,s)\,]
=\underbrace{(1-f(s)\Delta s)\,\mathbf{x}_s}_{\text{线性部分}}
-\underbrace{\Delta s\,\mathrm{N}(\mathbf{x}_s,s)}_{\text{非线性部分}}.
$$

Euler 中的线性因子是指数中线性部分的一阶 Taylor 近似：$e^{a}=1+a+\tfrac{a^2}{2}+\cdots$，$a:=-f(s)\Delta s$，故差距为 $e^{a}-(1+a)=\tfrac{a^2}{2}+\mathcal{O}(a^3)$。一旦 $|f(s)|\Delta s$ 不小（即步长 $\Delta s$ 不够小），Euler 的线性更新 $(1+a)\mathbf{x}_s$ 相对真实因子 $e^{a}\mathbf{x}_s$ 会产生量级为 $a/2$ 的相对误差。这是纯离散化带来的线性畸变。指数–Euler 步通过使用精确线性乘子避免该问题，在大步长时尤为重要。

### PF-ODE 数值求解器的几类方法

扩散模型的数值求解器可大致分为两类。

**时间步进方法。** 这类方法对时间区间 $[0,T]$ 离散化，并用各种为效率设计的数值积分格式近似 PF-ODE。下面以最具基础性、原则性且广泛采用的方法为代表：

- **去噪扩散隐式模型 (DDIM)。** DDIM 是最早的扩散快速采样器之一。从变分角度提出，引入一族非马尔可夫前向过程，其边缘与原始扩散一致，从而得到确定性反向过程与灵活跳步。从 ODE 角度看，DDIM 可理解为：对指数积分公式应用单次指数–Euler 步（即把积分内扩散模型项近似为常数），即得到该指数–Euler 更新。

- **扩散指数积分采样器 (DEIS)。** DEIS 首次利用 PF-ODE 的半线性结构，采用指数积分器。核心思想是通过积分因子精确处理线性部分，只近似非线性积分项。与在指数积分公式内假设被积函数为常数的 Euler 不同，DEIS 复用轨迹上先前估计点的历史，用**拉格朗日多项式**等高阶插值拟合过去求值，并用其近似下一步的积分。几何上，该多项式插值比常数近似更准确刻画轨迹曲率，从而在大步长下具有更高阶精度和更好稳定性。这种用过去求值锚定下一步更新（使每步只需一次新模型调用）称为**多步法**；**单步法**（如 DDIM）仅依赖最近状态进行更新，更简单但在达到高精度时通常代价更大。

- **DPM-Solver 系列。** DPM-Solver、DPM-Solver++ 和 DPM-Solver-v3 建立在 PF-ODE 的半线性结构之上，并引入关键的时间重参数化：**半对数信噪比 (SNR)**
  $$
  \lambda_t := \frac{1}{2}\log \frac{\alpha_t^2}{\sigma_t^2} = \log \frac{\alpha_t}{\sigma_t}.
  $$
  该变量替换将非线性项变为指数加权积分
  $$
  \int_{\lambda_s}^{\lambda_t} e^{-\lambda}\,\hat{\bm{\epsilon}}_{\bm{\phi}^*}(\hat{\mathbf{x}}_\lambda,\lambda) \mathrm{d} \lambda,
  $$
  其中 $\hat{\bm{\epsilon}}_{\bm{\phi}^*}$ 表示用重参数化时间 $\lambda$ 表达的模型。该表示使积分的高阶近似更易且更准。DPM-Solver 通过在 $\lambda$ 上的 Taylor 展开得到高阶求解器；DPM-Solver++ 适配无分类器引导与 $\mathbf{x}$-预测以提高稳定性；DPM-Solver-v3 进一步将参数化选择自动化，通过最小化局部误差的优化问题确定。

**（可选）时间并行方法。** 另一种策略是在不同时间区间上并行计算而非严格顺序执行。

- **ParaDiGMs。** 该方法将 ODE 解重新表述为不动点问题，从而积分项可并行求值，缓解标准时间步进求解器的顺序瓶颈。该思路不限于指数积分形式，对一般非线性漂移 $\mathbf{f}(\mathbf{x},t)$ 的 PF-ODE 同样适用，且与具体求解器无关：不动点形式用选定时刻的模型求值加权和替代积分，可包装任意时间步规则，Euler、DEIS 或 DPM-Solver 类更新均可使用，且求值可并行进行。

**真实计算成本 (NFE)。** 实践中，墙钟时间主要由**模型网络调用次数**决定，称为**函数求值次数 (NFE)**。若采样器每步做 $m$ 次求值、共 $N$ 步，则成本为 $\mathrm{NFE} = m\,N$。例如一阶 Euler 或指数–Euler 有 $m=1$；单步 $k$ 阶方法通常需 $m\geq k$。多步方法（如 DEIS、DPM-Solver++ 的多步版）复用过去求值，在短预热后平均 $m$ 接近 1。无分类器引导会使每步调用数约翻倍。因此“更快”的采样在实践中指更低的 NFE，而不仅是更少步数。

**关于使用 PF-ODE 等价形式的说明。** 下文中将使用等价参数化的结果：扰动核 $\mathbf{x}_t | \mathbf{x}_0 \sim \mathcal{N}(\cdot;\alpha_t \mathbf{x}_0,\sigma_t^2\mathbf{I})$ 的 $(f(t), g(t))$ 与 $(\alpha_t,\sigma_t)$ 可互换，由
$$
f(t) = \frac{\alpha_t'}{\alpha_t}, 
\quad
g^2(t) = \frac{\mathrm{d}}{\mathrm{d} t} \big(\sigma_t^2\big) 
         - 2 \frac{\alpha_t'}{\alpha_t} \sigma_t^2
       = 2\sigma_t\sigma_t' - 2 \frac{\alpha_t'}{\alpha_t} \sigma_t^2
$$
联系。在这些关系下，PF-ODE 可写成多种等价形式。

---

## DDIM

本节介绍扩散模型中加速采样的先驱方法之一：**去噪扩散隐式模型 (DDIM)**，也是最常用的基于 ODE 的求解器之一。尽管其名称带有变分渊源，如对 $(\mathbf{x},\bm{\epsilon})$-预测的推导所示，我们将说明其实际更新规则也可理解为对指数积分公式中积分应用 Euler 方法的直接结果。这一 ODE 视角不仅为 DDIM 提供了有原则的重新解释，也为设计更灵活、高效的快速采样器奠定基础。DDIM 的原始变分推导将在后文重访；并会建立 DDIM 更新与条件流匹配 (CFM) 的清晰对应，表明 DDIM 动力学可解释为 CFM 所学的流。

### 将 DDIM 理解为 ODE 求解器

设 $s > t$ 为两个离散时间步，$s$ 为起始时间、$t$ 为更新目标时间。为近似指数积分公式中的积分，自然选择是在步的起点 $s$ 处固定被积函数，即假设
$$
\bm{\epsilon}_{\bm{\phi}^*}(\mathbf{x}_\tau, \tau) \approx \bm{\epsilon}_{\bm{\phi}^*}(\mathbf{x}_s, s), \quad \text{对所有 } \tau \in [t, s].
$$
该假设导致 Euler 型近似更新，得到如下更新规则：
$$
\tilde{\mathbf{x}}_{t} = \mathcal{E}(s \to t)\tilde{\mathbf{x}}_{s} 
+ \left( \frac{1}{2} \int_{s}^{t} \frac{g^2(\tau)}{\sigma_\tau} \mathcal{E}(\tau \to t) \mathrm{d}\tau \right) 
\bm{\epsilon}_{\bm{\phi}^*}(\tilde{\mathbf{x}}_{s}, s),
$$
其中积分可解析求出，得到实用且高效的 DDIM 更新公式：

**命题（DDIM = Euler 方法（指数 Euler））。** 对指数积分形式应用 Euler 方法得到的更新规则给出如下 DDIM 更新：
$$
\tilde{\mathbf{x}}_{t} = \frac{\alpha_{t}}{\alpha_{s}}\tilde{\mathbf{x}}_{s} - \alpha_{t}\left(\frac{\sigma_{s}}{\alpha_{s}} - \frac{\sigma_{t}}{\alpha_{t}}\right)\bm{\epsilon}_{\bm{\phi}^*}(\tilde{\mathbf{x}}_{s}, s).
$$

（推导中使用 $\mathcal{E}(s \to t) = \alpha_t/\alpha_s$ 以及积分
$\int_s^t \frac{g^2(\tau)}{2\sigma_\tau} \frac{\alpha_t}{\alpha_\tau} \mathrm{d}\tau = -\alpha_t(\sigma_s/\alpha_s - \sigma_t/\alpha_t)$。）

这表明 DDIM 可解释为应用于半线性 PF-ODE 的指数积分器形式的一阶 Euler 方法。

### 不同参数化下 DDIM 的直观

DDIM 是加速扩散采样最常用的方法之一，除 $\bm{\epsilon}$-预测外还可采用不同参数化。本小节给出不同参数化下的改写及更直观的解释。

**不同参数化下的 DDIM。** 实践中使用以某种标准参数化表达的预训练扩散模型，并将相应预测子代入 PF-ODE 的 DDIM 离散化以替代理想目标。可实施版本由替换 $\bm{\epsilon}_{\bm{\phi}^*} \approx \bm{\epsilon}^*$、$\mathbf{x}_{\bm{\phi}^*} \approx \mathbf{x}^*$、$\mathbf{s}_{\bm{\phi}^*} \approx \mathbf{s}^*$、$\mathbf{v}_{\bm{\phi}^*} \approx \mathbf{v}^*$ 得到。

**结论（不同参数化下的 DDIM）。** 设 $s>t$。从 $\tilde{\mathbf{x}}_s\sim p_s$ 出发、在时刻 $t$ 结束的 DDIM 更新在不同参数化下为：
$$
\tilde{\mathbf{x}}_t  = \frac{\alpha_{t}}{\alpha_s} \tilde{\mathbf{x}}_s + \alpha_{t}\left(\frac{\sigma_{t}}{\alpha_{t}} -\frac{\sigma_s}{\alpha_s}  \right) \bm{\epsilon}^*(\tilde{\mathbf{x}}_s, s)
=\frac{\sigma_t}{\sigma_s}\tilde{\mathbf{x}}_s + \alpha_s \left(\frac{\alpha_t}{\alpha_s} - \frac{\sigma_t}{\sigma_s} \right) \mathbf{x}^*(\tilde{\mathbf{x}}_s, s)
= \frac{\alpha_{t}}{\alpha_s} \tilde{\mathbf{x}}_s + \sigma_s^2 \left(\frac{\alpha_{t}}{\alpha_s}- \frac{\sigma_{t}}{\sigma_s}\right) \mathbf{s}^*(\tilde{\mathbf{x}}_s, s)
= \alpha_t \mathbf{x}^*(\tilde{\mathbf{x}}_s, s)  + \sigma_t \bm{\epsilon}^*(\tilde{\mathbf{x}}_s, s).
$$

最后一个等式清楚表明：从 $\tilde{\mathbf{x}}_s\sim p_s$ 出发，（估计的）干净部分 $\mathbf{x}^*(\tilde{\mathbf{x}}_s,s)$ 与（估计的）噪声部分 $\bm{\epsilon}^*(\tilde{\mathbf{x}}_s,s)$ 作为插值端点，用系数 $(\alpha_t,\sigma_t)$ 重构出 $\tilde{\mathbf{x}}_t \sim p_t$。

事实上，DDIM 可视为 $\mathbf{v}$-参数化 PF-ODE 的**直接** Euler 离散，而无需使用指数积分器。由 PF-ODE 的 $\mathbf{v}$-预测形式，在 $\tau\in[t,s]$ 上有
$$
\frac{\mathrm{d} \mathbf{x}(\tau)}{\mathrm{d} \tau} 
= \alpha_\tau' \mathbf{x}^*(\mathbf{x}(\tau),\tau) 
+ \sigma_\tau' \bm{\epsilon}^*(\mathbf{x}(\tau),\tau).
$$
从 $\tilde{\mathbf{x}}_s$ 出发、在 $[t,s]$ 上积分，Euler 方法在右端点冻结预测子，得到
$$
\tilde{\mathbf{x}}_t = \alpha_t \mathbf{x}^*(\tilde{\mathbf{x}}_s,s) + \sigma_t \bm{\epsilon}^*(\tilde{\mathbf{x}}_s,s),
$$
与 DDIM 更新的最后一式一致。

![DDIM 作为 PF-ODE 的 Euler 离散的示意图。](../arXiv-2510.21890v1/Images/PartB/Euler.pdf)

**图：DDIM 作为 PF-ODE 的 Euler 离散。** 从时刻 $s$ 的状态 $\tilde{\mathbf{x}}_s$ 出发，理想 PF-ODE 轨迹（灰色曲线）确定性演化到时刻 $t$ 的 $\Psi_{s\to t}(\tilde{\mathbf{x}}_s)$。DDIM 更新（橙色）则直接将 $\tilde{\mathbf{x}}_s$ 映到 $\alpha_t \mathbf{x}^*(\tilde{\mathbf{x}}_s,s) + \sigma_t \bm{\epsilon}^*(\tilde{\mathbf{x}}_s,s)$。该 Euler 步与真实 PF-ODE 轨迹的偏差（蓝色）即离散化误差；当 $t$ 远离 $s$ 时偏差可能很大，导致生成质量下降。

在速度预测下，PF-ODE 中的线性项 $f(t)\mathbf{x}$ 被吸收进目标 $\mathbf{v}^*(\mathbf{x}(t), t)=\alpha_t'\mathbf{x}_0+\sigma_t'\bm{\epsilon}$。由微积分基本定理，$\int_s^t \alpha_\tau'\mathrm{d}\tau$ 与 $\int_s^t \sigma_\tau'\mathrm{d}\tau$ 简化为 $(\alpha_t-\alpha_s)$ 和 $(\sigma_t-\sigma_s)$，故单次 Euler 步即得闭式 DDIM 更新。也就是说，在 $\mathbf{v}$-预测下 PF-ODE 漂移中无单独线性项需要分离，普通 Euler 更新自然与 DDIM 形式一致。而在 $\bm{\epsilon}$-、$\mathbf{x}$-或 $\mathbf{s}$-预测下，漂移可分解为线性项与非线性修正的**半线性**形式；朴素 Euler 步只**近似**线性项而非精确计算，DDIM 则对应**指数–Euler**（积分因子）步，解析处理该线性部分。因此 $\mathbf{v}$-预测得到最简单、最直接的 Euler 积分，其他参数化则需指数–Euler 形式才能达到相同 DDIM 行为。

**观察（指数 Euler 与 DDIM 更新）。** 在相同调度 $(\alpha_t,\sigma_t)$ 下：
- $\mathbf{v}$-预测：Euler = DDIM；
- $\bm{\epsilon}$-、$\mathbf{x}$-或 $\mathbf{s}$-预测：指数–Euler = DDIM ≠ 普通 Euler。  
在 $\bm{\epsilon}$-/$\mathbf{x}$-/$\mathbf{s}$-预测下，普通 Euler 步不等价于 DDIM，因线性项仅被近似，可能降低稳定性。

**不同参数化下 DDIM 的示例。** 以理想替换（$\bm{\epsilon}^*$、$\mathbf{x}^*$、$\nabla_\mathbf{x}\log p_t$、$\mathbf{v}^*$）的简单例子说明。设前向核 $\alpha_t=1$、$\sigma_t=t$。则 DDIM（指数–Euler）更新
$\tilde{\mathbf{x}}_{t} = \tilde{\mathbf{x}}_s - (s-t)\,\bm{\epsilon}^*(\tilde{\mathbf{x}}_s,s)$。概念上，减去时间差 $(s-t)$ 乘以理想噪声估计，将当前样本 $\tilde{\mathbf{x}}_s$ 推向更干净的估计。用 $\mathbf{x}$-预测理想 $\mathbf{x}^*$ 与 $\bm{\epsilon}^*(\tilde{\mathbf{x}}_s,s) = (\tilde{\mathbf{x}}_s - \mathbf{x}^*(\tilde{\mathbf{x}}_s, s))/s$ 可得
$\tilde{\mathbf{x}}_t = \frac{t}{s}\,\tilde{\mathbf{x}}_s + (1-\frac{t}{s})\,\mathbf{x}^*(\tilde{\mathbf{x}}_s, s)$，即 $\tilde{\mathbf{x}}_t$ 是当前样本与干净数据理想估计的凸组合，且 $\tilde{\mathbf{x}}_t-\mathbf{x}^* = \frac{t}{s}\,(\tilde{\mathbf{x}}_s-\mathbf{x}^*)$，去噪残差每步以因子 $t/s\in(0,1)$ 收缩。用得分理想可得 $\tilde{\mathbf{x}}_t = \tilde{\mathbf{x}}_s + (s-t)\,s\,\nabla_{\mathbf{x}}\log p_s(\tilde{\mathbf{x}}_s)$，沿得分场向高似然区域移动。用速度理想 $\mathbf{v}^*(\tilde{\mathbf{x}}_s,s) = -\bm{\epsilon}^*(\tilde{\mathbf{x}}_s,s)$ 时，DDIM 更新为 $\tilde{\mathbf{x}}_t = \tilde{\mathbf{x}}_s + (t-s)\,\mathbf{v}^*(\tilde{\mathbf{x}}_s,s)$，割线斜率满足有限差分恒等式 $(\tilde{\mathbf{x}}_t-\tilde{\mathbf{x}}_s)/(t-s) = \mathbf{v}^*(\tilde{\mathbf{x}}_s,s)$，即更新沿局部 ODE 漂移的直线步进。

**DDIM 的挑战。** 一阶 Euler 离散的全局误差为 $\mathcal O(h)$，当最大步长 $h:=\max_{i}|t_i - t_{i-1}|$ 增大时精度下降。为提高精度，文献发展了高阶格式，通过更丰富的局部近似将全局阶提高到 $\mathcal O(h^{k})$（$k \ge 2$）。在合适的时间步分配下，这些方法可用更少步数达到目标质量。但需注意，高阶本身不保证更少步数或更低墙钟成本，因为每步可能需要多次模型求值。实践中效率的真实度量是函数求值次数 $\mathrm{NFE}=m\,N$，“更快”指用更小 NFE 达到所需质量，而不仅是更少步数。

### （可选）DDIM 的变分视角

DDIM 的动机来自从变分角度重看 DDPM。在 DDPM 中，反向过程与特定的马尔可夫前向转移核 $p(\mathbf{x}_t|\mathbf{x}_{t-\Delta t})$ 绑定，为正确近似多步后验需要小步长。DDIM 则观察到训练目标只依赖边缘扰动 $p_t(\mathbf{x}_t|\mathbf{x}_0)$，不依赖具体前向转移；因此可直接从边缘构造反向动力学，在保持边缘一致性的前提下跳过中间步。因为转移被定义为将 $p_s(\mathbf{x}_s|\mathbf{x}_0)$ 映到 $p_t(\mathbf{x}_t|\mathbf{x}_0)$（任意 $t<s$），可使用粗时间网格、大幅减少更新次数，从而减少模型求值并实现快速少步采样。

**重访 DDPM 的变分观点。** 在 DDPM 中，训练固定一族边缘扰动核 $p_t(\mathbf{x}_t|\mathbf{x}_0)$ 并优化只依赖这些边缘的代理目标。采样时反向条件却是在一步前向核下的贝叶斯后验，因此反向更新与**该**前向转移 $p(\mathbf{x}_t|\mathbf{x}_{t-\Delta t})$ 绑定。若通过增大 $\Delta t$ 并复用同一单步核来跳步，则不再匹配真实多步后验，通常会破坏边缘。

**DDIM 的原始动机。** DDIM 指出训练目标只约束边缘 $p_t(\mathbf{x}_t|\mathbf{x}_0)$，不约束中间反向转移。因此可**指定**一族反向条件 $\pi(\mathbf{x}_t|\mathbf{x}_s,\mathbf{x}_0)$（$t<s$），使其**一步边缘一致**：
$$
\int \pi(\mathbf{x}_t|\mathbf{x}_s,\mathbf{x}_0)\,p_s(\mathbf{x}_s|\mathbf{x}_0) \mathrm{d} \mathbf{x}_s
= p_t(\mathbf{x}_t|\mathbf{x}_0).
$$
当 $\pi$ 取为真实条件 $p(\mathbf{x}_t|\mathbf{x}_s, \mathbf{x}_0)$ 时，该条件即为全概率律。该构造去掉对前向单步核 $p(\mathbf{x}_t|\mathbf{x}_{t-\Delta t})$ 的依赖，并允许粗（跳过的）时间步。

**离散时间 DDIM 的推导。** 考虑一般前向扰动 $p_t(\mathbf{x}_t|\mathbf{x}_0) := \mathcal{N}(\mathbf{x}_t; \alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$。DDIM 不要求反向更新与单步前向核诱导的贝叶斯后验一致，只需**选择**一个保持边缘的反向条件。对任意 $t<s$ 设高斯族
$$
\pi(\mathbf{x}_t|\mathbf{x}_s,\mathbf{x}_0)
= \mathcal{N} \big(\mathbf{x}_t;\, a_{t,s}\,\mathbf{x}_0 + b_{t,s}\,\mathbf{x}_s,\; c_{t,s}^2\,\mathbf{I}\big),
$$
系数 $(a_{t,s},b_{t,s},c_{t,s})$ 由边缘一致性约束确定。由高斯性质可得
$$
\alpha_t = a_{t,s} + b_{t,s}\alpha_s,\quad \sigma_t^2 = b_{t,s}^2\sigma_s^2 + c_{t,s}^2.
$$
该系统欠定，将 $c_{t,s}$ 视为自由参数（$0\le c_{t,s} \le \sigma_t$）解得
$$
b_{t,s} = \frac{\sqrt{\sigma_t^2 - c_{t,s}^2}}{\sigma_s},\quad a_{t,s} = \alpha_t - \alpha_s\,b_{t,s}.
$$

**引理（DDIM 系数）。** 若边缘一致性成立，则系数即上述形式，$0\le c_{t,s}\le \sigma_t$。

**备注。** (1) DDIM 中我们**选择**反向核 $\pi(\mathbf{x}_t|\mathbf{x}_s,\mathbf{x}_0)$ 满足边缘一致性，一般 $\pi\neq p(\mathbf{x}_t|\mathbf{x}_s,\mathbf{x}_0)$，且该后验对指定 $\pi$ 或训练并非必需。(2) 仅当方差参数取为 DDPM 后验方差（即 $\eta=1$）时才有 $\pi=p$；否则不等。(3) 一般不假设马尔可夫，故 $p(\mathbf{x}_s|\mathbf{x}_t,\mathbf{x}_0)\neq p(\mathbf{x}_s|\mathbf{x}_t)$。

前向边缘 $\{p_t(\mathbf{x}_t|\mathbf{x}_0)\}_t$ 不能唯一确定反向条件转移；存在无穷多满足边缘一致性的核 $\pi(\mathbf{x}_t|\mathbf{x}_s,\mathbf{x}_0)$，由参数 $c_{t,s}$ 索引，控制每步 $s \to t$ 注入的噪声量。

**DDIM 采样器（步 $s\to t$）。** 由所选反向核并用预训练模型的预测子替代 $\mathbf{x}_0$ 得到。使用 $\bm{\epsilon}$-预测网络 $\bm{\epsilon}_{\bm{\phi}^*}$（即插即用、无需再训练），代入后得更新
$$
\mathbf{x}_t
= \frac{\alpha_t}{\alpha_s}\,\mathbf{x}_s
+\Big(\sqrt{\sigma_t^2 - c_{t,s}^2}-\frac{\alpha_t}{\alpha_s}\sigma_s\Big)\,
\bm{\epsilon}_{\bm{\phi}^*}(\mathbf{x}_s,s)
+ c_{t,s}\,\bm{\epsilon}_t,\quad \bm{\epsilon}_t\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
$$
$c_{t,s}\in[0,\sigma_t]$ 控制随机性。记 $\alpha_{t|s}:=\alpha_t/\alpha_s$，$\sigma_{t|s}^2:=\sigma_t^2-\alpha_{t|s}^2\,\sigma_s^2$。通过变化 $c_{t,s}$ 得到一族共享同一预训练模型、无需再训练的采样器：
- **DDPM 步（后验方差）：** $c_{t,s}=\frac{\sigma_s}{\sigma_t}\,\sigma_{t|s}$ 使 $\pi$ 等于单步前向核诱导的贝叶斯后验，即标准 DDPM 反向更新。
- **确定性 DDIM（$\eta=0$）：** $c_{t,s}=0$ 得 $\mathbf{x}_t = \alpha_{t|s}\mathbf{x}_s + (\sigma_t-\alpha_{t|s}\sigma_s)\bm{\epsilon}_{\bm{\phi}^*}(\mathbf{x}_s,s)$，与 ODE 视角的 DDIM 跳跃一致。
- **插值：** $c_{t,s}=\eta\,\frac{\sigma_s}{\sigma_t}\,\sigma_{t|s}$，$\eta\in[0,1]$，在随机 DDPM（$\eta=1$）与确定性 DDIM（$\eta=0$）之间平滑插值。

### DDIM 与条件流匹配

确定性 DDIM 可理解为寻找将 $p_s(\cdot|\mathbf{x}_0)$ 推前到 $p_t(\cdot|\mathbf{x}_0)$ 的条件流映射。该条件流的切向量与条件流匹配 (CFM) 中使用的条件速度一致。对该条件速度求边缘即得 PF-ODE 漂移，其普通 Euler 离散在 $\mathbf{v}$-预测下恢复边缘 DDIM 更新。

在反向核为确定性的情形，即求条件映射 $\Psi_{s\to t}(\cdot|\mathbf{x}_0)$ 使得 $(\Psi_{s\to t}(\cdot|\mathbf{x}_0))_{\#}p_s(\cdot|\mathbf{x}_0)=p_t(\cdot|\mathbf{x}_0)$。在线性–高斯路径 $\mathbf{x}_\tau=\alpha_\tau\mathbf{x}_0+\sigma_\tau\bm{\epsilon}$ 下可得**条件映射**
$$
\Psi_{s\to t}(\mathbf{x}_s|\mathbf{x}_0)
=\frac{\sigma_t}{\sigma_s}\,\mathbf{x}_s
+\Bigl(\alpha_t-\alpha_s\frac{\sigma_t}{\sigma_s}\Bigr)\mathbf{x}_0,
$$
其瞬时**条件速度**为
$$
\mathbf{v}_t^{*}(\mathbf{x}|\mathbf{x}_0)
=\frac{\sigma_t'}{\sigma_t}\,\mathbf{x}
+\Bigl(\alpha_t'-\alpha_t\frac{\sigma_t'}{\sigma_t}\Bigr)\mathbf{x}_0.
$$
称 $\Psi_{s\to t}(\cdot|\mathbf{x}_0)$ 为 DDIM 条件映射。CFM 将时间依赖场拟合到该目标速度，故 CFM 回归目标等于 DDIM 条件映射的条件速度。**观察：** 沿条件高斯路径，DDIM 条件映射与 CFM 目标生成同一条件流 $\Psi_{s\to t}(\cdot|\mathbf{x}_0)$。对 $\mathbf{x}_0$ 在给定 $\mathbf{x}_t=\mathbf{x}$ 下的后验求条件速度的平均即得边缘 PF-ODE 漂移 $\mathbf{v}^*(\mathbf{x},t)$，在线性–高斯调度下呈可分离预测形式；该边缘化 $\mathbf{v}$-预测的 PF-ODE 的普通 Euler 步正是 DDIM 更新。简言之，DDIM 是 (i) 切向量等于 CFM 目标的确定性条件输运，以及 (ii) 对该切向量边缘化后、与 DDIM 更新一致的 PF-ODE 的 Euler 步。

---

## DEIS

在指数积分器公式中，积分
$$
\int_{s}^{t}
\frac{g^2(\tau)}{2\,\sigma_\tau}\,
\mathcal{E}(\tau \to  t)\,
\bm{\epsilon}_{\bm{\phi}^*}(\mathbf{x}_\tau,\tau)\mathrm{d}\tau
$$
中唯一的未知量是模型输出 $\bm{\epsilon}_{\bm{\phi}^*}(\mathbf{x}_\tau,\tau)$；一旦 $(\alpha,\sigma,g)$ 固定，调度项与权重 $\mathcal{E}(\tau \to  t)$ 已知。DDIM（Euler 法）将模型输出视为常数来近似该积分，但这只是一阶精度，当模型输出随时间变化较快时会失效。自然的问题是：**能否更好利用已经计算过的模型求值？** 与经典多步求解器类似，我们复用先前的输出（锚点）在时间上拟合一条简单曲线，对该拟合曲线的积分可精确计算。这正是**扩散指数积分采样器 (DEIS)** 背后的原理。

### 多项式外推

得到一条恰好经过锚点的简单曲线的最自然方式是使用经过它们的最低次多项式；$n=0$ 为常数，$n=1$ 为直线，$n=2$ 为二次曲线，统一于**拉格朗日多项式**。

### DEIS：用拉格朗日多项式近似 PF-ODE 积分

设 $n \ge 0$ 为所选多项式次数。在第 $i$ 步，用由过去模型输出构成的 $n$ 次插值多项式近似未知映射，代入指数积分器更新即得 $\tilde{\mathbf{x}}_{t_i}$。

![DEIS 作为多步方法的示意图。](../arXiv-2510.21890v1/Images/PartC/DEIS.pdf)

$n$ 次更新需要 $n+1$ 个锚点；历史充足时使用完整 $n$ 次格式，前期用当前可行最高次数并随锚点增多提高次数。得到 **AB-DEIS-$n$** 更新及系数 $C_{i,j}$ 的闭式；过大的 $n$ 常降低表现，小次数（如 $n \in \{1,2,3\}$）通常更佳。

### DDIM = AB-DEIS-0

当 $n=0$（常数多项式）时退化为指数 Euler 步，与确定性 DDIM 更新一致。

---

## DPM-Solver

DPM-Solver 系列代表 PF-ODE 求解器的重要进展，目标是用少得多的步数达到相近样本质量；DPM-Solver++ 与 DPM-Solver-v3 还针对条件生成与 CFG 做了设计。

### DPM-Solver 的洞察：通过对数 SNR 的时间重参数化

用**半对数信噪比** $\lambda_t := \log(\alpha_t/\sigma_t)$ 重参数化时间，将非线性项变为指数加权积分，便于在 $\lambda$ 上做 Taylor 展开。

**命题（指数加权的精确解）：** 精确解可表为
$$
\widetilde{\Psi}_{s\to t}(\mathbf{x}_s) = \frac{\alpha_t}{\alpha_s}\mathbf{x}_s - \alpha_t \int_{\lambda_s}^{\lambda_t} e^{-\lambda} \hat{\bm{\epsilon}}_{\bm{\phi}^*}(\hat{\mathbf{x}}_\lambda,\lambda)\mathrm{d} \lambda.
$$
等价地，PF-ODE 在 $\lambda$ 下化为
$$
\frac{\mathrm{d} \hat{\mathbf{x}}_\lambda}{\mathrm{d} \lambda} = \frac{\alpha_\lambda'}{\alpha_\lambda} \hat{\mathbf{x}}_\lambda - \sigma_\lambda \hat{\bm{\epsilon}}_{\bm{\phi}^*}(\hat{\mathbf{x}}_\lambda, \lambda).
$$

### 用 Taylor 展开估计积分

在 $\lambda_s$ 处用 $(n-1)$ 阶 Taylor 展开近似被积函数，代入积分得到闭式近似，定义 **DPM-Solver-$n$**。$n=1$ 时退化为 DDIM；$n \geq 2$ 时需近似高阶导数，原文用有限差分与中间时刻实现。

### DPM-Solver-$n$ 的实现

预计算系数 $C_k$（与 $\varphi_{k+1}(h)$ 闭式）；用中间时刻 $s^{\mathrm{mid}}$ 与有限差分近似导数。**算法 DPM-Solver-2** 给出每步两次模型求值的伪代码。采样时间步建议在对数 SNR $\lambda$ 上均匀取点。

### DDIM = DPM-Solver-1

对固定调度，DPM-Solver-1 与确定性 DDIM（$\eta=0$）更新恒等。

### DPM-Solver-2 与经典 Heun 更新的讨论

在对数 SNR 下，$\mathbf{v}$-预测时 Heun = DPM-Solver-2；$\bm{\epsilon}$-/$\mathbf{x}$-/$\mathbf{s}$-预测时指数–Heun = DPM-Solver-2 ≠ 普通 Heun。

---

## DPM-Solver++

### 从 DPM-Solver 到带引导的 DPM-Solver++

高阶求解器在大引导尺度下存在稳定性问题。DPM-Solver++ 提出采用 $\mathbf{x}$-预测与阈值方法以提高稳定性。

### DPM-Solver++ 的方法

将精确解改写为数据参数化下的指数加权 $\hat{\mathbf{x}}_{\bm{\phi}^*}$ 积分；提供单步（Taylor）与多步（两步）两种变体。

### DPM-Solver++ 单步与多步

单步：用 $n$ 阶 Taylor 展开，$n=2$ 时得 DPM-Solver++(2S)。多步：复用两个历史锚点，每步一次新模型调用，用仿射近似被积函数得到带 $\mathrm{D}_i^{\mathrm{sim}}$ 的更新。**算法 DPM-Solver++(2M)** 给出完整伪代码。

---

## PF-ODE 求解器族及其数值类比

### PF-ODE 求解器族与经典对应

DDIM、DEIS、DPM-Solver、DPM-Solver++ 分别对应 Euler 型、指数 AB、指数 RK（对数 SNR）、指数 Heun 等经典格式。**表** 总结 PF-ODE 采样器与数值分析类比。

### 关于 DEIS 与 DPM-Solver++ 的讨论

二者均为指数积分器；DEIS 为多步、拉格朗日基，DPM++ 有 2S/2M、在对数 SNR 下用数据预测与后向差分。**表** 比较 DEIS 与 DPM++ 的各方面。

---

## （可选）DPM-Solver-v3

DPM-Solver-v3 在 PF-ODE 的 $\lambda$ 形式中引入自由变量，通过最小化离散化误差的优化确定参数化与系数。

### 洞察 1：调整线性项

引入 $\bm{\ell}_\lambda$ 将漂移分解，通过最小化 $\mathbb{E}\|\nabla_{\mathbf{x}}\mathrm{N}_{\bm{\phi}^*}\|_F^2$ 得 $\bm{\ell}_\lambda^*$，使非线性余项对 $\mathbf{x}$ 的敏感度降低。

### 洞察 2：引入自由变量最小化离散化误差

用新参数化 $\mathrm{N}_{\bm{\phi}^*}^{\text{new}}$ 等价重写解，其 $\lambda$-导数含自由变量 $\mathbf{a}_\lambda$、$\mathbf{b}_\lambda$；最小二乘得 $({\color{cyan}\mathbf{a}_\lambda^*}, {\color{green}\mathbf{b}_\lambda^*})$，可预计算。

### 综合与高阶 DPM-Solver-v3

先求 $\bm{\ell}_\lambda^*$ 再求 $\mathbf{a}_\lambda^*$、$\mathbf{b}_\lambda^*$；实践中用 MCMC 估计。这些统计量也可用于构造高阶求解器（Taylor 展开+有限差分近似导数）。

### 更多解释及与其他方法的联系

$\mathrm{N}_{\bm{\phi}^*}^{\text{new}}$ 是通用参数化；$\bm{\epsilon}$/$\mathbf{x}$-预测为其特例。一阶 DPM-Solver-v3 与 DDIM 不同。

---

## （可选）ParaDiGMs

### 从时间步进到时间并行求解器

将 PF-ODE 解写为积分形式的不动点，用 Picard 迭代在时间上并行求值。

### ParaDiGMs 的方法

定义轨迹到轨迹的算子 $\mathcal{L}$，真解为其不动点；离散化后得 $\mathbf{x}^{(k+1)}_j = \mathbf{x}^{(k)}_0 - \Delta t \sum_{i=0}^{j-1} \mathbf{v}_{\bm{\phi}^*} \big(\mathbf{x}^{(k)}_i, t_i\big)$。用滑动窗口：并行求窗口内漂移、左锚点累加更新、按误差与容差确定步长并推进窗口。**算法 ParaDiGMs** 给出带滑动窗口的伪代码。

### 与时间步进求解器的关系

$p=1$ 时退化为一阶时间步进（如 DDIM）；可与高阶求积或 DPM 族多步/指数积分器结合，并行方案与具体求解器选择无关。

---

## 结束语

本章直面扩散模型缓慢迭代采样的限制，通过更高效地求解 PF-ODE 加速生成：从 DDIM（一阶指数 Euler）到 DEIS（高阶多步）再到 DPM-Solver 系列（对数 SNR 重参数化）。NFE 已可降至 10–20。这些无训练方法仍是迭代的；能否在一步或极少数步内实现高质量生成？Part D 将探讨基于训练的加速：蒸馏类方法与从零学习的少步生成器（如一致性训练），将扩散质量与一步生成器速度结合。
