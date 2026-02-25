# 第 5 章 扩散模型的统一与系统视角 {#ch:all-equivalent}

> *数学是给不同事物取同一门名的艺术。*  
> — Henri Poincaré

本章在一个统一的图景下，系统地将变分视角、基于分数的视角与基于流的视角联系起来。尽管动机不同，这些方法在现代扩散方法的核心机制上是一致的。基于第 2–4 章的内容，我们可以看到一个共同的套路：定义一个刻画边际路径的前向扰动过程，然后学习一个时变向量场，将简单先验沿该路径输运到数据分布。

所有视角下的一个关键要素是 5.1 节引入的**条件化技巧**：它把难以处理的边际目标转化为可处理的条件目标，从而得到稳定、高效的训练。

在 **5.2 节**中，我们系统分析训练目标，识别其本质成分，并阐明变分、基于分数与基于流视角下损失函数的构造方式。

**5.3 节**表明，形如 $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \bm{\epsilon}$ 的任意仿射前向噪声注入，都可以等价地变换到标准线性进度 $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\bm{\epsilon}$。此外，常见的参数化——噪声预测、干净数据预测、分数预测与速度预测——在梯度意义下可以互相替换。因此，噪声调度与参数化的选择都遵循同一建模原则。

最后，**5.4 节**将讨论收束到一条统摄性规律：**Fokker–Planck 方程**。无论视作变分方案（离散时间去噪）、基于分数的方法（SDE 表述）还是基于流的方法（ODE 表述），都是在构造一个边际服从同一密度演化的生成器。Fokker–Planck 方程因此成为三种视角共同满足的普适约束，差异仅体现在参数化和训练目标上。

---

## 5.1 条件化技巧：扩散模型的“秘密武器” {#sec:conditional-trick}

截至目前，我们从三个看似不同的来源——变分、基于分数与基于流——考察了扩散模型。各自最初有不同的动机，并导出（在固定 $t$ 下）不同的训练目标：

- **变分视角：** 学习参数化密度 $p_{\bm{\phi}}(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t)$ 来逼近 oracle 逆向转移 $p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t)$，通过最小化：
  $$
  \mathcal{J}_{\text{KL}}(\bm{\phi}) := \mathbb{E}_{p_t(\mathbf{x}_t)}\left[
      \mathcal{D}_{\mathrm{KL}}\big(p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t) \| p_{\bm{\phi}}(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t)\big)
  \right];
  $$
- **基于分数的视角：** 学习分数模型 $\mathbf{s}_{\bm{\phi}}(\mathbf{x}_t, t)$ 来逼近边际分数 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t)$，通过：
  $$
  \mathcal{J}_{\text{SM}}(\bm{\phi}) := \mathbb{E}_{p_t(\mathbf{x}_t)}\left[
     \left\| \mathbf{s}_{\bm{\phi}}(\mathbf{x}_t, t) - \nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t) \right\|_2^2
  \right];
  $$
- **基于流的视角：** 学习速度模型 $\mathbf{v}_{\bm{\phi}}(\mathbf{x}_t, t)$ 以匹配 oracle 速度 $\mathbf{v}_t(\mathbf{x}_t)$（例如由边际 oracle 速度定义），通过最小化：
  $$
  \mathcal{J}_{\text{FM}}(\bm{\phi}) := \mathbb{E}_{p_t(\mathbf{x}_t)} \left[
      \left\| \mathbf{v}_{\bm{\phi}}(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t) \right\|_2^2
  \right].
  $$

乍看之下，这些目标似乎都难以处理，因为它们都依赖在一般情况下根本无法获知的 oracle 量。但这里有一个精彩的转折：每种方法都独立地得到了同一个优雅的解决办法——**对数据 $\mathbf{x}_0$ 做条件化**。这一技术将每个难以处理的训练目标转化为可处理的目标。

这一“条件化技巧”把目标重写为对已知高斯条件 $p_t(\mathbf{x}_t | \mathbf{x}_0)$ 的期望，得到梯度等价的闭式回归目标与可处理的训练目标：

- **变分视角**（条件 KL）：
  $$
  \mathcal{J}_{\text{KL}}(\bm{\phi}) = \underbrace{\mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{p_t(\mathbf{x}_t|\mathbf{x}_0)} \left[
      \mathcal{D}_{\mathrm{KL}}\big(p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t, \mathbf{x}_0) \| p_{\bm{\phi}}(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t)\big)
  \right]}_{\mathcal{J}_{\text{CKL}}(\bm{\phi})} + C;
  $$
- **基于分数的视角**（分数匹配）：
  $$
  \mathcal{J}_{\text{SM}}(\bm{\phi}) = \underbrace{\mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{p_t(\mathbf{x}_t|\mathbf{x}_0)} \left[
      \left\| \mathbf{s}_{\bm{\phi}}(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{x}_0) \right\|_2^2
  \right]}_{\mathcal{J}_{\text{DSM}}(\bm{\phi})} + C;
  $$
- **基于流的视角**（流匹配 / CFM）：
  $$
  \mathcal{J}_{\text{FM}}(\bm{\phi}) = \underbrace{\mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{p_t(\mathbf{x}_t|\mathbf{x}_0)} \left[
      \left\| \mathbf{v}_{\bm{\phi}}(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t | \mathbf{x}_0) \right\|^2
  \right]}_{\mathcal{J}_{\text{CFM}}(\bm{\phi})} + C.
  $$

为建立统一视角，下面我们系统性地重访条件 KL、分数与速度目标。关键的是，这些目标不仅可处理，而且与原始形式仅相差一个常数平移，在梯度上等价。条件形式（$\mathcal{J}_{\text{CKL}}$、$\mathcal{J}_{\text{DSM}}$、$\mathcal{J}_{\text{CFM}}$）与原始形式（$\mathcal{J}_{\text{KL}}$、$\mathcal{J}_{\text{SM}}$、$\mathcal{J}_{\text{FM}}$）只差这一平移，梯度不变，因而优化景观保持不变。因此，最小化元仍唯一对应真实的 oracle 目标，因为每个目标都归结为最小二乘回归，其解恢复相应的条件期望：

$$
\begin{aligned}
        p^*(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t) &= \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot|\mathbf{x}_t)} \big[p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t, \mathbf{x}_0)\big] &&= p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t), \\
    \mathbf{s}^*(\mathbf{x}_t, t) &= \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot|\mathbf{x}_t)} \big[\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0)\big] &&= \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t), \\
    \mathbf{v}^*(\mathbf{x}_t, t) &= \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot|\mathbf{x}_t)} \big[\mathbf{v}_t(\mathbf{x}_t|\mathbf{x}_0)\big] &&= \mathbf{v}_t(\mathbf{x}_t).
\end{aligned}
$$

这不是巧合：通过使训练可处理，这些条件形式揭示了一种深刻的统一。变分扩散、基于分数的 SDE 与流匹配，不过是同一原理的不同侧面。三种视角，一个洞见，被优雅地联系在一起。

本章后续将继续探讨它们之间的等价性。

---

## 5.2 阐明扩散模型训练损失的一条路线图 {#sec:elucidating}

本节系统梳理扩散模型中训练损失的结构。在 5.2.1 节中，我们将标准的三种目标推广到更广的四类参数化，说明它们如何从不同建模视角产生。在 5.2.2 节中，我们将这些结果提炼为一个一般框架，厘清扩散目标的结构，为 5.4 节的等价性结果做铺垫。

### 5.2.1 扩散模型中四种常见参数化 {#subsec:four-predictions}

如无特别说明，本节均考虑前向扰动核
$$
p_t(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t; \alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I} \right),
$$
其中 $\mathbf{x}_0 \sim p_{\mathrm{data}}$，即前向核定义。

记 $\omega: [0, T] \to \mathbb{R}_{>0}$ 为正的时间加权函数。四种标准参数化——噪声 $\bm{\epsilon}_{\bm{\phi}}$、干净 $\mathbf{x}_{\bm{\phi}}$、分数 $\mathbf{s}_{\bm{\phi}}$ 与速度 $\mathbf{v}_{\bm{\phi}}$——以及它们各自的最小化元 $\bm{\epsilon}^*$、$\mathbf{x}^*$、$\mathbf{s}^*$、$\mathbf{v}^*$，为清晰和后续讨论总结如下。

**变分视角。** 基于 DDPM 中的 KL 散度（见 DDPM 预测与 VDM 联系相关小节），该方法归结为预测产生 $\mathbf{x}_t$ 的期望噪声，或 $\mathbf{x}_t$ 所来自的期望干净信号。

1. **$\bm{\epsilon}$-预测（噪声预测）**：
   $$
   \bm{\epsilon}_{\bm{\phi}}(\mathbf{x}_t,t) \approx \mathbb{E}[\bm{\epsilon}|\mathbf{x}_t]=\bm{\epsilon}^*(\mathbf{x}_t, t)
   $$
   训练目标为
   $$
   \mathcal{L}_{\text{noise}}(\bm{\phi}) := \mathbb{E}_t \left[\omega(t) \mathbb{E}_{\mathbf{x}_0, \bm{\epsilon}} \left\| \bm{\epsilon}_{\bm{\phi}}(\mathbf{x}_t,t) - \bm{\epsilon} \right\|_2^2 \right].
   $$
   这里 $\bm{\epsilon}^*$ 表示为得到给定 $\mathbf{x}_t$ 所注入的平均噪声。

2. **$\mathbf{x}$-预测（干净预测）**：
   $$
   \mathbf{x}_{\bm{\phi}}(\mathbf{x}_t,t) \approx \mathbb{E}[\mathbf{x}_0|\mathbf{x}_t]=\mathbf{x}^*(\mathbf{x}_t, t)
   $$
   训练目标为
   $$
   \mathcal{L}_{\text{clean}}(\bm{\phi}) := \mathbb{E}_t \left[\omega(t) \mathbb{E}_{\mathbf{x}_0, \bm{\epsilon}} \left\| \mathbf{x}_{\bm{\phi}}(\mathbf{x}_t,t) - \mathbf{x}_0 \right\|_2^2 \right].
   $$
   这里 $\mathbf{x}^*$ 表示在给定噪声观测 $\mathbf{x}_t$ 下所有合理干净猜测的平均。

**基于分数的视角。** 预测噪声水平 $t$ 处的分数函数，其方向指向平均意义下将 $\mathbf{x}_t$ 去噪回所有可能生成它的干净样本的方向。

3. **分数预测**：
   $$
   \mathbf{s}_{\bm{\phi}}(\mathbf{x}_t,t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \mathbb{E}\left[\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0) |\mathbf{x}_t\right] = \mathbf{s}^*(\mathbf{x}_t, t)
   $$
   训练目标为
   $$
   \mathcal{L}_{\text{score}}(\bm{\phi}) := \mathbb{E}_t \left[\omega(t) \mathbb{E}_{\mathbf{x}_0, \bm{\epsilon}} \left\| \mathbf{s}_{\bm{\phi}}(\mathbf{x}_t,t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0) \right\|_2^2 \right],
   $$
   其中条件分数满足 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0) = -\frac{1}{\sigma_t} \bm{\epsilon}$。

**基于流的视角。** 预测数据沿 $\mathbf{x}_t$ 演化时的瞬时平均速度。

4. **$\mathbf{v}$-预测（速度预测）**：
   $$
   \mathbf{v}_{\bm{\phi}}(\mathbf{x}_t, t) \approx \mathbb{E}\left[\left. \frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} \right| \mathbf{x}_t\right] =\mathbf{v}^*(\mathbf{x}_t, t)
   $$
   训练目标为
   $$
   \mathcal{L}_{\text{velocity}}(\bm{\phi}) := \mathbb{E}_t \left[\omega(t) \mathbb{E}_{\mathbf{x}_0, \bm{\epsilon}} \left\| \mathbf{v}_{\bm{\phi}}(\mathbf{x}_t,t) - \mathbf{v}_t(\mathbf{x}_t | \mathbf{x}_0,\bm{\epsilon}) \right\|_2^2 \right],
   $$
   其中条件速度为 $\mathbf{v}_t(\mathbf{x}_t|\mathbf{x}_0,\bm{\epsilon}) = \alpha_t' \mathbf{x}_0 + \sigma_t' \bm{\epsilon}$。这里 $\mathbf{v}^*$ 表示穿过观测点 $\mathbf{x}_t$ 的平均速度向量。

基于“三种最小化元与 oracle 一致”的结论，所有四类预测最终都是在给定观测 $\mathbf{x}_t$ 下逼近某种条件期望：平均噪声、干净数据、分数或速度。

### 5.2.2 厘清扩散模型的训练目标 {#subsec:summary-disentagle}

如 5.2.1 节所示，四类预测的目标函数通常共享如下扩散模型训练的一般模板形式：

$$
\mathcal{L}(\bm{\phi}):=\mathbb{E}_{\mathbf{x}_0, \bm{\epsilon}} \underbrace{\mathbb{E}_{p_{\text{time}}(t)}}_{\substack{\text{时间} \\ \text{分布}}}\Big[ \underbrace{\omega(t)}_{\substack{\text{时间} \\ \text{加权}}}\underbrace{\| \mathrm{NN}_{\bm{\phi}}(\mathbf{x}_t, t) - (A_t \mathbf{x}_0 + B_t \bm{\epsilon}) \|_2^2}_{\text{MSE 部分}}\Big].
$$

为提高训练效率、优化扩散模型学习流程，以下几类设计选择至关重要：

- **(A)** 通过 $\alpha_t$ 和 $\sigma_t$ 的前向过程中 $\mathbf{x}_t$ 的噪声调度；
- **(B)** $\mathrm{NN}_{\bm{\phi}}$ 的预测类型及其回归目标 $(A_t \mathbf{x}_0 + B_t \bm{\epsilon})$；
- **(C)** 时间加权函数 $\omega(\cdot) \colon [0,T] \to \mathbb{R}_{\geq 0}$；
- **(D)** 时间分布 $p_{\text{time}}$。

下面分别展开这四部分，作为后续各节讨论的路线图。

**(A) 噪声调度 $\alpha_t$ 与 $\sigma_t$。** 用户可根据应用灵活选择调度，常见例子见插值实例表。重要的是，如我们将要证明的，所有形如 $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \bm{\epsilon}$ 的仿射流在数学上等价。具体而言，任意此类插值都可通过适当的时间重参数化与空间缩放，转化为典范线性调度（$\alpha_t = 1 - t$，$\sigma_t = t$）或三角调度（$\alpha_t = \cos t$，$\sigma_t = \sin t$）。

**(B) 参数化 $\mathrm{NN}_{\bm{\phi}}$ 与训练目标 $A_t \mathbf{x}_0 + B_t \bm{\epsilon}$。** 用户可灵活选择模型的预测目标：干净信号、噪声、分数或速度。如 5.2.1 节所述，这些预测类型共享形如
$$
\text{回归目标} = A_t \mathbf{x}_0 + B_t \bm{\epsilon}
$$
的公共回归目标，系数 $A_t$、$B_t$ 由所选预测类型与调度 $(\alpha_t, \sigma_t)$ 共同决定，关系见参数化 $A_t$–$B_t$ 表。

尽管这四种参数化看起来不同，我们将在后文证明它们可通过简单代数变换互相转化；并证明该模板中的平方 $\ell_2$ 损失项在所有预测类型下梯度等价，仅差一个仅依赖噪声调度 $(\alpha_t, \sigma_t)$ 的时间加权因子（除 $\omega(t) p_{\mathrm{time}}(t)$ 之外）。

**表：不同参数化之间关系的总结。** 四种参数化在数学上等价，可通过简单代数变换互相转换。

| 回归目标类型 | $A_t$ | $B_t$ |
|-------------|-------|-------|
| Clean（干净） | $1$  | $0$  |
| Noise（噪声） | $0$  | $1$  |
| Conditional Score（条件分数） | $0$  | $-\frac{1}{\sigma_t}$ |
| Conditional Velocity（条件速度） | $\alpha_t'$ | $\sigma_t'$ |

**(C) 时间分布 $p_{\text{time}}(t)$。** 由于训练损失是对 $t$ 的期望，从 $p_{\mathrm{time}}(t)$ 采样时间在数学上等价于用 $p_{\mathrm{time}}(t)$ 对每个 $t$ 的 MSE 加权；该因子可并入已有的时间加权 $\omega(t)$。但经验表明，$p_{\mathrm{time}}(t)$ 的不同选择会影响表现，因此我们将时间分布 $p_{\text{time}}(t)$ 与时间加权函数 $\omega(t)$ 分开讨论。

时间分布的常见选择是 $[0,T]$ 上的均匀分布。其他选择包括对数正态分布与自适应重要性采样方法。

**(D) 时间加权函数 $\omega(t)$。** 常见的加权是常数 $\omega \equiv 1$，也有自适应加权方案。某些 $\omega(t)$ 的选择会使上述目标成为负对数似然更紧的上界，从而将目标重新表述为最大似然训练。$\omega(t)$ 的典型方案包括令 $\omega(t) = g^2(t)$（其中 $g$ 为前向 SDE 的扩散系数）、信噪比（SNR）加权或单调加权函数。

总体而言，无论噪声调度、预测类型或时间采样分布如何选择，这些因素在理论上都归结为对目标函数中时间加权的影响，该时间加权会影响实际训练景观，进而影响模型表现。

---

## 5.3 扩散模型中的等价性 {#sec:equivalent-parametrizations}

5.2.1 节引入的四种预测类型将在 5.3.1 节中证明在梯度最小化意义下等价。随后在 5.3.2 节中我们将拓宽这一视角，说明不同的前向噪声调度可通过简单的时间与空间缩放相互联系。

### 5.3.1 四种预测类型等价 {#subsec:four-equiv-para}

我们先分析 5.2.2 节目标中成分 (B) 的设计选择。

我们已经看到，四种预测类型并非独立选择，而是同一底层量的不同视图。例如，噪声与干净预测直接相关，分数与噪声预测也直接相关。这一反复出现的模式指向更深的原理：四种参数化代数等价，可通过简单变换互相转化。为精确表述，我们给出以下命题（见四种参数化等价关系图）：

**命题（参数化等价）** 设最小化各自目标的最优预测为
$$
\bm{\epsilon}^*(\mathbf{x}_t,t) , \quad \mathbf{x}^*(\mathbf{x}_t,t), \quad \mathbf{s}^*(\mathbf{x}_t,t) , \quad \mathbf{v}^*(\mathbf{x}_t,t),
$$
分别对应噪声、干净、分数与速度参数化。它们满足如下等价关系：
$$
\begin{aligned}
\bm{\epsilon}^*(\mathbf{x}_t,t) &= -\sigma_t \mathbf{s}^*(\mathbf{x}_t,t), \\
\mathbf{x}^*(\mathbf{x}_t,t) &= \frac{1}{\alpha_t} \mathbf{x}_t
+ \frac{\sigma_t^2}{\alpha_t} \mathbf{s}^*(\mathbf{x}_t,t), \\
\mathbf{v}^*(\mathbf{x}_t,t) &= \alpha_t' \mathbf{x}^* + \sigma_t' \bm{\epsilon}^* 
= f(t)\mathbf{x}_t - \frac{1}{2} g^2(t)\mathbf{s}^*(\mathbf{x}_t,t).
\end{aligned}
$$
其中 $f(t)$ 与 $g(t)$ 通过前向 SDE 引理与 $\alpha_t$、$\sigma_t$ 相关。这些最小化元还满足噪声/干净/分数/速度定义式中的恒等式。

上述等价关系在给定前向加噪系数下，在每个 $t$ 处诱导出四种参数化
$$
\bm{\epsilon}_{\bm{\phi}}(\mathbf{x}_t,t), \quad \mathbf{x}_{\bm{\phi}}(\mathbf{x}_t,t), \quad \mathbf{s}_{\bm{\phi}}(\mathbf{x}_t,t), \quad   \mathbf{v}_{\bm{\phi}}(\mathbf{x}_t,t)
$$
之间的一一转换。实践中，我们仅用一种参数化（例如 $\bm\epsilon_{\bm\phi}$）训练一个网络，其他量则由该等价关系**事后定义**。

**图：四种参数化的等价关系。** $\mathbf{v}$-预测由 $\mathbf{v}^* = \alpha_t \mathbf{x}^* + \sigma_t \bm{\epsilon}^*$ 给出，其中干净与 $\bm{\epsilon}$-预测可通过 $\mathbf{x}_t = \alpha_t \mathbf{x}^* + \sigma_t \bm{\epsilon}^*$ 互换。

### 5.3.2 不同参数化下的 PF-ODE {#subsec:pf-ode-different-para}

PF-ODE 允许多种等价参数化（分数、噪声、去噪、速度）。尽管原则上可互换，选择会带来实际差异：改变向量场的刚度、离散误差的行为以及优化的难易。在使用先进 ODE 求解器进行快速采样时，实践者常采用 $\bm{\epsilon}$ 或 $\mathbf{x}$ 预测，因其与求解器输入一致并减少误差累积。在训练仅使用少量函数求值的生成器时，$\mathbf{x}$ 或 $\mathbf{v}$ 预测往往给出更平滑的目标和更好的步间一致性。

我们在每种参数化下写出 PF-ODE，并用参数化等价关系显式给出转换。结果总结为以下命题。

**命题（不同参数化下的 PF-ODE）** 设 $\alpha_t$、$\sigma_t$ 为前向扰动调度，记时间导数为 $\alpha_t' := \frac{\mathrm{d} \alpha_t}{\mathrm{d} t}$，$\sigma_t' := \frac{\mathrm{d} \sigma_t}{\mathrm{d} t}$。则经验 PF-ODE 有如下等价形式：
$$
\begin{aligned}
    \frac{\mathrm{d} \mathbf{x}(t)}{\mathrm{d} t}
&= \frac{\alpha_t'}{\alpha_t} \mathbf{x}(t)
 - \sigma_t \left(\frac{\alpha_t'}{\alpha_t}-\frac{\sigma_t'}{\sigma_t}\right)
   \bm{\epsilon}^*(\mathbf{x}(t), t) \\    
&= \frac{\sigma_t'}{\sigma_t} \mathbf{x}(t)
 + \alpha_t \left(\frac{\alpha_t'}{\alpha_t}-\frac{\sigma_t'}{\sigma_t}\right)
   \mathbf{x}^*(\mathbf{x}(t), t) \\
&= \frac{\alpha_t'}{\alpha_t} \mathbf{x}(t)
 + \sigma_t^{2} \left(\frac{\alpha_t'}{\alpha_t}-\frac{\sigma_t'}{\sigma_t}\right)
   \mathbf{s}^*(\mathbf{x}(t), t) \\ 
   &= \alpha_t'\mathbf{x}^*(\mathbf{x}(t), t) + \sigma_t' \bm{\epsilon}^*(\mathbf{x}(t), t)\\
&= \mathbf{v}^*(\mathbf{x}(t), t).
\end{aligned}
$$

若令
$$
f(t) = \frac{\alpha_t'}{\alpha_t}, 
\quad
g^2(t) = \frac{\mathrm{d}}{\mathrm{d} t} \big(\sigma_t^2\big) 
         - 2 \frac{\alpha_t'}{\alpha_t} \sigma_t^2
       = 2\sigma_t\sigma_t' - 2 \frac{\alpha_t'}{\alpha_t} \sigma_t^2,
$$
则 PF-ODE 可写成熟悉的 Score SDE 形式：
$$
\frac{\mathrm{d} \mathbf{x}(t)}{\mathrm{d} t}
= f(t) \mathbf{x}(t)  -  \frac{1}{2} g^2(t) \mathbf{s}^*(\mathbf{x}(t),t).
$$

为具体感受 PF-ODE 在采样时如何离散化，我们将在 DDIM 重访一节给出广泛使用的扩散 ODE 采样器 DDIM 的更新规则，说明 Euler 离散如何自然与 PF-ODE 相连。

### 5.3.3 所有仿射流等价 {#subsec:all-flows-are-equiv}

接下来我们分析 5.2.2 节目标中成分 (A) 的设计选择。

**状态层面等价。** FM 与 RF 中常用的典范插值为
$$
\mathbf{x}_t^{\mathrm{FM}}=(1-t) \mathbf{x}_0+t \bm{\epsilon} = \mathbf{x}_{0}+t(\bm{\epsilon}-\mathbf{x}_0),
$$
其速度为常向量 $\bm{\epsilon}-\mathbf{x}_0$。本小节的关键点是：这一选择的表面简洁性并非本质——任意仿射插值
$$
\mathbf{x}_t=\alpha_t \mathbf{x}_0+\sigma_t \bm{\epsilon}
$$
都可写成典范路径经时间重参数化与缩放后的形式。定义
$$
c(t):=\alpha_t+\sigma_t,
\qquad
\tau(t):=\frac{\sigma_t}{\alpha_t+\sigma_t}
\quad\big(c(t)\neq 0\big).
$$
直接代数改写可得
$$
\mathbf{x}_t
=\alpha_t \mathbf{x}_0+\sigma_t \bm{\epsilon}
=\big(\alpha_t+\sigma_t\big) \left(\frac{\alpha_t}{\alpha_t+\sigma_t}\mathbf{x}_0+\frac{\sigma_t}{\alpha_t+\sigma_t}\bm{\epsilon}\right)
=c(t) \left(\big(1-\tau(t)\big)\mathbf{x}_0+\tau(t)\bm{\epsilon}\right)
=c(t) \mathbf{x}_{\tau(t)}^{\mathrm{FM}}.
$$
因此每条仿射路径都是典范 FM 路径在变量替换 $t\mapsto\tau(t)$ 与空间缩放 $\mathbf{x}\mapsto c(t)\mathbf{x}$ 下的像。等式逐点成立，故在分布意义下也成立。

对相应速度，对 $\mathbf{x}_t=c(t)\mathbf{x}_{\tau(t)}^{\mathrm{FM}}$ 应用链式法则可得
$$
\mathbf{v}(\mathbf{x}_t,t)
= \frac{\mathrm{d}}{\mathrm{d} t}\left(\alpha_t\mathbf{x}_0 +\sigma_t \bm{\epsilon}\right)
= \frac{\mathrm{d}}{\mathrm{d} t}\big(c(t)\mathbf{x}_{\tau(t)}^{\mathrm{FM}}\big)
= c'(t) \mathbf{x}_{\tau(t)}^{\mathrm{FM}}+c(t) \tau'(t) \frac{\mathrm{d}}{\mathrm{d} s}\mathbf{x}^{\mathrm{FM}}_{s}\bigg|_{s=\tau(t)} 
= c'(t) \mathbf{x}_{\tau(t)}^{\mathrm{FM}}+c(t) \tau'(t) \mathbf{v}^{\mathrm{FM}} \left(\mathbf{x}_{\tau(t)}^{\mathrm{FM}},\tau(t)\right),
$$
因为沿典范路径有 $\mathbf{v}^{\mathrm{FM}}(\mathbf{x}^{\mathrm{FM}}_{\tau},\tau)=-\mathbf{x}_0+\bm{\epsilon}$。

我们将上述推导总结为以下形式命题。

**命题（仿射流等价）** 设 $\mathbf{x}_t^{\mathrm{FM}}=(1-t)\mathbf{x}_0+t \bm{\epsilon}$，$\mathbf{x}_t=\alpha_t\mathbf{x}_0+\sigma_t\bm{\epsilon}$，且 $c(t):=\alpha_t+\sigma_t\neq 0$，$\tau(t):=\sigma_t/(\alpha_t+\sigma_t)$。则
$$
\begin{aligned}
\mathbf{x}_t&=c(t) \mathbf{x}_{\tau(t)}^{\mathrm{FM}},\\
\mathbf{v}(\mathbf{x}_t,t)
&=c'(t) \mathbf{x}_{\tau(t)}^{\mathrm{FM}}
 + c(t) \tau'(t) \mathbf{v}^{\mathrm{FM}} \left(\mathbf{x}_{\tau(t)}^{\mathrm{FM}},\tau(t)\right).
\end{aligned}
$$
特别地，所有仿射插值在时间重参数化与空间缩放意义下等价。

**与三角流的等价。** 另一种常用的仿射流是三角插值。作为具体例子，我们也说明**任意**仿射流都可用这种形式表示。三角路径定义为
$$
\mathbf{x}^{\mathrm{Trig}}_u := \cos(u) \mathbf{x}_0 + \sin(u)\bm{\epsilon} .
$$
令 $R_t := \sqrt{\alpha_t^2 + \sigma_t^2}$，设 $R_t>0$。选取角 $\tau_t$ 使得
$$
\cos \tau_t = \frac{\alpha_t}{R_t},
\qquad
\sin \tau_t = \frac{\sigma_t}{R_t}.
$$
则任意仿射插值 $\mathbf{x}_t=\alpha_t\mathbf{x}_0+\sigma_t\bm{\epsilon}$ 都是三角路径的缩放与重时版本：
$$
\mathbf{x}_t
= \alpha_t\mathbf{x}_0+\sigma_t\bm{\epsilon}
= R_t \left(\frac{\alpha_t}{R_t}\mathbf{x}_0+\frac{\sigma_t}{R_t}\bm{\epsilon}\right)
= R_t \mathbf{x}^{\mathrm{Trig}}_{\tau_t}.
$$
$(\alpha_t,\sigma_t)$ 是平面上的点，用 $R_t$ 归一化后落在单位圆上，从而确定角 $\tau_t$ 进而确定状态 $\mathbf{x}^{\mathrm{Trig}}_{\tau_t}$；半径 $R_t$ 给出整体尺度。对 $\mathbf{x}^{\mathrm{Trig}}_u$ 关于 $u$ 求导得其速度
$$
\mathbf{v}^{\mathrm{Trig}}_u = -\sin(u) \mathbf{x}_0 + \cos(u) \bm{\epsilon}.
$$
通过与上式相同的变量替换，可得到速度（以及其他参数化）的闭式转换。

综上可得：

**结论** 无论调度 $(\alpha_t,\sigma_t)$ 如何（包括 VE、VP（如三角）、FM 或 RF），仿射插值都可通过适当的时间变量替换与标量缩放互相转换。

**四种参数化的训练目标。** 设 $\mathbf{x}_t=\alpha_t \mathbf{x}_0+\sigma_t \bm{\epsilon}$，$\sigma_t>0$，$(\alpha_t,\sigma_t)$ 可微且 $\alpha_t'\sigma_t-\alpha_t\sigma_t'\neq 0$。考虑 oracle 目标
$$
\bm{\epsilon}^*(\mathbf{x}_t,t)=\mathbb{E}[\bm{\epsilon}|\mathbf{x}_t],\quad 
\mathbf{x}_0^*(\mathbf{x}_t,t)=\mathbb{E}[\mathbf{x}_0|\mathbf{x}_t],\quad
\mathbf{v}^*(\mathbf{x}_t,t)=\mathbb{E}[\alpha_t'\mathbf{x}_0+\sigma_t'\bm{\epsilon}|\mathbf{x}_t].
$$
由参数化等价命题，它们满足
$$
\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)
= -\frac{1}{\sigma_t} \bm{\epsilon}^*(\mathbf{x}_t,t)
= \frac{\alpha_t}{\sigma_t^2} \left(\mathbf{x}_0^*(\mathbf{x}_t,t)-\frac{\mathbf{x}_t}{\alpha_t}\right),
\quad
\mathbf{v}^*=\alpha_t' \mathbf{x}_0^*+\sigma_t' \bm{\epsilon}^*.
$$
在头转换
$$
\mathbf{s}_{\bm{\phi}} \equiv -\frac{1}{\sigma_t} \bm{\epsilon}_{\bm{\phi}}
\equiv \frac{\alpha_t}{\sigma_t^2} \left(\mathbf{x}_{\bm{\phi}}-\frac{\mathbf{x}_t}{\alpha_t}\right),
$$
以及速度到分数的转换
$$
\mathbf{s}_{\bm{\phi}}
= \frac{\alpha_t}{\sigma_t(\alpha_t'\sigma_t-\alpha_t\sigma_t')} \mathbf{v}_{\bm{\phi}}
- \frac{\alpha_t'}{\sigma_t(\alpha_t'\sigma_t-\alpha_t\sigma_t')} \mathbf{x}_t,
$$
下，每样本平方损失在时间依赖权重意义下一致：
$$
\begin{aligned}
\big\|\mathbf{s}_{\bm{\phi}}-\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)\big\|_2^2
&=\frac{1}{\sigma_t^2} \big\|\bm{\epsilon}_{\bm{\phi}}-\bm{\epsilon}^*\big\|_2^2 \\
&=\frac{\alpha_t^2}{\sigma_t^4} \big\|\mathbf{x}_{\bm{\phi}}-\mathbf{x}_0^*\big\|_2^2 \\
&=\left(\frac{\alpha_t}{\sigma_t(\alpha_t'\sigma_t-\alpha_t\sigma_t')}\right)^{ 2}
\big\|\mathbf{v}_{\bm{\phi}}-\mathbf{v}^*\big\|_2^2 .
\end{aligned}
$$

由仿射流等价命题，任意仿射流 $\mathbf{x}_t=\alpha_t\mathbf{x}_0+\sigma_t\bm{\epsilon}$ 可通过 $\mathbf{x}_t=c(t)\mathbf{x}^{\mathrm{FM}}_{\tau(t)}$（$c(t)=\alpha_t+\sigma_t$，$\tau(t)=\sigma_t/(\alpha_t+\sigma_t)$）转换到典范 FM 路径。求导得
$$
\mathbf{v}_{\bm{\phi}}(\mathbf{x}_t,t)=c'(t)\mathbf{x}^{\mathrm{FM}}_{\tau(t)}
+c(t)\tau'(t)\mathbf{v}^{\mathrm{FM}}_{\bm{\phi}} \left(\mathbf{x}_{\tau(t)}^{\mathrm{FM}},\tau(t)\right),
\qquad
\mathbf{x}_{\tau(t)}^{\mathrm{FM}}=\frac{\mathbf{x}_t}{c(t)},
$$
$\mathbf{v}^*$ 满足相同关系。因此速度损失按
$$
\big\|\mathbf{v}_{\bm{\phi}}(\mathbf{x}_t,t)-\mathbf{v}^*(\mathbf{x}_t,t)\big\|_2^2
= \big(c(t) \tau'(t)\big)^2
\Big\|\mathbf{v}^{\mathrm{FM}}_{\bm{\phi}} \left(\frac{\mathbf{x}_t}{c(t)},\tau(t)\right)
-\big(\mathbf{v}^{\mathrm{FM}}\big)^* \left(\frac{\mathbf{x}_t}{c(t)},\tau(t)\right)\Big\|_2^2
$$
变换。

由此可得：

**结论** 分数、噪声、干净与速度的训练目标在理论上等价，仅差由 $(\alpha_t,\sigma_t)$ 决定的时间依赖权重（以及速度情形下涉及 $\mathbf{x}_t$ 的仿射头转换）。

### 5.3.4 （选读）参数化与典范流的概念分析 {#subsec:concept-why-v-affine}

尽管前文已证明四种参数化数学等价、可互相转化，且前向仿射噪声注入流等价于典范形式
$$
\mathbf{x}_t^{\mathrm{FM}} = (1-t) \mathbf{x}_0 + t \bm{\epsilon},
$$
本小节仍进一步提供直觉，并分析将 $\mathbf{v}$-预测参数化与此典范仿射流结合使用的潜在优势。

本小节回答一个简单问题：**不同参数化与调度如何塑造模型所学以及我们如何采样？** 分三步进行：

- **回归目标与调度。** 我们说明为何将 $\mathbf{v}$-预测与典范线性调度 $(\alpha_t,\sigma_t)=(1-t,t)$ 结合是自然的：它保持目标尺度随时间稳定，并消除动力学中的曲率效应。
- **求解器含义。** 我们概念性地考察该参数化如何与数值积分格式相互作用，具体例子（如 Euler 求解器与 Heun 方法）留到相应小节。

为避免歧义，先区分两类速度场。**条件速度**（作为可处理的训练目标）定义为
$$
\mathbf{v}_t(\mathbf{x}_t |\mathbf{z}) = \mathbf{x}_t' = \alpha_t' \mathbf{x}_0 + \sigma_t' \bm{\epsilon}, 
\quad \text{其中 } \mathbf{z} = (\mathbf{x}_0, \bm{\epsilon}),
$$
而**oracle（边际）速度**（用于 PF-ODE 推理时移动样本）为
$$
\mathbf{v}^*(\mathbf{x}, t) = \mathbb{E}\big[\mathbf{v}_t(\cdot |\mathbf{z})  \big|  \mathbf{x}_t = \mathbf{x}\big].
$$

**视角 1：为何 $(\alpha_t,\sigma_t)=(1-t,t)$ 是自然调度。** 令 $\sigma_t:=\rho(t)$，$\alpha_t:=1-\rho(t)$，则条件速度为
$$
\mathbf{v}_t(\mathbf{x}_t |\mathbf{z})
=\rho'(t)(\bm{\epsilon}-\mathbf{x}_0),
\quad\text{其中 } \mathbf{z}=(\mathbf{x}_0,\bm{\epsilon}).
$$

*单位尺度回归目标。* 对典范调度 $\rho(t)=t$，条件速度 $\mathbf{v}_t(\cdot |\mathbf{z})$ 满足
$$
\mathbb{E}\left[\|\mathbf{v}_t(\cdot |\mathbf{z})\|_2^2\right]
=\mathbb{E}_{\bm{\epsilon}}\|\bm{\epsilon}\|_2^2+\mathbb{E}_{\mathbf{x}_0}\|\mathbf{x}_0\|^2_2 = D + \underbrace{\operatorname{Tr}\operatorname{Cov}[\mathbf{x}_0]}_{\text{总方差}}+\underbrace{\|\mathbb{E}\mathbf{x}_0\|^2_2}_{\text{均值}}.
$$
因此期望目标范数在 $t$ 上恒定。将数据标准化为零均值、单位协方差（即 $\operatorname{Cov}[\mathbf{x}_0]=I$）后，$\alpha_t'\mathbf{x}_0$ 与 $\sigma_t'\bm{\epsilon}$ 在所有 $t$ 上贡献相当，避免端点附近的梯度爆炸/消失。从扩散的训练目标
$$
\mathcal{L}_{\mathrm{velocity}}(\bm{\phi})
= \mathbb{E}_{t}  \mathbb{E}_{\mathbf{x}_0,\bm{\epsilon}}\left[\big\|\mathbf{v}_{\bm{\phi}}(\mathbf{x}_t,t) - \mathbf{v}_t(\mathbf{x}_t|\mathbf{z})\big\|_2^2
\right]
$$
出发，由链式法则，损失对参数 $\bm{\phi}$ 的梯度为
$$
\nabla_{\bm{\phi}}\mathcal{L}_{\mathrm{velocity}}(\bm{\phi})
= 2  \mathbb{E}_{t}  \mathbb{E}_{\mathbf{x}_0,\bm{\epsilon}} \left[
  \partial_{\bm{\phi}}\mathbf{v}_{\bm{\phi}}(\mathbf{x}_t,t)^{\top} \left(\mathbf{v}_{\bm{\phi}}(\mathbf{x}_t,t) - \mathbf{v}_t(\mathbf{x}_t|\mathbf{z})\right)
\right].
$$
因此目标 $\|\mathbf{v}_t(\mathbf{x}_t|\mathbf{z})\|_2$ 的尺度影响梯度稳定性：若在某个 $t$ 坍缩为 $0$ 或爆炸，梯度倾向于消失或爆炸。在典范选择 $\rho(t)=t$ 下，上式给出与 $t$ 无关的目标范数，因此不会出现由回归信号引起的端点（$t=0$ 或 $t=1$）坍缩或爆炸（假设 $\mathbb{E}\|\partial_{\bm{\phi}}\mathbf{v}_{\bm{\phi}}(\mathbf{x}_t,t)\|^2$ 及时间权重受控）。

*典范调度与 $\mathbf{v}$-预测的相互作用。* 在仿射路径 $\mathbf{x}_t=\alpha_t \mathbf{x}_0+\sigma_t\bm{\epsilon}$ 下，oracle 速度分解为
$$
\mathbf{v}^*(\mathbf{x}, t)=\alpha_t' \mathbf{x}^*(\mathbf{x},t)+\sigma_t' \bm{\epsilon}^*(\mathbf{x},t),
$$
其中 $\mathbf{x}^*=\mathbb{E}[\mathbf{x}_0|\mathbf{x}_t=\mathbf{x}]$，$\bm{\epsilon}^*=\mathbb{E}[\bm{\epsilon}|\mathbf{x}_t=\mathbf{x}]$。在固定 $\mathbf{x}$ 下对 $t$ 求导得
$$
\partial_t \mathbf{v}_t^*
=\underbrace{\alpha_t'' \mathbf{x}^*+\sigma_t'' \bm{\epsilon}^*}_{\text{调度曲率}}
+ \alpha_t' \partial_t \mathbf{x}^*+\sigma_t' \partial_t \bm{\epsilon}^*.
$$
对线性调度 $\alpha_t=1-t,\ \sigma_t=t$，曲率项消失（$\alpha_t''=\sigma_t''=0$），故 $\mathbf{v}_t^*$ 随时间的变化主要反映后验演化 $(\partial_t \mathbf{x}^*,\partial_t \bm{\epsilon}^*)$ 而非调度。这对 $\mathbf{v}$-预测尤其干净：系数 $\alpha_t',\sigma_t'$ 为常数（$-1$ 与 $+1$），避免了漂移中额外的 $t$ 依赖缩放。相比之下，分数、$\mathbf{x}_0$ 或 $\bm{\epsilon}$-参数化常引入 $\sigma_t'/\sigma_t$、$\alpha_t'/\alpha_t$ 等比值，在线性调度下在端点附近仍可能剧烈变化。因此，尽管原则上并非独家，线性 $(1-t,t)$ 调度与 $v$-预测的结合为 oracle 速度提供了特别稳定、透明的时间依赖。

*最小化条件能量。* 下面从最优传输（见 OT/EOT 章）的更具理论性的视角出发。**条件动能**刻画沿前向路径条件速度的总期望运动量，即从 $\mathbf{x}_0$ 到 $\bm{\epsilon}$ 所需的瞬时运动（或动能）量：
$$
\mathcal K[\rho]
:=\int_0^1 \mathbb{E}_{\mathbf{x}_0,\bm{\epsilon}}\!\big[\|\mathbf{v}_t(\cdot |\mathbf{z})\|_2^2\big]\mathrm{d} t
=\Big(D+\operatorname{Tr}\operatorname{Cov}[\mathbf{x}_0]+\|\mathbb{E}\mathbf{x}_0\|_2^2\Big)
\int_0^1 \big(\rho'(t)\big)^2\mathrm{d} t.
$$
因此最小化 $\mathcal K[\rho]$ 对应在期望意义下找最平滑、能耗最低的路径。在边界条件 $\rho(0)=0$、$\rho(1)=1$ 下，Euler–Lagrange 方程 $\rho''(t)=0$ 给出最小化元 $\rho(t)=t$，即直线条件路径。换言之，在所有连接 $\mathbf{x}_0$ 与 $\bm{\epsilon}$ 的光滑插值中，典范流 $\rho(t)=t$ 是两者之间能量效率最高的方式。我们将在“条件流的最优传输”命题中更详细地回到这一点。

*关于 oracle 速度的注记。* 若改为用边际速度定义能量
$$
\int_0^1 \mathbb{E}_{\mathbf{x}_t} \big[\|\mathbf{v}^*(\mathbf{x}_t, t)\|^2\big]\mathrm{d} t,
$$
其中 $\mathbf{z}=(\mathbf{x}_0,\bm{\epsilon})$，$\mathbf{v}_t(\mathbf{x}_t |\mathbf{z})=\rho'(t)(\bm{\epsilon}-\mathbf{x}_0)$，有
$$
\mathbf{v}^*(\mathbf{x}, t)
=\mathbb{E}[\mathbf{v}_t(\cdot|\mathbf{z})|\mathbf{x}_t=\mathbf{x}]
=\rho'(t) \mathbb{E}[\bm{\epsilon}-\mathbf{x}_0|\mathbf{x}_t=\mathbf{x}];
$$
因此边际速度的能量为
$$
\int_0^1 \mathbb{E}_{\mathbf{x}_t\sim p_t} \big[\|\mathbf{v}^*(\mathbf{x}_t, t)\|^2_2\big]\mathrm{d} t
 = \int_0^1 \mathbb{E}_{\mathbf{x}_t} \big[\| \rho'(t) \mathbb{E}[\bm{\epsilon}-\mathbf{x}_0|\mathbf{x}_t] \|^2_2\big]\mathrm{d} t  = \int_0^1 \big(\rho'(t)\big)^2 \kappa(t) \mathrm{d} t,
$$
其中 $\kappa(t):=\mathbb{E}_{\mathbf{x}_t\sim p_t} \left[\big\|\mathbb{E}[\bm{\epsilon}-\mathbf{x}_0|\mathbf{x}_t]\big\|^2_2\right]$。因此**边际**意义下的最优调度 $\rho(t)$ 不必是线性的。当且仅当 $\kappa(t)$ 为常数时它为线性；一般地，Euler–Lagrange 条件
$$
(\kappa(t)\rho'(t))' = 0  \Rightarrow  \rho'(t)\propto \frac{1}{\kappa(t)}
$$
表明 oracle 最优调度对时间做自适应重参数化。直观上，$\kappa(t)$ 刻画标签 $(\bm{\epsilon}-\mathbf{x}_0)$ 中有多少可从 $\mathbf{x}_t\sim p_t$ 预测：oracle 流在 $\kappa(t)$ 大处减速（反映 oracle 速度期望范数高的区域），在 $\kappa(t)$ 小处加速。因此，尽管条件流使用线性调度 $(1-t,t)$，对应的边际（oracle）动力学一般是非线性的。

**视角 2：为何速度预测可视为对采样是自然的。**

*$\mathbf{x}$-、$\bm{\epsilon}$-与 $\mathbf{s}$-预测下 PF-ODE 的半线性形式。* 在干净、噪声与分数参数化下，漂移呈半线性形式（见 PF-ODE 的前三个等式）：
$$
\frac{\mathrm{d} \mathbf{x}(t)}{\mathrm{d} t}= \underbrace{L(t) \mathbf{x}(t)}_{\text{线性部分}} + \underbrace{N_{\bm{\phi}}(\mathbf{x}(t),t)}_{\text{非线性部分}},
\quad
N_{\bm{\phi}}\in\{\mathbf{x}_{\bm{\phi}}, \bm{\epsilon}_{\bm{\phi}}, \mathbf{s}_{\bm{\phi}}\}.
$$
当线性漂移 $L(t) \mathbf{x}(t)$ 在若干方向上驱动 $\mathbf{x}(t)$ 变化的速率与非线性的速率相差很大时，系统是**刚性的**，即漂移的 Jacobian（对 $\mathbf{x}$）
$$
J(\mathbf{x},t):=L(t)+\nabla_{\mathbf{x}}N_{\bm{\phi}}(\mathbf{x},t)
$$
的特征值实部量级相差很大（量级大对应更快方向）。例如，动力学可能同时包含 $\mathbf{x}(t)$ 的“快速线性”变化与“慢速非线性”变化。此时显式求解器必须取很小的时间步才能保持数值稳定。

为应对这种不平衡，高阶稳定求解器常采用**积分因子**，将线性项 $L(t) \mathbf{x}$ 解析处理，仅对较慢的非线性余项离散化，代价是代数与实现更复杂。求解器一章将专门讨论该主题。

*$\mathbf{v}$-预测下的 PF-ODE。* 在 $\mathbf{v}$-预测下，模型直接学习速度场并积分
$$
\frac{\mathrm{d} \mathbf{x}(t)}{\mathrm{d} t}
=\mathbf{v}_{\bm{\phi}}(\mathbf{x}(t),t)
\approx \mathbf{v}^*(\mathbf{x}(t),t).
$$
在这一表述中，显式线性项被吸收进单一学习场，动力学不再分成两部分。步长因此由学习场 $\mathbf{v}_{\bm{\phi}}(\mathbf{x},t)$ 随 $\mathbf{x}$ 与 $t$ 的平滑程度决定，而非由给定标量系数 $L(t)$ 的量级决定。换言之，潜在的快速线性漂移被折叠进一个一致的速度场，减少了时间尺度差异并简化了数值积分。

后文将用简单例子说明 $\mathbf{v}$-预测在采样过程中的结构简洁性。为得到与 DDIM（扩散建模中最广泛使用的快速采样器之一）相同的 PF-ODE 离散更新，在 $\bm{\epsilon}$-、$\mathbf{x}$-或 $\mathbf{s}$-参数化下，朴素 Euler 步仅**近似**线性项而非精确计算。因此这些参数化需要更高级的方法——**指数积分器**——来分离并精确计算线性项。相比之下，在 $\mathbf{v}$-预测下 PF-ODE 漂移中没有单独的线性项需要分离，朴素 Euler 更新自然与 DDIM 形式一致。类似地，二阶 DPM-Solver 与经典 Heun 方法一致：对 $\mathbf{v}$-预测是朴素 Heun，对 $\bm{\epsilon}$-、$\mathbf{x}$-或 $\mathbf{s}$-预测则是指数 Heun。详细讨论见各自小节。

我们指出，生成上的任何改进（例如在更少的模型求值下获得更高质量样本）都既依赖 $\mathbf{v}_{\bm{\phi}}$ 对 oracle 速度的逼近精度，也依赖采样算法（包括数值积分器、离散调度与步长控制）与它的交互效果。因此，采用 $\mathbf{v}$-参数化本身并不保证更好的采样表现。

**小结。** 尽管 $\mathbf{v}$-预测与典范线性调度结合具有若干理论优势（如恒定目标范数、无调度曲率），这些性质并不必然使其普遍更优。实践中，模型表现取决于网络结构、归一化方式、损失随时间加权、采样器与离散步数、引导强度、正则策略、数据尺度与总训练预算等诸多因素的交互。不同数据集与目标可能更适合其他参数化或调度，最优配置**最终**是经验问题，应通过验证与消融研究确定。

---

## 5.4 底层规律：Fokker–Planck 方程 {#sec:comparison-connection}

**图：通过连续性方程将变分、SDE 与 ODE 表述统一起来的视角，其中所有 $p_t(\mathbf{x})$ 在同一动力学下演化。** 速度场 $\mathbf{v}^*(\mathbf{x}, t) = f(t)\mathbf{x} - \frac{1}{2}g^2(t)\mathbf{s}^*(\mathbf{x}, t)$ 由分数函数 $\mathbf{s}^*(\mathbf{x}, t) := \nabla_\mathbf{x} \log p_t(\mathbf{x})$ 决定。系数 $f(t)$、$g(t)$、$\sigma_t$、$\alpha_t$ 为预先给定的时间依赖函数，$\gamma(t)$ 为可调的时间变化超参数。

本节说明扩散模型的三种主要视角——变分、基于分数与基于流——并非彼此独立的构造，而是源于同一条统一原理：在所选前向过程下支配密度演化的**连续性（Fokker–Planck）方程**。

首先回顾：基于离散核与 Bayes 法则的变分视角，与基于连续动力学的分数 SDE 视角，已在“启发式 FPE-SDE”等式中被统一。我们通过说明变分模型充当底层前向与逆向 SDE 的一致离散化来建立这一联系。具体地，由离散核逐步计算的边际密度，其演化方式与支配连续时间动力学的 Fokker–Planck 方程一致。这确认了两种视角在根本上是等价的。

接着我们连接基于流与基于分数的视角。在 5.4.1 节中说明：一个 ODE 流确定一条密度路径，其边际总可以由一族随机过程实现。这将确定性流与随机 SDE 纳入同一族。

综合这些结果，三种视角被统一在一个框架下（见统一框架图）。最后，我们在 5.5 节总结本章。

### 5.4.1 基于流的方法与 Score SDE 的联系 {#subsec:flow-score}

扩散模型的一个突出之处在于：不同的动力系统——确定性或随机——如何描出相同的概率分布演化。本节揭示第 4 章 ODE 流与 Score SDE 之间自然、优雅的联系。具体地，我们说明定义生成 ODE 的速度场可以转化为遵循同一 Fokker–Planck 动力学的随机对应物，从而在确定性插值与随机采样之间架起有原则的桥梁。这为我们提供了一族从 ODE 到 SDE 的连续模型，它们生成相同的数据分布路径。

考虑连续时间设定，扰动核为
$$
p_t(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \alpha_t \mathbf{x}_0, \sigma_t^2 I),
$$
其中 $\mathbf{x}_0 \sim p_{\mathrm{data}}$。该条件分布按常规定义边际密度路径 $p_t(\mathbf{x}_t) = \mathbb{E}_{\mathbf{x}_0 \sim p_{\mathrm{data}}}[p(\mathbf{x}_t|\mathbf{x}_0)]$，且 $p_T\approx p_{\mathrm{prior}}$。

为匹配该密度路径，考虑 ODE
$$
\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d} t} = \mathbf{v}_t(\mathbf{x}(t)), \quad t \in [0,T],
$$
其中 $\mathbf{v}_t(\mathbf{x}) =\mathbb{E}\left[\alpha_t'\mathbf{x}_0+\sigma_t'\bm{\epsilon}|\mathbf{x}\right]$ 为 oracle 速度（见边际 oracle 速度式；注意时间已按扩散模型惯例翻转）。从 $\mathbf{x}(T)\sim p_{\mathrm{prior}}$ 反向积分该 ODE 即得 $p_0$ 的样本。

尽管该 ODE 足以生成高质量样本，引入随机性可能提高样本多样性。这引出如下问题：**是否存在一个 SDE，其从 $p_{\mathrm{prior}}$ 出发的动力学与上述 ODE 产生相同的边际密度？** 该论断肯定：存在一族逆向时间 SDE，它们诱导的边际密度路径与相应 PF-ODE 相同。这些 SDE 诱导的密度满足该路径的同一 Fokker–Planck 方程，因此其单时刻边际与给定的插值 $\{p_t\}_{t\in[0,T]}$ 一致。

**命题（逆向时间 SDE 与插值生成相同边际）** 设 $\gamma(t) \geq 0$ 为任意时间依赖系数。考虑逆向时间 SDE
$$
\mathrm{d} \bar{\mathbf{x}}(t)  =  \Big[ \mathbf{v}^*(\bar{\mathbf{x}}(t), t)  -  \tfrac{1}{2} \gamma^2(t) \mathbf{s}^*(\bar{\mathbf{x}}(t), t) \Big] \mathrm{d} \bar{t}  +  \gamma(t)  \mathrm{d} \bar{\mathbf{w}}(t),
$$
从 $\bar{\mathbf{x}}(T) \sim p_{T}$ 向后演化到 $t = 0$。则过程 $\{\bar{\mathbf{x}}(t)\}_{t\in[0,T]}$ 与 ODE 密度路径诱导的给定边际 $\{p_t\}_{t \in [0,T]}$ 一致。这里 $\mathbf{s}(\mathbf{x}, t) := \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ 为分数函数，与速度场 $\mathbf{v}(\mathbf{x}, t)$ 的关系为
$$
\mathbf{v}^*(\mathbf{x}, t)  =  f(t) \mathbf{x}  -  \frac{1}{2} g^2(t) \mathbf{s}^*(\mathbf{x}, t),
\quad
\mathbf{s}^*(\mathbf{x}, t)  =  \frac{1}{\sigma_t} \frac{\alpha_t \mathbf{v}^*(\mathbf{x}, t) - \alpha_t' \mathbf{x}}{\alpha_t' \sigma_t - \alpha_t \sigma_t'}.
$$
（证明：逆向时间 Fokker–Planck 方程中二阶项相消，得到与 PF-ODE 相关的一阶（仅漂移）Fokker–Planck 方程，故逆向 SDE 与 ODE 诱导相同的边际密度路径。详见文献附录 A.2–A.3。）

超参数 $\gamma(t)$ 可任意选取，与 $\alpha_t$、$\sigma_t$ 无关，甚至在训练后也可改，因为它不改变速度 $\mathbf{v}(\mathbf{x}, t)$ 或分数 $\mathbf{s}(\mathbf{x}, t)$。例如：

- 令 $\gamma(t) = 0$ 即得上述 ODE。
- 当 $\gamma(t) = g(t)$ 时，该 SDE 变为逆向时间 SDE（因为 oracle 速度满足 $\mathbf{v}^*(\mathbf{x}, t) = f(t)\mathbf{x} - \frac{1}{2}g^2(t)\mathbf{s}^*(\mathbf{x}, t)$）。
- 其他 $\gamma(t)$ 也有研究；例如有工作选择 $\gamma(t)$ 以最小化 $p_{\mathrm{data}}$ 与从 $t=T$ 解该逆向 SDE 得到 $t=0$ 密度之间的 KL 间隙。

按照 Score SDE，训练好的速度场 $\mathbf{v}_{\bm{\phi}^*}(\mathbf{x}, t)$ 可通过上述关系转化为参数化分数 $\mathbf{s}_{\bm{\phi}^*}(\mathbf{x}, t)$。代入逆向 SDE 即得**经验逆向时间 SDE**，可从 $t=T$、$\bar{\mathbf{x}}(T) \sim p_{\mathrm{prior}}$ 数值积分采样。

该命题凸显扩散模型的显著灵活性：一旦**边际密度路径** $\{p_t\}_{t\in[0,T]}$ 固定，一整族动力学都可以复现它，包括 PF-ODE 与逆向时间 SDE
$$
\mathrm{d}\bar{\mathbf{x}}(t)
=\big[\mathbf{v}^*(\bar{\mathbf{x}},t)-\tfrac12\gamma^2(t)\mathbf{s}^*(\bar{\mathbf{x}},t)\big]\mathrm{d}\bar t
+\gamma(t)\mathrm{d}\bar{\mathbf{w}}(t),\qquad \gamma(t)\ge0.
$$
所有这些动力学满足同一逆向时间 Fokker–Planck 方程，因而给出相同的边际演化。函数 $\gamma(t)$ 连续调节随机性水平而不影响单时刻分布，揭示了确定性基于流的 ODE 与其随机 SDE 对应物之间的深刻联系，如统一框架图所示。

---

## 5.5 小结 {#subsec:takeaway-dm-gaussianfm}

本章作为我们理论探索的拱心石，将变分、基于分数与基于流的视角综合进一个统一、连贯的框架。我们表明，这三种看似不同的进路并非**仅仅**并行，而是深刻、根本地相互联系。

我们的统一基于两条核心洞见。第一，我们指出了所有框架共有的“秘密武器”：一种**条件化技巧**，将难以处理的边际训练目标转化为可处理的条件目标，从而实现稳定、高效的学习。第二，我们确立了 **Fokker-Planck 方程**是支配概率密度演化的普适规律。三种视角各自构造的生成过程都服从这一基本动力学。

此外，我们证明了各种模型参数化——即噪声、干净数据、分数或速度预测——均可互换。这表明预测目标的选择是实现与稳定性的问题，而非根本的建模差异。最终的结论是：现代扩散方法尽管来源多样，都实例化了同一条核心原则：学习一个时变向量场，将简单先验输运到数据分布。

在这一统一且有原则的基础牢固建立之后，我们得以从基础理论转向扩散模型的实际应用与加速。生成等价于求解微分方程这一中心洞见，为控制与优化提供了有力平台。本专著后续部分将利用这一统一理解应对关键实际挑战：

1. **Part C** 将聚焦于在推理阶段改进采样过程。我们将探讨如何引导生成轨迹以实现可控生成，并研究先进数值求解器以大幅加速缓慢、迭代的采样过程。
2. **Part D** 将超越迭代求解器，直接学习快速生成器。我们将考察仅用一步或少数几步即可产生高质量样本的方法，既包括从教师模型蒸馏，也包括从零训练。

在统一了扩散模型的*是什么*与*为什么*之后，我们接下来将把注意力转向*如何*——激动人心且实用的前沿。
