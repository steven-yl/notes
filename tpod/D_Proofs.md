# 附录：补充材料与证明
<!-- \label{app:proof} -->

## 变分视角
<!-- \label{app-sec:var-proof} -->

### 定理~\ref{thm:equiv-marginal-kl}：边际 KL 与条件 KL 最小化的等价性
<!-- \label{subsec:proof-equiv-marginal-kl} -->

**证明.** **\Cref{eq:kl-matching} 的推导。**

我们从展开右侧的期望开始：
$$
\begin{align*}
    &\mathbb{E}_{p(\rvx_0, \rvx_i)}\left[
    \mathcal{D}_{\text{KL}} \big(p(\rvx_{i-1}|\rvx_i, \rvx_0) \| p_\phi(\rvx_{i-1}|\rvx_i)\big)
    \right]
    \\
    =& \int \int p(\rvx_0, \rvx_i)   
    \mathcal{D}_{\text{KL}} \big(p(\rvx_{i-1}|\rvx_i, \rvx_0) \| p_\phi(\rvx_{i-1}|\rvx_i)\big) 
    \diff\rvx_0 \diff\rvx_i.
\end{align*}
$$
由 KL 散度的定义，
$$
\begin{align*}
    &\mathcal{D}_{\text{KL}} \big(p(\rvx_{i-1}|\rvx_i, \rvx_0) \| p_\phi(\rvx_{i-1}|\rvx_i)\big)
    = \int p(\rvx_{i-1}|\rvx_i, \rvx_0) 
    \log \frac{p(\rvx_{i-1}|\rvx_i, \rvx_0)}{p_\phi(\rvx_{i-1}|\rvx_i)} 
    \diff\rvx_{i-1}.
\end{align*}
$$
将其代入期望，得到
$$
\begin{align*}
    &\int \int \int p(\rvx_0, \rvx_i)  p(\rvx_{i-1}|\rvx_i, \rvx_0) 
    \log \frac{p(\rvx_{i-1}|\rvx_i, \rvx_0)}{p_\phi(\rvx_{i-1}|\rvx_i)} 
    \diff\rvx_{i-1} \diff\rvx_0 \diff\rvx_i.
\end{align*}
$$
由概率的链式法则，
$$
\begin{align*}
    p(\rvx_0, \rvx_i) 
    &= p(\rvx_i)   p(\rvx_0|\rvx_i),
\end{align*}
$$
我们将积分改写为
$$
\begin{align*}
    &\int p(\rvx_i) \int p(\rvx_0|\rvx_i) \int p(\rvx_{i-1}|\rvx_i, \rvx_0)
    \log \frac{p(\rvx_{i-1}|\rvx_i, \rvx_0)}{p_\phi(\rvx_{i-1}|\rvx_i)} 
    \diff\rvx_{i-1} \diff\rvx_0 \diff\rvx_i.
\end{align*}
$$
由此可将期望写成嵌套形式：
$$
\begin{align*}
    &\mathbb{E}_{p(\rvx_i)} \left[
    \mathbb{E}_{p(\rvx_0|\rvx_i)} \left[
    \mathbb{E}_{p(\rvx_{i-1}|\rvx_i, \rvx_0)} \left[
    \log \frac{p(\rvx_{i-1}|\rvx_i, \rvx_0)}{p_\phi(\rvx_{i-1}|\rvx_i)}
    \right]
    \right]
    \right].
\end{align*}
$$

接下来，我们对对数进行分解：
$$
\begin{align*}
    \log \frac{p(\rvx_{i-1}|\rvx_i, \rvx_0)}{p_\phi(\rvx_{i-1}|\rvx_i)}
    =\log \frac{p(\rvx_{i-1}|\rvx_i, \rvx_0)}{p(\rvx_{i-1}|\rvx_i)} 
    + \log \frac{p(\rvx_{i-1}|\rvx_i)}{p_\phi(\rvx_{i-1}|\rvx_i)}.
\end{align*}
$$
将其代回期望得到两项：
$$
\begin{align*}
    &\mathbb{E}_{p(\rvx_i)} \left[
    \mathbb{E}_{p(\rvx_0|\rvx_i)} \left[
    \mathbb{E}_{p(\rvx_{i-1}|\rvx_i, \rvx_0)} \left[
    \log \frac{p(\rvx_{i-1}|\rvx_i, \rvx_0)}{p(\rvx_{i-1}|\rvx_i)}
    \right]
    \right]
    \right]
    \\
    &+
    \mathbb{E}_{p(\rvx_i)} \left[
    \mathbb{E}_{p(\rvx_0|\rvx_i)} \left[
    \mathbb{E}_{p(\rvx_{i-1}|\rvx_i, \rvx_0)} \left[
    \log \frac{p(\rvx_{i-1}|\rvx_i)}{p_\phi(\rvx_{i-1}|\rvx_i)}
    \right]
    \right]
    \right].
\end{align*}
$$
由于第二项对数不依赖于 $\rvx_0$，由全概率公式
$$
\begin{align*}
    \mathbb{E}_{p(\rvx_0|\rvx_i)} \left[
    \mathbb{E}_{p(\rvx_{i-1}|\rvx_i, \rvx_0)} \left[
    \log \frac{p(\rvx_{i-1}|\rvx_i)}{p_\phi(\rvx_{i-1}|\rvx_i)}
    \right]
    \right]
    = \mathbb{E}_{p(\rvx_{i-1}|\rvx_i)} \left[
    \log \frac{p(\rvx_{i-1}|\rvx_i)}{p_\phi(\rvx_{i-1}|\rvx_i)}
    \right].
\end{align*}
$$
类似地，第一项是 KL 散度
$$
\begin{align*}
    \mathbb{E}_{p(\rvx_0|\rvx_i)} \left[
    \mathcal{D}_{\text{KL}} \big(p(\rvx_{i-1}|\rvx_i, \rvx_0)  \|  p(\rvx_{i-1}|\rvx_i)\big)
    \right].
\end{align*}
$$
综合以上，得到分解：
$$
\begin{align*}
    &\mathbb{E}_{p(\rvx_0, \rvx_i)} \left[
    \mathcal{D}_{\text{KL}} \big(p(\rvx_{i-1}|\rvx_i, \rvx_0)  \|  p_\phi(\rvx_{i-1}|\rvx_i)\big)
    \right]
    \\
    =& \mathbb{E}_{p(\rvx_i)} \left[
    \mathbb{E}_{p(\rvx_0|\rvx_i)} \left[
    \mathcal{D}_{\text{KL}} \big(p(\rvx_{i-1}|\rvx_i, \rvx_0)  \|  p(\rvx_{i-1}|\rvx_i)\big)
    \right]
    \right]
    \\
    &+
    \mathbb{E}_{p(\rvx_i)} \left[
    \mathcal{D}_{\text{KL}} \big(p(\rvx_{i-1}|\rvx_i)  \|  p_\phi(\rvx_{i-1}|\rvx_i)\big)
    \right].
\end{align*}
$$

**最优性证明。** 欲证：
$$
p^*(\mathbf{x}_{i-1}|\mathbf{x}_i) = p(\mathbf{x}_{i-1}|\mathbf{x}_i) = \mathbb{E}_{p(\mathbf{x}|\mathbf{x}_i)} \left[ p(\mathbf{x}_{i-1}|\mathbf{x}_i, \mathbf{x}) \right], \quad \mathbf{x}_i \sim p_i.
$$
第一个等式成立是因为当 $p^* = p$ 时 KL 散度 $\mathcal{D}_{\mathrm{KL}}(p \| p_{\bm{\phi}})$ 达到最小（假定参数化足够表达）。第二个等式直接由全概率公式得到。
□

---

### 定理~\ref{thm:ddpm-elbo}：扩散模型的 ELBO
<!-- \label{app:vlb-proof} -->

**证明.** 为记号简便，记 $\rvx_{0:L} := (\rvx_0, \rvx_1, \ldots, \rvx_L)$。

**第一步：应用 Jensen 不等式。**  
边际对数似然为：
$$
\log p_{\bm{\phi}}(\rvx) = \log \int p_{\bm{\phi}}(\rvx, \rvx_{0:L})   \diff\rvx_0 \cdots \diff\rvx_L,
$$
其中联合分布为：
$$
p_{\bm{\phi}}(\rvx, \rvx_{0:L}) = p_{\text{prior}}(\rvx_L) \prod_{i=1}^L p_{\bm{\phi}}(\rvx_{i-1} | \rvx_i) \cdot p_{\bm{\phi}}(\rvx | \rvx_0).
$$

我们引入变分分布 $p(\rvx_{0:L} | \rvx)$ 并改写：
$$
\log p_{\bm{\phi}}(\rvx) = \log \int p(\rvx_{0:L} | \rvx) \frac{p_{\bm{\phi}}(\rvx, \rvx_{0:L})}{p(\rvx_{0:L} | \rvx)}   \diff\rvx_0 \cdots \diff\rvx_L.
$$
应用 Jensen 不等式（$\log \mathbb{E}[Z] \geq \mathbb{E}[\log Z]$），得到 ELBO：
$$
\log p_{\bm{\phi}}(\rvx) \geq \mathbb{E}_{p(\rvx_{0:L} | \rvx)} \left[ \log \frac{p_{\bm{\phi}}(\rvx, \rvx_{0:L})}{p(\rvx_{0:L} | \rvx)} \right] =: \mathcal{L}_{\text{ELBO}},
$$
因此
$$
-\log p_{\bm{\phi}}(\rvx) \leq -\mathcal{L}_{\text{ELBO}}.
$$

**第二步：展开 ELBO。**  
设变分分布因子分解为：
$$
p(\rvx_{0:L} | \rvx) = p(\rvx_L | \rvx) \prod_{i=1}^L p(\rvx_{i-1} | \rvx_i, \rvx).
$$
将联合分布与变分分布代入 ELBO：
$$
\begin{align*}
    \mathcal{L}_{\text{ELBO}} = \mathbb{E}_{p(\rvx_{0:L} | \rvx)} \Big[ 
    & \log p_{\text{prior}}(\rvx_L) 
    + \sum_{i=1}^L \log p_{\bm{\phi}}(\rvx_{i-1} | \rvx_i)
    + \log p_{\bm{\phi}}(\rvx | \rvx_0) \\
    & - \log p(\rvx_L | \rvx) 
    - \sum_{i=1}^L \log p(\rvx_{i-1} | \rvx_i, \rvx)
    \Big].
\end{align*}
$$

我们按依赖关系分组并做边际化，计算负 ELBO：
$$
\begin{align*}
    -\mathcal{L}_{\text{ELBO}} = 
    &\mathbb{E}_{p(\rvx_0 | \rvx)} \left[ -\log p_{\bm{\phi}}(\rvx | \rvx_0) \right]
    + \mathbb{E}_{p(\rvx_L | \rvx)} \left[ \log \frac{p(\rvx_L | \rvx)}{p_{\text{prior}}(\rvx_L)} \right] \\
    &+ \sum_{i=1}^L \mathbb{E}_{p(\rvx_i | \rvx)} \left[
        \mathbb{E}_{p(\rvx_{i-1} | \rvx_i, \rvx)} \left[
            \log \frac{p(\rvx_{i-1} | \rvx_i, \rvx)}{p_{\bm{\phi}}(\rvx_{i-1} | \rvx_i)}
        \right]
    \right].
\end{align*}
$$
为说明最后一项，我们使用因子分解
$$
p(\rvx_i, \rvx_{i-1} | \rvx) = p(\rvx_i | \rvx) \cdot p(\rvx_{i-1} | \rvx_i, \rvx),
$$
得到：
$$
\begin{align*}
&\mathbb{E}_{p(\rvx_{0:L} | \rvx)} \left[
    \log \frac{p(\rvx_{i-1} | \rvx_i, \rvx)}{p_{\bm{\phi}}(\rvx_{i-1} | \rvx_i)}
\right]
\\ = &\int p(\rvx_i | \rvx) \left[ \int p(\rvx_{i-1} | \rvx_i, \rvx)
\log \frac{p(\rvx_{i-1} | \rvx_i, \rvx)}{p_{\bm{\phi}}(\rvx_{i-1} | \rvx_i)} \diff\rvx_{i-1} \right] \diff\rvx_i\\ = & \mathbb{E}_{p(\rvx_i | \rvx)} \left[
    \mathbb{E}_{p(\rvx_{i-1} | \rvx_i, \rvx)} \left[
        \log \frac{p(\rvx_{i-1} | \rvx_i, \rvx)}{p_{\bm{\phi}}(\rvx_{i-1} | \rvx_i)}
    \right]
\right].
\end{align*}
$$

我们将 $-\mathcal{L}_{\text{ELBO}}$ 中的三项分别记为：
$$
\mathcal{L}_{\text{recon.}}, \quad
\mathcal{L}_{\text{prior}}, \quad
\mathcal{L}_{\text{diffusion}},
$$
依次对应重构损失、先验 KL 与逐步扩散 KL。推导完成。
□

---

## 基于分数的视角
<!-- \label{app-sec:score-proof} -->

### 命题~\ref{sm-trace}：通过分部积分的可处理分数匹配
<!-- \label{app-sec:sm-trace} -->

**证明.**

**展开 $\mathcal{L}_{\text{SM}}(\bm{\phi})$。**  
在期望内展开平方差：
$$
\begin{align*}
        \mathcal{L}_{\text{SM}}(\bm{\phi}) &= \frac{1}{2}\mathbb{E}_{\rvx \sim p_{\text{data}}(\rvx)} \left[ \norm{\rvs_{\bm{\phi}}(\rvx)}_2^2 - 2 \langle \rvs_{\bm{\phi}}(\rvx), \rvs(\rvx) \rangle + \norm{\rvs(\rvx)}_2^2 \right] \\
        &= \frac{1}{2}\mathbb{E}_{\rvx \sim p_{\text{data}}(\rvx)} \left[ \norm{\rvs_{\bm{\phi}}(\rvx)}_2^2 \right] - \mathbb{E}_{\rvx \sim p_{\text{data}}(\rvx)} \left[ \langle \rvs_{\bm{\phi}}(\rvx), \rvs(\rvx) \rangle \right] \\
        &\quad + \frac{1}{2}\mathbb{E}_{\rvx \sim p_{\text{data}}(\rvx)} \left[ \norm{\rvs(\rvx)}_2^2 \right].
        \end{align*}
$$
我们集中处理交叉项：
$$
\mathbb{E}_{\rvx \sim p_{\text{data}}(\rvx)} \left[ \langle \rvs_{\bm{\phi}}(\rvx), \rvs(\rvx) \rangle \right].
$$
利用
$$
\nabla_{\rvx} \log p_{\text{data}}(\rvx) = \frac{\nabla_{\rvx}p_{\text{data}}(\rvx)}{p_{\text{data}}(\rvx)},
$$
并设 $p_{\text{data}}(\rvx)$ 在其支撑上非零，交叉项变为：
$$
\begin{align*}
            \mathbb{E}_{\rvx \sim p_{\text{data}}(\rvx)} \left[ \langle \rvs_{\bm{\phi}}(\rvx), \rvs(\rvx) \rangle \right] 
             &= \int \rvs_{\bm{\phi}}(\rvx)^\top  \nabla_{\rvx} \log p_{\text{data}}(\rvx) p_{\text{data}}(\rvx) \diff \rvx \\
             &= \int \rvs_{\bm{\phi}}(\rvx)^\top  \nabla_{\rvx}p_{\text{data}}(\rvx) \diff \rvx \\
             &= \sum_{i=1}^{D} \int  s_{\bm{\phi}}^{(i)}(\rvx) \partial_{x_i}p_{\text{data}}(\rvx) \diff \rvx,
        \end{align*}
$$
其中 $s_{\bm{\phi}}^{(i)}(\rvx)$ 是分数函数
$$
\rvs_{\bm{\phi}} = \left( s_{\bm{\phi}}^{(1)}, s_{\bm{\phi}}^{(2)}, \dots, s_{\bm{\phi}}^{(D)} \right).
$$
的第 $i$ 个分量。

**分部积分。**  
我们使用以下分部积分公式~\citep{evans10}（由标准微积分推导）：

**引理.** 设 $u, v$ 为半径 $R>0$ 的球 $\mathbb{B}(\bm{0}, R) \subset \mathbb{R}^D$ 上的可微实值函数。则对 $i = 1, \ldots, D$ 有：
$$
\int_{\mathbb{B}(\bm{0}, R)} u   \partial_{x_i} v   \diff \rvx = -\int_{\mathbb{B}(\bm{0}, R)} v   \partial_{x_i} u   \diff \rvx + \int_{\partial \mathbb{B}(\bm{0}, R)} u v \nu_i   \diff S,
$$
其中 $\nu = (\nu_1, \ldots, \nu_D)$ 为边界 $\partial \mathbb{B}(\bm{0}, R)$（半径 $R>0$ 的球面）的单位外法向，$\diff S$ 为 $\partial \mathbb{B}(\bm{0}, R)$ 上的面积元。

对所有 $i = 1, \dots, D$，令 $u(\rvx) := s_{\bm{\phi}}^{(i)}(\rvx)$，$v(\rvx) = p_{\text{data}}(\rvx)$ 并应用该公式，并假设
$$
|u(\rvx) v(\rvx)| \to 0 \quad \text{as } R \to \infty.
$$
对 $i = 1, \dots, D$ 求和得：
$$
\begin{align*}
            \mathbb{E}_{\rvx \sim p_{\text{data}}(\rvx)} \left[ \langle \rvs_{\bm{\phi}}(\rvx), \rvs(\rvx) \rangle \right] 
             &= - \sum_{i=1}^{D} \int  \partial_{x_i}s_{\bm{\phi}}^{(i)}(\rvx) p_{\text{data}}(\rvx) \diff \rvx \\
             &= - \mathbb{E}_{\rvx \sim p_{\text{data}}(\rvx)} \left[\Tr\left(\nabla_{\rvx}\rvs_{\bm{\phi}}(\rvx)\right) \right].
        \end{align*}
$$
综合以上结果：
$$
\begin{align*}
        \mathcal{L}_{\text{SM}}(\bm{\phi}) &= \underbrace{\mathbb{E}_{\rvx \sim p_{\text{data}}(\rvx)} \left[\Tr\left(\nabla_{\rvx}\rvs_{\bm{\phi}}(\rvx)\right) + \frac{1}{2}  \norm{\rvs_{\bm{\phi}}(\rvx)}_2^2 \right]}_{\widetilde{\mathcal{L}}_{\text{SM}}(\bm{\phi})} \\
        &\quad + \underbrace{\frac{1}{2}\mathbb{E}_{\rvx \sim p_{\text{data}}(\rvx)} \left[ \norm{\rvs(\rvx)}_2^2 \right]}_{=:C},
        \end{align*}
$$
其中 $C$ 仅依赖于分布 $p_{\text{data}}$，证明完毕。
□

### 定理~\ref{thm:sm-dsm}：SM 与 DSM 最小化的等价性

**证明.** 展开 $\mathcal{L}_{\text{SM}}(\bm{\phi}; \sigma)$ 与 $\mathcal{L}_{\text{DSM}}(\bm{\phi}; \sigma)$，有：
$$
\begin{align*}
 &\mathcal{L}_{\text{SM}}(\bm{\phi}; \sigma) = \frac{1}{2} \mathbb{E}_{\tilde{\rvx} \sim p_\sigma(\tilde{\rvx})} 
 \begin{aligned}[t]
         \Big[ 
    \norm{\rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)}_2^2 
    &- 2 \rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)^\top  \nabla_{\tilde{\rvx}} \log p_\sigma(\tilde{\rvx}) \\
    &+ \norm{\nabla_{\tilde{\rvx}} \log p_\sigma(\tilde{\rvx})}_2^2 
    \Big], 
 \end{aligned}
\\
&\mathcal{L}_{\text{DSM}}(\bm{\phi}; \sigma) = \frac{1}{2} \mathbb{E}_{ p_{\text{data}}(\rvx)p_{\sigma}(\tilde{\rvx} | \rvx)}
 \begin{aligned}[t]
         \Big[
    \norm{\rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)}_2^2 
    &- 2 \rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)^\top \nabla_{\tilde{\rvx}} \log p_{\sigma}(\tilde{\rvx} | \rvx) \\
    &+ \norm{\nabla_{\tilde{\rvx}} \log p_{\sigma}(\tilde{\rvx} | \rvx)}_2^2 
    \Big].
 \end{aligned}
\end{align*}
$$
两式相减得：
$$
\begin{align*}
 &\mathcal{L}_{\text{SM}}(\bm{\phi}; \sigma) -\mathcal{L}_{\text{DSM}}(\bm{\phi}; \sigma) 
 \\= \frac{1}{2} &\Bigg( \mathbb{E}_{\tilde{\rvx} \sim p_\sigma(\tilde{\rvx})} \norm{\rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)}_2^2 - \mathbb{E}_{ p_{\text{data}}(\rvx)p_{\sigma}(\tilde{\rvx} | \rvx)} \norm{\rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)}_2^2\Bigg)
 \\ \quad - &  
 \begin{aligned}[t]
 \Bigg(
      \mathbb{E}_{\tilde{\rvx} \sim p_\sigma(\tilde{\rvx})}  &\left[ \rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)^\top  \nabla_{\tilde{\rvx}} \log p_\sigma(\tilde{\rvx}) \right] \\ 
      &- \mathbb{E}_{ p_{\text{data}}(\rvx)p_{\sigma}(\tilde{\rvx} | \rvx)}\left[ \rvs_{\bm{\phi}}(\tilde{\rvx};\rvx)^\top  \nabla_{\tilde{\rvx}} \log p_\sigma(\tilde{\rvx}| \rvx)\right]
      \Bigg)
 \end{aligned}
 \\ \quad +  \frac{1}{2}&\Bigg(\mathbb{E}_{\tilde{\rvx} \sim p_\sigma(\tilde{\rvx})}  \norm{\nabla_{\tilde{\rvx}} \log p_\sigma(\tilde{\rvx})}_2^2  -  \mathbb{E}_{ p_{\text{data}}(\rvx)p_{\sigma}(\tilde{\rvx} | \rvx)} \norm{\nabla_{\tilde{\rvx}} \log p_{\sigma}(\tilde{\rvx} | \rvx)}_2^2   \Bigg).
\end{align*}
$$
下面逐项处理三项。对第一项，由于 $p_\sigma(\tilde{\rvx}) = \int p_{\sigma}(\tilde{\rvx} | \rvx) p_{\text{data}}(\rvx) \diff\rvx$，可改写为：
$$
\begin{align*}
    \mathbb{E}_{\tilde{\rvx} \sim p_\sigma(\tilde{\rvx})} \norm{\rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)}_2^2 
    &= \int \Big( \int p_{\sigma}(\tilde{\rvx} | \rvx) p_{\text{data}}(\rvx)  \diff\rvx \Big) \norm{\rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)}_2^2 \diff \tilde{\rvx} 
    \\ &= \int p_{\text{data}}(\rvx) \int p_{\sigma}(\tilde{\rvx} | \rvx) \norm{\rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)}_2^2 \diff \tilde{\rvx} \diff\rvx
    \\ &= \mathbb{E}_{ p_{\text{data}}(\rvx)p_{\sigma}(\tilde{\rvx} | \rvx)} \norm{\rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)}_2^2.
\end{align*}
$$
故第一项为零。对第二项：
$$
\begin{aligned}
          &\mathbb{E}_{\tilde{\rvx} \sim p_\sigma(\tilde{\rvx})}\left[ \rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)^\top  \nabla_{\tilde{\rvx}} \log p_\sigma(\tilde{\rvx}) \right] 
      \\=&\int  p_\sigma(\tilde{\rvx}) \rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)^\top  \frac{\nabla_{\tilde{\rvx}} p_\sigma(\tilde{\rvx})}{p_\sigma(\tilde{\rvx})}  \diff\tilde\rvx
        \\=&\int   \rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)^\top  \nabla_{\tilde{\rvx}} \int p_{\sigma}(\tilde{\rvx} | \rvx) p_{\text{data}}(\rvx) \diff\rvx \diff\tilde\rvx
    \\=&\int \int  \rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)^\top    \nabla_{\tilde{\rvx}}p_{\sigma}(\tilde{\rvx} | \rvx) p_{\text{data}}(\rvx) \diff\tilde\rvx \diff\rvx 
      \\=  &\mathbb{E}_{ p_{\text{data}}(\rvx)p_{\sigma}(\tilde{\rvx} | \rvx)}\left[ \rvs_{\bm{\phi}}(\tilde{\rvx}; \sigma)^\top  \nabla_{\tilde{\rvx}} \log p_\sigma(\tilde{\rvx}| \rvx)\right].
\end{aligned}
$$
因此第二项也为零。对第三项，注意到
$$
C:=\frac{1}{2}\Bigg(\mathbb{E}_{\tilde{\rvx} \sim p_\sigma(\tilde{\rvx})}  \norm{\nabla_{\tilde{\rvx}} \log p_\sigma(\tilde{\rvx})}_2^2  -  \mathbb{E}_{ p_{\text{data}}(\rvx)p_{\sigma}(\tilde{\rvx} | \rvx)} \norm{\nabla_{\tilde{\rvx}} \log p_{\sigma}(\tilde{\rvx} | \rvx)}_2^2   \Bigg)
$$
仅依赖于 $p_{\text{data}}(\rvx)$ 与 $p_{\sigma}(\tilde{\rvx}|\rvx)$，故关于 $\bm{\phi}$ 为常数。
□

### 引理~\ref{tweedie}：Tweedie 公式
<!-- \label{app-sec:tweedie} -->

我们先给出 Tweedie 公式的更一般形式，考虑时间依赖的高斯扰动，并在下面给出证明。

**带时间依赖参数的 Tweedie 恒等式。**  
设 $\rvx_t \sim \mathcal{N}\big(\cdot; \alpha_t\rvx_0, \sigma_t^2\rmI\big)$ 为高斯随机向量。则 Tweedie 公式给出
$$
\alpha_t \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0|\rvx_t)}[\mathbf{x}_0|\rvx_t] = \rvx_t + \sigma_t^2 \nabla_{\rvx_t} \log p_t(\rvx_t),
$$
其中期望对给定观测 $\rvx_t$ 下 $\mathbf{x}_0$ 的后验分布 $p(\mathbf{x}_0|\rvx_t)$ 求取，$p_t(\rvx_t)$ 为 $\rvx_t$ 的边际密度。

**证明.**

**边际密度及其分数。**  
我们回顾 $\rvx_t$ 的边际密度为
$$
\begin{align*}
p_t(\rvx_t) = \int p_t(\rvx_t|\rvx_0)  p_0(\rvx_0)  \diff\rvx_0.
\end{align*}
$$
现计算分数函数：
$$
\begin{align*}
\nabla_{\rvx_t} \log p_t(\rvx_t) 
= \frac{\nabla_{\rvx_t} p_t(\rvx_t)}{p_t(\rvx_t)} = \frac{1}{p_t(\rvx_t)} \int \nabla_{\rvx_t} p_t(\rvx_t|\rvx_0)  p_0(\rvx_0)  \diff\rvx_0.
\end{align*}
$$
因此需要计算条件密度的梯度。

**条件密度的梯度与重排。**  
条件高斯密度的梯度为：
$$
\begin{align*}
\nabla_{\rvx_t} p_t(\rvx_t|\rvx_0) 
= -p_t(\rvx_t|\rvx_0) \cdot \sigma_t^{-2} (\rvx_t - \alpha_t \rvx_0).
\end{align*}
$$
代入前式得：
$$
\begin{align*}
\nabla_{\rvx_t} p_t(\rvx_t) 
&= \int \nabla_{\rvx_t} p_t(\rvx_t|\rvx_0)  p_0(\rvx_0)  \diff\rvx_0 \\
&= -\sigma_t^{-2} \int (\rvx_t - \alpha_t \rvx_0)  p_t(\rvx_t|\rvx_0)  p_0(\rvx_0)  \diff\rvx_0 \\
&= -\sigma_t^{-2} \int (\rvx_t - \alpha_t \rvx_0)  p(\rvx_0|\rvx_t)  p_t(\rvx_t)  \diff\rvx_0 \\
&= -p_t(\rvx_t)  \sigma_t^{-2} \left(\rvx_t - \alpha_t  \mathbb{E}_{p(\rvx_0|\rvx_t)}[\rvx_0\vert \rvx_t] \right).
\end{align*}
$$
两边除以 $p_t(\rvx_t)$ 得：
$$
\begin{align*}
\nabla_{\rvx_t} \log p_t(\rvx_t) 
= -\sigma_t^{-2} \left(\rvx_t - \alpha_t  \mathbb{E}_{p(\rvx_0|\rvx_t)}[\rvx_0\vert \rvx_t] \right).
\end{align*}
$$
重排即得：
$$
\begin{align*}
\rvx_t + \sigma_t^{2} \nabla_{\rvx_t} \log p_t(\rvx_t) 
= \alpha_t  \mathbb{E}_{p(\rvx_0|\rvx_t)}[\rvx_0\vert \rvx_t].
\end{align*}
$$
推导完成。

### Stein 恒等式与代理 SURE 目标
<!-- \label{app-sec:stein-identity} -->

**Stein 恒等式。** Stein 恒等式是一种分部积分技术，将未知密度下的期望转化为可观测函数及其导数的期望，从而消去配分函数，得到无偏、可处理的目标与检验，而无需显式计算未知密度或配分函数。我们从最简单的一维情形出发，再推广到证明 SURE 代理损失所需的形式。

**一维标准正态情形。**  
若 $z \sim \mathcal{N}(0,1)$ 且 $f$ 具有适当的衰减性，则 Stein 恒等式为：
\[
\E[f'(z)]  =  \E[Z f(z)] .
\]
记 $\phi(z) := \tfrac{1}{\sqrt{2\pi}}  e^{-z^2/2}$ 为一维标准正态密度。证明由分部积分及 $\phi'(z) = -z\phi(z)$ 与边界项消失得到。精确地，有
\[
\E[f'(Z)] 
= \int_{-\infty}^{\infty} f'(z)  \phi(z)  \diff z.
\]
令 $u=f(z)$、$\diff v=\phi'(z)  \diff z$ 做分部积分得
\[
\int f'(z)  \phi(z)  \diff z
= \Big[ f(z)  \phi(z) \Big]_{-\infty}^{\infty}
- \int f(z)  \phi'(z)  \diff z.
\]
由 $\phi'(z) = -z\phi(z)$ 及 $|z|\to\infty$ 时 $f(z)\phi(z)\to 0$（衰减条件），边界项为零，故
\[
\E[f'(z)]  =  \int f(z)  z  \phi(z)  \diff z  =  \E[z f(z)].
\]
一维 Stein 恒等式得证。

**多维标准正态情形。**  
若 $\rvz \sim \mathcal{N}(\mathbf{0},\rmI_D)$ 且 $g:\R^D \to \R$，则 Stein 恒等式为
\[
\E[\nabla g(\rvz)]  =  \E[\rvz g(\rvz)].
\]
等价地，对 $\mathbf{u}:\R^D \to \R^D$，
\begin{align}\label{eq:stein-multi}
    \E[ \nabla_{\tilde{\rvx}} \cdot\mathbf{u}(\rvz)]  =  \E[\rvz^\top \mathbf{u}(\rvz)] .
\end{align}

**SURE 所需恒等式。**  
令 $\tilde{\rvx}=\rvx+\sigma\rvz$，其中 $\rvz \sim \mathcal{N}(\mathbf{0},\rmI_D)$，对任意满足适当正则性的向量函数 $\mathbf{a}$，
\begin{align}\label{eq:stein-for-sure}
    \E\big[(\tilde{\rvx}-\rvx)^\top \mathbf{a}(\tilde{\rvx})  \big|  \rvx\big]
 =  \sigma^2 \E\big[ \nabla_{\tilde{\rvx}} \cdot\mathbf{a}(\tilde{\rvx})  \big|  \rvx\big]. 
\end{align}
此式由 \Cref{eq:stein-multi} 及链式法则得到。

**由条件 MSE 导出 SURE。**  
设 $\rmD:\R^D \to \R^D$ 为去噪器，定义
\[
R(\rmD;\rvx):=\E \left[\|\rmD(\tilde{\rvx})-\rvx\|_2^2| \rvx\right].
\]
在 $\tilde{\rvx}$ 处展开：
\[
\begin{aligned}
&~R(\rmD;\rvx) \\
=&~\E \left[\|\rmD(\tilde{\rvx})-\tilde{\rvx}\|^2  \big|  \rvx\right]
+2 \E \left[(\rmD(\tilde{\rvx})-\tilde{\rvx})^\top(\tilde{\rvx}-\rvx)  \big|  \rvx\right]
+\E \left[\|\tilde{\rvx}-\rvx\|^2  \big|  \rvx\right] \\
=&~\E \left[\|\rmD(\tilde{\rvx})-\tilde{\rvx}\|^2  \big|  \rvx\right]
+2\Big(\underbrace{\E[(\tilde{\rvx}-\rvx)^\top \rmD(\tilde{\rvx})|\rvx]}_{\substack{\sigma^2 \E[ \nabla_{\tilde{\rvx}} \cdot\rmD(\tilde{\rvx})|\rvx] \\  \text{ by \Cref{eq:stein-for-sure}}
}}
-\underbrace{\E[(\tilde{\rvx}-\rvx)^\top \tilde{\rvx}|\rvx]}_{ \sigma^2 D}\Big)
\\&\qquad\qquad\qquad+\underbrace{\E[\|\tilde{\rvx}-\rvx\|^2|\rvx]}_{ \sigma^2 D} \\
=&~\E \left[\|\rmD(\tilde{\rvx})-\tilde{\rvx}\|^2
+2\sigma^2  \nabla_{\tilde{\rvx}} \cdot\rmD(\tilde{\rvx})
-D\sigma^2 | \rvx\right].
\end{aligned}
\]
因此*可观测*的代理
\[ 
\mathrm{SURE}(\rmD;\tilde{\rvx})
:= \|\rmD(\tilde{\rvx})-\tilde{\rvx}\|_2^2
+2\sigma^2  \nabla_{\tilde{\rvx}} \cdot\rmD(\tilde{\rvx})
-D\sigma^2 
\]
满足 $\E \left[\mathrm{SURE}(\rmD;\tilde{\rvx}) \big| \rvx\right]=R(\rmD;\rvx)$。
在期望或经验意义下最小化 SURE，等价于仅用带噪观测最小化真实条件 MSE。

### 定理~\ref{thm:fpe}：经 Fokker–Planck 的边际对齐
<!-- \label{app-sec:scoresde-fpe-proof} -->

**证明.**

**第一部分：前向 SDE 的 Fokker-Planck 方程。**  
考虑前向 SDE：
\[
\diff \rvx(t) = \rvf(\rvx(t), t)  \diff t + g(t)  \diff \rvw(t).
\]
扩散矩阵为 $\sigma(t) = g(t) I_D$，故 $\sigma(t) \sigma(t)^T = g^2(t) I_D$。$\rvx(t)$ 的边际密度 $p_t(\rvx)$ 满足的 Fokker-Planck 方程为：
\[
\partial_t p_t(\rvx) = -\nabla_{\rvx} \cdot \bigl[ \rvf(\rvx, t) p_t(\rvx) \bigr] + \frac{1}{2} \sum_{i,j=1}^D \frac{\partial^2}{\partial x_i \partial x_j} \bigl[ (g^2(t) \delta_{ij}) p_t(\rvx) \bigr].
\]
计算扩散项：
\[
\sum_{i,j=1}^D \frac{\partial^2}{\partial x_i \partial x_j} \bigl[ g^2(t) \delta_{ij} p_t(\rvx) \bigr] = \sum_{i=1}^D \frac{\partial^2}{\partial x_i^2} \bigl[ g^2(t) p_t(\rvx) \bigr] = g^2(t) \Delta_{\rvx} p_t(\rvx).
\]
因此：
\[
\partial_t p_t(\rvx) = -\nabla_{\rvx} \cdot \bigl[ \rvf(\rvx, t) p_t(\rvx) \bigr] + \frac{1}{2} g^2(t) \Delta_{\rvx} p_t(\rvx).
\]
现用
\[
\tilde \rvf(\rvx, t) = \rvf(\rvx, t) - \frac{1}{2} g^2(t) \nabla_{\rvx} \log p_t(\rvx).
\]
重写。由 $\nabla_{\rvx} \log p_t(\rvx) = \frac{\nabla_{\rvx} p_t(\rvx)}{p_t(\rvx)}$ 得：
\[
\nabla_{\rvx} \cdot \bigl[ \tilde \rvf(\rvx, t) p_t(\rvx) \bigr] = \nabla_{\rvx} \cdot \biggl[ \rvf(\rvx, t) p_t(\rvx) - \frac{1}{2} g^2(t) \nabla_{\rvx} p_t(\rvx) \biggr].
\]
第二项为
\[
\nabla_{\rvx} \cdot \biggl[ -\frac{1}{2} g^2(t) \nabla_{\rvx} p_t(\rvx) \biggr] = -\frac{1}{2} g^2(t) \Delta_{\rvx} p_t(\rvx).
\]
故
\[
\nabla_{\rvx} \cdot \bigl[ \tilde \rvf(\rvx, t) p_t(\rvx) \bigr] = \nabla_{\rvx} \cdot \bigl[ \rvf(\rvx, t) p_t(\rvx) \bigr] - \frac{1}{2} g^2(t) \Delta_{\rvx} p_t(\rvx).
\]
因此
\[
\partial_t p_t(\rvx) = -\nabla_{\rvx} \cdot \bigl[ \tilde \rvf(\rvx, t) p_t(\rvx) \bigr],
\]
两种形式的 Fokker-Planck 方程均得验证。

**第二部分：PF-ODE 的边际密度。**  
考虑 PF-ODE：
\[
\frac{\diff \tilde\rvx(t)}{\diff t} = \tilde \rvf(\tilde\rvx(t), t), \quad \tilde \rvf(\rvx, t) = \rvf(\rvx, t) - \frac{1}{2} g^2(t) \nabla_{\rvx} \log p_t(\rvx).
\]

**前向：$\tilde\rvx(0) \sim p_0$。** 设 $\tilde\rvx(t)$ 沿 PF-ODE 且 $\tilde\rvx(0) \sim p_0$。流映射 $\bm{\Psi}_t: \mathbb{R}^D \to \mathbb{R}^D$ 定义为
\[
\frac{\diff}{\diff t} \bm{\Psi}_t(\mathbf{x}_0) = \tilde \rvf(\bm{\Psi}_t(\mathbf{x}_0), t), \quad \bm{\Psi}_0(\mathbf{x}_0) = \mathbf{x}_0.
\]
由 $\tilde\rvx(t) = \bm{\Psi}_t(\tilde\rvx(0))$，$\tilde\rvx(t)$ 的密度 $\tilde p_t(\rvx)$ 满足连续性方程
\[
\partial_t \tilde p_t(\rvx) = -\nabla_{\rvx} \cdot \bigl[ \tilde \rvf(\rvx, t) \tilde p_t(\rvx) \bigr].
\]
因 $\tilde\rvx(0) \sim p_0$，有 $\tilde p_0(\rvx) = p_0(\rvx)$。由第一部分，$p_t(\rvx)$ 满足
\[
\partial_t p_t(\rvx) = -\nabla_{\rvx} \cdot \bigl[ \tilde \rvf(\rvx, t) p_t(\rvx) \bigr].
\]
$\tilde p_t$ 与 $p_t$ 满足同一连续性方程且初值均为 $p_0$。在适当光滑性下（如 $\tilde \rvf \in C^1$）解在相应函数空间中唯一，故 $\tilde p_t = p_t$。因此对所有 $t \in [0, T]$ 有 $\tilde\rvx(t) \sim p_t$。

**后向：$\tilde\rvx(T) \sim p_T$。** 设 $\tilde\rvx(t)$ 沿 PF-ODE 从 $t = T$ 反向到 $t = 0$，且 $\tilde\rvx(T) \sim p_T$。令 $s = T - t$，则反向 ODE 为
\[
\frac{\diff}{\diff s} \tilde\rvx(T - s) = -\tilde \rvf(\tilde\rvx(T - s), T - s).
\]
$\tilde\rvx(T - s)$ 的密度 $\tilde p_{T-s}(\rvx)$ 满足
\[
\partial_s \tilde p_{T-s}(\rvx) = \nabla_{\rvx} \cdot \bigl[ \tilde \rvf(\rvx, T - s) \tilde p_{T-s}(\rvx) \bigr].
\]
由 $\tilde\rvx(T) \sim p_T$ 得 $\tilde p_T = p_T$。$p_t$ 在 $t = T - s$ 的 Fokker-Planck 方程为
\[
\partial_t p_{T-s}(\rvx) = -\nabla_{\rvx} \cdot \bigl[ \tilde \rvf(\rvx, T - s) p_{T-s}(\rvx) \bigr].
\]
因 $\partial_t = -\partial_s$，有
\[
\partial_s p_{T-s}(\rvx) = \nabla_{\rvx} \cdot \bigl[ \tilde \rvf(\rvx, T - s) p_{T-s}(\rvx) \bigr].
\]
$\tilde p_{T-s}$ 与 $p_{T-s}$ 满足同一 PDE 且在 $s = 0$ 有相同初值（$\tilde p_T = p_T$）。由唯一性得 $\tilde p_{T-s} = p_{T-s}$，故对所有 $t \in [0, T]$ 有 $\tilde\rvx(t) = \tilde\rvx(T - s) \sim p_{T-s} = p_t$。

**第三部分：逆时 SDE 的边际密度。** 考虑逆时 SDE：
\[
\diff \bar\rvx(t) = \bigl[ \rvf(\bar\rvx(t), t) - g^2(t) \nabla_{\rvx} \log p_t(\bar\rvx(t)) \bigr] \diff t + g(t) \diff \bar\rvw(t),
\]
其中 $\bar\rvx(0) \sim p_T$，$\bar\rvw(t)$ 为逆时标准 Wiener 过程，定义为 $\bar\rvw(t) = \rvw(T - t) - \rvw(T)$。需证 $\bar\rvx(t) \sim p_{T-t}$。  
将漂移改写为
\[
\rvf(\rvx, t) = \tilde \rvf(\rvx, t) + \frac{1}{2} g^2(t) \nabla_{\rvx} \log p_t(\rvx),
\]
故
\[
\rvf(\rvx, t) - g^2(t) \nabla_{\rvx} \log p_t(\rvx) = \tilde \rvf(\rvx, t) - \frac{1}{2} g^2(t) \nabla_{\rvx} \log p_t(\rvx).
\]
逆时 SDE 为
\[
\diff \bar\rvx(t) = \biggl[ \tilde \rvf(\bar\rvx(t), t) - \frac{1}{2} g^2(t) \nabla_{\rvx} \log p_t(\bar\rvx(t)) \biggr] \diff t + g(t) \diff \bar\rvw(t).
\]
令 $s = T - t$，则 $\bar\rvx(t) = \bar\rvx(T - s)$，$\diff t = -\diff s$。代入并利用 $\bar\rvw(t) = \rvw(T - t) - \rvw(T)$ 得 $\diff \bar\rvw(T - s) = -\diff \rvw(s)$，令 $\bar\rvw'(s) = -\rvw(s)$ 得
\[
\diff \bar\rvx(s) = \biggl[ \tilde \rvf(\bar\rvx(s), T - s) - \frac{1}{2} g^2(T - s) \nabla_{\rvx} \log p_{T-s}(\bar\rvx(s)) \biggr] \diff s + g(T - s) \diff \bar\rvw'(s).
\]
$\bar\rvx(s)$ 的密度 $\bar p_s(\rvx)$ 满足的 Fokker-Planck 方程为
\[
\partial_s \bar p_s(\rvx) = -\nabla_{\rvx} \cdot \biggl[ \biggl( \tilde \rvf(\rvx, T - s) - \frac{1}{2} g^2(T - s) \nabla_{\rvx} \log p_{T-s}(\rvx) \biggr) \bar p_s(\rvx) \biggr] + \frac{1}{2} g^2(T - s) \Delta_{\rvx} \bar p_s(\rvx).
\]
令 $\bar p_s = p_{T-s}$。$p_{T-s}$ 的 Fokker-Planck 方程为 $\partial_t p_{T-s}(\rvx) = -\nabla_{\rvx} \cdot \bigl[ \tilde \rvf(\rvx, T - s) p_{T-s}(\rvx) \bigr]$，由 $\partial_t = -\partial_s$ 得 $\partial_s p_{T-s}(\rvx) = \nabla_{\rvx} \cdot \bigl[ \tilde \rvf(\rvx, T - s) p_{T-s}(\rvx) \bigr]$。将 $\bar p_s = p_{T-s}$ 代入 $\bar p_s$ 的 FPE，多出的项相互抵消，故 $\bar p_s = p_{T-s}$ 满足该 FPE。由 $\bar\rvx(0) \sim p_T$ 得 $\bar p_0 = p_T$，初值一致。在适当光滑性下解唯一，故 $\bar p_s = p_{T-s}$，即 $\bar\rvx(t) = \bar\rvx(T - s) \sim p_{T-t}$。
□

### 命题~\ref{dsm-minimizer}：SM 与 DSM 的最小元
<!-- \label{app-sec:min-dsm} -->

**证明.** 为求最小元 $\mathbf{s}^*$，先固定时刻 $t$ 并分析目标函数中的内层期望。将该期望用 $X_0$ 与 $X_t$ 的联合分布重写后，对每个固定的 $\rvx_t$ 需最小化关于 $\mathbf{s}_{\bm{\phi}}(\rvx_t, t)$ 的期望平方误差。当 $\mathbf{s}_{\bm{\phi}}(\rvx_t, t)$ 等于条件期望
\[
\mathbf{s}^*(\rvx_t, t) 
= \mathbb{E}_{X_0 \sim p(X_0 | X_t = \rvx_t)} 
\left[ \nabla_{\rvx_t} \log p(\rvx_t | X_0) \right]
\]
时达到最小。再证其等于 $\nabla_{\rvx_t} \log p_t(\rvx_t)$：由 Bayes 法则与边际概率定义有 $p_t(\rvx_t) = \int p_t(\rvx_t | \rvx_0) p_{\text{data}}(\rvx_0)   \diff\rvx_0$，对 $\rvx_t$ 取对数再求梯度，在适当正则条件下可交换梯度与积分，即得 $\nabla_{\rvx_t} \log p_t(\rvx_t)$ 的积分表示，与 $\mathbf{s}^*(\rvx_t,t)$ 一致。
□

### 高斯的闭式分数函数

以下引理给出一般多元正态分布分数公式，供后文引用。

**引理（高斯的分数，Score of Gaussian）。** 设 $\rvx \in \mathbb{R}^D$，考虑多元正态分布
\[
p(\tilde{\rvx} | \rvx) := \mathcal{N}(\tilde{\rvx}; \bm{\mu}, \bm{\Sigma}),
\]
其中 $\bm{\mu} \in \mathbb{R}^D$ 为均值，$\bm{\Sigma} \in \mathbb{R}^{D \times D}$ 为可逆协方差矩阵。其分数函数为
\begin{align}\label{eq:score-gaussian-formula}
    \nabla_{\tilde{\rvx}} \log p(\tilde{\rvx} | \rvx) = -\bm{\Sigma}^{-1} (\tilde{\rvx} - \bm{\mu}).
\end{align}

*证明.* $p(\tilde{\rvx} | \rvx)$ 的密度为
\[
p(\tilde{\rvx} | \rvx) = \frac{1}{(2\pi)^{D/2} |\bm{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\tilde{\rvx} - \bm{\mu})^\top \bm{\Sigma}^{-1} (\tilde{\rvx} - \bm{\mu}) \right).
\]
对 $\log p(\tilde{\rvx} | \rvx)$ 关于 $\tilde{\rvx}$ 求梯度，利用链式法则得 $\nabla_{\tilde{\rvx}} \left( (\tilde{\rvx} - \bm{\mu})^\top \bm{\Sigma}^{-1} (\tilde{\rvx} - \bm{\mu}) \right) = 2 \bm{\Sigma}^{-1} (\tilde{\rvx} - \bm{\mu})$，故分数函数为
\begin{align}\label{eq:gaussian-score}
    \nabla_{\tilde{\rvx}} \log p(\tilde{\rvx} | \rvx) = - \bm{\Sigma}^{-1} (\tilde{\rvx} - \bm{\mu}).
\end{align}

---

## 基于流的视角
<!-- \label{app-sec:flow-proof} -->

### 引理~\ref{instant-change-of-var}：瞬时变量替换
<!-- \label{instant-change-of-var} -->

**证明.**

**方法一：变量替换公式。** 记 $p(\rvx(t), t)$ 为 $p_t(\rvx_t)$。从 ODE 离散化
\[
\rvz_{t+\Delta t} = \rvz_t + \Delta t  \rmF(\rvz_t, t)
\]
出发，归一化流的变量替换公式（\Cref{eq:nf-change-of-var-log}）给出
\[
\log p_{t+\Delta t}(\rvz_{t+\Delta t}) 
= \log p_t(\rvz_t) - \log \Big| \det\big(\rmI + \Delta t \nabla_{\rvz} \rmF(\rvz_t, t)\big) \Big|
= \log p_t(\rvz_t) - \Tr\Big(\log(\rmI + \Delta t \nabla_{\rvz} \rmF(\rvz_t, t)) \Big)
= \log p_t(\rvz_t) - \Delta t  \Tr\big(\nabla_{\rvz} \rmF(\rvz_t, t)\big) + \mathcal{O}(\Delta t^2),
\]
其中用到了 $\log \det = \Tr \log$ 以及小 $\Delta t$ 的展开。令 $\Delta t \to 0$ 即得对数密度的连续时间微分方程；与 \Cref{eq:log-det-tr} 中的技巧相同。

**方法二：连续性方程。** 也可利用连续性方程，其本质上充当变量替换公式：
\[
\partial_t p(\rvz,t) = - \nabla_{\rvz} \cdot \big(\rmF(\rvz,t) p(\rvz,t)\big).
\]
展开散度得
\[
\partial_t p = - \big( (\nabla_{\rvz} \cdot \rmF) p + \rmF \cdot \nabla_{\rvz} p \big).
\]
沿满足 $\frac{\diff \rvz}{\diff t} = \rmF(\rvz(t), t)$ 的轨迹 $\rvz(t)$，对时间的全导数为
\[
\frac{\diff}{\diff t} p(\rvz(t), t) 
= \nabla_{\rvz} p \cdot \frac{\diff \rvz}{\diff t} + \partial_t p 
= \nabla_{\rvz} p \cdot \rmF - \big( (\nabla_{\rvz} \cdot \rmF) p + \rmF \cdot \nabla_{\rvz} p \big) 
= - (\nabla_{\rvz} \cdot \rmF) p.
\]
两边除以 $p(\rvz(t), t)$ 得
\[
\frac{\diff}{\diff t} \log p(\rvz(t), t) = - \nabla_{\rvz} \cdot \rmF(\rvz(t), t).
\]
□

### 定理~\ref{thm:continuity-mass}：质量守恒准则
<!-- \label{app-sec:mass-conservation} -->

**预备：流映射与流诱导密度。** 对任意初始位置 $\mathbf{x}_0\in\mathbb{R}^D$，流映射 $\bm{\Psi}_t:\mathbb{R}^D\to\mathbb{R}^D$ 是下列 ODE 的唯一解：
\[
\frac{\diff}{\diff t} \bm{\Psi}_t(\mathbf{x}_0)
= \mathbf{v}_t\bigl(\bm{\Psi}_t(\mathbf{x}_0)\bigr), 
\quad \bm{\Psi}_0(\mathbf{x}_0)=\mathbf{x}_0.
\]
在所用正则性假设下，$\bm{\Psi}_t$ 对 $t$ 和 $\mathbf{x}_0$ 连续可微。

流诱导密度 $p_t^{\mathrm{fwd}}$ 是初始密度 $p_0$ 经 $\bm{\Psi}_t$ 的推前：
\[
p_t^{\mathrm{fwd}}(\mathbf{x})
= p_0\bigl(\bm{\Psi}_t^{-1}(\mathbf{x})\bigr)
  \Bigl|\det \bigl(\nabla\bm{\Psi}_t^{-1}(\mathbf{x})\bigr)\Bigr|.
\]
它给出从 $p_0=p_{\mathrm{data}}$ 出发、在速度场 $\mathbf{v}_t$ 下演化到时刻 $t$、位于 $\mathbf{x}$ 的粒子密度。

**非形式证明：**  
**充分条件：$p_t^{\mathrm{fwd}} = p_t$ $\Rightarrow$ 连续性方程。**  
在 \Cref{subsec:cov-continuity-eq} 中，我们通过离散变量替换公式的连续时间极限得到了连续性方程的**强解**；那里假定密度 $p_t$ 足够光滑使得所有导数在经典意义下存在且 PDE 逐点成立。此处在**弱意义**下给出一个互补推导：连续性方程仅在与任意光滑紧支撑试验函数积分后才施加，从而放宽对 $p_t$ 和速度场 $\mathbf{v}_t$ 的正则性要求。弱形式不仅更严谨（可容纳更不光滑的解），也是 PDE 理论与数值分析中的标准框架。

对任意紧支撑光滑试验函数 $\varphi(\mathbf{x})$，推前性质给出：
\[
\int p_t^{\mathrm{fwd}}(\mathbf{x}) \varphi(\mathbf{x})  \diff\mathbf{x} = \int p_0(\bm{\Psi}_t^{-1}(\mathbf{x})) \left| \det\left( \nabla \bm{\Psi}_t^{-1}(\mathbf{x}) \right) \right| \varphi(\mathbf{x}) \diff\mathbf{x} = \int p_0(\mathbf{y}) \varphi(\bm{\Psi}_t(\mathbf{y})) \diff\mathbf{y},
\]
其中第二式由变量替换 $\mathbf{x} = \bm{\Psi}_t(\mathbf{y})$ 得到，$\diff\mathbf{y} = \left| \det\left( \nabla \bm{\Psi}_t^{-1}(\mathbf{x}) \right) \right| \diff\mathbf{x}$。

对 $t$ 求导：
\[
\frac{\diff}{\diff t} \int p_t^{\mathrm{fwd}}(\mathbf{x}) \varphi(\mathbf{x}) \diff\mathbf{x} = \frac{\diff}{\diff t} \int p_0(\mathbf{y}) \varphi(\bm{\Psi}_t(\mathbf{y})) \diff\mathbf{y}.
\]
左边为 $\int \frac{\partial p_t^{\mathrm{fwd}}}{\partial t}(\mathbf{x}) \varphi(\mathbf{x}) \diff\mathbf{x}$。右边因 $\frac{\partial \bm{\Psi}_t}{\partial t}(\mathbf{y}) = \mathbf{v}_t(\bm{\Psi}_t(\mathbf{y}))$ 为
\[
\int p_0(\mathbf{y}) \nabla \varphi(\bm{\Psi}_t(\mathbf{y})) \cdot \mathbf{v}_t(\bm{\Psi}_t(\mathbf{y})) \diff\mathbf{y}.
\]
做变量替换 $\mathbf{x} = \bm{\Psi}_t(\mathbf{y})$，则 $\diff\mathbf{y} = \left| \det\left( \nabla \bm{\Psi}_t^{-1}(\mathbf{x}) \right) \right| \diff\mathbf{x}$，且
\[
p_0(\mathbf{y}) = p_t^{\mathrm{fwd}}(\mathbf{x}) \left| \det\left( \nabla \bm{\Psi}_t(\mathbf{y}) \right) \right| = \frac{p_t^{\mathrm{fwd}}(\mathbf{x})}{\left| \det\left( \nabla \bm{\Psi}_t^{-1}(\mathbf{x}) \right) \right|}.
\]
因此右边变为 $\int p_t^{\mathrm{fwd}}(\mathbf{x}) \nabla \varphi(\mathbf{x}) \cdot \mathbf{v}_t(\mathbf{x}) \diff\mathbf{x}$。利用 $\varphi$ 的紧支撑做分部积分：
\[
\int p_t^{\mathrm{fwd}}(\mathbf{x}) \nabla \varphi(\mathbf{x}) \cdot \mathbf{v}_t(\mathbf{x}) \diff\mathbf{x} = -\int \varphi(\mathbf{x}) \nabla \cdot (p_t^{\mathrm{fwd}}(\mathbf{x}) \mathbf{v}_t(\mathbf{x})) \diff\mathbf{x}.
\]
令两边相等：
\[
\int \left[ \frac{\partial p_t^{\mathrm{fwd}}}{\partial t}(\mathbf{x}) + \nabla \cdot (p_t^{\mathrm{fwd}}(\mathbf{x}) \mathbf{v}_t(\mathbf{x})) \right] \varphi(\mathbf{x}) \diff\mathbf{x} = 0.
\]
由 $\varphi$ 的任意性得 $\frac{\partial p_t^{\mathrm{fwd}}}{\partial t} + \nabla \cdot (p_t^{\mathrm{fwd}} \mathbf{v}_t) = 0$。在 $p_t^{\mathrm{fwd}} = p_t$ 时即有 $\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \mathbf{v}_t) = 0$。

**必要条件：连续性方程 $\Rightarrow p_t^{\mathrm{fwd}} = p_t$。**  
设 $p_t$ 满足连续性方程 $\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \mathbf{v}_t) = 0$，初值为 $p_0(\mathbf{x}) = p_{\mathrm{data}}(\mathbf{x})$。如上所述，$p_t^{\mathrm{fwd}}$ 满足同一连续性方程，且 $p_0^{\mathrm{fwd}}(\mathbf{x}) = p_0(\bm{\Psi}_0^{-1}(\mathbf{x})) \left| \det\left( \nabla \bm{\Psi}_0^{-1}(\mathbf{x}) \right) \right| = p_0(\mathbf{x})$（因 $\bm{\Psi}_0(\mathbf{x}) = \mathbf{x}$），故两者初值均为 $p_0 = p_{\mathrm{data}}$。

连续性方程可改写为 $\frac{\partial p}{\partial t} + \mathbf{v}_t \cdot \nabla p + p \nabla \cdot \mathbf{v}_t = 0$。这是一阶线性 PDE。若 $\mathbf{v}_t$ 连续可微且全局 Lipschitz，$p_t$ 足够光滑，则特征线法保证光滑函数空间中解唯一。因 $p_t$ 与 $p_t^{\mathrm{fwd}}$ 满足同一 PDE 与初值，故对所有 $t\in[0,1]$ 和 $\mathbf{x} \in \mathbb{R}^D$ 有 $p_t(\mathbf{x}) = p_t^{\mathrm{fwd}}(\mathbf{x})$。等价性证毕。
□

---

## 理论补充：扩散模型的统一系统视角
<!-- \label{app-sec:equiv-para} -->

### 命题~\ref{equi-para}：参数化的等价性
<!-- \label{equi-para} -->

**证明.** 与 DSM 损失的定理~\ref{dsm-minimizer} 类似，匹配损失
\[
\mathbb{E}_t\left[\omega(t)\,\mathbb{E}_{\rvx_0,\bm{\epsilon}}\left[\|\cdot\|_2^2\right]\right]
\]
的全局最优在每固定 $t$ 下内层期望 $\mathbb{E}_{\rvx_0,\bm{\epsilon}}\left[\|\cdot\|_2^2\right]$ 最小时达到。这是标准的均方误差问题，最小元唯一。由去噪分数匹配~\citep{vincent2011connection} 与定理~\ref{dsm-minimizer}，最优分数函数满足
\[
\rvs^*(\rvx_t,t) = \mathbb{E}_{p(\rvx_0|\rvx_t)}\left[\nabla_{\rvx}\log p_t(\rvx_t|\rvx_0)\right] = \nabla_{\rvx_t} \log p_t(\rvx_t).
\]
利用恒等式 $\nabla_{\rvx}\log p_t(\rvx_t|\rvx_0) = -\frac{1}{\sigma_t} \bm{\epsilon}$（$\bm{\epsilon} \sim \mathcal{N}(\bm{0},\rmI)$）得 $\rvs^*(\rvx_t,t) = -\frac{1}{\sigma_t}\, \bm{\epsilon}^*(\rvx_t,t)$，其中 $\bm{\epsilon}^*(\rvx_t,t) = \mathbb{E}_{p(\rvx_0|\rvx_t)}[\bm{\epsilon}|\rvx_t]$ 是 $\mathcal{L}_{\text{noise}}({\bm{\phi}})$ 的最优 $\bm{\epsilon}$ 预测。对 $\rvx$ 预测损失 $\mathcal{L}_{\text{clean}}$，最优估计为 $\rvx^*(\rvx_t,t) = \mathbb{E}_{p(\rvx_0|\rvx_t)}[\rvx_0|\rvx_t]$，由 Tweedie 公式与分数相联系：$\alpha_t\, \rvx^*(\rvx_t,t) = \rvx_t + \sigma_t^2\, \rvs^*(\rvx_t,t)$。对速度预测损失 $\mathcal{L}_{\text{velocity}}$，最优估计为
\[
\rvv^*(\rvx_t,t) = \mathbb{E}_{p(\rvx_0|\rvx_t)}[\alpha_t' \rvx_0 + \sigma_t' \bm{\epsilon}|\rvx_t] = \alpha_t' \rvx^* + \sigma_t' \bm{\epsilon}^*.
\]
将 $\rvx^*$ 和 $\bm{\epsilon}^*$ 用 $\rvs^*$ 表示代入，得
\[
\rvv^*(\rvx_t,t) = \frac{\alpha_t'}{\alpha_t} \rvx_t + \left( \frac{\alpha_t'}{\alpha_t} \sigma_t^2 - \sigma_t \sigma_t' \right) \rvs^*(\rvx_t,t) = f(t) \rvx_t - \frac{1}{2}g^2(t) \rvs^*(\rvx_t,t),
\]
推导完成。
□

---

## 理论补充：学习快速扩散生成器
<!-- \label{app-sec:flow-map} -->

### 知识蒸馏损失作为一般框架 \Cref{eq:general-cm-loss} 的实例

我们从 oracle KD 损失出发：
\[
\mathcal{L}_{\mathrm{KD}}^{\text{oracle}}(\btheta)
=\E_{\rvx_T\sim p_T}\,
\big\|\rmG_\btheta(\rvx_T,T,0)-\bPsi_{T\to 0}(\rvx_T)\big\|_2^2,
\]
其中 $p_T=p_{\mathrm{prior}}$。对确定性 ODE 流映射 $\bPsi$（半群、沿轨迹双射），边际满足推前恒等式
\[
p_t = \bPsi_{0\to t}\,\sharp\,p_{\mathrm{data}} = \bPsi_{T\to t}\,\sharp\,p_{\mathrm{prior}};
\]
故 $\bPsi_{s\to T}\,\sharp\,p_s=p_T$ 且 $\bPsi_{T\to 0}\circ\bPsi_{s\to T}=\bPsi_{s\to 0}$。做变量替换 $\rvx_T=\bPsi_{s\to T}(\rvx_s)$，$\rvx_s\sim p_s$，得
\[
\mathcal{L}_{\mathrm{KD}}^{\text{oracle}}(\btheta)
= \E_{\rvx_s\sim p_s} 
\Big\|\rmG_\btheta \big(\bPsi_{s\to T}(\rvx_s),T,0\big) - \bPsi_{s\to 0}(\rvx_s)\Big\|_2^2.
\]
定义拉回的学生 $\widetilde{\rmG}_\btheta(\rvx_s,s,0):=\rmG_\btheta(\bPsi_{s\to T}(\rvx_s),T,0)$，则同一损失可写成统一流映射形式（在 $t=0$）：
\[
\mathcal{L}_{\mathrm{KD}}^{\text{oracle}}(\btheta)
= \E_{\rvx_s\sim p_s} 
\big\|\widetilde{\rmG}_\btheta(\rvx_s,s,0)-\bPsi_{s\to 0}(\rvx_s)\big\|_2^2.
\]
该推导依赖于经 oracle 流的变量替换与半群性质。
□

### 命题~\ref{vsd-grad} 的参数–流解释

由命题~\ref{vsd-grad} 的推导可知，VSD 的梯度可解释为参数诱导的传输流：调整模型参数会在数据空间中移动粒子，使其运动与学生分布和教师分布之间的分数差异对齐。

令 $t\sim p(t)$，$\rvz\sim p(\rvz)$，$\beps\sim\mathcal N(\bm{0},\rmI)$，且 $\hat\rvx_t=\alpha_t\,\rmG_\btheta(\rvz)+\sigma_t\,\beps$。定义**样本（粒子）速度** $\widehat{\rvv}_\btheta(\hat\rvx_t):=\partial_\btheta \hat\rvx_t=\alpha_t\,\partial_\btheta \rmG_\btheta(\rvz)$，以及 $\rvx$ 空间中的**速度场**为条件平均 $\rvv_\btheta(\rvx):=\E \big[\widehat{\rvv}_\btheta(\hat\rvx_t) \big| \hat\rvx_t=\rvx\big]$。在此定义下，每个固定 $t$ 时密度满足参数–流连续性方程 $\partial_\btheta p_t^{\btheta}(\rvx)+\nabla_{\rvx}\cdot\big(p_t^{\btheta}(\rvx) \rvv_\btheta(\rvx)\big)=0$。这里 $\rvv_\btheta(\hat\rvx_t)=\partial_\btheta \hat\rvx_t$ 是数据空间中**参数诱导的粒子速度**（$t$ 固定）。因此 $\mathcal{L}_{\text{VSD}}$ 的梯度具有传输形式：$\nabla_{\btheta}\mathcal{L}_{\text{VSD}} = \E\big[\omega(t)\langle \nabla_{\rvx}\log p_t^{\btheta}-\nabla_{\rvx}\log p_t,\;\rvv_\btheta\rangle\big]$，即调整参数流使粒子运动与局部分数差异对齐。

### 定理~\ref{thm:ct-cm}：CM 等于 CT 至 $\smallO(\Delta s)$ 阶

**证明.** **步骤 1：** 带 oracle 分数的 DDIM 更新给出条件均值 $\hat\rvx_{s'} = \E[\rvx_{s'}|\rvx_s]$（利用 Tweedie 公式与 $\rvx_s=\alpha_s\rvx_0+\sigma_s\beps$）。**步骤 2：** 在 $\hat\rvx_{s'}$ 处对 CT 损失做二阶 Taylor 展开，用塔性质与 $\E[\rvx_{s'}-\hat\rvx_{s'}|\rvx_s]=\bm{0}$，线性–高斯调度下 $\E[\|\rvx_{s'}-\hat\rvx_{s'}\|^2|\rvx_s]=\mathcal{O}(\Delta s^2)$，故 $\mathcal{L}_{\mathrm{CT}} = \mathcal{L}_{\mathrm{CM}} + \smallO (\Delta s)$。
□

### 命题~\ref{continuous-time-ct}：CT 梯度的连续时间极限

**证明.** 对 $\mathcal{L}_{\mathrm{CM}}^{\Delta s}(\btheta,\btheta^-) := \E \left[\omega(s) \big\| \rvf_{\btheta}(\rvx_s,s) - \rvf_{\btheta^-} \big(\bPsi_{s\to s-\Delta s}(\rvx_s), s-\Delta s\big) \big\|_2^2\right]$，记 $\tilde\rvx_{s-\Delta s}:=\bPsi_{s\to s-\Delta s}(\rvx_s)$。展开 MSE 并在 $(\rvx_s, s)$ 处对 $\rvf_{\btheta^-}$ 做 Taylor 展开，结合 oracle 流的一阶展开与全微分恒等式可得 $\nabla_{\btheta} \mathcal{L}_{\mathrm{CM}}^{\Delta s}(\btheta, \btheta^-)=2  \mathbb{E} \left[ \omega(s) \, \nabla_{\btheta} \rvf_{\btheta}(\rvx_s, s)^\top \cdot \frac{\diff}{\diff s} \rvf_{\btheta^-}(\rvx_s, s) \right] \Delta s + \mathcal{O}\left(\abs{\Delta s}^2\right)$。除以 $\Delta s$ 并取极限得 $\lim_{\Delta s\rightarrow0}\frac{1}{\Delta s}\nabla_{\btheta} \mathcal{L}_{\mathrm{CM}}^{\Delta s}(\btheta, \btheta^-) = \nabla_{\btheta}  \mathbb{E} \left[ 2\omega(t) \rvf_{\btheta}^\top(\rvx_s, s) \cdot \frac{\diff}{\diff s} \rvf_{\btheta^-}(\rvx_s, s) \right]$。
□

---

## （可选）Elucidating Diffusion Model (EDM)
<!-- \label{sec:edm} -->

本节介绍 $\rvx$ 预测模型中神经网络参数化设计的具体准则，来自 EDM~\citep{karras2022elucidating}。EDM 提供了一套简单配方，使训练更易优化、更稳定。$\rvx$ 预测模型写为时间依赖的跳跃连接与缩放残差的组合（\Cref{eq:D-parametrization}）。核心思想是：在所有时刻将输入和回归目标都归一化为单位方差，并调整跳跃路径使残差误差随时间不被放大。该配方已成为扩散模型实现中广泛采用的默认设置，并自然推广到流映射学习，尤其是一致性模型族。

### 神经网络 $\rvx_{\bm{\phi}}$ 的设计准则
<!-- \label{subsec:D-design-edm} -->

EDM 考虑参数化 $\rvx_{\bm{\phi}}(\rvx, t) := c_{\mathrm{skip}}(t) \rvx + c_{\mathrm{out}}(t) \rmF_{\bm{\phi}}\left(c_{\mathrm{in}}(t) \rvx, c_{\mathrm{noise}}(t)\right)$。代入扩散损失并经代数化简，目标变为 $\mathbb{E}_{\rvx_0, \bm{\epsilon}, t} \left[\omega(t)c_{\mathrm{out}}^2(t)\norm{ \rmF_{\bm{\phi}}\left(c_{\mathrm{in}}(t)\rvx_t, c_{\mathrm{noise}}(t)\right) - \rvx_{\mathrm{tgt}}(t)}_2^2\right]$，回归目标 $\rvx_{\mathrm{tgt}}(t) = \frac{\left(1-c_{\mathrm{skip}}(t)\alpha_t\right)\rvx_0-c_{\mathrm{skip}}(t)\sigma_t \bm{\epsilon}}{c_{\mathrm{out}}(t)}$。结合 $p_{\text{data}}$ 的标准差 $\sigma_{\mathrm{d}}$，EDM 提出**单位方差准则**：**输入单位方差** $c_{\mathrm{in}}(t) = \frac{1}{\sqrt{\sigma_{\mathrm{d}}^2 \alpha_t^2 + \sigma_t^2}}$；**训练目标单位方差**（数据中心化）$c_{\mathrm{out}}^2(t) = \left(1-c_{\mathrm{skip}}(t)\alpha_t\right)^2 \sigma_{\mathrm{d}}^2 + c_{\mathrm{skip}}^2(t)\sigma_t^2$；**最小化误差放大**：$c_{\mathrm{skip}}^* = \frac{\alpha_t \sigma_{\mathrm{d}}^2 }{\alpha_t^2 \sigma_{\mathrm{d}}^2  + \sigma_t^2}$，$c_{\mathrm{out}}^*(t) = \frac{\sigma_t \sigma_{\mathrm{d}} }{\sqrt{\alpha_t^2 \sigma_{\mathrm{d}}^2  + \sigma_t^2}} \ge 0$。记 $R_t := \alpha_t^2 \sigma_{\mathrm{d}}^2+ \sigma_t^2$，则 $c_{\mathrm{in}}(t) = \frac{1}{\sqrt{R_t}}$，$c_{\mathrm{skip}}(t) = \frac{\alpha_t \sigma_{\mathrm{d}}^2 }{R_t}$，$c_{\mathrm{out}}(t) = \frac{\sigma_t \sigma_{\mathrm{d}} }{\sqrt{R_t}}$。$c_{\mathrm{noise}}(t)$ 为噪声水平嵌入，可取 $c_{\mathrm{noise}}(t)=\log \sigma_t$ 等一一映射。

### EDM 的常见特例：$\alpha_t = 1$，$\sigma_t = t$

考虑 $\alpha_t = 1$、$\sigma_t = t$（亦见 \Cref{sec:CTM}）。前向为 $\rvx_t = \rvx_0 + t\bm{\epsilon}$，扰动核 $p_t(\rvx_t|\rvx_0) = \mathcal{N}(\rvx_t; \rvx_0, t^2\rmI)$，先验 $p_{\text{prior}}(\rvx_T) := \mathcal{N}(\rvx_T; \bm{0}, T^2 \rmI)$。基于 $\rvx$ 预测的 PF-ODE 简化为 $\frac{\diff \rvx(t)}{\diff t} = \frac{\rvx(t) - \rvx_{\bm{\phi}^\times}(\rvx(t), t)}{t}$。系数为 $c_{\mathrm{in}}(t) = \frac{1}{\sqrt{\sigma_{\mathrm{d}}^2 + t^2}}$，$c_{\mathrm{skip}}(t) = \frac{\sigma_{\mathrm{d}}^2}{\sigma_{\mathrm{d}}^2 + t^2}$，$c_{\mathrm{out}}(t) = \frac{t \sigma_{\mathrm{d}}}{\sqrt{\sigma_{\mathrm{d}}^2 + t^2}}$。当 $t \approx 0$ 时 $c_{\mathrm{skip}} \approx 1$、$c_{\mathrm{out}} \approx 0$，$\rvx_{\bm{\phi}}(\rvx, t) \approx \rvx$；当 $t \gg 0$ 时 $c_{\mathrm{skip}} \approx 0$、$c_{\mathrm{out}} \approx \sigma_{\mathrm{d}}$，$\rvx_{\bm{\phi}}(\rvx, t) \approx \sigma_{\mathrm{d}} \rmF_{\bm{\phi}}\left(c_{\mathrm{in}}(t)\rvx, c_{\mathrm{noise}}(t)\right)$。参数化从小 $t$ 的恒等映射平滑过渡到大 $t$ 时标准化输入上的缩放残差预测器。
