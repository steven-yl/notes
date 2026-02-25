# 符号说明

本文档为 *The Principles of Diffusion Models* 中符号约定的中文说明，公式与记号与原文一致。源文件：`arXiv-2510.21890v1/notations.tex`。

---

## 数与数组

| 符号 | 含义 |
|------|------|
| $a$ | 标量。 |
| $\mathbf{a}$ | 列向量（如 $\mathbf{a}\in\mathbb{R}^D$）。 |
| $\mathbf{A}$ | 矩阵（如 $\mathbf{A}\in\mathbb{R}^{m\times n}$）。 |
| $\mathbf{A}^\top$ | $\mathbf{A}$ 的转置。 |
| $\operatorname{Tr}(\mathbf{A})$ | $\mathbf{A}$ 的迹。 |
| $\mathbf{I}_D$ | $D\times D$ 单位阵。 |
| $\mathbf{I}$ | 单位阵；维数由上下文给出。 |
| $\mathrm{diag}(\mathbf{a})$ | 对角线由 $\mathbf{a}$ 给出的对角矩阵。 |
| $\boldsymbol{\phi},\,\boldsymbol{\theta}$ | 可学习的神经网络参数。 |
| $\boldsymbol{\phi}^{\times},\,\boldsymbol{\theta}^{\times}$ | 训练后的参数（推断时固定）。 |
| $\boldsymbol{\phi}^{*},\,\boldsymbol{\theta}^{*}$ | 优化问题的最优参数。 |

---

## 微积分

| 符号 | 含义 |
|------|------|
| $\displaystyle \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ | $\mathbf{y}$ 对 $\mathbf{x}$ 的偏导数（按分量）。 |
| $\displaystyle \frac{\mathrm{d} \mathbf{y}}{\mathrm{d} \mathbf{x}}\ \text{或}\ \mathrm{D}\mathbf{y}(\mathbf{x})$ | $\mathbf{y}$ 对 $\mathbf{x}$ 的（Fréchet）全导数。 |
| $\displaystyle \nabla_\mathbf{x} y$ | 标量 $y:\mathbb{R}^D \to\mathbb{R}$ 的梯度；$\mathbb{R}^D$ 中的列向量。 |
| $\displaystyle \frac{\partial \mathbf{F}}{\partial \mathbf{x}}\ \text{或}\ \nabla_\mathbf{x} \mathbf{F}$ | $\mathbf{F}:\mathbb{R}^n \to\mathbb{R}^m$ 的雅可比矩阵；形状 $m\times n$。 |
| $\displaystyle \nabla\cdot\mathbf{y}$ | 向量场 $\mathbf{y}:\mathbb{R}^D \to\mathbb{R}^D$ 的散度；标量。 |
| $\displaystyle \nabla^2_\mathbf{x} f(\mathbf{x})\ \text{或}\ \mathbf{H}(f)(\mathbf{x})$ | $f:\mathbb{R}^D \to\mathbb{R}$ 的黑塞矩阵；形状 $D\times D$。 |
| $\displaystyle \int f(\mathbf{x}) \,\mathrm{d}\mathbf{x}$ | $f$ 在 $\mathbf{x}$ 定义域上的积分。 |

---

## 概率与信息论

| 符号 | 含义 |
|------|------|
| $p(\mathbf{a})$ | 连续向量 $\mathbf{a}$ 上的密度/分布。 |
| $p_{\mathrm{data}}$ | 数据分布。 |
| $p_{\mathrm{prior}}$ | 先验分布（如标准正态）。 |
| $p_{\mathrm{src}}$ | 源分布。 |
| $p_{\mathrm{tgt}}$ | 目标分布。 |
| $\mathbf{a} \sim p$ | 随机向量 $\mathbf{a}$ 服从分布 $p$。 |
| $\mathbb{E}_{\mathbf{x}\sim p} \big[\mathbf{f}(\mathbf{x})\big]$ | $p(\mathbf{x})$ 下 $\mathbf{f}(\mathbf{x})$ 的期望。 |
| $\mathbb{E} \big[\mathbf{f}(\mathbf{x})|\mathbf{z}\big]$ 或 $\mathbb{E}_{\mathbf{x}\sim p(\cdot|\mathbf{z})} \big[\mathbf{f}(\mathbf{x})\big]$ | 给定 $\mathbf{z}$ 时 $\mathbf{f}(\mathbf{x})$ 的条件期望（$\mathbf{x}$ 服从 $p(\cdot|\mathbf{z})$）。 |
| $\operatorname{Var} \big(\mathbf{f}(\mathbf{x})\big)$ | 在 $p(\mathbf{x})$ 下的方差。 |
| $\operatorname{Cov} \big(\mathbf{f}(\mathbf{x}),\mathbf{g}(\mathbf{x})\big)$ | 在 $p(\mathbf{x})$ 下的协方差。 |
| $\mathcal{D}_{\mathrm{KL}} \left(p\Vert q\right)$ | $q$ 到 $p$ 的 Kullback–Leibler 散度。 |
| $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ | 标准正态样本。 |
| $\mathcal{N}(\mathbf{x};\boldsymbol{\mu},\boldsymbol{\Sigma})$ | 均值为 $\boldsymbol{\mu}$、协方差为 $\boldsymbol{\Sigma}$ 的关于 $\mathbf{x}$ 的高斯分布。 |

---

## 说明

同一符号既可用于随机向量，也可用于其实现值。这一约定在深度学习和生成建模中常见，使记号简洁。具体含义由上下文决定。

例如，在 $p(\mathbf{x})$ 这类表达中，$\mathbf{x}$ 是哑变量，整体表示作为输入函数的分布或密度，因此 $p(\mathbf{x})$ 指的是函数形式，而不是在某个具体样本处的取值。若需表示在给定点处的取值，我们会明确写出（例如「在给定点 $\mathbf{x}$ 处计算 $p$」）。

条件表达式由上下文理解。对 $p(\mathbf{x}|\mathbf{y})$，固定 $\mathbf{y}$ 则视为 $\mathbf{x}$ 的密度；固定 $\mathbf{x}$ 则视为 $\mathbf{y}$ 的函数。

对条件期望，$\mathbb{E}[\mathbf{f}(\mathbf{x})|\mathbf{z}]$ 表示 $\mathbf{z}$ 的函数，给出在给定 $\mathbf{z}$ 时 $\mathbf{f}(\mathbf{x})$ 的期望。当条件为某个具体实现值时，记为 $\mathbb{E}[\mathbf{f}(\mathbf{x})| \mathbf{Z}=\mathbf{z}]$。等价地，可写为对条件分布的积分：
$$
\mathbb{E}_{\mathbf{x}\sim p(\cdot|\mathbf{z})}[\mathbf{f}(\mathbf{x})]
=
\int \mathbf{f}(\mathbf{x})\,p(\mathbf{x}|\mathbf{z})\,\mathrm{d}\mathbf{x}.
$$
这一区分明确了 $\mathbf{z}$ 是定义函数 $\mathbf{z}\mapsto \mathbb{E}[\mathbf{f}(\mathbf{x})|\mathbf{z}]$ 的变量，还是该函数在某固定点处的取值。
