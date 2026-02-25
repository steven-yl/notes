# The Principles of Diffusion Models（扩散模型原理）— 翻译记录

本文档汇总 **tpod** 目录下对 *The Principles of Diffusion Models*（arXiv:2510.21890）Part B 各章的中文翻译记录与约定。

---

## 一、源与译稿对应关系

| 译稿（tpod） | 源文件（arXiv-2510.21890v1） | 状态 | 备注 |
|-------------|------------------------------|------|------|
| [Preface.md](Preface.md) | `Preface.tex` | ✅ 已完成 | 序言与路线图；Part A/B/C/D 与附录概要；图 1–2 为 caption 说明 |
| [Preface_notations.md](Preface_notations.md) | `notations.tex` | ✅ 已完成 | 符号说明（数与数组、微积分、概率与信息论、说明段落） |
| [Part-B-diagram.md](Part-B-diagram.md) | `Part-B-diagram.tex` / Preface 中 Part B 图 | ✅ 已完成 | Part B 结构图文字版与术语中英对照 |
| [Chapter-1-Varational-Perspective-From-VAE-to-DDPM.md](Chapter-1-Varational-Perspective-From-VAE-to-DDPM.md) | `Chapter-1-Varational-Perspective-From-VAE-to-DDPM.tex` | ✅ 已完成 | 约 218 行；含 1.1–1.3、表 1、图 1–6 |
| [Chapter-2-Score-Based-Perspective-From-EBMs-to-NCSN.md](Chapter-2-Score-Based-Perspective-From-EBMs-to-NCSN.md) | `Chapter-2-Score-Based-Perspective-From-EBMs-to-NCSN.tex` | ✅ 已完成 | 约 394 行；含 2.1–2.6、表 1、图 1–7、算法框 |
| [Chapter-3-Diffusion-Models-Today-Score-SDE-Framework.md](Chapter-3-Diffusion-Models-Today-Score-SDE-Framework.md) | `Chapter-3-Diffusion-Models-Today-Score-SDE-Framework.tex` | ✅ 已完成 | 约 233 行；含 3.1–3.4、表 1、图 1–6、可选节 |

图片路径统一为相对路径：`../arXiv-2510.21890v1/Images/PartB/<文件名>.pdf`（或 .png），与原文 `Images/PartB/` 对应。

---

## 二、翻译约定

- **正文**：所有英文叙述性文字译为中文，段落与原文一一对应。
- **公式**：不翻译，保持与 tex 一致（含 `\mathbf{x}`、`\boldsymbol{\phi}`、`\mathrm{d}` 等）。
- **表格与图片**：只翻译必要的 caption；表格表头与表题已译，单元格内公式不变。
- **代码 / 算法框**：只翻译必要注释（如「用上一噪声水平的输出初始化 Langevin」等）。
- **人名**：不翻译（如 Richard P. Feynman、Anderson、Song、Hyvärinen、Vincent、Bayes、Markov、Jensen 等）。
- **引用**：`\ref` / `\Cref` 改为「上式」「前文」「附录」「见 3.1.5 节」等中文说明。
- **可编译性**：Markdown 可正常渲染；公式为 `$$...$$`（块级）与 `$...$`（行内）。

---

## 三、各章结构摘要

### 第 1 章 变分视角：从 VAE 到 DDPM

- **1.1** 变分自编码器（1.1.1 概率编码器与解码器 → 1.1.5 从标准 VAE 到层次化 VAE）
- **1.2** 变分视角：DDPM（前向过程、逆向去噪、逆向核建模、ε-预测/损失、ELBO、采样）
- **1.3** 结语  
- **表 1**：VAE 中 KL 与重建的折中  
- **图 1–6**：VAE 示意、HVAE 计算图、DDPM 示意、前向/逆向过程、采样与干净预测

### 第 2 章 基于分数的视角：从 EBM 到 NCSN

- **2.1** 基于能量的模型（能量函数、分数、分数匹配、Langevin 采样）
- **2.2** 从 EBM 到基于分数的生成模型（分数匹配训练、Langevin 采样、引言）
- **2.3** 去噪分数匹配（动机、训练、采样、Tweedie、SURE、广义分数匹配）
- **2.4** NCSN（动机、训练、退火 Langevin 采样）
- **2.5** 小结：NCSN 与 DDPM 对比  
- **2.6** 结语  
- **表 1**：NCSN 与 DDPM 对比  
- **图 1–7**：EBM 训练、分数向量场、Langevin、分数匹配、DSM、SM 准确性、NCSN  
- **算法**：退火 Langevin 动力学

### 第 3 章 扩散模型的当下：分数 SDE 框架

- **3.1** 分数 SDE 原理（离散→连续、前向 SDE、逆向 SDE、PF-ODE、边缘对齐、高斯例子）
- **3.2** 训练与采样
- **3.3** SDE 的实例化（VE/VP SDE、表 1）
- **（可选）** 前向核再认识、Fokker–Planck 与逆向 SDE
- **3.4** 结语  
- **表 1**：前向 SDE 小结（VE / VP）  
- **图 1–6**：离散加噪、前向过程、逆向过程、密度演化、采样示意、前向核引理

---

## 四、修订与检查记录

- **Ch1**：补全 1.2、1.3 的续译；补表 1；统一图 1–6 的图片语法与 caption；修正「enable」→「为使生成过程可行」、「下标 ×」→「$\boldsymbol{\phi}^*$（已训练并冻结）」。
- **Ch2**：全文由 tex 译出；图 4–7、表 1 与原文一致；算法注释已译；修正「strikingly」→「极为」等。
- **Ch1 & Ch3**：查漏补缺；Ch1 图 4–6 的 alt 统一为「图 N：…」；Ch3「bar」→「上划线 $\bar{\cdot}$」；Ch3 结语节号定为 3.4。

---

## 五、未译 / 待译

- **Prologue**（`Prologue.tex`）若需纳入 tpod，可沿用上述约定单独建译稿并在此表补充。
- 当前 tpod 已包含：序言与符号说明、Part B 结构图、Part B 第 1–3 章译稿。

---

*最后更新：按当前 tpod 目录与对话中的翻译与修订整理。*
