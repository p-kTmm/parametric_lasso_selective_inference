\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[a4paper, left=2.5cm, right=2.5cm, top=3cm, bottom=3cm]{geometry}

\title{Selective Inference on Oracle Trans Lasso}
\date{November 2024}

\begin{document}

\maketitle

\section{Algorithm}

\noindent \textbf{Input}: Primary data \((\mathbf{X}^{0}, \mathbf{y}^{0})\) and auxiliary samples \((\mathbf{X}^{\mathcal{A}}, \mathbf{y}^{\mathcal{A}})\)

\noindent \textbf{Output}: \(\hat{\boldsymbol{\beta}}\)

\begin{itemize}
    \item Compute:
    \[
    \hat{\boldsymbol{w}} = \arg \min_{\boldsymbol{w}} \left\{ \frac{1}{2} \| \mathbf{y}^{\mathcal{A}} - \mathbf{X}^{\mathcal{A}} \boldsymbol{w} \|_2^2 + \lambda_{\boldsymbol{w}} \| \boldsymbol{w} \|_1 \right\}
    \]
    \item Let:
    \[
    \hat{\boldsymbol{\beta}} = \hat{\boldsymbol{w}} + \hat{\boldsymbol{\delta}}, \quad \hat{\boldsymbol{\delta}} = \arg \min_{\boldsymbol{\delta}} \left\{ \frac{1}{2} \| \mathbf{y}^{0} - \mathbf{X}^{0} (\hat{\boldsymbol{w}} + \boldsymbol{\delta}) \|_2^2 + \lambda_{\boldsymbol{\delta}} \| \boldsymbol{\delta} \|_1 \right\}
    \]
\end{itemize}

\section{Setup Problem}
\[
\mathbf{y}^0 \sim \mathcal{N}(\boldsymbol{\mu}^0, \Sigma^0), \quad
\mathbf{y}^\mathcal{A} \sim \mathcal{N}(\boldsymbol{\mu}^\mathcal{A}, \Sigma^\mathcal{A})
\]
\[
\mathbf{y} = \begin{pmatrix}
\mathbf{y}^0 \\ \mathbf{y}^\mathcal{A}
\end{pmatrix}, \quad
\Sigma = \begin{pmatrix}
\Sigma^0 & \mathbf{0} \\
\mathbf{0} & \Sigma^\mathcal{A}
\end{pmatrix}
\]

\section{Test Statistic and Selection}
\textbf{Hypothesis:}
\[
\mathrm{H}_{0,j} : \beta_j = 0 \quad \text{vs.} \quad \mathrm{H}_{1,j} : \beta_j \neq 0, \quad \forall j \in \mathcal{M}_{obs}
\]
\textbf{Test statistic:}
\[
\hat{\beta}_j = \boldsymbol{\eta}_j^\top \mathbf{y}_{obs}, \quad
\boldsymbol{\eta}_j = \mathcal{X} \mathbf{e}_j, \quad
\mathcal{X} = \begin{pmatrix}
\mathbf{X}^0_{\mathcal{M}_{obs}}(\mathbf{X}^0_{\mathcal{M}_{obs}}^\top \mathbf{X}^0_{\mathcal{M}_{obs}})^{-1} \\
\mathbf{0}
\end{pmatrix}
\]

\section{Transition Points}
The transition point \(t_z\) is given by:
\[
t_z = \min \{ t_z^1, t_z^2, t_z^3, t_z^4 \}
\]
where:
\[
t_z^1 = \min_{j \in \mathcal{W}_z} \left( -\frac{\hat{w}_j(z)}{\psi_j(z)} \right)_{++}, \quad
t_z^2 = \min_{j \in \mathcal{W}_z^c} \left( \lambda_{\boldsymbol{w}}\frac{\text{sign}(\gamma_j(z)) - s_j(z)}{\gamma_j(z)} \right)_{++}
\]
\[
t_z^3 = \min_{j \in \Delta_z} \left( -\frac{\hat{\delta}_j(z)}{\nu_j(z)} \right)_{++}, \quad
t_z^4 = \min_{j \in \Delta_z^c} \left( \lambda_{\boldsymbol{\delta}} \frac{\text{sign}(\kappa_j(z)) - s'_j(z)}{\kappa_j(z)} \right)_{++}
\]
Here, \((m)_{++} = m\) if \(m > 0\), and \(+\infty\) otherwise.

\end{document}
