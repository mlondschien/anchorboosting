# Anchor Boosting

This repository combines ideas from [1, 2 and 3].
[1] suggests regularizing linear regression with the correlation between a so-called anchor variable and the regression's residuals.
The anchor variable is assumed to be exogenous to the system, i.e., not causally affected by covariates, the outcome, or relevant hidden variables.
[1] show that such regularization induces robustness of the linear regression model to shift interventions with limited magnitude in directions as observed in the data.

[2] proposes to generalize this to nonlinear regression by simply optimizing, e.g., via boosting, the regularized anchor regression loss, replacing linear regression with a more flexible model class.

Lastly, [3] give ideas to generalize anchor regression to classification.

This repository combines these approaches and implements non-linear anchor (multiclass-) classification using LGBM.

## Simple multiclass classification
Consider a setup with observations $x_i \in \mathbb{R}^p$ with outcomes $y_i \in \{1, \ldots, K\}$ for $i=1, \ldots, n$.
We assign raw scores $f_i = (f_{i, 1}, \ldots, f_{i, K}) \in \mathbb{R}^K$ to each observation and use cross-entropy to obtain probability predictions.
For $k = 1, \ldots, K$ and $i=1, \ldots, n$ the estimated probability of observation $i$ with raw score $f_i$ to belong to class $k$ is $p_{i, k} := \exp(f_{i, k}) / \sum_{j=1}^K \exp(f_{i, j})$. The log-likelihood of raw scores $(f_i)_{i=1}^n$ is then 
$$\ell\left((f_{i, k})_{i=1, \ldots, n}^{k=1, \ldots, K}, (y_i)_{i=1}^n\right) = \sum_{i=1}^n \left(f_{i, y} - \log\left(\sum_{j=1}^K \exp(f_{i,j})\right)\right).$$

The gradient of the log-likelihood is
$$
\frac{d}{df_{i, k}} \ell(f, y) = \begin{cases}
1 - p_{i, k} &  y_i = k \\
- p_{i, k} &  y_i \neq k \\
\end{cases}
$$

The (diagonal of the) Hessian of the log-likelihood is
$$
\frac{d^2}{d^2f_{i, k}} \ell(f, y) = (1 - p_{i, k}) p_{i, k}
$$

## Anchorized multiclass classification
[1] suggests to add a regularization term based on an "anchor variable" to the linear least squares optimization problem to improve distributional robustness.
[2] describes how this idea could be applied to nonlinear regression and [3] presents an idea to generalize this to (two-class) classification.

Say additional to features and outcomes we observe anchor values $a_i \in \mathbb{R}^q, i=1,\ldots K$.
Write $A = (a_i)_{i=1}^n \in \mathbb{R}^{n \times q}$ and $\pi_A$ for the plinear rojection onto the column space of $A$.
Motivated by [3], define residuals 
$$r_{i, k} = \begin{cases}
1 - p_{i, k} &  y_i = k \\
- p_{i, k} &  y_i \neq k.
\end{cases}$$
such that for all $i$ we have $\sum_{i, j} r_{i, j} = 0$.

For some tuning parameter $\gamma$, we add the additional regularization term $\gamma \| \pi_A r \|_2^2$ to our optimization problem.
This encourages uncorrelatedness between the residuals and the anchor and, hopefully, better domain generalization. To optimize this, we also calculate the gradient of the regularization term. First, note that

$$
\frac{d}{d f_{i, k}} p_{i, j} = 
\frac{d}{d f_{i, k}} \frac{\exp(f_j)}{\sum_l \exp(f_l)} =
\begin{cases}
\frac{\exp(f_k)}{\sum_l \exp(f_l)} - \frac{\exp(f_k)^2}{(\sum_l \exp(f_l))^2} & j = k \\
 - \frac{\exp(f_k)\exp{(f_j)}}{(\sum_l \exp(f_l))^2} & j \neq k
\end{cases}
=
\begin{cases}
p_{i, j} (1 - p_{i, j}) & j = k \\
p_{i, j} p_{i, k} & j \neq k
\end{cases}
$$

and

$$
\frac{d^2}{d^2 f_{i, k}} p_{i, j} =
\begin{cases}
(1 - 2 p_{i, j}) p_{i, j} (1 - p_{i, j}) & j = k \\
p_{i, j}^2 p_{i, k} & i \neq k3
\end{cases}
$$

such that

$$
\frac{d}{d f_{i, k}} \|\pi_A r\|_2^2 = 2 \pi_A r \cdot \pi_A \frac{d}{d f_{i, k}}(p_{i, j})_{i, j}
$$

Furthermore, the (diagonal of the) Hessian is ???

$$
\frac{d^2}{d^2 f_{i, k}} \|\pi_A r\|_2^2 = 2 \pi_A
$$


## References

[1] Rothenhäusler, D., N. Meinshausen, P. Bühlmann, and J. Peters (2021). Anchor regression: Heterogeneous data meet causality. Journal of the Royal Statistical Society Series B (Statistical Methodology) 83(2), 215–246.

[2] Bühlmann, P. (2020). Invariance, causality and robustness. Statistical Science 35(3), 404– 426.

[3] Kook, L., B. Sick, and P. Bühlmann (2022). Distributional anchor regression. Statistics and Computing 32(3), 1–19.