# Anchor Boosting

This repository combines ideas from [1, 2 and 3].
[1] suggests regularizing linear regression with the correlation between a so-called anchor variable and the regression's residuals.
The anchor variable is assumed to be exogenous to the system, i.e., not causally affected by covariates, the outcome, or relevant hidden variables.
[1] show that such regularization induces robustness of the linear regression model to shift interventions with limited magnitude in directions as observed in the data.

[2] proposes to generalize this to nonlinear regression by simply optimizing, e.g., via boosting, the regularized anchor regression loss, replacing linear regression with a more flexible model class.

Lastly, [3] give ideas to generalize anchor regression to classification.

This repository combines these approaches and implements non-linear anchor (multiclass-) classification using LGBM.

## Classification via boosting

We first describe how to explicitly solve a classification problem with boosting.
This includes calculating the gradient and the diagonal of the hessian of the loss with respect to model parameters.
In each boosting step, the gradient is fit with a weak learner (CART tree).
The diagonal of the hessian is used as a stopping criterium only (`min_hessian_in_leaf`).

We separately describe two-class and multi-class classification, even though the former is a special case of the latter.
$K$-class classification involves learning $K$ scores (parameters), each fitted with a single weak learner.
I.e., the boosting algorithm will train $K$ trees in each set.

### Two-class classification

Consider a setup with observations $x_i \in \mathbb{R}^p$ with outcomes $y_i \in \{-1, 1\}$ for $i=1, \ldots, n$. 
We assign raw scores $f_i \in \mathbb{R}$ to each observation and use the expit to obtain probability predictions.
The estimated probability of observation $x_i$ with raw score $f_i$ to belong to class 1 is $p_i = \frac{e^f}{1 + e^f} = (1 + e^{-f})^{-1}$.
The log-likelihood of raw scores $`f=(f_i)_{i=1}^n`$ is

$$\ell(f, y) = -\sum_{i=1}^n \log(1 + e^{-yf}).$$

The gradient of the log-likelihood is

$$
\frac{d}{d f_i} \ell(f, y) = y \frac{e^{-yf}}{1 + e^{-yf}} =
\begin{cases}
1 - p & y = 1 \\
-p & y = 0
\end{cases}
$$

The (diagonal of the) Hessian of the log-likelihood is

$$
\frac{d^2}{d^2f_i} \ell(f, y) = (1 - p_i) p_i.
$$

### Multiclass classification
Consider a setup with observations $x_i \in \mathbb{R}^p$ with outcomes $y_i \in \{1, \ldots, K\}$ for $i=1, \ldots, n$.
We assign raw scores $f_i = (f_{i, 1}, \ldots, f_{i, K}) \in \mathbb{R}^K$ to each observation and use cross-entropy to obtain probability predictions.
For $k = 1, \ldots, K$ and $i=1, \ldots, n$ the estimated probability of observation $x_i$ with raw score $f_i$ to belong to class $k$ is $p_{i, k} := \exp(f_{i, k}) / \sum_{j=1}^K \exp(f_{i, j})$. The log-likelihood of raw scores $`(f_i)_{i=1}^n`$ is then
```math
\ell\left((f_{i, k})_{i=1, \ldots, n}^{k=1, \ldots, K}, (y_i)_{i=1}^n\right) = \sum_{i=1}^n \left(f_{i, y} - \log\left(\sum_{j=1}^K \exp(f_{i,j})\right)\right).
```

The gradient of the log-likelihood is

$$
\frac{d}{df_{i, k}} \ell(f, y) = \begin{cases}
1 - p_{i, k} &  y_i = k \\
-p_{i, k} &  y_i \neq k \\
\end{cases}
$$

The (diagonal of the) Hessian of the log-likelihood is
```math
\frac{d^2}{d^2 f_{i, k}} \ell(f, y) = (1 - p_{i, k}) p_{i, k}
```

## Anchorized classification

[1] suggest adding a regularization term based on an "anchor variable" $A$ to the linear least squares optimization problem to improve distributional robustness.

Say additional to features and outcomes we observe anchor values $a_i \in \mathbb{R}^q, i=1,\ldots K$.
Write $A = (a_i)_{i=1}^n \in \mathbb{R}^{n \times q}$ and $\pi_A$ for the linear projection onto the column space of $A$.
[1] show that

$$
b^\gamma = \underset{b}{\arg\min} \|(\mathrm{Id} - \pi_A)(Y - Xb)\|_2^2 + \gamma\|\pi_A(Y - Xb)\|_2^2
$$

minimizes the linear model's worst-case risk with respect to certain shift interventions as seen in the data.

[2] suggests applying anchor regression to nonlinear regression by boosting the above anchor loss with a flexible (nonlinear) learner.
[3] presents an idea to generalize anchor regression to different distributions than the Gaussian distribution, i.e., using different losses than the squared error.
One interesting case is the logistic loss for (two-class) classification.
This is mostly based on intelligently defining residuals to be used in the equivalent classification anchor loss.

[4] proposes the usage of a different type of residuals for classification outside of the use with anchor regression.
We apply their residuals in the context of anchor classification.

### Anchorized two-class classification according to [3]

Note that again $y \in \{-1, 1\}$. 
[3] suggest defining residuals as

$$r_i = \frac{d}{d f}(\ell(f, y)) = y \frac{e^{-yf}}{1 + e^{-yf}} =
\begin{cases}
1 - p & y = 1 \\
-p & y = 0
\end{cases}
$$

For some tuning parameter $\gamma$, we add the regularization term $(\gamma - 1) \| \pi_A r \|_2^2$ to our optimization problem.

$$
\hat f = \underset{f}{\arg \min} \ \ell(f, Y) + (\gamma - 1) \|\pi_A r\|_2^2
$$

This encourages uncorrelatedness between the residuals and the anchor and, hopefully, better domain generalization. To optimize this, we also calculate the gradient of the regularization term. First, note that

$$
\frac{d}{d f_i} p_i = p_i (1 - p_i)
$$

such that

$$
\frac{d}{d f_i} \| \pi_A r \|_2^2 = 2 \pi_A r \cdot p (1 - p)
$$

### Anchorized multiclass classification according to [3]

We combine the notions of [1, 2, 3].
Motivated by [3], define residuals 
```math
r_{i, k} =
\frac{d}{df_{i, k}} \ell(f, y) =
 \begin{cases}
1 - p_{i, k} &  y_i = k \\
-p_{i, k} &  y_i \neq k
\end{cases}
```
such that for all $i$ we have $\sum_{k} r_{i, k} = 0$.

For some tuning parameter $\gamma$, we add the regularization term $(\gamma - 1) \| \pi_A r \|_2^2$ to our optimization problem.

$$
\hat f = \underset{f}{\arg \min} \ \ell(f, Y) + (\gamma - 1) \|\pi_A r\|_2^2
$$

This encourages uncorrelatedness between the residuals and the anchor and, hopefully, better domain generalization. To optimize this, we also calculate the gradient of the regularization term. First, note that

$$
\frac{d}{d f_{i, k}} p_{i, j} = 
\frac{d}{d f_{i, k}} \frac{\exp(f_{i, j})}{\sum_l \exp(f_{i, l})} =
\begin{cases}
\frac{\exp(f_{i, k})}{\sum_l \exp(f_{i, l})} - \frac{\exp(f_{i, k})^2}{(\sum_l \exp(f_{i, l}))^2} & j = k \\
-\frac{\exp(f_{i, k})\exp{(f_{i, j})}}{(\sum_l \exp(f_{i, l}))^2} & j \neq k
\end{cases} =
\begin{cases}
p_{i, j} - p_{i, j}p_{i, k} & j = k \\
-p_{i, j} p_{i, k} & j \neq k
\end{cases}.
$$

Then,

```math
\frac{d}{d f_{i, k}} \| \pi_A r\|_2^2 = 2 \pi_A r \cdot \left(\begin{cases}
p_{i, j} - p_{i, j}p_{i, k} & l=i, j = k \\
-p_{i, j} p_{i, k} & l=i, j \neq k \\
0 & l \neq i
\end{cases}\right)_{l=1, \ldots, n}^{j=1, \ldots K}
=(\pi_A r)_{i, k} \ p_{i, k} - \sum_{l=1}^K (\pi_A r)_{i, l} \ p_{i, l}.
```

<!---
The diagonal of the hessian is equal to 

$$
\frac{d^2}{d f_{i, k}^2} \| \pi_A r\|^2 = 2 \pi_A r \cdot
\left(\begin{cases}
p_{i, j} - 3 p_{i, j}^2 + 2 p_{i, j}^3 & l=i, j = k \\
2 p_{i, j} p_{i, k}^2 - p_{i, j} p{i, k} & l=i, j \neq k \\
0 & l \neq i
\end{cases}\right)_{l=1, \ldots, n}^{j=1, \ldots K} +
2 \pi_A \left(\begin{cases}
p_{i, j} - p_{i, j}p_{i, k} & l=i, j = k \\
- p_{i, j} p_{i, k} & l=i, j \neq k \\
0 & l \neq i
\end{cases}\right)_{l=1, \ldots, n}^{j=1, \ldots K} \cdot \left(\begin{cases}
p_{i, j} - p_{i, j}p_{i, k} & l=i, j = k \\
- p_{i, j} p_{i, k} & l=i, j \neq k \\
0 & l \neq i
\end{cases}\right)_{l=1, \ldots, n}^{j=1, \ldots K}
$$

$$
\frac{d^2}{d f_{i, k} d f_{i, l}} p_{i, j} =
\begin{cases}
(1 - 2 p_{i, j}) p_{i, j} (1 - p_{i, j}) & j = k \\
p_{i, j}^2 p_{i, k} & j \neq k
\end{cases}
$$ -->

## Anchorized two-class classification with residuals motivated by [4]



## References

[1] Rothenhäusler, D., N. Meinshausen, P. Bühlmann, and J. Peters (2021). Anchor regression: Heterogeneous data meet causality. Journal of the Royal Statistical Society Series B (Statistical Methodology) 83(2), 215–246.

[2] Bühlmann, P. (2020). Invariance, causality and robustness. Statistical Science 35(3), 404– 426.

[3] Kook, L., B. Sick, and P. Bühlmann (2022). Distributional anchor regression. Statistics and Computing 32(3), 1–19.
