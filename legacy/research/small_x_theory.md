## Mean-Squared Bias Analysis of Greedy Flipping

We analyze only the mean-bias term:

$$
\mathcal L_\mu(e) = (\mu^\top e)^2
$$

where:

- $\mu = \mathbb{E}[x]$
- $e = w - w_q$
- $b = \mu^\top e$

Thus,

$$
\mathcal L_\mu = b^2
$$

---

## Effect of Flipping a Single Weight

Consider flipping the quantized weight at dimension $i$.
This changes the quantization error:

$$
e_i \rightarrow e_i \pm s
$$

where $s$ is the quantization step size.

The bias becomes:

$$
b' = \mu^\top e' = b \pm \mu_i s
$$

Therefore,

$$
\Delta \mathcal L_\mu
=
(b \pm \mu_i s)^2 - b^2
=
\pm 2 b \mu_i s + \mu_i^2 s^2
$$

This fully characterizes how a single flip affects the squared mean bias.

---

## Choosing the Flip Direction

We select the sign that reduces $|b|$.

- If $b > 0$, choose $b' = b - |\mu_i| s$.
- If $b < 0$, choose $b' = b + |\mu_i| s$.

With the optimal sign:

$$
\Delta \mathcal L_\mu
=
- 2 |b| |\mu_i| s + \mu_i^2 s^2
$$

---

## When Does Flipping Improve the Loss?

Flipping reduces the loss if:

$$
\Delta \mathcal L_\mu < 0
$$

Substituting:

$$
- 2 |b| |\mu_i| s + \mu_i^2 s^2 < 0
$$

Dividing by $s > 0$:

$$
2 |b| |\mu_i| > |\mu_i|^2 s
$$

If $|\mu_i| > 0$, divide again by $|\mu_i|$:

$$
\boxed{
2 |b| > |\mu_i| s
}
$$

---

## Interpretation

Each flip changes the bias by exactly:

$$
\Delta b = \pm \mu_i s
$$

Thus, $|\mu_i| s$ is the **correction step size**.

The condition

$$
2|b| > |\mu_i| s
$$

means:

> The correction step must be sufficiently small relative to the current residual.

If $|\mu_i|$ is large, then $|\mu_i| s$ is large.
This produces a coarse correction that may overshoot zero:

$$
|b - |\mu_i| s| > |b|
$$

In that case, the squared bias increases.

---

## Why Greedy Flipping on Small-Mean Dimensions Is Stable

For dimensions with small $|\mu_i|$:

$$
|\mu_i| s \ll |b|
$$

This implies:

- Small correction step
- Low overshoot risk
- Smooth monotonic decrease of $b^2$

In contrast, large-mean dimensions:

- Produce large discrete jumps
- Frequently violate $2|b| > |\mu_i| s$
- Cause oscillation or instability in greedy updates

---

## Practical Implication

Sorting $|\mu|$ and restricting flipping to non-outlier dimensions ensures:

1. Controlled step size in bias correction
2. Stable greedy descent on $(\mu^\top e)^2$
3. Progressive reduction of systematic mean error

Thus, knee-point filtering can be interpreted as 
**adaptive step-size control for discrete bias minimization**.
