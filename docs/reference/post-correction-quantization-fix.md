## Paper Idea: James-Stein Heuristic AWQ with Global Greedy Flipping

We are working on a new quantization method and plan to submit it to the upcoming EMNLP conference.

---

### Motivation

Most existing low-bit methods (INT4/INT8, AWQ) rely on nearest rounding, which does not explicitly control
the signed error under the actual activation distribution. Optimization-based methods (GPTQ, AdaRound)
reduce error but are slow and expensive.

Our goal is to keep the efficiency of existing quantization methods while using a lightweight, statistics-guided rounding scheme
that directly controls the signed expectation of the quantization error.

---

### Core Idea

We keep the per-channel scaling, but replace pure nearest rounding with a **global greedy flipping**
procedure that uses activation statistics to decide which quantized integers should move by +1 or -1.

Key ingredients from the implementation:
- **James-Stein mean**: per-channel activation means use James-Stein shrinkage instead of the raw sample mean.
- **Dynamic outlier masking**: outlier channels in `|E[X]|` are detected with a Kneedle-like knee finder and
  excluded from flipping.
- **Global greedy flipping**: for each output channel, select the best set of integer flips that minimizes
  the residual of a linear error objective.

---

### Objective (per output channel)

The heuristic targets the signed expectation:

```
sum_i E[X_i] * (w_i - wq_i)  ->  0
```

This is a **linear expectation objective**, not a squared error and not based on `E[|X|]`.
Flips are chosen to reduce the magnitude of the above sum for each output channel.

**Link to** `|E(X)W - E(X)W_q|` (channel-wise notation):
when we focus on a single output channel, `w` and `w_q` denote the weight vector of that channel.
Let `E(X)` be the per-input-channel mean vector. Then:

```
|E(X)W - E(X)W_q|  =  | sum_i E[X_i] * (w_i - wq_i) |
```

So driving `sum_i E[X_i] * (w_i - wq_i) -> 0` is equivalent to minimizing the scalar
`|E(X)W - E(X)W_q|` for that single channel.

---


---

### Dynamic Outlier Detection (Kneedle)

We compute `|E[X]|` (James-Stein shrinkage) and sort in descending order.
We apply a Kneedle-style knee finder on the **first half** of the sorted list to determine where
outliers end and normal channels begin.

Channels above the knee threshold are treated as outliers and **excluded from flipping**.

---

### Global Greedy Flipping (per output channel)

1. Start from group-wise asymmetric quantization (per-group min/max, with scale and zero-point).
2. Compute the current signed error:
   ```
   current_error = sum_i (w_i - wq_i) * E[X_i]
   ```
3. For each weight element, compute the **flip direction** (+1 or -1 in integer space) that would
   reduce the error, and its **impact** `E[X_i] * scale_i`.
4. Mask invalid candidates:
   - Wrong sign (does not reduce the error)
   - Would go out of integer range
   - Outlier channels (from Kneedle)
5. Sort candidates by rounding cost (distance to nearest integer) and take the best prefix length `k`
   that minimizes the residual error.
6. Apply flips, with a hard cap of `max_flip_percent` per output channel.

This is **not** "farther rounding by default"; flips are chosen by a greedy search to reduce the signed
error, with outlier masking and a per-row flip budget.

---

### Practical Notes (from the code)

- Group-wise asymmetric quantization with `bits in {3,4}`.
- Calibration uses token subsampling and batched sequential quantization.
- `lm_head` is processed in chunks to avoid OOM.

---

### Summary

The method is a James-Stein enhanced quantization variant:
- Activation mean estimation via James-Stein shrinkage.
- Dynamic outlier masking with Kneedle.
- Global greedy integer flips to reduce the signed expectation of quantization error.

This keeps other quantization method efficiency while providing a more targeted control of quantization error than nearest rounding.
