# Phân tích lý thuyết dựa trên L2 reconstruction loss cho weight-only post-training quantization

## 1. Thiết lập bài toán quantization

Xét một linear layer (bỏ qua bias để đơn giản ký hiệu):

$$
y = x^\top W
$$

trong đó:

- $x \in \mathbb{R}^d$ là activation vector (đầu vào của layer), được lấy từ calibration set,
- $W \in \mathbb{R}^{d \times m}$ là weight full-precision,
- $W_q \in \mathbb{R}^{d \times m}$ là weight sau khi quantization (weight-only post-training quantization).

Hầu hết các phương pháp weight-only PTQ hiện nay đều thực hiện **channel-wise quantization theo output channel**. Do đó, ta xét riêng một output channel $j$, tương ứng với một vector weight:

$$
w := W_{:,j}, \qquad
w_q := (W_q)_{:,j}.
$$

Sai số do quantization của channel này được ký hiệu là:

$$
e := w - w_q.
$$

Với một activation $x$, sai số trên logit của channel đó là:

$$
\delta(x) = x^\top w - x^\top w_q = x^\top e.
$$

---

## 2. L2 reconstruction loss cho một output channel

Một cách tự nhiên để đo lường sai số lượng tử hóa là dùng **L2 reconstruction loss** trên output:

$$
\mathcal{L}_2(e) := \mathbb{E}\big[(x^\top e)^2\big],
$$

trong đó kỳ vọng được lấy theo phân phối activation $x$ trong calibration set.

Khai triển biểu thức trên:

$$
\mathcal{L}_2(e)
= \mathbb{E}\big[e^\top (x x^\top) e\big]
= e^\top \mathbb{E}[x x^\top] e.
$$

Như vậy, với L2 loss, bài toán lượng tử hóa weight cho một channel tương đương với việc tối thiểu hóa một **quadratic form** theo sai số $e$, với ma trận:

$$
H := \mathbb{E}[x x^\top].
$$

---

## 3. Decomposition theo mean và covariance của activation

Đặt:

$$
\mu := \mathbb{E}[x], \qquad
\varepsilon := x - \mu.
$$

Theo định nghĩa, ta có $\mathbb{E}[\varepsilon] = 0$ và

$$
\mathrm{Cov}(x)
:= \mathbb{E}\big[(x-\mu)(x-\mu)^\top\big]
= \mathbb{E}[\varepsilon \varepsilon^\top].
$$

Từ $x = \mu + \varepsilon$, ta suy ra:

$$
\mathbb{E}[x x^\top]
= \mathbb{E}[(\mu+\varepsilon)(\mu+\varepsilon)^\top]
= \mu \mu^\top + \mathrm{Cov}(x).
$$

Thay vào biểu thức của $\mathcal{L}_2$:

$$
\mathcal{L}_2(e)
= e^\top (\mu\mu^\top + \mathrm{Cov}(x)) e
= (\mu^\top e)^2 + e^\top \mathrm{Cov}(x) e.
$$

Đây là một **đẳng thức chính xác**, không phải xấp xỉ.

---

## 4. Ý nghĩa của hai hạng trong L2 loss

Biểu thức trên cho thấy L2 reconstruction loss tự nhiên tách thành hai thành phần:

### 4.1 Bias (mean) term

$$
(\mu^\top e)^2
= \big(\mathbb{E}[x]^\top (w-w_q)\big)^2.
$$

Hạng này đo lường sai lệch có hướng của weight quantized so với weight gốc, theo **mean activation**. Nó phản ánh việc rounding/quantization có tạo ra một sai số có hệ thống (systematic bias) trên output hay không.

### 4.2 Variance / covariance term

$$
e^\top \mathrm{Cov}(x) e.
$$

Hạng này đo lường mức độ mà sai số weight $e$ được khuếch đại bởi độ phân tán và tương quan của activation. Nó phụ thuộc vào năng lượng của activation cũng như các tương quan giữa các chiều đầu vào.

---

## 5. GPTQ dưới góc nhìn L2 decomposition

Các phương pháp như **GPTQ** có thể được hiểu là cố gắng tối thiểu hóa trực tiếp:

$$
e^\top \mathbb{E}[x x^\top] e,
$$

tức là tối ưu toàn bộ quadratic form dựa trên:

$$
H = \mathbb{E}[x x^\top].
$$

Do:

$$
\mathbb{E}[x x^\top] = \mu\mu^\top + \mathrm{Cov}(x),
$$

về mặt lý thuyết GPTQ đồng thời kiểm soát:

- bias term $(\mu^\top e)^2$,
- covariance term $e^\top \mathrm{Cov}(x) e$.

Tuy nhiên, việc xử lý full covariance (bao gồm các phần tử off-diagonal) dẫn đến chi phí tính toán lớn, khiến GPTQ thường chậm và phức tạp hơn trong thực tế.

---

## 7. Phương pháp correction dựa trên signed expectation

Khác với các phương pháp trên, phương pháp của chúng tôi tập trung trực tiếp vào **bias term** của L2 loss.

Cụ thể, chúng tôi xét objective tuyến tính theo signed expectation:

$$
\mathbb{E}[x]^\top (w-w_q)
= \sum_i \mathbb{E}[x_i] \, (w_i - w_{q,i}),
$$

và thực hiện các bước correction (flipping trong không gian integer) để đưa đại lượng này về gần $0$.

Từ decomposition ở Mục 3, việc tối thiểu hóa:

$$
\big|\mathbb{E}[x]^\top (w-w_q)\big|
$$

tương đương với việc trực tiếp giảm **bias term** $(\mu^\top e)^2$ trong L2 reconstruction loss.

---
