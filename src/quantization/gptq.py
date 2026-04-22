from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import transformers

from src.post_correction.bias_correction import BiasCorrectionCorrection
from src.post_correction.smart_flip import SmartFlipCorrection
from src.quantization.state import IntegerQuantizedTensorState


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=False,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                elif len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]

        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = 1

        self.scale = (xmax - xmin) / self.maxq
        self.scale = self.scale.clamp(min=1e-8)
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                scale1 = scale1.clamp(min=1e-8)
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q = (q - x).abs_().pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.perchannel:
            tmp = shape[0] if weight else shape[-1]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return

        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        elif len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def ready(self):
        return torch.all(self.scale != 0)


@dataclass
class GPTQConfig:
    bits: int = 4
    group_size: int = -1
    percdamp: float = 0.01
    sym: bool = False
    act_order: bool = False
    true_sequential: bool = True
    static_groups: bool = False
    mse: bool = False
    max_bias_samples: int = 4096


class GPTQ:
    def __init__(
        self,
        layer,
        smart_flip: Optional[SmartFlipCorrection] = None,
        bias_correction: Optional[BiasCorrectionCorrection] = None,
        max_bias_samples: int = 4096,
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.smart_flip = smart_flip
        self.bias_correction = bias_correction
        self.max_bias_samples = max_bias_samples

        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.activation_sums = torch.zeros(self.columns, dtype=torch.float64)
        self.activation_count = 0
        self.activation_samples = None
        self.last_stats = {
            "outlier_percent": 0.0,
            "flip_stats": {"total": 0},
            "bias_delta_norm": 0.0,
        }
        self.dead = None

    def _reshape_inputs(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            return inp.t(), tmp, inp

        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2]).flatten(1)
            return inp, tmp, inp.t()

        return inp, tmp, None

    def _update_activation_stats(self, activation_rows):
        if activation_rows is None or activation_rows.numel() == 0:
            return

        rows_cpu = activation_rows.detach().reshape(-1, activation_rows.shape[-1]).float().cpu()
        self.activation_sums += rows_cpu.sum(dim=0, dtype=torch.float64)
        self.activation_count += rows_cpu.shape[0]

        if self.bias_correction is None or self.max_bias_samples <= 0:
            return

        if self.activation_samples is None:
            take = min(self.max_bias_samples, rows_cpu.shape[0])
            if rows_cpu.shape[0] > take:
                indices = torch.randperm(rows_cpu.shape[0])[:take]
                rows_cpu = rows_cpu[indices]
            self.activation_samples = rows_cpu
            return

        combined = torch.cat([self.activation_samples, rows_cpu], dim=0)
        if combined.shape[0] > self.max_bias_samples:
            indices = torch.randperm(combined.shape[0])[: self.max_bias_samples]
            combined = combined[indices]
        self.activation_samples = combined

    def add_batch(self, inp, out):
        del out
        inp, tmp, activation_rows = self._reshape_inputs(inp)
        self._update_activation_stats(activation_rows)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def _build_quant_state(self, original_weights, pre_round, integer_weights, scale, zero):
        return IntegerQuantizedTensorState(
            float_weights=original_weights,
            pre_round=pre_round,
            integer_weights=integer_weights,
            scale=scale,
            zero_point=zero,
            max_int=int(self.quantizer.maxq.item()),
            min_int=0,
            in_features=original_weights.shape[1],
            padded_in_features=original_weights.shape[1],
            original_dtype=self.layer.weight.data.dtype,
        )

    def _run_post_correction(self, original_weights, quant_weights, pre_round, integer_weights, scale, zero, dead_mask=None):
        corrected_weights = quant_weights
        outlier_percent = 0.0
        flip_stats = {"total": 0}
        bias_delta_norm = 0.0

        if self.smart_flip is not None and self.activation_count > 0:
            act_mean = (self.activation_sums / self.activation_count).to(device=self.dev, dtype=scale.dtype)
            if dead_mask is not None and torch.any(dead_mask):
                act_mean = act_mean.clone()
                act_mean[dead_mask] = 0
            act_mean = self.smart_flip.prepare_activation_means(act_mean)
            quant_state = self._build_quant_state(original_weights, pre_round, integer_weights, scale, zero)
            corrected_weights, outlier_percent, flip_stats = self.smart_flip.apply(quant_state, act_mean)
            if dead_mask is not None and torch.any(dead_mask):
                corrected_weights[:, dead_mask] = quant_weights[:, dead_mask]

        if self.bias_correction is not None and isinstance(self.layer, nn.Linear):
            bias_delta = self.bias_correction.compute_bias_delta(
                self.layer,
                corrected_weights,
                self.activation_samples,
                device=str(self.dev),
            )
            self.bias_correction.apply_bias_delta(
                self.layer,
                bias_delta,
                device=str(self.dev),
                dtype=self.layer.weight.data.dtype,
            )
            bias_delta_norm = float(bias_delta.norm().item())

        self.last_stats = {
            "outlier_percent": outlier_percent,
            "flip_stats": flip_stats,
            "bias_delta_norm": bias_delta_norm,
        }
        return corrected_weights

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        W_orig = W.clone()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        self.dead = dead.clone()
        H[dead, dead] = 1
        W[:, dead] = 0
        W_orig[:, dead] = 0

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            W_orig = W_orig[:, perm]
            dead = dead[perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        Q_pre = torch.zeros_like(W)
        Q_int = torch.zeros_like(W)
        Q_scale = torch.zeros_like(W)
        Q_zero = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Qpre1 = torch.zeros_like(W1)
            Qint1 = torch.zeros_like(W1)
            Qscale1 = torch.zeros_like(W1)
            Qzero1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                scale_col = self.quantizer.scale.flatten().to(w.dtype)
                zero_col = self.quantizer.zero.flatten().to(w.dtype)
                pre_col = w / scale_col + zero_col
                int_col = torch.clamp(torch.round(pre_col), 0, int(self.quantizer.maxq.item()))
                q = quantize(
                    w.unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                ).flatten()

                Q1[:, i] = q
                Qpre1[:, i] = pre_col
                Qint1[:, i] = int_col
                Qscale1[:, i] = scale_col
                Qzero1[:, i] = zero_col
                losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Q_pre[:, i1:i2] = Qpre1
            Q_int[:, i1:i2] = Qint1
            Q_scale[:, i1:i2] = Qscale1
            Q_zero[:, i1:i2] = Qzero1
            losses[:, i1:i2] = losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print(f"time {time.time() - tick:.2f}")
        print("error", torch.sum(losses).item())

        if actorder:
            Q = Q[:, invperm]
            Q_pre = Q_pre[:, invperm]
            Q_int = Q_int[:, invperm]
            Q_scale = Q_scale[:, invperm]
            Q_zero = Q_zero[:, invperm]
            W_orig = W_orig[:, invperm]
            dead = dead[invperm]

        if self.smart_flip is not None or self.bias_correction is not None:
            Q = self._run_post_correction(W_orig, Q, Q_pre, Q_int, Q_scale, Q_zero, dead_mask=dead)

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        self.activation_sums = None
        self.activation_samples = None
        torch.cuda.empty_cache()


class GPTQQuantizer:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        config: Optional[GPTQConfig] = None,
        post_correction: Optional[object] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or GPTQConfig()
        self.post_correction = post_correction
        self.layer_stats: Dict[str, dict] = {}

        print("\n[GPTQ Quantizer Initialized]")
        print(f"  Config: {self.config}")
        print(f"  Post correction: {type(self.post_correction).__name__ if self.post_correction else 'none'}")

    def _get_layers(self):
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise ValueError("GPTQ backend currently supports decoder-only models with model.model.layers")
        return self.model.model.layers

    def _move_model_inputs(self, dev):
        model_body = self.model.model
        if hasattr(model_body, "embed_tokens") and model_body.embed_tokens is not None:
            model_body.embed_tokens = model_body.embed_tokens.to(dev)
        if hasattr(model_body, "norm") and model_body.norm is not None:
            model_body.norm = model_body.norm.to(dev)
        if hasattr(model_body, "rotary_emb") and model_body.rotary_emb is not None:
            model_body.rotary_emb = model_body.rotary_emb.to(dev)

    @staticmethod
    def _cleanup_cuda():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _prepare_layer_kwargs(kwargs):
        prepared = {}
        for key, value in kwargs.items():
            if key == "use_cache":
                continue
            if isinstance(value, torch.Tensor):
                prepared[key] = value
            else:
                prepared[key] = value
        return prepared

    def _get_sequential_groups(self, full):
        default = [list(full.keys())]
        preferred = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
        if not self.config.true_sequential:
            return default
        if all(all(name in full for name in group) for group in preferred):
            return preferred
        return default

    @staticmethod
    def find_layers(module, layers=(nn.Linear,), name=""):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(
                GPTQQuantizer.find_layers(
                    child,
                    layers=layers,
                    name=name + "." + name1 if name != "" else name1,
                )
            )
        return res

    @torch.no_grad()
    def quantize_model_sequential(self, calibration_data, n_samples: int = 128):
        del n_samples
        print("\n" + "=" * 80)
        print("GPTQ Sequential Quantization")
        print("=" * 80)

        if not calibration_data:
            raise ValueError("GPTQ requires tensor calibration data")

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        layers = self._get_layers()

        self._move_model_inputs(self.device)
        layers[0] = layers[0].to(self.device)

        dtype = next(iter(self.model.parameters())).dtype
        nsamples = len(calibration_data)
        hidden_size = self.model.config.hidden_size
        seqlen = calibration_data[0].shape[1]
        inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=self.device)
        cache = {"i": 0, "kwargs": {}}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["kwargs"] = kwargs
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in calibration_data:
            try:
                self.model(batch.to(self.device))
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        self._move_model_inputs("cpu")
        self._cleanup_cuda()

        outs = torch.zeros_like(inps)
        layer_kwargs = self._prepare_layer_kwargs(cache["kwargs"])

        for layer_idx in range(len(layers)):
            print(f"\n[Layer {layer_idx + 1}/{len(layers)}]")
            layer = layers[layer_idx].to(self.device)
            full = self.find_layers(layer)

            for group in self._get_sequential_groups(full):
                subset = {name: full[name] for name in group}
                gptq_layers = {}
                for name, module in subset.items():
                    smart_flip = self.post_correction if isinstance(self.post_correction, SmartFlipCorrection) else None
                    bias_correction = self.post_correction if isinstance(self.post_correction, BiasCorrectionCorrection) else None
                    gptq_layer = GPTQ(
                        module,
                        smart_flip=smart_flip,
                        bias_correction=bias_correction,
                        max_bias_samples=self.config.max_bias_samples,
                    )
                    gptq_layer.quantizer = Quantizer()
                    gptq_layer.quantizer.configure(
                        self.config.bits,
                        perchannel=True,
                        sym=self.config.sym,
                        mse=self.config.mse,
                    )
                    gptq_layers[name] = gptq_layer

                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq_layers[name].add_batch(inp[0].data, out.data)
                    return tmp

                handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
                for sample_idx in range(nsamples):
                    outs[sample_idx] = layer(inps[sample_idx].unsqueeze(0), **layer_kwargs)[0]
                for handle in handles:
                    handle.remove()

                for name, gptq_layer in gptq_layers.items():
                    print(f"  Quantizing {name}")
                    gptq_layer.fasterquant(
                        percdamp=self.config.percdamp,
                        groupsize=self.config.group_size,
                        actorder=self.config.act_order,
                        static_groups=self.config.static_groups,
                    )
                    self.layer_stats[f"model.layers.{layer_idx}.{name}"] = gptq_layer.last_stats
                    gptq_layer.free()

            for sample_idx in range(nsamples):
                outs[sample_idx] = layer(inps[sample_idx].unsqueeze(0), **layer_kwargs)[0]

            layers[layer_idx] = layer.cpu()
            del layer
            self._cleanup_cuda()
            inps, outs = outs, inps

        self.model.config.use_cache = use_cache
        print("\n" + "=" * 80)
        print("GPTQ Quantization Complete")
        print("=" * 80)
