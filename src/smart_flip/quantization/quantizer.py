"""
Refactored quantizer that stays close to awq_js_xl.py while separating:
- AWQ raw alpha search
- AWQ raw quantization
- Smart flip post-correction
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def find_knee_point(values, tolerance_offset: float = 0.0) -> int:
    n = len(values)
    if n < 3:
        return n // 2

    if torch.is_tensor(values):
        y = values.detach().cpu().float().numpy()
    else:
        y = np.asarray(values, dtype=np.float32)

    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-10:
        return n // 2

    y_norm = (y - y_min) / (y_max - y_min)
    x_norm = np.linspace(0.0, 1.0, n)
    y_line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    distances = np.abs(y_norm - y_line)

    knee_idx = int(np.argmax(distances))
    if knee_idx < n - 1:
        offset_indices = int(tolerance_offset * n)
        knee_idx = min(knee_idx + offset_indices, n - 1)
        knee_idx = max(knee_idx, 0)
    return knee_idx


def compute_james_stein_mean(raw_means: torch.Tensor, variance_estimate: Optional[torch.Tensor] = None) -> torch.Tensor:
    p = len(raw_means)
    if p < 3:
        return raw_means

    grand_mean = raw_means.mean()
    deviations = raw_means - grand_mean
    sum_sq_dev = (deviations ** 2).sum()
    if sum_sq_dev < 1e-10:
        return raw_means

    if variance_estimate is None:
        variance_estimate = ((raw_means - grand_mean).abs().mean()) ** 2
        variance_estimate = variance_estimate.clamp(min=1e-8)

    shrinkage_factor = ((p - 2) * variance_estimate) / sum_sq_dev
    shrinkage_factor = shrinkage_factor.clamp(0, 1)
    return grand_mean + (1 - shrinkage_factor) * deviations


@dataclass
class QuantizationConfig:
    bits: int = 4
    n_grid: int = 20
    group_size: int = 128
    use_flip: bool = True
    knee_tolerance: float = 0.0
    max_tokens_per_sample: int = 2048
    layer_batch_size: int = 16
    lmhead_chunks: int = 4
    max_flip_percent: float = 0.05
    use_james_stein: bool = True


class SmartFlipAWQQuantizerXL:
    def __init__(self, model, tokenizer, device: str = "cuda", config: Optional[QuantizationConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or QuantizationConfig()
        self.activation_data: Dict[str, list] = {}
        self.layer_stats: Dict[str, dict] = {}

        print("\n[Smart Flip AWQ Quantizer XL Initialized]")
        print(f"  Config: {asdict(self.config)}")

    def get_hook(self, name: str):
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []

            inp = input[0] if isinstance(input, tuple) else input
            if inp.dim() == 3 and inp.shape[1] > self.config.max_tokens_per_sample:
                seq_len = inp.shape[1]
                indices = torch.randperm(seq_len)[: self.config.max_tokens_per_sample]
                indices = indices.sort()[0]
                inp = inp[:, indices, :]

            self.activation_data[name].append(inp.detach().cpu().float())

        return hook

    @torch.no_grad()
    def get_activation_stats(self, name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if name not in self.activation_data or not self.activation_data[name]:
            return None, None

        x_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in x_list)
        in_features = x_list[0].shape[-1]

        l2_sum = torch.zeros(in_features, dtype=torch.float32)
        mean_sum = torch.zeros(in_features, dtype=torch.float32)

        for x in x_list:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            l2_sum += x_flat.pow(2).sum(dim=0)
            mean_sum += x_flat.sum(dim=0)

        salience = l2_sum / total_samples
        raw_mean = mean_sum / total_samples
        js_mean = compute_james_stein_mean(raw_mean) if self.config.use_james_stein else raw_mean
        return salience, js_mean

    @torch.no_grad()
    def compute_dynamic_outlier_threshold(self, activation_means: torch.Tensor, debug: bool = False):
        sorted_means, _ = torch.sort(activation_means.abs(), descending=True)
        n = len(sorted_means)
        first_half = sorted_means[: n // 2]

        if len(first_half) < 3:
            threshold_idx = int(0.05 * n)
            threshold = sorted_means[threshold_idx].item()
            return threshold, 0.05

        knee_idx = find_knee_point(first_half, tolerance_offset=self.config.knee_tolerance)
        threshold = sorted_means[knee_idx].item()
        num_outliers = (activation_means.abs() >= threshold).sum().item()
        outlier_percent = num_outliers / n

        if debug:
            print(f"    DEBUG threshold={threshold:.6f}, outlier_percent={outlier_percent*100:.2f}%")

        return threshold, outlier_percent

    @torch.no_grad()
    def quantize_weight_groupwise(self, W: torch.Tensor, group_activation_means: torch.Tensor, apply_flip: bool = False, debug: bool = False):
        out_features, in_features = W.shape
        device = W.device

        n_groups = (in_features + self.config.group_size - 1) // self.config.group_size
        padded_in_features = n_groups * self.config.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=W.dtype)
            W_padded[:, :in_features] = W
            act_padded = torch.zeros(padded_in_features, device=device, dtype=W.dtype)
            act_padded[:in_features] = group_activation_means
        else:
            W_padded = W
            act_padded = group_activation_means

        W_g = W_padded.reshape(out_features, n_groups, self.config.group_size)
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        max_int = 2 ** self.config.bits - 1

        scale = (w_max - w_min) / max_int
        scale = scale.clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        scale_flat = scale.repeat(1, 1, self.config.group_size).reshape(out_features, padded_in_features)
        zp_flat = zp.repeat(1, 1, self.config.group_size).reshape(out_features, padded_in_features)

        W_div = W_padded / scale_flat
        W_int = torch.round(W_div + zp_flat).clamp(0, max_int)
        W_quant = (W_int - zp_flat) * scale_flat

        flip_stats = {
            "total": 0,
            "per_channel_mean": 0.0,
            "per_channel_median": 0.0,
            "per_channel_min": 0.0,
            "per_channel_max": 0.0,
            "per_channel_std": 0.0,
            "per_channel_p25": 0.0,
            "per_channel_p75": 0.0,
            "per_channel_p90": 0.0,
            "per_channel_p95": 0.0,
            "per_channel_p99": 0.0,
            "per_channel_zero_pct": 100.0,
        }

        if not apply_flip:
            W_dequant = (W_int - zp_flat) * scale_flat
            if padded_in_features > in_features:
                W_dequant = W_dequant[:, :in_features]
            return W_dequant.to(W.dtype), None, flip_stats

        W_diff = W_padded - W_quant
        current_error = (W_diff * act_padded.unsqueeze(0)).sum(dim=1)

        flip_dir = torch.sign(W_div + zp_flat - W_int)
        flip_dir[flip_dir == 0] = 1.0
        flip_impacts = act_padded.unsqueeze(0) * flip_dir * scale_flat

        target_sign = torch.sign(current_error).unsqueeze(1)
        valid_mask = torch.sign(flip_impacts) == target_sign

        w_int_proposed = W_int + flip_dir
        in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
        valid_mask = valid_mask & in_range

        outlier_threshold, outlier_percent = self.compute_dynamic_outlier_threshold(act_padded, debug=debug)
        is_outlier = act_padded.abs() > outlier_threshold
        valid_mask = valid_mask & (~is_outlier).unsqueeze(0)

        rounding_costs = (W_div + zp_flat - W_int).abs()
        rounding_costs_masked = rounding_costs.clone()
        rounding_costs_masked[~valid_mask] = -1.0

        sorted_indices = torch.argsort(rounding_costs_masked, dim=1, descending=True)
        sorted_impacts = torch.gather(flip_impacts, 1, sorted_indices)
        sorted_validity = torch.gather(valid_mask.long(), 1, sorted_indices)
        sorted_impacts = sorted_impacts * sorted_validity

        cumsum_impacts = torch.cumsum(sorted_impacts, dim=1)
        residuals = torch.abs(current_error.unsqueeze(1) - cumsum_impacts)
        all_residuals = torch.cat([torch.abs(current_error).unsqueeze(1), residuals], dim=1)
        best_k = torch.argmin(all_residuals, dim=1)

        idx_range = torch.arange(padded_in_features, device=device).unsqueeze(0)
        flip_mask_sorted = idx_range < best_k.unsqueeze(1)
        final_flips_sorted = flip_mask_sorted & sorted_validity.bool()

        sorted_flip_dir = torch.gather(flip_dir, 1, sorted_indices)
        sorted_flip_dir[~final_flips_sorted] = 0.0

        max_flips_per_output = int(self.config.max_flip_percent * in_features)
        cumsum_flips = final_flips_sorted.long().cumsum(dim=1)
        within_limit = cumsum_flips <= max_flips_per_output
        sorted_flip_dir[~within_limit] = 0.0

        W_int.scatter_add_(1, sorted_indices, sorted_flip_dir)
        W_int.clamp_(0, max_int)

        num_flips_total = int((sorted_flip_dir != 0).sum().item())
        flips_per_channel = (sorted_flip_dir != 0).sum(dim=0).float()
        if padded_in_features > in_features:
            flips_per_channel = flips_per_channel[:in_features]

        if flips_per_channel.numel() > 0:
            flip_stats = {
                "total": num_flips_total,
                "per_channel_mean": flips_per_channel.mean().item(),
                "per_channel_median": flips_per_channel.median().item(),
                "per_channel_min": flips_per_channel.min().item(),
                "per_channel_max": flips_per_channel.max().item(),
                "per_channel_std": flips_per_channel.std().item(),
                "per_channel_p25": torch.quantile(flips_per_channel, 0.25).item(),
                "per_channel_p75": torch.quantile(flips_per_channel, 0.75).item(),
                "per_channel_p90": torch.quantile(flips_per_channel, 0.90).item(),
                "per_channel_p95": torch.quantile(flips_per_channel, 0.95).item(),
                "per_channel_p99": torch.quantile(flips_per_channel, 0.99).item(),
                "per_channel_zero_pct": (flips_per_channel == 0).float().mean().item() * 100,
            }

        W_dequant = (W_int - zp_flat) * scale_flat
        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]
        return W_dequant.to(W.dtype), outlier_percent, flip_stats

    @torch.no_grad()
    def search_best_scale(self, name: str, module: nn.Linear):
        if name not in self.activation_data or not self.activation_data[name]:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience, js_mean = self.get_activation_stats(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience = activation_salience.to(self.device).to(module.weight.dtype)
        js_mean = js_mean.to(self.device).to(module.weight.dtype)

        x_list = self.activation_data[name]
        x_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in x_list], dim=0)
        max_samples = min(2048, x_cpu.shape[0])
        if x_cpu.shape[0] > max_samples:
            indices = torch.randperm(x_cpu.shape[0])[:max_samples]
            x_search = x_cpu[indices].to(self.device)
        else:
            x_search = x_cpu.to(self.device)

        del x_cpu

        if x_search.dtype != module.weight.dtype:
            x_search = x_search.to(module.weight.dtype)

        W = module.weight.data
        y_orig = torch.matmul(x_search, W.t())

        best_error = float("inf")
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        activation_salience = activation_salience.clamp(min=1e-5)
        for grid_idx in range(self.config.n_grid + 1):
            alpha = grid_idx / self.config.n_grid
            scales = activation_salience.pow(alpha)
            W_scaled = W * scales.unsqueeze(0)
            scaled_act_mean = js_mean / scales

            W_quant, _, _ = self.quantize_weight_groupwise(
                W_scaled,
                scaled_act_mean,
                apply_flip=False,
            )

            W_recon = W_quant / scales.unsqueeze(0)
            y_quant = torch.matmul(x_search, W_recon.t())
            error = (y_orig - y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

            del W_scaled, W_quant, W_recon, y_quant, scales

        del x_search, y_orig
        torch.cuda.empty_cache()
        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_layer(self, name: str, module: nn.Linear):
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data
        original_dtype = W.dtype
        W_scaled = W * best_scales.unsqueeze(0)

        _, js_mean = self.get_activation_stats(name)
        if js_mean is not None:
            scaled_act_mean = js_mean.to(self.device).to(W.dtype) / best_scales
        else:
            scaled_act_mean = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)

        W_quant, outlier_pct, flip_stats = self.quantize_weight_groupwise(
            W_scaled,
            scaled_act_mean,
            apply_flip=self.config.use_flip,
        )

        W_final = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)
        module.weight.data = W_final

        self.layer_stats[name] = {
            "alpha": best_alpha,
            "error": best_error,
            "outlier_percent": outlier_pct if outlier_pct is not None else 0.0,
            "flip_stats": flip_stats,
        }

        del best_scales, scaled_act_mean, W_scaled, W_quant, W_final
        if name in self.activation_data:
            del self.activation_data[name]
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def quantize_lmhead_half_by_half(self, name: str, module: nn.Linear, debug: bool = False, num_chunks: int = 4):
        W = module.weight.data
        original_dtype = W.dtype
        out_features, in_features = W.shape

        _, js_mean = self.get_activation_stats(name)
        if js_mean is None:
            js_mean = torch.zeros(in_features, device=self.device, dtype=W.dtype)
        else:
            js_mean = js_mean.to(self.device).to(W.dtype)

        chunk_size = out_features // num_chunks
        chunk_boundaries = [
            (i * chunk_size, out_features if i == num_chunks - 1 else (i + 1) * chunk_size)
            for i in range(num_chunks)
        ]

        W_final_chunks = []
        chunk_stats = []
        for start_idx, end_idx in chunk_boundaries:
            W_chunk = W[start_idx:end_idx, :]
            activation_salience, _ = self.get_activation_stats(name)
            activation_salience = activation_salience.to(self.device).to(W.dtype).clamp(min=1e-5)

            x_list = self.activation_data[name]
            x_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in x_list], dim=0)
            max_samples = min(1024, x_cpu.shape[0])
            if x_cpu.shape[0] > max_samples:
                indices = torch.randperm(x_cpu.shape[0])[:max_samples]
                x_search = x_cpu[indices].to(self.device)
            else:
                x_search = x_cpu.to(self.device)
            if x_search.dtype != W.dtype:
                x_search = x_search.to(W.dtype)

            y_orig = torch.matmul(x_search, W_chunk.t())
            best_error = float("inf")
            best_alpha = 0.0
            best_scales = torch.ones(W_chunk.shape[1], device=self.device)

            for grid_idx in range(self.config.n_grid + 1):
                alpha = grid_idx / self.config.n_grid
                scales = activation_salience.pow(alpha)
                W_scaled = W_chunk * scales.unsqueeze(0)
                scaled_act_mean = js_mean / scales

                W_quant, _, _ = self.quantize_weight_groupwise(
                    W_scaled,
                    scaled_act_mean,
                    apply_flip=False,
                    debug=(debug and grid_idx == 0),
                )
                W_recon = W_quant / scales.unsqueeze(0)
                y_quant = torch.matmul(x_search, W_recon.t())
                error = (y_orig - y_quant).pow(2).mean().item()

                if error < best_error:
                    best_error = error
                    best_alpha = alpha
                    best_scales = scales.clone()

                del W_scaled, W_quant, W_recon, y_quant, scales

            W_scaled = W_chunk * best_scales.unsqueeze(0)
            scaled_act_mean = js_mean / best_scales
            W_quant, outlier_pct, flip_stats = self.quantize_weight_groupwise(
                W_scaled,
                scaled_act_mean,
                apply_flip=self.config.use_flip,
            )
            W_final_chunk = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)

            W_final_chunks.append(W_final_chunk)
            chunk_stats.append(
                {
                    "alpha": best_alpha,
                    "error": best_error,
                    "outlier_percent": outlier_pct if outlier_pct is not None else 0.0,
                    "flip_stats": flip_stats,
                }
            )

            del x_search, y_orig, W_chunk, W_scaled, W_quant, W_final_chunk, scaled_act_mean
            torch.cuda.empty_cache()

        module.weight.data = torch.cat(W_final_chunks, dim=0)
        self.layer_stats[name] = {
            "alpha": float(np.mean([s["alpha"] for s in chunk_stats])),
            "error": float(np.mean([s["error"] for s in chunk_stats])),
            "outlier_percent": float(np.mean([s["outlier_percent"] for s in chunk_stats])),
            "flip_stats": {"total": int(sum(s["flip_stats"]["total"] for s in chunk_stats))},
        }

        del W_final_chunks, chunk_stats, js_mean
        torch.cuda.empty_cache()

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples: int = 500):
        print(f"  Calibrating {len(layer_names_batch)} layers...")
        self.model.eval()
        handles = []

        for name, module in layer_names_batch:
            handle = module.register_forward_hook(self.get_hook(name))
            handles.append(handle)

        successful = 0
        with torch.no_grad():
            for text in tqdm(calibration_data[:n_samples], desc="  Calibration", leave=False):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1

                    if (successful + 1) % 32 == 0:
                        torch.cuda.empty_cache()
                except Exception:
                    continue

        for handle in handles:
            handle.remove()

        torch.cuda.empty_cache()
        gc.collect()

    def quantize_model_sequential(self, calibration_data, n_samples: int = 500):
        print("\n" + "=" * 80)
        print("Batched Sequential Quantization")
        print("=" * 80)
        print(f"  Strategy: Process {self.config.layer_batch_size} layers per batch")

        if HAS_PSUTIL:
            print(f"  Initial System RAM: {psutil.virtual_memory().percent:.1f}%")

        layer_names = [(name, module) for name, module in self.model.named_modules() if isinstance(module, nn.Linear)]
        num_layers = len(layer_names)
        num_batches = (num_layers + self.config.layer_batch_size - 1) // self.config.layer_batch_size

        quantized_count = 0
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.config.layer_batch_size
            batch_end = min(batch_start + self.config.layer_batch_size, num_layers)
            batch_layers = layer_names[batch_start:batch_end]

            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Layers {batch_start}-{batch_end - 1}")
            self.calibrate_layer_batch(batch_layers, calibration_data, n_samples)

            print(f"  Quantizing {len(batch_layers)} layers...")
            for name, module in tqdm(batch_layers, desc="  Quantization", leave=False):
                try:
                    is_lmhead = "lm_head" in name.lower() or name.endswith("lm_head")
                    if is_lmhead:
                        self.quantize_lmhead_half_by_half(
                            name,
                            module,
                            debug=(quantized_count < 2),
                            num_chunks=self.config.lmhead_chunks,
                        )
                    else:
                        self.quantize_layer(name, module)
                    quantized_count += 1
                except Exception as exc:
                    print(f"\nWarning: error quantizing {name}: {exc}")
                    continue

            self.activation_data = {}
            torch.cuda.empty_cache()
            gc.collect()

        print("\n" + "=" * 80)
        print("Quantization Complete")
        print(f"  Total layers quantized: {quantized_count}/{num_layers}")
        print("=" * 80)
