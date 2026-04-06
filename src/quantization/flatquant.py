from __future__ import annotations

import gc
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from src.quantization.state import IntegerQuantizedTensorState

if TYPE_CHECKING:
    from src.post_correction.bias_correction import BiasCorrectionCorrection
    from src.post_correction.smart_flip import SmartFlipCorrection


@dataclass
class FlatQuantConfig:
    w_bits: int = 4
    a_bits: int = 4
    q_bits: int = 16
    k_bits: int = 16
    v_bits: int = 16
    epochs: int = 15
    cali_bsz: int = 4
    flat_lr: float = 5e-3
    cali_trans: bool = True
    add_diag: bool = True
    lwc: bool = True
    lac: bool = True
    diag_init: str = "sq_style"
    diag_alpha: float = 0.3
    w_asym: bool = False
    deactive_amp: bool = False
    direct_inv: bool = False
    separate_vtrans: bool = False
    warmup: bool = False
    gptq_mse: bool = False
    w_groupsize: int = -1


class _FlatQuantLogger:
    def info(self, message):
        print(message)


class _ActivationCollector:
    def __init__(self, sample_limit: int, track_mean: bool, storage_dtype: torch.dtype = torch.float16):
        self.sample_limit = max(0, int(sample_limit))
        self.track_mean = track_mean
        self.storage_dtype = storage_dtype
        self._samples = []
        self._stored_rows = 0
        self._feature_dim = None
        self._sum = None
        self._count = 0

    def add(self, batch: torch.Tensor):
        if batch.numel() == 0:
            return

        rows = batch.detach().reshape(-1, batch.shape[-1]).cpu()
        if self._feature_dim is None:
            self._feature_dim = int(rows.shape[-1])

        if self.track_mean:
            rows_fp32 = rows.to(dtype=torch.float32)
            batch_sum = rows_fp32.sum(dim=0)
            self._sum = batch_sum if self._sum is None else self._sum + batch_sum
            self._count += int(rows_fp32.shape[0])

        if self._stored_rows >= self.sample_limit:
            return

        remaining = self.sample_limit - self._stored_rows
        if remaining <= 0:
            return

        kept = rows[:remaining].to(dtype=self.storage_dtype)
        self._samples.append(kept)
        self._stored_rows += int(kept.shape[0])

    def get_sample_rows(self):
        if not self._samples:
            if self._feature_dim is None:
                return None
            return torch.empty((0, self._feature_dim), dtype=torch.float32)
        return torch.cat(self._samples, dim=0).to(dtype=torch.float32)

    def get_mean(self):
        if not self.track_mean or self._sum is None or self._count == 0:
            return None
        return self._sum / self._count

class FlatQuantRTNQuantizer:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        config: Optional[FlatQuantConfig] = None,
        post_correction: Optional["SmartFlipCorrection | BiasCorrectionCorrection"] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or FlatQuantConfig()
        self.post_correction = post_correction
        self.layer_stats: Dict[str, dict] = {}
        self._imports = None
        self._prepared = False
        self._artifact_dir: str | None = None

        if not hasattr(self.model, "seqlen"):
            self.model.seqlen = 2048

        print("\n[FlatQuant RTN Quantizer Initialized]")
        print(f"  Config: {asdict(self.config)}")
        print(f"  Post correction: {type(self.post_correction).__name__ if self.post_correction else 'none'}")

    @staticmethod
    def empty_flip_stats() -> dict:
        return {"total": 0}

    @staticmethod
    def _flatquant_root() -> Path:
        return Path(__file__).resolve().parents[2] / "flatquant"

    def _ensure_flatquant_importable(self):
        flatquant_root = str(self._flatquant_root())
        if flatquant_root not in sys.path:
            sys.path.insert(0, flatquant_root)

    def _load_flatquant_imports(self):
        if self._imports is not None:
            return self._imports

        self._ensure_flatquant_importable()
        from flatquant.flat_utils import load_flat_parameters, reparameterize_model
        from flatquant.model_tools.llama31_utils import apply_flatquant_to_llama_31
        from flatquant.model_tools.llama_utils import apply_flatquant_to_llama
        from flatquant.model_tools.qwen_utils import apply_flatquant_to_qwen
        from src.quantization.flatquant_mistral import apply_flatquant_to_mistral
        from flatquant.train_utils import cali_flat_quant

        self._imports = {
            "load_flat_parameters": load_flat_parameters,
            "reparameterize_model": reparameterize_model,
            "apply_flatquant_to_llama": apply_flatquant_to_llama,
            "apply_flatquant_to_llama_31": apply_flatquant_to_llama_31,
            "apply_flatquant_to_qwen": apply_flatquant_to_qwen,
            "apply_flatquant_to_mistral": apply_flatquant_to_mistral,
            "cali_flat_quant": cali_flat_quant,
        }
        return self._imports

    def set_artifact_dir(self, artifact_dir: str | Path | None):
        self._artifact_dir = None if artifact_dir is None else str(artifact_dir)

    def _build_flatquant_args(self, nsamples: int) -> SimpleNamespace:
        model_name = getattr(self.model, "name_or_path", None) or getattr(self.model.config, "_name_or_path", "model")
        artifact_dir = self._artifact_dir or "./results/models/smart-flip-flatquant"
        return SimpleNamespace(
            model=model_name,
            seed=0,
            hf_token=None,
            a_bits=self.config.a_bits,
            a_groupsize=-1,
            a_asym=False,
            w_bits=self.config.w_bits,
            w_groupsize=self.config.w_groupsize,
            w_asym=self.config.w_asym,
            gptq=False,
            gptq_mse=self.config.gptq_mse,
            percdamp=0.01,
            act_order=False,
            epochs=self.config.epochs,
            nsamples=nsamples,
            cali_bsz=self.config.cali_bsz,
            flat_lr=self.config.flat_lr,
            cali_trans=self.config.cali_trans,
            add_diag=self.config.add_diag,
            lwc=self.config.lwc,
            lac=self.config.lac,
            resume=False,
            save_matrix=False,
            reload_matrix=False,
            matrix_path=None,
            diag_init=self.config.diag_init,
            diag_alpha=self.config.diag_alpha,
            warmup=self.config.warmup,
            deactive_amp=self.config.deactive_amp,
            direct_inv=self.config.direct_inv,
            separate_vtrans=self.config.separate_vtrans,
            q_bits=self.config.q_bits,
            q_asym=False,
            q_groupsize=-1,
            k_bits=self.config.k_bits,
            k_asym=False,
            k_groupsize=-1,
            v_bits=self.config.v_bits,
            v_asym=False,
            v_groupsize=-1,
            output_dir=artifact_dir,
            exp_name=Path(artifact_dir).name,
            lm_eval=False,
            tasks=[],
            lm_eval_batch_size=1,
            distribute_model=False,
            quantized_save=False,
            quantize=True,
            model_name=model_name.split("/")[-1],
            exp_dir=artifact_dir,
        )

    def _select_apply_fn(self):
        imports = self._load_flatquant_imports()
        model_name = (getattr(self.model, "name_or_path", None) or getattr(self.model.config, "_name_or_path", "")).lower()
        model_type = getattr(self.model.config, "model_type", "")

        if model_type == "qwen2":
            return imports["apply_flatquant_to_qwen"]
        if model_type == "mistral":
            return imports["apply_flatquant_to_mistral"]
        if model_type == "llama" and "3.1" in model_name:
            return imports["apply_flatquant_to_llama_31"]
        if model_type == "llama":
            return imports["apply_flatquant_to_llama"]
        raise ValueError(f"FlatQuant backend does not support model type: {model_type}")

    @staticmethod
    def _infer_wrapped_weight_device(module, projection_names):
        for projection_name in projection_names:
            projection = getattr(module, projection_name, None)
            linear = getattr(projection, "linear", None)
            weight = getattr(linear, "weight", None)
            device = getattr(weight, "device", None)
            if device is not None:
                return device
        return None

    @classmethod
    def _align_reused_flatquant_module_devices(cls, model):
        layers = getattr(getattr(model, "model", None), "layers", [])
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            attn_device = None if attn is None else cls._infer_wrapped_weight_device(attn, ("q_proj", "k_proj", "v_proj", "o_proj"))
            if attn is not None and attn_device is not None:
                layer.self_attn = attn.to(attn_device)
                input_layernorm = getattr(layer, "input_layernorm", None)
                if input_layernorm is not None and hasattr(input_layernorm, "to"):
                    layer.input_layernorm = input_layernorm.to(attn_device)

            mlp = getattr(layer, "mlp", None)
            mlp_device = None if mlp is None else cls._infer_wrapped_weight_device(mlp, ("up_proj", "gate_proj", "down_proj"))
            if mlp is not None and mlp_device is not None:
                layer.mlp = mlp.to(mlp_device)
                post_attention_layernorm = getattr(layer, "post_attention_layernorm", None)
                if post_attention_layernorm is not None and hasattr(post_attention_layernorm, "to"):
                    layer.post_attention_layernorm = post_attention_layernorm.to(mlp_device)

    def _build_calibration_loader(self, calibration_data: Iterable[str]):
        loader = []
        seqlen = int(getattr(self.model, "seqlen", 2048))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        for text in calibration_data:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=seqlen,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"]
            if input_ids.shape[1] < seqlen:
                pad = torch.full((1, seqlen - input_ids.shape[1]), pad_id, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, pad], dim=1)
            elif input_ids.shape[1] > seqlen:
                input_ids = input_ids[:, :seqlen]
            target = input_ids.clone()
            loader.append((input_ids, target))
        return loader

    def _prepare_model(self, calibration_loader, reuse_flat_parameters_path: str | None = None):
        if self._prepared:
            return

        imports = self._load_flatquant_imports()
        apply_fn = self._select_apply_fn()
        flatquant_args = self._build_flatquant_args(len(calibration_loader))
        self.model = apply_fn(flatquant_args, self.model)
        Path(flatquant_args.exp_dir).mkdir(parents=True, exist_ok=True)
        if reuse_flat_parameters_path:
            imports["load_flat_parameters"](flatquant_args, self.model, path=reuse_flat_parameters_path)
            self._align_reused_flatquant_module_devices(self.model)
        else:
            imports["cali_flat_quant"](flatquant_args, self.model, calibration_loader, self.device, logger=_FlatQuantLogger())
        imports["reparameterize_model"](self.model)
        self.model.eval()
        self._prepared = True

    def _get_layer_inputs(self, calibration_loader):
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        layers = self.model.model.layers
        model_device = self.device
        model_dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros((len(calibration_loader), self.model.seqlen, self.model.config.hidden_size), dtype=model_dtype, device=model_device)
        cache = {"i": 0, "attention_mask": None, "position_ids": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs.get("attention_mask")
                cache["position_ids"] = kwargs.get("position_ids")
                raise ValueError

        self.model.model.embed_tokens = self.model.model.embed_tokens.to(model_device)
        if hasattr(self.model.model, "norm"):
            self.model.model.norm = self.model.model.norm.to(model_device)
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.to(model_device)
        layers[0] = layers[0].to(model_device)
        layers[0] = Catcher(layers[0])

        with torch.no_grad():
            for batch in calibration_loader:
                try:
                    self.model(batch[0].to(model_device), use_cache=False)
                except ValueError:
                    pass

        layers[0] = layers[0].module.cpu()
        self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
        if hasattr(self.model.model, "norm"):
            self.model.model.norm = self.model.model.norm.cpu()
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.config.use_cache = use_cache
        return inps, cache["attention_mask"], cache["position_ids"]

    @staticmethod
    def _find_linear_modules(module, name=""):
        if type(module) is nn.Linear:
            return {name: module}
        result = {}
        for child_name, child in module.named_children():
            child_path = f"{name}.{child_name}" if name else child_name
            result.update(FlatQuantRTNQuantizer._find_linear_modules(child, child_path))
        return result

    @staticmethod
    def _extract_hidden_states(output):
        if isinstance(output, tuple):
            return output[0]
        return output

    @staticmethod
    def _flatten_activations(activation_batches) -> Optional[torch.Tensor]:
        if isinstance(activation_batches, _ActivationCollector):
            samples = activation_batches.get_sample_rows()
            if samples is None or samples.numel() == 0:
                return None
            return samples

        batches = [batch.reshape(-1, batch.shape[-1]) for batch in activation_batches if batch.numel() > 0]
        if not batches:
            return None
        return torch.cat(batches, dim=0)

    def _collect_subset_activations(self, layer, subset, inps, attention_mask, position_ids):
        sample_limit = 1024
        if self.post_correction is not None and hasattr(self.post_correction, "config"):
            sample_limit = max(sample_limit, int(getattr(self.post_correction.config, "max_samples", sample_limit)))
        track_mean = self.post_correction is not None and type(self.post_correction).__name__ != "BiasCorrectionCorrection"
        activation_data = {
            name: _ActivationCollector(sample_limit=sample_limit, track_mean=track_mean)
            for name in subset
        }

        def make_hook(name):
            def hook(_module, inputs, _output):
                inp = inputs[0] if isinstance(inputs, tuple) else inputs
                activation_data[name].add(inp)
            return hook

        handles = [module.register_forward_hook(make_hook(name)) for name, module in subset.items()]
        outs = torch.zeros_like(inps)
        with torch.no_grad():
            for idx in range(inps.shape[0]):
                layer_output = layer(
                    inps[idx].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                outs[idx] = self._extract_hidden_states(layer_output)
        for handle in handles:
            handle.remove()
        return activation_data, outs

    def _compute_output_mse(self, float_weight: torch.Tensor, quant_weight: torch.Tensor, activation_batches: Iterable[torch.Tensor]) -> float:
        activation_rows = self._flatten_activations(activation_batches)
        if activation_rows is None:
            return 0.0

        max_samples = min(1024, activation_rows.shape[0])
        if activation_rows.shape[0] > max_samples:
            activation_rows = activation_rows[:max_samples]
        x_samples = activation_rows.to(device=float_weight.device, dtype=float_weight.dtype)
        y_orig = torch.matmul(x_samples, float_weight.t())
        y_quant = torch.matmul(x_samples, quant_weight.t())
        return float((y_orig - y_quant).pow(2).mean().item())

    def _quantize_weight_rtn_raw(self, weight: torch.Tensor) -> IntegerQuantizedTensorState:
        out_features, in_features = weight.shape
        if self.config.w_asym:
            w_min = torch.minimum(weight.min(dim=1, keepdim=True)[0], torch.zeros((out_features, 1), device=weight.device, dtype=weight.dtype))
            w_max = torch.maximum(weight.max(dim=1, keepdim=True)[0], torch.zeros((out_features, 1), device=weight.device, dtype=weight.dtype))
            max_int = 2 ** self.config.w_bits - 1
            scale = (w_max - w_min).clamp(min=1e-5) / max_int
            zero_point = torch.round(-w_min / scale).clamp(0, max_int)
        else:
            maxq = 2 ** (self.config.w_bits - 1) - 1
            offset = maxq + 1
            max_int = 2 ** self.config.w_bits - 1
            scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / maxq
            zero_point = torch.full((out_features, 1), float(offset), device=weight.device, dtype=weight.dtype)

        scale_flat = scale.expand(out_features, in_features)
        zero_flat = zero_point.expand(out_features, in_features)
        pre_round = weight / scale_flat + zero_flat
        integer_weights = torch.round(pre_round).clamp(0, max_int)

        return IntegerQuantizedTensorState(
            float_weights=weight,
            pre_round=pre_round,
            integer_weights=integer_weights,
            scale=scale_flat,
            zero_point=zero_flat,
            max_int=max_int,
            in_features=in_features,
            padded_in_features=in_features,
            original_dtype=weight.dtype,
        )

    def _build_post_mean(self, activation_batches: Iterable[torch.Tensor], dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
        if self.post_correction is None:
            return None

        if isinstance(activation_batches, _ActivationCollector):
            raw_mean = activation_batches.get_mean()
            if raw_mean is None:
                return None
        else:
            activation_rows = self._flatten_activations(activation_batches)
            if activation_rows is None:
                return None
            raw_mean = activation_rows.mean(dim=0)

        prepared = self.post_correction.prepare_activation_means(raw_mean)
        return prepared.to(device=device, dtype=dtype)

    @property
    def bias_correction(self):
        return self.post_correction

    def _quantize_module(self, name: str, module: nn.Linear, activation_batches: Iterable[torch.Tensor]):
        float_weight = module.weight.data.clone()
        quant_state = self._quantize_weight_rtn_raw(float_weight)
        quant_weight = quant_state.dequantize_truncated()
        quant_error = self._compute_output_mse(float_weight, quant_weight, activation_batches)

        if self.post_correction is None:
            module.weight.data = quant_weight.to(module.weight.data.dtype)
            self.layer_stats[name] = {
                "error": quant_error,
                "outlier_percent": 0.0,
                "flip_stats": self.empty_flip_stats(),
            }
            return

        correction_name = type(self.post_correction).__name__
        if correction_name == "BiasCorrectionCorrection":
            bias_delta = self.bias_correction.compute_bias_delta(
                module,
                quant_weight,
                activation_batches,
                device=self.device,
            )
            module.weight.data = quant_weight.to(module.weight.data.dtype)
            self.bias_correction.apply_bias_delta(module, bias_delta, self.device, module.weight.data.dtype)
            self.layer_stats[name] = {
                "error": quant_error,
                "outlier_percent": 0.0,
                "flip_stats": {"total": 0, "bias_corrected": True},
                "bias_delta_norm": float(bias_delta.norm().item()),
            }
            return

        post_mean = self._build_post_mean(activation_batches, float_weight.dtype, float_weight.device)
        if post_mean is None:
            post_mean = torch.zeros(float_weight.shape[1], device=float_weight.device, dtype=float_weight.dtype)
        corrected_weight, outlier_percent, flip_stats = self.post_correction.apply(quant_state, post_mean)
        module.weight.data = corrected_weight.to(module.weight.data.dtype)
        self.layer_stats[name] = {
            "error": quant_error,
            "outlier_percent": outlier_percent,
            "flip_stats": flip_stats,
        }

    def quantize_model_sequential(self, calibration_data, n_samples: int = 128, reuse_flat_parameters_path: str | None = None):
        calibration_subset = list(calibration_data[:n_samples])
        calibration_loader = self._build_calibration_loader(calibration_subset)
        self._prepare_model(calibration_loader, reuse_flat_parameters_path=reuse_flat_parameters_path)

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        inps, attention_mask, position_ids = self._get_layer_inputs(calibration_loader)
        outs = torch.zeros_like(inps)
        layers = self.model.model.layers

        print("\n" + "=" * 80)
        print("FlatQuant RTN Sequential Quantization")
        print("=" * 80)
        print(f"  Weight bits: {self.config.w_bits}")
        print(f"  Activation bits: {self.config.a_bits}")
        print("  KV cache quantization: disabled")

        for layer_idx in tqdm(range(len(layers)), desc="Quantizing FlatQuant layers"):
            layer = layers[layer_idx].to(self.device)
            full = self._find_linear_modules(layer)
            subset = {name: module for name, module in full.items() if name.endswith(".linear")}
            activation_data, _ = self._collect_subset_activations(layer, subset, inps, attention_mask, position_ids)

            for name, module in subset.items():
                full_name = f"model.layers.{layer_idx}.{name}"
                self._quantize_module(full_name, module, activation_data.get(name, []))

            with torch.no_grad():
                for sample_idx in range(inps.shape[0]):
                    layer_output = layer(
                        inps[sample_idx].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )
                    outs[sample_idx] = self._extract_hidden_states(layer_output)

            layers[layer_idx] = layer.cpu()
            del layer
            inps, outs = outs, inps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        self.model.config.use_cache = use_cache
        self.model.eval()

    def build_evaluation_target(self) -> dict:
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
        }


