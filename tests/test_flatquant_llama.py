import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


class FlatQuantLlamaWrapperTests(unittest.TestCase):
    def _install_stub_modules(self):
        backups = {
            name: sys.modules.get(name)
            for name in [
                "flatquant",
                "flatquant.quant_utils",
                "flatquant.utils",
                "flatquant.function_utils",
                "flatquant.trans_utils",
                "flatquant.flat_linear",
                "transformers",
                "transformers.models",
                "transformers.models.llama",
                "transformers.models.llama.modeling_llama",
            ]
        }

        flatquant_pkg = types.ModuleType("flatquant")
        quant_utils = types.ModuleType("flatquant.quant_utils")
        utils = types.ModuleType("flatquant.utils")
        function_utils = types.ModuleType("flatquant.function_utils")
        trans_utils = types.ModuleType("flatquant.trans_utils")
        flat_linear = types.ModuleType("flatquant.flat_linear")

        class ActivationQuantizer:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, x):
                return x

        class IdentityTrans(nn.Module):
            def __init__(self, *args, add_diag=False, **kwargs):
                super().__init__()
                self.dim = args[0] if args else 1
                self.add_diag = add_diag
                self.use_diag = add_diag
                self.diag_scale = nn.Parameter(torch.ones(self.dim))

            def forward(self, x, inv_t=False):
                return x

            def to_eval_mode(self):
                return None

            def get_matrix(self, inv_t=False):
                return torch.eye(self.dim)

        class FlatQuantizedLinear(nn.Module):
            def __init__(self, args, linear):
                super().__init__()
                self.linear = linear

            def forward(self, x, qa_trans=None, out_trans=None):
                return self.linear(x)

            def _ori_forward(self, x):
                return self.linear(x)

            def reparameterize(self, qa_trans=None, out_trans=None):
                return None

        quant_utils.ActivationQuantizer = ActivationQuantizer
        utils.skip_initialization = lambda: None
        function_utils.get_init_scale = lambda w, a, alpha: torch.ones_like(w)
        function_utils.get_decompose_dim = lambda dim: (dim, dim)
        trans_utils.SVDSingleTransMatrix = IdentityTrans
        trans_utils.SVDDecomposeTransMatrix = IdentityTrans
        trans_utils.InvSingleTransMatrix = IdentityTrans
        trans_utils.InvDecomposeTransMatrix = IdentityTrans
        flat_linear.FlatQuantizedLinear = FlatQuantizedLinear

        transformers_pkg = types.ModuleType("transformers")
        transformers_models = types.ModuleType("transformers.models")
        llama_pkg = types.ModuleType("transformers.models.llama")
        modeling = types.ModuleType("transformers.models.llama.modeling_llama")

        class LlamaMLP(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.act_fn = nn.SiLU()
                self.up_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                self.down_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        class RotaryEmbedding(nn.Module):
            def forward(self, x, position_ids):
                batch, seq_len = position_ids.shape
                cos = torch.ones(batch, seq_len, x.shape[-1], dtype=x.dtype, device=x.device)
                sin = torch.zeros(batch, seq_len, x.shape[-1], dtype=x.dtype, device=x.device)
                return cos, sin

        class LlamaAttention(nn.Module):
            def __init__(self, config, layer_idx):
                super().__init__()
                self.config = config
                self.layer_idx = layer_idx
                self.hidden_size = config.hidden_size
                self.head_dim = config.hidden_size // config.num_attention_heads
                self.num_heads = config.num_attention_heads
                self.num_key_value_heads = config.num_key_value_heads
                self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
                self.attention_dropout = config.attention_dropout
                self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
                self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
                self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
                self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
                self.rotary_emb = RotaryEmbedding()

        def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
            return q, k

        def repeat_kv(x, num_key_value_groups):
            return x.repeat_interleave(num_key_value_groups, dim=1)

        modeling.LlamaMLP = LlamaMLP
        modeling.LlamaAttention = LlamaAttention
        modeling.apply_rotary_pos_emb = apply_rotary_pos_emb
        modeling.repeat_kv = repeat_kv

        sys.modules["flatquant"] = flatquant_pkg
        sys.modules["flatquant.quant_utils"] = quant_utils
        sys.modules["flatquant.utils"] = utils
        sys.modules["flatquant.function_utils"] = function_utils
        sys.modules["flatquant.trans_utils"] = trans_utils
        sys.modules["flatquant.flat_linear"] = flat_linear
        sys.modules["transformers"] = transformers_pkg
        sys.modules["transformers.models"] = transformers_models
        sys.modules["transformers.models.llama"] = llama_pkg
        sys.modules["transformers.models.llama.modeling_llama"] = modeling
        return backups

    def _restore_modules(self, backups):
        for name, module in backups.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _load_module(self):
        module_path = Path(__file__).resolve().parents[2] / "FlatQuant" / "flatquant" / "model_tools" / "llama_utils.py"
        spec = importlib.util.spec_from_file_location("test_flatquant_llama_module", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def _make_config(self):
        return SimpleNamespace(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_dropout=0.0,
            num_hidden_layers=1,
        )

    def _make_args(self):
        return SimpleNamespace(
            direct_inv=False,
            w_bits=4,
            a_bits=4,
            add_diag=True,
            q_bits=16,
            q_asym=False,
            lac=True,
            k_bits=16,
            k_asym=False,
            v_bits=16,
            v_asym=False,
            separate_vtrans=False,
            diag_init="disabled",
        )

    def test_llama_wrapper_uses_position_ids_for_rotary_embeddings(self):
        backups = self._install_stub_modules()
        try:
            module = self._load_module()
            config = self._make_config()
            args = self._make_args()
            base_attn = module.LlamaAttention(config, 0)
            wrapped = module.FlatQuantLlamaAttention(args, base_attn)

            hidden_states = torch.randn(1, 3, config.hidden_size)
            position_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)

            attn_output, attn_weights, cache = wrapped(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

            self.assertEqual(attn_output.shape, hidden_states.shape)
            self.assertIsNone(attn_weights)
            self.assertIsNone(cache)
        finally:
            self._restore_modules(backups)


if __name__ == "__main__":
    unittest.main()
