import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


class FlatQuantMistralWrapperTests(unittest.TestCase):
    def _install_stub_modules(self, rotary_ctor_style='legacy'):
        backups = {name: sys.modules.get(name) for name in [
            'flatquant',
            'flatquant.quant_utils',
            'flatquant.utils',
            'flatquant.function_utils',
            'flatquant.trans_utils',
            'flatquant.flat_linear',
            'transformers',
            'transformers.models',
            'transformers.models.mistral',
            'transformers.models.mistral.modeling_mistral',
        ]}

        flatquant_pkg = types.ModuleType('flatquant')
        quant_utils = types.ModuleType('flatquant.quant_utils')
        utils = types.ModuleType('flatquant.utils')
        function_utils = types.ModuleType('flatquant.function_utils')
        trans_utils = types.ModuleType('flatquant.trans_utils')
        flat_linear = types.ModuleType('flatquant.flat_linear')

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

        transformers_pkg = types.ModuleType('transformers')
        transformers_models = types.ModuleType('transformers.models')
        mistral_pkg = types.ModuleType('transformers.models.mistral')
        modeling = types.ModuleType('transformers.models.mistral.modeling_mistral')

        class MistralMLP(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.act_fn = nn.SiLU()
                self.up_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                self.down_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        class MistralAttention(nn.Module):
            def __init__(self, config, layer_idx):
                super().__init__()
                self.config = config
                self.layer_idx = layer_idx
                self.head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
                self.num_heads = config.num_attention_heads
                self.num_key_value_heads = config.num_key_value_heads
                self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
                self.scaling = self.head_dim ** -0.5
                self.attention_dropout = config.attention_dropout
                self.is_causal = True
                self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
                self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
                self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
                self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

            def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, cache_position=None, **kwargs):
                raise NotImplementedError

        class MistralDecoderLayer(nn.Module):
            def __init__(self, config, layer_idx):
                super().__init__()
                self.hidden_size = config.hidden_size
                self.self_attn = MistralAttention(config, layer_idx)
                self.mlp = MistralMLP(config)
                self.input_layernorm = nn.Identity()
                self.post_attention_layernorm = nn.Identity()

            def forward(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
                **kwargs,
            ):
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                hidden_states, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states
                return hidden_states

        class MistralRotaryEmbedding(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                if rotary_ctor_style == 'config_only':
                    if 'config' in kwargs:
                        config = kwargs['config']
                    elif args:
                        config = args[0]
                    else:
                        raise TypeError('config is required')
                    self.dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
                else:
                    dim = args[0] if args else kwargs.get('dim')
                    if dim is None:
                        raise TypeError('dim is required')
                    self.dim = dim

            def forward(self, x, position_ids):
                batch, seq_len = position_ids.shape
                cos = torch.ones(batch, seq_len, x.shape[-1], dtype=x.dtype, device=x.device)
                sin = torch.zeros(batch, seq_len, x.shape[-1], dtype=x.dtype, device=x.device)
                return cos, sin

        def apply_rotary_pos_emb(q, k, cos, sin):
            return q, k

        def eager_attention_forward(module, query_states, key_states, value_states, attention_mask, dropout=0.0, scaling=1.0, sliding_window=None, **kwargs):
            attn_output = value_states.transpose(1, 2).contiguous()
            return attn_output, None

        def repeat_kv(x, num_key_value_groups):
            return x.repeat_interleave(num_key_value_groups, dim=1)

        modeling.MistralMLP = MistralMLP
        modeling.MistralAttention = MistralAttention
        modeling.MistralDecoderLayer = MistralDecoderLayer
        modeling.MistralRotaryEmbedding = MistralRotaryEmbedding
        modeling.apply_rotary_pos_emb = apply_rotary_pos_emb
        modeling.eager_attention_forward = eager_attention_forward
        modeling.ALL_ATTENTION_FUNCTIONS = {'eager': eager_attention_forward}
        modeling.repeat_kv = repeat_kv

        sys.modules['flatquant'] = flatquant_pkg
        sys.modules['flatquant.quant_utils'] = quant_utils
        sys.modules['flatquant.utils'] = utils
        sys.modules['flatquant.function_utils'] = function_utils
        sys.modules['flatquant.trans_utils'] = trans_utils
        sys.modules['flatquant.flat_linear'] = flat_linear
        sys.modules['transformers'] = transformers_pkg
        sys.modules['transformers.models'] = transformers_models
        sys.modules['transformers.models.mistral'] = mistral_pkg
        sys.modules['transformers.models.mistral.modeling_mistral'] = modeling
        return backups

    def _restore_modules(self, backups):
        for name, module in backups.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _load_module(self):
        module_path = Path(__file__).resolve().parents[1] / 'src' / 'quantization' / 'flatquant_mistral.py'
        spec = importlib.util.spec_from_file_location('test_flatquant_mistral_module', module_path)
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
            head_dim=4,
            _attn_implementation='eager',
            sliding_window=None,
            max_position_embeddings=32,
            rope_theta=10000,
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
            diag_init='disabled',
        )

    def test_mistral_wrapper_supports_position_embeddings_api(self):
        backups = self._install_stub_modules()
        try:
            module = self._load_module()
            config = self._make_config()
            args = self._make_args()
            base_attn = module.MistralAttention(config, 0)
            wrapped = module.FlatQuantMistralAttention(args, base_attn)

            hidden_states = torch.randn(1, 2, config.hidden_size)
            position_embeddings = (torch.ones(1), torch.ones(1))
            attn_output, attn_weights = wrapped(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )

            self.assertEqual(attn_output.shape, hidden_states.shape)
            self.assertIsNone(attn_weights)
        finally:
            self._restore_modules(backups)

    def test_mistral_wrapper_builds_position_embeddings_from_position_ids(self):
        backups = self._install_stub_modules()
        try:
            module = self._load_module()
            config = self._make_config()
            args = self._make_args()
            base_attn = module.MistralAttention(config, 0)
            wrapped = module.FlatQuantMistralAttention(args, base_attn)

            hidden_states = torch.randn(1, 2, config.hidden_size)
            position_ids = torch.tensor([[0, 1]])
            attn_output, attn_weights = wrapped(
                hidden_states,
                position_ids=position_ids,
                attention_mask=None,
            )

            self.assertEqual(attn_output.shape, hidden_states.shape)
            self.assertIsNone(attn_weights)
        finally:
            self._restore_modules(backups)


    def test_mistral_wrapper_handles_eager_attention_output_layout(self):
        backups = self._install_stub_modules()
        try:
            module = self._load_module()
            config = SimpleNamespace(
                hidden_size=12,
                num_attention_heads=3,
                num_key_value_heads=3,
                attention_dropout=0.0,
                head_dim=4,
                _attn_implementation='eager',
                sliding_window=None,
                max_position_embeddings=32,
                rope_theta=10000,
            )
            args = self._make_args()
            base_attn = module.MistralAttention(config, 0)
            wrapped = module.FlatQuantMistralAttention(args, base_attn)

            hidden_states = torch.randn(1, 5, config.hidden_size)
            position_embeddings = (torch.ones(1, 5, config.head_dim), torch.zeros(1, 5, config.head_dim))
            attn_output, attn_weights = wrapped(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )

            self.assertEqual(attn_output.shape, hidden_states.shape)
            self.assertIsNone(attn_weights)
        finally:
            self._restore_modules(backups)


    def test_mistral_decoder_wrapper_returns_tuple_for_direct_layer_calls(self):
        backups = self._install_stub_modules()
        try:
            module = self._load_module()
            config = self._make_config()
            args = self._make_args()
            base_layer = module.MistralDecoderLayer(config, 0)
            wrapped = module.FlatQuantMistralDecoderLayer(args, base_layer)

            hidden_states = torch.randn(1, 2, config.hidden_size)
            direct_output = wrapped(hidden_states, attention_mask=None, position_ids=torch.tensor([[0, 1]]))
            self.assertIsInstance(direct_output, tuple)
            self.assertEqual(direct_output[0].shape, hidden_states.shape)

            model_output = wrapped(
                hidden_states,
                attention_mask=None,
                position_ids=torch.tensor([[0, 1]]),
                position_embeddings=(torch.ones(1, 2, config.head_dim), torch.zeros(1, 2, config.head_dim)),
            )
            self.assertEqual(model_output.shape, hidden_states.shape)
        finally:
            self._restore_modules(backups)

    def test_mistral_wrapper_supports_config_style_rotary_constructor(self):
        backups = self._install_stub_modules(rotary_ctor_style='config_only')
        try:
            module = self._load_module()
            config = self._make_config()
            args = self._make_args()
            base_attn = module.MistralAttention(config, 0)
            wrapped = module.FlatQuantMistralAttention(args, base_attn)

            hidden_states = torch.randn(1, 2, config.hidden_size)
            position_ids = torch.tensor([[0, 1]])
            attn_output, attn_weights = wrapped(
                hidden_states,
                position_ids=position_ids,
                attention_mask=None,
            )

            self.assertEqual(attn_output.shape, hidden_states.shape)
            self.assertIsNone(attn_weights)
        finally:
            self._restore_modules(backups)


if __name__ == '__main__':
    unittest.main()
