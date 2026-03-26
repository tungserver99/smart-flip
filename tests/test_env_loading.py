import os
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class EnvLoadingTests(unittest.TestCase):
    def test_load_runtime_env_reads_wandb_and_hf_tokens(self):
        env_dir = Path('data/cache/test_env_loading')
        env_dir.mkdir(parents=True, exist_ok=True)
        env_path = env_dir / '.env'
        env_path.write_text("WANDB_API_KEY=wandb-secret\nHF_TOKEN=hf-secret\n", encoding='utf-8')

        with patch.dict(os.environ, {}, clear=True):
            main.load_runtime_env(env_path)
            self.assertEqual(os.environ['WANDB_API_KEY'], 'wandb-secret')
            self.assertEqual(os.environ['HF_TOKEN'], 'hf-secret')
            self.assertEqual(main.resolve_hf_token(), 'hf-secret')

    def test_resolve_hf_token_falls_back_to_huggingface_hub_token(self):
        with patch.dict(os.environ, {'HUGGINGFACE_HUB_TOKEN': 'hub-secret'}, clear=True):
            self.assertEqual(main.resolve_hf_token(), 'hub-secret')


if __name__ == '__main__':
    unittest.main()
