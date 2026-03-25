import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main


class MetadataSerializationTests(unittest.TestCase):
    def test_build_metadata_config_excludes_func_and_callables(self):
        args = SimpleNamespace(
            model_path="dummy-model",
            variant="awq_raw",
            func=lambda x: x,
            run_name="demo",
            seed=42,
        )

        config = main.build_metadata_config(args)

        self.assertNotIn("func", config)
        self.assertEqual(config["model_path"], "dummy-model")
        self.assertEqual(config["variant"], "awq_raw")
        json.dumps(config)


if __name__ == "__main__":
    unittest.main()
