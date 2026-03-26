import json
import unittest
from pathlib import Path


class NotebookCommandTests(unittest.TestCase):
    def test_notebook_uses_current_bash_wrapper_commands(self):
        notebook = json.loads(Path("Untitled0.ipynb").read_text(encoding="utf-8"))
        command_cell = next(
            cell
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code" and any("run_float_model.sh" in line for line in cell.get("source", []))
        )

        expected = [
            "# !python main.py float_model --model-path mistralai/Mistral-7B-v0.3 --run-name float_eval\n",
            "# !python main.py raw_quantize --model-path mistralai/Mistral-7B-v0.3 --origin-method awq --run-name awq_raw_run\n",
            "# !python main.py flip_quantize --model-path mistralai/Mistral-7B-v0.3 --origin-method awq --run-name awq_flip_run\n",
            "!MODEL_PATH=mistralai/Mistral-7B-v0.3 bash scripts/bash/run_float_model.sh\n",
            "!MODEL_PATH=mistralai/Mistral-7B-v0.3 bash scripts/bash/run_raw_quantize.sh\n",
            "!MODEL_PATH=mistralai/Mistral-7B-v0.3 bash scripts/bash/run_flip_quantize.sh",
        ]

        self.assertEqual(command_cell["source"], expected)


if __name__ == "__main__":
    unittest.main()
