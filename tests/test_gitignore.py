from pathlib import Path
import unittest


class GitignoreTest(unittest.TestCase):
    def test_wandb_artifacts_are_ignored(self):
        content = Path(".gitignore").read_text(encoding="utf-8")

        self.assertIn("wandb/", content)
        self.assertIn("*.wandb", content)


if __name__ == "__main__":
    unittest.main()
