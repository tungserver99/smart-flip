import unittest
from pathlib import Path


class RequirementsTests(unittest.TestCase):
    def test_requirements_txt_exists_and_lists_runtime_dependencies(self):
        path = Path("requirements.txt")
        self.assertTrue(path.exists(), "requirements.txt is missing")

        lines = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

        required = {"numpy", "torch", "transformers", "datasets", "tqdm"}
        self.assertTrue(required.issubset(lines), f"Missing dependencies: {sorted(required - lines)}")


if __name__ == "__main__":
    unittest.main()
