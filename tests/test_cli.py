import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path.cwd() / "test_project_cli"
        self.test_dir.mkdir(exist_ok=True)
        (self.test_dir / ".git").mkdir(exist_ok=True)
        os.chdir(self.test_dir)
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
        # Use the modern CLI entry point via the aicache package
        self.cli_path = [sys.executable, "-m", "aicache"]
        # Clear the cache before each test (use --confirm to skip prompt)
        subprocess.run(
            [*self.cli_path, "clear", "--confirm"],
            env=self.env,
            capture_output=True,
        )

    def tearDown(self):
        os.chdir("..")
        shutil.rmtree(self.test_dir)

    def test_cli_stats(self):
        # Set a value via the cache API directly since modern CLI
        # doesn't have a simple set command like the old CLI
        result = subprocess.run(
            [*self.cli_path, "stats"],
            capture_output=True,
            text=True,
            env=self.env,
        )
        self.assertEqual(result.returncode, 0)
        # Modern CLI outputs rich-formatted stats
        self.assertIn("Cache", result.stdout)

    def test_cli_clear(self):
        result = subprocess.run(
            [*self.cli_path, "clear", "--confirm"],
            capture_output=True,
            text=True,
            env=self.env,
        )
        self.assertEqual(result.returncode, 0)

    def test_cli_help(self):
        result = subprocess.run(
            [*self.cli_path, "--help"],
            capture_output=True,
            text=True,
            env=self.env,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("AI Cache", result.stdout)


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
    unittest.main()
