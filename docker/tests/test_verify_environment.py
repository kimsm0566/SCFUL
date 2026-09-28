from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "docker"))

import verify_environment


class VerifyEnvironmentTests(unittest.TestCase):
    def test_scu_tree_compiles(self):
        self.assertTrue(
            verify_environment.compile_scu_tree(REPOSITORY_ROOT / "third_party" / "SCU")
        )

    def test_fedosd_package_imports(self):
        verify_environment.import_fedosd(REPOSITORY_ROOT / "third_party" / "FedOSD")


if __name__ == "__main__":
    unittest.main()
