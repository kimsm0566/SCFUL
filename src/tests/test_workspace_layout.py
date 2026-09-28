import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class WorkspaceLayoutTests(unittest.TestCase):
    def test_external_baseline_clones_are_outside_research_src_package(self):
        self.assertTrue((ROOT / "third_party/SCU/.git").is_dir())
        self.assertTrue((ROOT / "third_party/FedOSD/.git").is_dir())
        self.assertFalse((ROOT / "src/SCU").exists())
        self.assertFalse((ROOT / "src/FedOSD").exists())

    def test_readmes_document_main_entrypoint_and_third_party_boundary(self):
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        source_readme = (ROOT / "src/README.md").read_text(encoding="utf-8")

        self.assertIn("python src/main.py", root_readme)
        self.assertIn("third_party/", root_readme)
        self.assertIn("src/main.py", source_readme)
        self.assertIn("src/data/", source_readme)

    def test_main_uses_one_common_data_and_training_flow(self):
        main_source = (ROOT / "src/main.py").read_text(encoding="utf-8")

        self.assertIn("data_state = trainer.prepare_data(comm)", main_source)
        self.assertIn("result = trainer.train(data_state", main_source)
        self.assertNotIn('args.algorithm == "FSCU"', main_source)
        self.assertNotIn("def run_fscu", main_source)

    def test_fscu_data_preparation_lives_in_data_module(self):
        data_source = (ROOT / "src/data/data.py").read_text(encoding="utf-8")
        runtime_source = (ROOT / "src/utils/fscu_runtime.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("def prepare_fscu_data", data_source)
        self.assertIn("def build_fscu_split_manifest", data_source)
        self.assertNotIn("def prepare_fscu_data", runtime_source)
        self.assertNotIn("def build_split_manifest", runtime_source)


if __name__ == "__main__":
    unittest.main()
