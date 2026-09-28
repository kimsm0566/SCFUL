import unittest

from src.utils.config import (
    DEFAULT_ORACLE_CONFIG_PATH,
    build_effective_config,
    load_json_config,
)


class ConfigTests(unittest.TestCase):
    def test_default_config_is_cifar10_only_without_mutating_source(self):
        source = load_json_config(DEFAULT_ORACLE_CONFIG_PATH)
        smoke = build_effective_config(source, smoke_test=True)

        self.assertEqual(source["dataset"]["name"], "CIFAR10")
        self.assertEqual(smoke["dataset"]["name"], "CIFAR10")
        self.assertEqual(smoke["dataset"]["smoke_train_per_class"], 20)
        self.assertEqual(smoke["dataset"]["smoke_test_per_class"], 12)
        self.assertEqual(source["training"]["epochs"], 200)


if __name__ == "__main__":
    unittest.main()
