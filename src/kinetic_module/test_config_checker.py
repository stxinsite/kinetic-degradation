import unittest
import yaml
from calc_full_config import KineticParameters


class MyTestCase(unittest.TestCase):
    def test_fully_defined(self):
        with open(file=f'./data/unit_test_config.yml', mode='r') as file:
            fully_defined_params: dict[str, float] = yaml.safe_load(file)

        fully_defined_kp = KineticParameters(fully_defined_params)
        self.assertTrue(fully_defined_kp.is_fully_defined())  # add assertion here

    def test_insufficient(self):
        with open(file=f'./data/insufficient_test_config.yml', mode='r') as file:
            insufficient_params: dict[str, float] = yaml.safe_load(file)

        insufficient_kp = KineticParameters(insufficient_params)
        self.assertFalse(insufficient_kp.is_fully_defined())

    def test_inconsistent(self):
        with open(file=f'./data/inconsistent_test_config.yml', mode='r') as file:
            inconsistent_params: dict[str, float] = yaml.safe_load(file)

        inconsistent_kp = KineticParameters(inconsistent_params)
        self.assertFalse(inconsistent_kp.is_fully_defined())


if __name__ == '__main__':
    unittest.main()
