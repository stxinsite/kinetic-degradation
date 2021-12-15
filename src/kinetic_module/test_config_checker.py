import unittest
import yaml
from calc_full_config import KineticParameters


class MyTestCase(unittest.TestCase):
    def test_fully_defined(self):
        """Tests that KineticParameters recognizes fully defined and consistent config.
        """
        with open(file='./data/unit_test_config.yml', mode='r') as file:
            fully_defined_params: dict[str, float] = yaml.safe_load(file)

        fully_defined_kp = KineticParameters(fully_defined_params)
        self.assertTrue(fully_defined_kp.is_fully_defined())

    def test_insufficient(self):
        """Tests that KineticParameters recognizes insufficient config.
        """

        # not enough parameters are defined to calculate all of them
        with open(file='./data/insufficient_test_config.yml', mode='r') as file:
            insufficient_params: dict[str, float] = yaml.safe_load(file)

        insufficient_kp = KineticParameters(insufficient_params)
        self.assertFalse(insufficient_kp.is_fully_defined())

    def test_inconsistent(self):
        """Tests that KineticParameters recognizes inconsistent config.
        """

        # parameters are provided but don't satisfy ratios
        with open(file='./data/inconsistent_test_config.yml', mode='r') as file:
            inconsistent_params: dict[str, float] = yaml.safe_load(file)

        inconsistent_kp = KineticParameters(inconsistent_params)
        self.assertFalse(inconsistent_kp.is_fully_defined())

    def test_missing(self):
        """Tests that KineticParameters recognizes and handles missing parameters.
        """

        # only alpha is missing from otherwise fully defined and consistent config
        # KineticParameters should be able to create the missing key and calculate its value
        with open(file='./data/missing_test_config.yml', mode='r') as file:
            missing_params: dict[str, float] = yaml.safe_load(file)

        missing_kp = KineticParameters(missing_params)
        self.assertTrue(missing_kp.is_fully_defined())


if __name__ == '__main__':
    unittest.main()
