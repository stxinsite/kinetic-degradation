import numpy as np
from sigfig import round

class RatioGroup(object):
    """
    Calculates the relation x = y / z given at least two of x, y, z are known.
    """

    def __init__(self, x, y, z):
        super(RatioGroup, self).__init__()
        self.x = x
        self.y = y
        self.z = z
        if self.is_sufficient():
            self.calc_unknown()

    def is_sufficient(self):
        """Checks whether at least two of x, y, z are known."""
        return sum(attr is None for attr in self.__dict__.values()) < 2

    def calc_unknown(self):
        """Calculates unknown variable."""
        if self.x is None:
            self.x = round(self.y / self.z, sigfigs = 3)
        elif self.y is None:
            self.y = round(self.x * self.z, sigfigs = 3)
        elif self.z is None:
            self.z = round(self.y / self.x, sigfigs = 3)

    def test_ratio(self):
        """Checks whether relation is satisfied."""
        if not self.is_sufficient():
            print("FAIL: Too few parameters are known.")
        elif not np.isclose(self.x, self.y / self.z, rtol = 0.05):
            print("FAIL: Ratio of kinetic rates is not satisfied.")
        else:
            return True
        return False

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

class KineticParameters(object):
    """
    Solves for any dependent kinetic parameters given sufficient independent parameters.
    """

    def __init__(self, params):
        """params is a kinetic model config dictionary."""
        self.params = params.copy()
        self.ratio_groups = {}
        for _ in range(2):
            # All kinetic parameters should be known by second iteration.
            self.update_kinetic_groups()  # Update rate constants.
            self.update_coop_groups()  # Update using thermodynamic closure.

    def update_group(self, attr_x, attr_y, attr_z):
        """Updates attributes in a RatioGroup."""
        ratio_group = RatioGroup(self.params[attr_x], self.params[attr_y], self.params[attr_z])
        self.params[attr_x] = ratio_group.get_x()
        self.params[attr_y] = ratio_group.get_y()
        self.params[attr_z] = ratio_group.get_z()
        return ratio_group

    def update_kinetic_groups(self):
        """Updates Kd = koff / kon for T and E3 in binary and ternary complexes."""
        self.ratio_groups['T_binary_group'] = self.update_group('Kd_T_binary', 'koff_T_binary', 'kon_T_binary')
        self.ratio_groups['E3_binary_group'] = self.update_group('Kd_E3_binary', 'koff_E3_binary', 'kon_E3_binary')
        self.ratio_groups['T_ternary_group'] = self.update_group('Kd_T_ternary', 'koff_T_ternary', 'kon_T_ternary')
        self.ratio_groups['E3_ternary_group'] = self.update_group('Kd_E3_ternary', 'koff_E3_ternary', 'kon_E3_ternary')

    def update_coop_groups(self):
        """Updates alpha = Kd_binary / Kd_ternary for T and E3."""
        # if not self.params['Kd_E3_binary'] and not self.params['Kd_E3_ternary']:
        self.ratio_groups['T_coop_group'] = self.update_group('alpha', 'Kd_T_binary', 'Kd_T_ternary')
        self.ratio_groups['E3_coop_group'] = self.update_group('alpha', 'Kd_E3_binary', 'Kd_E3_ternary')

    def test_closure(self):
        """Checks whether kinetic parameters are consistent."""
        is_closed = True
        for group_name, ratio_group in self.ratio_groups.items():
            if not ratio_group.test_ratio():
                # Either too few parameters in group are known or the ratio is not satisfied.
                print(f"For parameters in {group_name}")
                is_closed = False
        return is_closed

    def get_dict(self):
        return self.params
