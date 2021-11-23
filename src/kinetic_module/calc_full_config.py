import numpy as np

class KineticParameters(object):
    """
    Solve for any dependent kinetic parameters given sufficient independent parameters.
    """
    binding_equilibrium_expr = [
        ('koff_T_binary', 'kon_T_binary', 'Kd_T_binary', 1),
        ('kon_T_binary', 'koff_T_binary', 'Kd_T_binary', -1),
        ('Kd_T_binary', 'koff_T_binary', 'kon_T_binary', -1),
        ('koff_E3_binary', 'kon_E3_binary', 'Kd_E3_binary', 1),
        ('kon_E3_binary', 'koff_E3_binary', 'Kd_E3_binary', -1),
        ('Kd_E3_binary', 'koff_E3_binary', 'kon_E3_binary', -1),
        ('koff_T_ternary', 'kon_T_ternary', 'Kd_T_ternary', 1),
        ('kon_T_ternary', 'koff_T_ternary', 'Kd_T_ternary', -1),
        ('Kd_T_ternary', 'koff_T_ternary', 'kon_T_ternary', -1),
        ('koff_E3_ternary', 'kon_E3_ternary', 'Kd_E3_ternary', 1),
        ('kon_E3_ternary', 'koff_E3_ternary', 'Kd_E3_ternary', -1),
        ('Kd_E3_ternary', 'koff_E3_ternary', 'kon_E3_ternary', -1)
    ]

    cooperativity_expr = [
        ('Kd_T_binary', 'Kd_T_ternary', 'alpha', 1),
        ('Kd_T_ternary', 'Kd_T_binary', 'alpha', -1),
        ('alpha', 'Kd_T_binary', 'Kd_T_ternary', -1),
        ('Kd_E3_binary', 'Kd_E3_ternary', 'alpha', 1),
        ('Kd_E3_ternary', 'Kd_E3_binary', 'alpha', -1),
        ('alpha', 'Kd_E3_binary', 'Kd_E3_ternary', -1)
    ]

    expression_list = binding_equilibrium_expr + cooperativity_expr

    def __init__(self, params):
        """params is a kinetic model config dictionary"""
        self.params = params.copy()
        self.forward_pass()
        self.backward_pass()

    def test_and_calc_params(self, direction):
        for LH_Key, RH_Key1, RH_Key2, power in self.expression_list[::direction]:
            value = self.params[LH_Key]

            if self.params[RH_Key1] and self.params[RH_Key2]:
                proposed_value = self.params[RH_Key1] * (self.params[RH_Key2] ** power)
            else:
                proposed_value = None

            if proposed_value is not None:
                if value is not None:
                    # if key already has value, check consistency with proposed value
                    assert np.isclose(value, proposed_value, rtol = 0.05), (
                        f"{LH_Key} = {value} is not consistent with {RH_Key1} {'*' if power == 1 else '/'} {RH_Key2} = {proposed_value}"
                    )
                else:
                    # if value is nan, update with proposed value
                    self.params[LH_Key] = proposed_value

    def forward_pass(self):
        self.test_and_calc_params(1)

    def backward_pass(self):
        self.test_and_calc_params(-1)

    def is_fully_defined(self):
        for LH_Key, RH_Key1, RH_Key2, power in self.binding_equilibrium_expr:
            if self.params[LH_Key] is None:
                return False
        return True

    def get_dict(self):
        return self.params
