from typing import Literal, Optional
import numpy as np
import json


class KineticParameters(object):
    """A class used to check and solve for model parameters.

    This class checks for the consistency of kinetic rate constants and binding
    affinities and solves for any unknown parameters if given a sufficient
    number of known parameters.

    ...

    Attributes
    ----------
    binding_equilibrium_expr : list[tuple[str, str, str, int]]]
        expressions relating kinetic rate constants to each other
    cooperativity_expr : list[tuple[str, str, str, int]]]
        expressions relating binding affinities and cooperativity to each other
    expression_list : list[tuple[str, str, str, int]]]
        a list of tuples (lhs_key, rhs_key1, rhs_key2, pwr) where
        lhs_key = rhs_key1 * (rhs_key2 ** pwr)
    _params : dict[str, float]
        kinetic rate constants and model parameters for rate equations
    warning_messages: set[str]
        messages for inconsistent parameters

    Methods
    -------
    test_and_calc_params(direction)
        iterate through expression_list and test or solve for parameters
    forward_pass()
        iterate forward through expression_list
    backward_pass()
        iterate backward through expression_list
    is_fully_defined()
        test whether all parameters in expression_list are known and consistent
    """

    binding_equilibrium_expr: list[tuple[str, str, str, int]] = [
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

    cooperativity_expr: list[tuple[str, str, str, int]] = [
        ('Kd_T_binary', 'Kd_T_ternary', 'alpha', 1),
        ('Kd_T_ternary', 'Kd_T_binary', 'alpha', -1),
        ('alpha', 'Kd_T_binary', 'Kd_T_ternary', -1),
        ('Kd_E3_binary', 'Kd_E3_ternary', 'alpha', 1),
        ('Kd_E3_ternary', 'Kd_E3_binary', 'alpha', -1),
        ('alpha', 'Kd_E3_binary', 'Kd_E3_ternary', -1)
    ]

    expression_list: list[tuple[str, str, str, int]] = binding_equilibrium_expr + cooperativity_expr

    def __init__(self, params: dict[str, float]):
        """
        Parameters
        ----------
        params : dict[str, float]
            kinetic rate constants and model parameters for rate equations
        """
        self._params: dict[str, float] = params.copy()
        self.warning_messages: set[str] = set()
        self.forward_pass()
        self.backward_pass()

    def __str__(self):
        return json.dumps(self._params, indent=4)

    @property
    def params(self) -> dict[str, float]:
        return self._params

    def test_and_calc_params(self, direction: Literal[1, -1]) -> None:
        """Checks or calculates parameters.

        Parameters
        ----------
        direction: Literal[1, -1]
            Traverse forward (1) or backward (-1) through parameters.
        """

        for lhs_key, rhs_key1, rhs_key2, pwr in self.expression_list[::direction]:
            value: Optional[float]
            proposed_value: Optional[float]

            try:
                self._params[lhs_key]
            except KeyError:
                print(f"{lhs_key} was not provided")
            finally:
                value = self._params.setdefault(lhs_key)

            rhs_value1 = self._params.get(rhs_key1)
            rhs_value2 = self._params.get(rhs_key2)
            try:
                proposed_value = rhs_value1 * (rhs_value2 ** pwr)
            except TypeError:
                proposed_value = None

            if proposed_value:
                if value:
                    # if left-hand side key already has value, check consistency with proposed value
                    if not np.isclose(value, proposed_value, rtol = 0.05):
                        self.warning_messages.add(
                            f"{lhs_key} = {value} is inconsistent with {rhs_key1} {'*' if pwr == 1 else '/'} {rhs_key2} = {proposed_value}"
                        )
                else:
                    # if left-hand side value is None, update with proposed value
                    self._params[lhs_key] = proposed_value

    def forward_pass(self) -> None:
        """Checks or calculates parameters in forward direction.
        """
        self.test_and_calc_params(direction=1)

    def backward_pass(self) -> None:
        """Checks or calculates parameters in backward direction.
        """
        self.test_and_calc_params(direction=-1)

    def is_fully_defined(self) -> bool:
        """Checks whether all parameters are known and consistent.
        """
        # check for any values left None
        for lhs_key, rhs_key1, rhs_key2, pwr in self.expression_list:
            if self._params[lhs_key] is None:
                self.warning_messages.add(f"{lhs_key} is undefined")

        if len(self.warning_messages):
            # check for any warning messages
            print(*self.warning_messages, sep='\n')
            return False

        return True
