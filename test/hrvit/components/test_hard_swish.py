import numpy as np
import pytest

from hrvit.components.hard_swish import HardSwish


@pytest.mark.parametrize('input_value, expected_output_value', [
    (-100.0, 0.0),
    (-2.5, -0.0),
    (-2.0, -0.2),
    (0.0, 0.0),
    (2.0, 1.8),
    (2.5, 2.5),
    (100.0, 100.0),
])
def test_hard_swish_produces_expected_values(
    input_value: float,
    expected_output_value: float,
):
    hard_swish = HardSwish()

    hard_swish_value = hard_swish(input_value)

    assert np.allclose(hard_swish_value, expected_output_value)
