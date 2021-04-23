import os

import numpy as np
import pytest

from src.jit_functions import (
    assign_conditions,
    sum_true,
    first_true_value,
    update_community_probability,
)
from src.tests.fixtures import model


def test_assign_conditions(model):
    """Concurrent conditions are assigned when the model is initialized. Make sure the appropriate amount is assigned"""
    ages = np.array([0] * 5000 + [1] * 5000 + [2] * 5000)
    concurrent_conditions = assign_conditions(ages, model.rng.rand(len(ages)))

    # Those ages 1 (50-64) should be assigned concurrent conditions 23.74% of the time
    age_1 = concurrent_conditions[ages == 1]
    assert sum(age_1) / len(age_1) == pytest.approx(0.2374, abs=0.03)

    # Those ages 2 (65+) should be assigned concurrent conditions 54.97% of the time
    age_2 = concurrent_conditions[ages == 2]
    assert sum(age_2) / len(age_2) == pytest.approx(0.5497, abs=0.03)


def test_update_community_probability(model):
    """Community probabilities are updated before the model runs. Movement should be based on concurrent conditions"""
    concurrent_conditions = np.array([1] * model.population.shape[0])

    # Probability of movement before update:
    age_1 = model.movement.location.probabilities[model.age_groups == 1]
    age_1_before = age_1.mean()
    age_2 = model.movement.location.probabilities[model.age_groups == 2]
    age_2_before = age_2.mean()

    new_probabilities = update_community_probability(
        cp=model.movement.location.probabilities,
        age=model.age_groups,
        cc=concurrent_conditions,
    )

    # After Updates: Probabilities should go up
    age_1 = new_probabilities[model.age_groups == 1]
    assert age_1.mean() / age_1_before == pytest.approx(2.316, abs=0.01)
    age_2 = new_probabilities[model.age_groups == 2]
    assert age_2.mean() / age_2_before == pytest.approx(1.437, abs=0.01)


def test_first_true_value():
    a = np.array([False, False, False, True])
    assert first_true_value(a) == (True, 3)

    b = np.array([True, False, False, False])
    assert first_true_value(b) == (True, 0)

    c = np.array([False, False])
    assert first_true_value(c) == (False, -1)


@pytest.mark.skipif(os.environ.get("NUMBA_DISABLE_JIT") == "1", reason="Skipping NUMBA JIT Compiling.")
def test_first_true_value_nonbool():
    with pytest.raises(TypeError):
        d = np.array([1, 2, 3, np.nan])  # non-boolean array
        first_true_value(d)


def test_sum_true():
    size = 1000
    a = np.full(size, True)
    assert sum_true(a) == size

    b = np.array([True, False, True, False])
    assert sum_true(b) == 2


__all__ = ["model"]
