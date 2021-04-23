import src.data_input as di
import pytest


def test_values():
    sp = di.read_population()
    # All types should be represented
    assert set(sp.Age_Group.unique()) == {0, 1, 2}
    assert set(sp.Sex.unique()) == {1, 2}
    assert set(sp.Race.unique()) == {1, 2, 3}
    assert sp.shape[0] > 10_400_000
    assert list(sp.Start_Location.unique()) == [0]
