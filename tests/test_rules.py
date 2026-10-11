import numpy as np

from cellular_automata.rules import (
    canonical_number,
    elementary,
    elementary_representatives,
    from_number,
    mutate,
    random_rule,
    to_number,
)
from cellular_automata.wolfram_classes import CLASS_OF


def test_number_round_trip():
    for k, n in [(2, 3), (2, 5), (3, 3)]:
        for number in [0, 1, 7, k ** (k**n) - 1]:
            assert to_number(from_number(number, k, n), k) == number


def test_wolfram_bit_convention():
    # Rule 30 = 00011110: neighbourhoods 100, 011, 010, 001 map to 1.
    assert elementary(30).tolist() == [0, 1, 1, 1, 1, 0, 0, 0]


def test_88_equivalence_classes():
    assert len(elementary_representatives()) == 88


def test_known_equivalences():
    # Rule 30's mirror is 86, its complement 135, both 149.
    assert {canonical_number(n) for n in (30, 86, 135, 149)} == {30}
    assert canonical_number(110) == 110 and canonical_number(124) == 110


def test_class_reference_covers_representatives_exactly():
    assert set(CLASS_OF) == set(elementary_representatives())


def test_mutation_changes_exactly_the_hit_entries():
    rng = np.random.default_rng(1)
    table = random_rule(3, 3, rng)
    mutated = mutate(table, 3, rng, rate=1.0)
    assert (mutated != table).all()
    assert mutated.max() < 3
    assert (mutate(table, 3, rng, rate=0.0) == table).all()


def test_random_rule_lambda():
    rng = np.random.default_rng(2)
    lams = [np.mean(random_rule(2, 7, rng, lam=0.3) != 0) for _ in range(50)]
    assert abs(np.mean(lams) - 0.3) < 0.03
