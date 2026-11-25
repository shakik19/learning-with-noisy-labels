import numpy as np
import pytest

from noisylabels.noise import corrupt_labels, transition_matrix

K = 10
N = 50_000


@pytest.fixture
def y():
    return np.arange(N) % K


def test_none_changes_nothing(y):
    noisy, corrupted = corrupt_labels(y, "none", 0.0, K, np.random.default_rng(0))
    assert np.array_equal(noisy, y) and not corrupted.any()


@pytest.mark.parametrize("eta", [0.2, 0.4, 0.8])
def test_symmetric(y, eta):
    noisy, corrupted = corrupt_labels(y, "sym", eta, K, np.random.default_rng(0))
    assert np.array_equal(corrupted, noisy != y)
    assert abs(corrupted.mean() - eta) < 0.01  # realized rate equals eta (flips never hit the true class)
    t = transition_matrix(y, noisy, K)
    assert np.allclose(np.diag(t), 1 - eta, atol=0.02)
    off = t[~np.eye(K, dtype=bool)]
    p = eta / (K - 1)
    sd = np.sqrt(p * (1 - p) / (N / K))  # sampling error of one entry (N/K samples per class)
    assert np.allclose(off, p, atol=5 * sd)  # uniform over the other classes


def test_pairflip(y):
    noisy, corrupted = corrupt_labels(y, "pair", 0.4, K, np.random.default_rng(0))
    assert np.all(noisy[corrupted] == (y[corrupted] + 1) % K)
    assert abs(corrupted.mean() - 0.4) < 0.01
    t = transition_matrix(y, noisy, K)
    allowed = np.eye(K, dtype=bool) | np.roll(np.eye(K, dtype=bool), 1, axis=1)
    assert np.all(t[~allowed] == 0)


def test_random_labels(y):
    noisy, corrupted = corrupt_labels(y, "random", 0.0, K, np.random.default_rng(0))
    assert abs(corrupted.mean() - (1 - 1 / K)) < 0.01
    assert np.allclose(np.bincount(noisy, minlength=K) / N, 1 / K, atol=0.01)


def test_deterministic_given_seed(y):
    a, _ = corrupt_labels(y, "sym", 0.4, K, np.random.default_rng([3, 2024]))
    b, _ = corrupt_labels(y, "sym", 0.4, K, np.random.default_rng([3, 2024]))
    c, _ = corrupt_labels(y, "sym", 0.4, K, np.random.default_rng([4, 2024]))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
