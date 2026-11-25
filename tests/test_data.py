import numpy as np
import pytest

from noisylabels.data import load_cifar10, stratified_split, stratified_subset

from fake_cifar import write_fake_cifar


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    return write_fake_cifar(tmp_path_factory.mktemp("cifar"), per_batch=200, n_test=100)


def test_loader_layout(fake):
    x_tr, y_tr, x_te, y_te = load_cifar10(fake)
    assert x_tr.shape == (1000, 3, 32, 32) and x_tr.dtype == np.uint8
    assert y_tr.shape == (1000,) and x_te.shape == (100, 3, 32, 32)


def test_split_is_stratified_disjoint_and_fixed():
    y = np.arange(50_000) % 10
    tr, va = stratified_split(y, 5000, seed=0)
    assert len(va) == 5000 and len(tr) == 45_000
    assert len(np.intersect1d(tr, va)) == 0
    assert np.all(np.bincount(y[va]) == 500)
    _, va2 = stratified_split(y, 5000, seed=0)
    assert np.array_equal(va, va2)


def test_subset_is_stratified_and_inside_train():
    y = np.arange(50_000) % 10
    tr, _ = stratified_split(y, 5000, seed=0)
    sub = stratified_subset(tr, y, 10_000, seed=0)
    assert np.all(np.bincount(y[sub]) == 1000) and np.isin(sub, tr).all()
