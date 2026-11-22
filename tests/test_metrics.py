import torch

from simheart.evaluation import mean_pairwise_distance, mmd_rbf, nearest_distances


def sample(n: int, shift: float = 0.0, seed: int = 0) -> torch.Tensor:
    return torch.randn(n, 216, generator=torch.Generator().manual_seed(seed)) + shift


def test_mmd_is_near_zero_for_the_same_distribution_and_grows_with_a_shift():
    same = mmd_rbf(sample(300, seed=1), sample(300, seed=2))
    small = mmd_rbf(sample(300, seed=1), sample(300, 0.2, seed=2))
    large = mmd_rbf(sample(300, seed=1), sample(300, 1.0, seed=2))
    assert abs(same) < 0.01
    assert same < small < large


def test_pairwise_distance_is_zero_for_a_collapsed_set():
    collapsed = sample(1).repeat(50, 1)
    assert mean_pairwise_distance(collapsed) == 0
    assert mean_pairwise_distance(sample(50)) > 0


def test_nearest_distance_is_zero_for_copies():
    reference = sample(100)
    distances = nearest_distances(reference[:10], reference)
    assert torch.all(distances == 0)
    assert torch.all(nearest_distances(sample(10, seed=5), reference) > 0)
