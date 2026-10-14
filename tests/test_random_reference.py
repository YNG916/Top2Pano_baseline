"""Random image references must have the stated distribution and paired seeds."""
import numpy as np
from scripts.evaluate_random_reference import psnr, random_scores


def test_uniform_reference_matches_exact_quantized_distribution():
    target = np.zeros((256, 448, 3), dtype=np.float64)
    draws, expected = random_scores(target, 'same-physical-query', repetitions=8)
    exact_mse = np.mean((np.arange(256, dtype=np.float64) / 255.) ** 2)
    np.testing.assert_allclose(expected, -10 * np.log10(exact_mse), atol=1e-12)
    assert abs(np.mean(draws) - expected) < .02
    paired, _ = random_scores(target, 'same-physical-query', repetitions=8)
    assert paired == draws
    other, _ = random_scores(target, 'other-physical-query', repetitions=8)
    assert other != draws


def test_full_image_psnr_and_fixed_gray_are_distinct_from_noise():
    # Known independent reference: all errors .1 give MSE=.01 and PSNR=20 dB.
    target = np.full((64, 128, 3), .4, dtype=np.float64)
    np.testing.assert_allclose(psnr(np.full_like(target, .5), target), 20., atol=1e-12)
    noise, _ = random_scores(target, 'query', repetitions=4)
    assert np.mean(noise) < 12.
