"""Regress empty histogram division without changing nonempty upstream results."""
import torch
from ldm.models.diffusion.ddpm import LatentDiffusion, compute_color_histogram


def test_nonempty_histogram_matches_original_range_and_batch_aggregation():
    torch.manual_seed(7)
    image = torch.rand(2, 4, 5, 3) * 3 - 1  # Also keep original out-of-range exclusion.
    expected = []
    for channel in range(3):
        counts = torch.histc(image[..., channel], bins=16, min=0, max=1)
        assert counts.sum() > 0
        expected.append(counts / counts.sum())
    torch.testing.assert_close(compute_color_histogram(image, bins=16), torch.stack(expected, dim=-1), rtol=0, atol=0)


def test_empty_channel_is_finite_without_rescaling_or_clipping_colors():
    image = torch.full((1, 2, 3, 3), -.5)
    image[..., 1] = .25
    image[..., 2] = 1.5
    histogram = compute_color_histogram(image, bins=16)
    assert torch.isfinite(histogram).all()
    assert histogram[:, 0].sum() == histogram[:, 2].sum() == 0
    assert histogram[:, 1].sum() == 1


def test_actual_render_loss_handles_empty_decoded_histogram_and_preserves_gradient():
    class LossFixture:
        training = True
        device = torch.device('cpu')
        parameterization = 'eps'
        logvar = torch.zeros(1)
        learn_logvar = False
        l_simple_weight = 1.
        lvlb_weights = torch.ones(1)
        original_elbo_weight = 0.
        loss_type = 'l2'
        def __init__(self):
            self.weight = torch.tensor(.2, requires_grad=True)
        def q_sample(self, x_start, t, noise):
            return x_start
        def apply_model(self, x, t, cond):
            return self.weight * torch.ones_like(x)
        def get_loss(self, prediction, target, mean=False):
            return (prediction - target).square() if self.loss_type == 'l2' else (prediction - target).abs()
        def predict_start_from_noise(self, x, t, noise):
            return x
        def decode_first_stage(self, x):
            # Every RGB value lies outside the upstream histogram's [0, 1] range.
            return torch.full((1, 4, 2, 3), -.5)
    model = LossFixture()
    noise = torch.full((1, 4, 2, 3), .7)
    image = torch.linspace(-1, 1, 18).reshape(1, 2, 3, 3)
    batch = {'ground_truth': image, 'ground_truth_depth': torch.ones(1, 2, 3, 3)}
    loss, metrics = LatentDiffusion.p_losses(model, torch.zeros_like(noise), None, batch,
                                            torch.tensor([0]), noise=noise)
    assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in metrics.values())
    loss.backward()
    # The auxiliary losses remain detached; the original diffusion gradient survives.
    torch.testing.assert_close(model.weight.grad, torch.tensor(-1.))
