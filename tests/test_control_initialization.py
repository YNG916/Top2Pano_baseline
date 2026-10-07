import pytest
import torch

from scripts.init_control_sd21 import build_initial_state


def test_sd_parameters_are_copied_to_control_but_new_zero_layers_stay_zero():
    scratch = {
        "model.diffusion_model.input_blocks.0.weight": torch.zeros(2, 3),
        "control_model.input_blocks.0.weight": torch.zeros(2, 3),
        "control_model.zero_convs.0.weight": torch.zeros(2, 2),
        "first_stage_model.decoder.depth_map.weight": torch.ones(1, 2),
    }
    pretrained = {"model.diffusion_model.input_blocks.0.weight": torch.full((2, 3), 7.)}
    target, copied, initialized = build_initial_state(scratch, pretrained)
    torch.testing.assert_close(target["control_model.input_blocks.0.weight"], pretrained["model.diffusion_model.input_blocks.0.weight"])
    torch.testing.assert_close(target["model.diffusion_model.input_blocks.0.weight"], pretrained["model.diffusion_model.input_blocks.0.weight"])
    assert target["control_model.zero_convs.0.weight"].sum() == 0
    assert "first_stage_model.decoder.depth_map.weight" in initialized
    assert len(copied) == 2


def test_wrong_checkpoint_dimensions_fail_before_saving():
    with pytest.raises(ValueError, match="shape mismatch"):
        build_initial_state({"control_model.attn.weight": torch.zeros(4, 1024)},
                            {"model.diffusion_model.attn.weight": torch.zeros(4, 768)})
