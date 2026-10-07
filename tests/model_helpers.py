"""Small frozen text fixture: no external CLIP weights for CPU tests."""
import torch


class FrozenTextFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("context", torch.zeros(1, 1, 32))

    def encode(self, texts):
        return self.context.expand(len(texts), -1, -1)

    forward = encode
