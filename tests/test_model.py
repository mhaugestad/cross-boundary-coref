import torch
from types import SimpleNamespace

from cross_boundary_coref.model import AffinityModel, _checkpoint_state


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(11, 4)
        self.config = SimpleNamespace(dim=4)

    def forward(self, *, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))


def test_model_forward_shape_and_bilinear_head():
    model = AffinityModel.__new__(AffinityModel)
    torch.nn.Module.__init__(model)
    model.encoder = TinyEncoder()
    model.affinity = torch.nn.Bilinear(4, 4, 1, bias=False)
    model.encoder_name = "test"
    output = model(
        torch.tensor([[1, 2, 0]]),
        torch.tensor([[1, 1, 0]]),
    )
    assert output.shape == (1, 3, 3)
    assert model.affinity.bias is None


def test_lightning_checkpoint_prefix_is_removed():
    state = {"model.encoder.weight": torch.ones(2, 2)}
    assert list(_checkpoint_state({"state_dict": state})) == ["encoder.weight"]


def test_uninitialized_encoder_does_not_change_checkpoint_training_rng(monkeypatch):
    class RandomEncoder(TinyEncoder):
        def __init__(self):
            super().__init__()
            self.random_parameter = torch.nn.Parameter(torch.rand(4))

    monkeypatch.setattr(
        "cross_boundary_coref.model.AutoConfig.from_pretrained",
        lambda _name: object(),
    )
    monkeypatch.setattr(
        "cross_boundary_coref.model.AutoModel.from_config",
        lambda _config: RandomEncoder(),
    )
    torch.manual_seed(42)
    torch.nn.Bilinear(4, 4, 1, bias=False)
    expected_rng_state = torch.get_rng_state()

    torch.manual_seed(42)
    AffinityModel("test", load_pretrained=False)

    assert torch.equal(torch.get_rng_state(), expected_rng_state)
