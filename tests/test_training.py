import torch
from types import SimpleNamespace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from dataset import collate_examples  # noqa: E402
from loss import affinity_loss  # noqa: E402
from train import AffinityTrainingModule, validation_metrics  # noqa: E402


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(11, 4)
        self.config = SimpleNamespace(dim=4)

    def forward(self, *, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))


def test_weighted_bce_masks_non_context_chunk_pairs():
    logits = torch.zeros(1, 3, 3, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0, 1] = 1
    loss = affinity_loss(
        logits,
        targets,
        torch.tensor([[1, 0, 0]]),
        torch.tensor([[0, 1, 0]]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 2, 2] == 0


def test_collator_builds_affinity_target_matrix():
    batch = collate_examples(
        [
            {
                "id": "one",
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "context_mask": [1, 0, 0],
                "chunk_mask": [0, 1, 0],
                "target_boxes": [[0, 1, 1, 2]],
            }
        ],
        pad_token_id=0,
    )
    assert batch["affinity_targets"].shape == (1, 3, 3)
    assert batch["affinity_targets"][0, 0, 1] == 1


def test_validation_decoder_threshold_selection():
    scores = torch.tensor([[0.95, 0.1], [0.1, 0.1]]).numpy()
    selected = validation_metrics([(scores, [((0, 1), (0, 1))])], [0.5, 0.9], [4, 8])
    assert selected["f1"] == 1
    assert selected["threshold"] == 0.9


def test_training_step_optimizer_and_checkpoint_round_trip(tmp_path, monkeypatch):
    from cross_boundary_coref.model import AffinityModel, _checkpoint_state

    def tiny_model():
        model = AffinityModel.__new__(AffinityModel)
        torch.nn.Module.__init__(model)
        model.encoder = TinyEncoder()
        model.affinity = torch.nn.Bilinear(4, 4, 1, bias=False)
        model.encoder_name = "test"
        model.tokenizer = None
        return model

    model = tiny_model()
    module = AffinityTrainingModule(
        model,
        learning_rate=2e-5,
        negative_weight=5.0,
        thresholds=[0.5],
        connectivities=[4],
    )
    monkeypatch.setattr(module, "log", lambda *_args, **_kwargs: None)
    batch = collate_examples(
        [
            {
                "id": "smoke",
                "input_ids": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
                "context_mask": [0, 1, 0, 0],
                "chunk_mask": [0, 0, 1, 0],
                "target_boxes": [[1, 2, 2, 3]],
            }
        ],
        pad_token_id=0,
    )
    optimizer = module.configure_optimizers()
    loss = module.training_step(batch, 0)
    loss.backward()
    optimizer.step()

    checkpoint = tmp_path / "smoke.ckpt"
    torch.save(
        {"state_dict": {f"model.{key}": value for key, value in model.state_dict().items()}},
        checkpoint,
    )
    restored = tiny_model()
    restored.load_state_dict(
        _checkpoint_state(torch.load(checkpoint, weights_only=True)),
        strict=True,
    )
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_published_span_labels_are_matched_by_relation_suffix():
    from dataset import tokenize_example

    class Encoding(dict):
        def sequence_ids(self):
            return [None, 0, 0, None, 1, 1, None]

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return Encoding(
                input_ids=[1, 2, 3, 4, 5, 6, 7],
                attention_mask=[1] * 7,
                offset_mapping=[
                    (0, 0), (0, 5), (6, 9), (0, 0),
                    (0, 3), (4, 8), (0, 0),
                ],
            )

    row = {
        "id": "published",
        "context": "Alice met Bob.",
        "chunk": "She left.",
        "context_spans": [{"label": "ANTECEDENT_1", "start": 0, "end": 5}],
        "chunk_spans": [{"label": "ANAPHOR_1", "start": 0, "end": 3}],
    }
    encoded = tokenize_example(row, Tokenizer())
    assert encoded["target_boxes"] == [[1, 2, 4, 5]]
