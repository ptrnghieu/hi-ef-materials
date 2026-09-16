import unittest

import torch

from train_contextual_affective_residual import (
    ContextualAffectiveResidual,
    compute_losses,
    conditional_supervised_contrastive_loss,
    null_consistency_loss,
)


def synthetic_batch(batch_size: int = 4) -> dict[str, object]:
    batch: dict[str, object] = {
        "target": torch.tensor([0, 1, 2, 3][:batch_size]),
        "party_a_target": torch.tensor([0, 0, 1, 1][:batch_size]),
        "source_folder": ["01", "02", "01", "02"][:batch_size],
    }
    for position in (1, 2, 3):
        batch[f"clip{position}"] = {
            "face": torch.randn(batch_size, 16, 512),
            "ori": torch.randn(batch_size, 16, 512),
            "text": torch.randn(batch_size, 512),
            "audio": torch.randn(batch_size, 527),
            "face_mask": torch.ones(batch_size, 16, dtype=torch.bool),
            "audio_found": torch.ones(batch_size, dtype=torch.bool),
        }
    return batch


class CanonicalArchitectureTests(unittest.TestCase):
    def build_model(self, variant: str) -> ContextualAffectiveResidual:
        return ContextualAffectiveResidual(
            variant=variant,
            num_source_folders=2,
            d_model=16,
            temporal_layers=1,
            context_layers=1,
            dropout=0.0,
            face_pooling="masked",
        )

    def test_final_logits_are_exact_context_plus_delta(self) -> None:
        batch = synthetic_batch()
        for variant in ("context", "affect", "interaction", "both"):
            with self.subTest(variant=variant):
                output = self.build_model(variant)(batch)
                torch.testing.assert_close(
                    output["final_logits"],
                    output["context_logits"] + output["delta_logits"],
                )

    def test_context_ablation_has_zero_delta(self) -> None:
        output = self.build_model("context")(synthetic_batch())
        torch.testing.assert_close(
            output["delta_logits"], torch.zeros_like(output["delta_logits"])
        )
        torch.testing.assert_close(output["final_logits"], output["context_logits"])

    def test_null_delta_depends_only_on_context_and_zero_a(self) -> None:
        model = self.build_model("both").eval()
        batch = synthetic_batch()
        altered = synthetic_batch()
        altered["clip1"] = batch["clip1"]
        altered["clip2"] = batch["clip2"]
        with torch.no_grad():
            first = model(batch)
            second = model(altered)
        torch.testing.assert_close(first["null_logits"], second["null_logits"])

    def test_affect_loss_does_not_update_interaction_branch(self) -> None:
        model = self.build_model("both")
        output = model(synthetic_batch())
        output["affect_logits"].sum().backward()
        interaction_gradients = [
            parameter.grad for parameter in model.interaction_encoder.parameters()
        ]
        self.assertTrue(all(gradient is None for gradient in interaction_gradients))

    def test_all_agreed_losses_are_finite(self) -> None:
        batch = synthetic_batch()
        model = self.build_model("both")
        losses = compute_losses(
            model(batch),
            batch,
            "both",
            {"01": 0, "02": 1},
            {"context": 1.0, "emotion": 1.0, "contrastive": 1.0,
             "null": 1.0, "nuisance": 1.0},
            0.1,
            "symmetric",
        )
        self.assertEqual(
            set(losses),
            {"total", "final", "context", "emotion", "contrastive", "null", "nuisance"},
        )
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))

    def test_conditional_contrastive_requires_matched_pairs(self) -> None:
        representations = torch.randn(4, 8, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        loss = conditional_supervised_contrastive_loss(
            representations, labels, ["01", "02", "01", "02"], 0.1
        )
        self.assertGreaterEqual(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(representations.grad)

    def test_null_divergence_is_zero_for_identical_logits(self) -> None:
        logits = torch.randn(5, 7)
        for direction in ("context-to-null", "null-to-context", "symmetric"):
            with self.subTest(direction=direction):
                self.assertAlmostEqual(
                    float(null_consistency_loss(logits, logits, direction)), 0.0, places=6
                )


if __name__ == "__main__":
    unittest.main()
