import unittest
from pathlib import Path

import pandas as pd
import torch

from train_reliability_gated_residual import (
    ReliabilityGatedResidual,
    compute_reliability_losses,
    context_consistency_loss,
    relative_gate_ranking_loss,
    uses_counterfactual,
    uses_gate,
    uses_relative_ranking,
)


ROOT = Path(__file__).resolve().parents[1]


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


class ReliabilityArchitectureTests(unittest.TestCase):
    def build_model(self, variant: str) -> ReliabilityGatedResidual:
        return ReliabilityGatedResidual(
            variant=variant,
            num_source_folders=2,
            d_model=16,
            temporal_layers=1,
            context_layers=1,
            dropout=0.0,
            face_pooling="masked",
        )

    def test_frozen_v09_flags(self) -> None:
        self.assertFalse(uses_gate("ungated"))
        self.assertFalse(uses_relative_ranking("ungated"))
        self.assertTrue(uses_gate("relative_gate"))
        self.assertTrue(uses_relative_ranking("relative_gate"))
        self.assertFalse(uses_counterfactual("relative_gate"))
        self.assertTrue(uses_gate("relative_gate_cf"))
        self.assertTrue(uses_relative_ranking("relative_gate_cf"))
        self.assertTrue(uses_counterfactual("relative_gate_cf"))

    def test_effective_residual_invariant(self) -> None:
        batch = synthetic_batch()
        for variant in ("ungated", "relative_gate", "relative_gate_cf"):
            with self.subTest(variant=variant):
                output = self.build_model(variant)(batch)
                torch.testing.assert_close(
                    output["delta_logits"],
                    output["reliability_gate"] * output["raw_delta_logits"],
                )
                torch.testing.assert_close(
                    output["final_logits"],
                    output["context_logits"] + output["delta_logits"],
                )

    def test_ungated_variants_are_exactly_one(self) -> None:
        output = self.build_model("ungated")(synthetic_batch())
        torch.testing.assert_close(output["reliability_gate"], torch.ones(4, 1))
        torch.testing.assert_close(output["delta_logits"], output["raw_delta_logits"])

    def test_learned_gate_is_scalar_and_bounded(self) -> None:
        output = self.build_model("relative_gate_cf")(synthetic_batch())
        self.assertEqual(tuple(output["reliability_gate"].shape), (4, 1))
        self.assertTrue(bool((output["reliability_gate"] > 0).all()))
        self.assertTrue(bool((output["reliability_gate"] < 1).all()))
        torch.testing.assert_close(
            output["reliability_gate"], torch.full((4, 1), 0.5)
        )

    def test_all_factorial_cells_have_finite_backward_path(self) -> None:
        batch = synthetic_batch()
        for variant in ("ungated", "relative_gate", "relative_gate_cf"):
            with self.subTest(variant=variant):
                model = self.build_model(variant)
                losses = compute_reliability_losses(
                    model,
                    model(batch),
                    batch,
                    {"01": 0, "02": 1},
                    {
                        "context": 1.0, "emotion": 0.5, "contrastive": 0.1,
                        "null": 1.0, "nuisance": 0.05,
                        "counterfactual": 1.0, "ranking": 0.1,
                    },
                    torch.ones(7),
                    0.1,
                    "context-to-null",
                    0.2,
                )
                self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                losses["total"].backward()
                self.assertTrue(any(
                    parameter.grad is not None for parameter in model.parameters()
                ))
                if uses_relative_ranking(variant):
                    self.assertGreaterEqual(float(losses["ranking"]), 0.0)
                else:
                    self.assertEqual(float(losses["ranking"]), 0.0)
                if uses_counterfactual(variant):
                    self.assertGreaterEqual(float(losses["counterfactual"]), 0.0)
                else:
                    self.assertEqual(float(losses["counterfactual"]), 0.0)

    def test_invalid_party_a_is_paired_cyclically(self) -> None:
        model = self.build_model("relative_gate_cf")
        captured: dict[str, torch.Tensor] = {}

        def fake_residual(context: torch.Tensor, affect: torch.Tensor, interaction: torch.Tensor) -> torch.Tensor:
            captured["context"] = context
            captured["affect"] = affect
            captured["interaction"] = interaction
            return context.new_zeros((context.size(0), 7))

        model._residual = fake_residual  # type: ignore[method-assign]
        model.reliability_logit = lambda c, a, i: c.new_zeros((c.size(0), 1))  # type: ignore[method-assign]
        context = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        affect = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 100
        interaction = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 200
        model.invalid_party_a({
            "context_representation": context,
            "affect_representation": affect,
            "interaction_representation": interaction,
            "context_logits": torch.zeros(3, 7),
        })
        torch.testing.assert_close(captured["context"], context)
        torch.testing.assert_close(captured["affect"], affect.roll(1, dims=0))
        torch.testing.assert_close(captured["interaction"], interaction.roll(1, dims=0))
        self.assertFalse(bool((captured["affect"] == affect).all(dim=1).any()))

    def test_singleton_invalid_party_a_uses_zero(self) -> None:
        model = self.build_model("relative_gate_cf")
        captured: list[torch.Tensor] = []

        def fake_residual(context: torch.Tensor, affect: torch.Tensor, interaction: torch.Tensor) -> torch.Tensor:
            captured.extend([affect, interaction])
            return context.new_zeros((1, 7))

        model._residual = fake_residual  # type: ignore[method-assign]
        model.invalid_party_a({
            "context_representation": torch.randn(1, 16),
            "affect_representation": torch.randn(1, 16),
            "interaction_representation": torch.randn(1, 16),
            "context_logits": torch.zeros(1, 7),
        })
        for value in captured:
            torch.testing.assert_close(value, torch.zeros_like(value))

    def test_context_teacher_is_detached(self) -> None:
        context = torch.randn(4, 7, requires_grad=True)
        invalid = torch.randn(4, 7, requires_grad=True)
        context_consistency_loss(context, invalid).backward()
        self.assertIsNone(context.grad)
        self.assertGreater(float(invalid.grad.abs().sum()), 0.0)

    def test_relative_ranking_pushes_real_up_and_counterfactual_down(self) -> None:
        real = torch.full((4, 1), 0.5, requires_grad=True)
        counterfactual = torch.full((4, 1), 0.5, requires_grad=True)
        relative_gate_ranking_loss(real, counterfactual, 0.2).backward()
        self.assertTrue(bool((real.grad < 0).all()))
        self.assertTrue(bool((counterfactual.grad > 0).all()))

    def test_relative_ranking_is_autocast_safe(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        real = torch.full((4, 1), 0.5, device=device, requires_grad=True)
        counterfactual = torch.full((4, 1), 0.5, device=device, requires_grad=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            loss = relative_gate_ranking_loss(real, counterfactual, 0.2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(real.grad)
        self.assertIsNotNone(counterfactual.grad)

    def test_satisfied_relative_margin_has_zero_loss(self) -> None:
        real = torch.full((4, 1), 0.8)
        counterfactual = torch.full((4, 1), 0.5)
        self.assertEqual(float(relative_gate_ranking_loss(real, counterfactual, 0.2)), 0.0)


class InnerManifestTests(unittest.TestCase):
    def test_inner_manifest_contains_only_original_train(self) -> None:
        path = ROOT / "experiments/manifests/inner_development_seed8042.csv"
        rows = pd.read_csv(path, dtype={"source_folder": str})
        self.assertEqual(set(rows["original_split"]), {"train"})
        self.assertEqual(set(rows["split"]), {"inner_train", "inner_development"})
        self.assertEqual(len(rows), 1993)
        train = rows[rows["split"] == "inner_train"]
        development = rows[rows["split"] == "inner_development"]
        self.assertFalse(set(train["source_folder"]) & set(development["source_folder"]))
        self.assertEqual(train["source_folder"].nunique(), 29)
        self.assertEqual(development["source_folder"].nunique(), 8)


if __name__ == "__main__":
    unittest.main()
