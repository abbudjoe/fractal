from __future__ import annotations

from dataclasses import dataclass

from .common import StringEnum, ValidationError, ensure_non_negative, ensure_positive


class SymbolicModelFamily(StringEnum):
    PAPER_COMPLEX_EML = "paper-complex-eml"
    STABLE_REAL_EML = "stable-real-eml"
    GENERIC_TREE = "generic-tree"
    SMALL_MLP = "small-mlp"


class SymbolicPreset(StringEnum):
    SMOKE = "smoke"
    COMPACT = "compact"
    TIER0_EXACT = "tier0-exact"


class SymbolicTreeOptimizer(StringEnum):
    AUTODIFF = "autodiff"
    TORCH_AUTODIFF = "torch-autodiff"
    SPSA = "spsa"


@dataclass(frozen=True)
class SymbolicDatasetSpec:
    train_samples: int = 48
    validation_samples: int = 96
    extrapolation_samples: int = 96
    tasks_per_depth: int = 2

    def validate(self) -> None:
        ensure_positive(self.train_samples, "symbolic_dataset.train_samples")
        ensure_positive(self.validation_samples, "symbolic_dataset.validation_samples")
        ensure_positive(self.extrapolation_samples, "symbolic_dataset.extrapolation_samples")
        ensure_positive(self.tasks_per_depth, "symbolic_dataset.tasks_per_depth")
        if self.tasks_per_depth > 2:
            raise ValidationError("symbolic_dataset.tasks_per_depth currently supports at most 2")


@dataclass(frozen=True)
class SymbolicTrainSpec:
    steps: int = 120
    tree_learning_rate: float = 0.045
    mlp_learning_rate: float = 0.025
    spsa_perturbation: float = 0.035
    initial_temperature: float = 1.0
    final_temperature: float = 0.28
    snap_penalty_weight: float = 0.01
    hardening_tolerance_multiplier: float = 6.0
    hidden_units: int = 12
    tree_optimizer: SymbolicTreeOptimizer = SymbolicTreeOptimizer.AUTODIFF
    paper_restarts: int = 1

    def validate(self) -> None:
        ensure_positive(self.steps, "symbolic_train.steps")
        ensure_positive(self.hidden_units, "symbolic_train.hidden_units")
        ensure_positive(self.paper_restarts, "symbolic_train.paper_restarts")
        if self.tree_learning_rate <= 0.0:
            raise ValidationError("symbolic_train.tree_learning_rate must be greater than zero")
        if self.mlp_learning_rate <= 0.0:
            raise ValidationError("symbolic_train.mlp_learning_rate must be greater than zero")
        if self.spsa_perturbation <= 0.0:
            raise ValidationError("symbolic_train.spsa_perturbation must be greater than zero")
        if self.initial_temperature <= 0.0 or self.final_temperature <= 0.0:
            raise ValidationError("symbolic_train temperatures must be greater than zero")
        if self.final_temperature > self.initial_temperature:
            raise ValidationError("symbolic_train.final_temperature must be <= initial_temperature")
        if self.snap_penalty_weight < 0.0:
            raise ValidationError("symbolic_train.snap_penalty_weight must be non-negative")
        if self.hardening_tolerance_multiplier < 1.0:
            raise ValidationError("symbolic_train.hardening_tolerance_multiplier must be >= 1")
        if not isinstance(self.tree_optimizer, SymbolicTreeOptimizer):
            raise ValidationError("symbolic_train.tree_optimizer must be a SymbolicTreeOptimizer")


@dataclass(frozen=True)
class SymbolicBenchmarkManifest:
    run_label: str
    preset: SymbolicPreset
    model_families: tuple[SymbolicModelFamily, ...]
    seeds: tuple[int, ...]
    dataset: SymbolicDatasetSpec
    train: SymbolicTrainSpec
    implementation_kind: str = "python_stdlib_symbolic"
    backend: str = "cpu"
    note: str = ""

    def validate(self) -> None:
        if not self.run_label.strip():
            raise ValidationError("symbolic_manifest.run_label must not be empty")
        if not self.model_families:
            raise ValidationError("symbolic_manifest.model_families must not be empty")
        if not self.seeds:
            raise ValidationError("symbolic_manifest.seeds must not be empty")
        for seed in self.seeds:
            ensure_non_negative(seed, "symbolic_manifest.seed")
        if self.backend not in {"cpu", "mps", "auto"}:
            raise ValidationError("symbolic benchmark backend must be one of cpu|mps|auto")
        if self.backend != "cpu" and self.train.tree_optimizer is not SymbolicTreeOptimizer.TORCH_AUTODIFF:
            raise ValidationError("symbolic benchmark backend=mps|auto requires tree_optimizer=torch-autodiff")
        self.dataset.validate()
        self.train.validate()


def preset_manifest(
    *,
    preset: SymbolicPreset,
    run_label: str,
    seeds: tuple[int, ...],
    model_families: tuple[SymbolicModelFamily, ...],
) -> SymbolicBenchmarkManifest:
    if preset is SymbolicPreset.SMOKE:
        dataset = SymbolicDatasetSpec(
            train_samples=20,
            validation_samples=32,
            extrapolation_samples=32,
            tasks_per_depth=1,
        )
        train = SymbolicTrainSpec(steps=24, hidden_units=8)
    elif preset is SymbolicPreset.COMPACT:
        dataset = SymbolicDatasetSpec()
        train = SymbolicTrainSpec()
    elif preset is SymbolicPreset.TIER0_EXACT:
        dataset = SymbolicDatasetSpec(
            train_samples=40,
            validation_samples=80,
            extrapolation_samples=80,
            tasks_per_depth=1,
        )
        train = SymbolicTrainSpec(
            steps=240,
            tree_learning_rate=0.035,
            mlp_learning_rate=0.018,
            initial_temperature=0.75,
            final_temperature=0.12,
            snap_penalty_weight=0.02,
            hidden_units=10,
            tree_optimizer=SymbolicTreeOptimizer.AUTODIFF,
            paper_restarts=6,
        )
    else:
        raise ValidationError(f"unsupported symbolic preset: {preset}")
    return SymbolicBenchmarkManifest(
        run_label=run_label,
        preset=preset,
        model_families=model_families,
        seeds=seeds,
        dataset=dataset,
        train=train,
        note=(
            "Paper-aligned symbolic benchmark for EML-style trees. The paper-complex arm "
            "uses complex EML internally with bounded exp/log guards; the stable-real arm "
            "is the repo's practical real-valued gated square surrogate family."
        ),
    )
