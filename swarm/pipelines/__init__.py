"""Dataset collection, causal supervision, unrolled DBN training, statistical evaluation, & scaling benchmarks."""

from __future__ import annotations

from .benchmark import (
    DEFAULT_BENCHMARK_METHODS,
    DEFAULT_SWARM_SIZES,
    build_benchmark_grid_topology,
    run_scaling_benchmark,
)
from .evaluator import (
    EVAL_SUMMARY_METRICS,
    evaluate_experiment_matrix,
    run_single_episode,
    validate_and_hash_checkpoint,
    validate_eval_seeds,
)
from .labels import (
    StepSupervisionLabels,
    compute_episode_causal_labels,
    compute_step_causal_labels,
)
from .rollout_collector import (
    RolloutSplit,
    collect_rollout_dataset,
    collect_single_rollout_episode,
    load_rollout_shards,
    select_behavior_policy,
    validate_rollout_split,
)
from .statistics import (
    DEFAULT_SUMMARY_METRICS,
    METRIC_LOWER_IS_BETTER,
    bootstrap_mean_ci,
    compute_paired_statistics,
    exact_wilcoxon_two_sided,
    holm_bonferroni_adjust,
    significance_stars,
)
from .trainer import (
    RolloutSequenceDataset,
    compute_step_multitask_loss,
    evaluate_heldout_predictive,
    shard_step_to_feature_batch,
    train_neural_dbn,
)

__all__ = [
    "DEFAULT_BENCHMARK_METHODS",
    "DEFAULT_SUMMARY_METRICS",
    "DEFAULT_SWARM_SIZES",
    "EVAL_SUMMARY_METRICS",
    "METRIC_LOWER_IS_BETTER",
    "RolloutSequenceDataset",
    "RolloutSplit",
    "StepSupervisionLabels",
    "bootstrap_mean_ci",
    "build_benchmark_grid_topology",
    "collect_rollout_dataset",
    "collect_single_rollout_episode",
    "compute_episode_causal_labels",
    "compute_paired_statistics",
    "compute_step_causal_labels",
    "compute_step_multitask_loss",
    "evaluate_experiment_matrix",
    "evaluate_heldout_predictive",
    "exact_wilcoxon_two_sided",
    "holm_bonferroni_adjust",
    "load_rollout_shards",
    "run_scaling_benchmark",
    "run_single_episode",
    "select_behavior_policy",
    "shard_step_to_feature_batch",
    "significance_stars",
    "train_neural_dbn",
    "validate_and_hash_checkpoint",
    "validate_eval_seeds",
    "validate_rollout_split",
]
