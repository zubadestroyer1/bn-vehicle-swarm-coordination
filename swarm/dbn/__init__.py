"""Neurally Updated Dynamic Bayesian Network (``swarm.dbn``) Package.

Exports candidate graph construction, schema-locked pure-observable feature extraction,
``STGraphTransformerDBN``, Generalized Causal Noisy-OR CPDs, exact Division-Free Prefix-Suffix
Sum-Product Loopy Belief Propagation, post-hoc temperature calibration, non-neural DBN baselines,
and ``NeuralDBNUpdater``.
"""

from __future__ import annotations

from .baselines_dbn import (
    HeuristicRadiusDBNUpdater,
    IndependentDBNUpdater,
    OnlineBICDBNUpdater,
)
from .calibration import (
    TEMP_MAX,
    TEMP_MIN,
    TemperatureScales,
    calibrate_temperature,
    calibrate_temperatures,
    compute_predictive_metrics,
    fit_binary_temperature,
)
from .candidates import (
    CandidateGraph,
    TopologyLookupCache,
    build_candidate_graph,
    get_topology_cache,
)
from .cpd import (
    CorridorSpillbackCPD,
    PrunedDAGStructure,
    expected_downstream_spillback,
    gate_and_prune_edges,
    generalized_noisy_or_base_survival,
    heuristic_corridor_spillback_prob,
    parent_survival_factors,
    static_edge_cpd_parameters,
    temporal_edge_prior,
)
from .features import (
    FeatureNormalizer,
    GraphFeatureBatch,
    extract_graph_features,
)
from .inference import (
    LBPInferenceResult,
    compute_cascade_scores,
    multiplicative_sum_product_lbp,
)
from .transformer import (
    ContinuousDecayGRU,
    MultiScaleBasisEncoder,
    RMSNorm,
    STGraphTransformerBlock,
    STGraphTransformerDBN,
    SwiGLU,
    TorchTemporalMemory,
    TransformerStepOutputs,
    segment_softmax,
)
from .updater import (
    DBNCheckpointBundle,
    DifferentiableStepOutput,
    NeuralDBNUpdater,
    build_dbn_updater,
    forward_dbn_step,
    load_dbn_checkpoint,
    save_dbn_checkpoint,
)

__all__ = [
    "CandidateGraph",
    "ContinuousDecayGRU",
    "CorridorSpillbackCPD",
    "DBNCheckpointBundle",
    "DifferentiableStepOutput",
    "FeatureNormalizer",
    "GraphFeatureBatch",
    "HeuristicRadiusDBNUpdater",
    "IndependentDBNUpdater",
    "LBPInferenceResult",
    "MultiScaleBasisEncoder",
    "NeuralDBNUpdater",
    "OnlineBICDBNUpdater",
    "PrunedDAGStructure",
    "RMSNorm",
    "STGraphTransformerBlock",
    "STGraphTransformerDBN",
    "SwiGLU",
    "TEMP_MAX",
    "TEMP_MIN",
    "TemperatureScales",
    "TopologyLookupCache",
    "TorchTemporalMemory",
    "TransformerStepOutputs",
    "build_candidate_graph",
    "build_dbn_updater",
    "calibrate_temperature",
    "calibrate_temperatures",
    "compute_cascade_scores",
    "compute_predictive_metrics",
    "expected_downstream_spillback",
    "extract_graph_features",
    "fit_binary_temperature",
    "forward_dbn_step",
    "gate_and_prune_edges",
    "generalized_noisy_or_base_survival",
    "get_topology_cache",
    "heuristic_corridor_spillback_prob",
    "load_dbn_checkpoint",
    "multiplicative_sum_product_lbp",
    "parent_survival_factors",
    "save_dbn_checkpoint",
    "segment_softmax",
    "static_edge_cpd_parameters",
    "temporal_edge_prior",
]
