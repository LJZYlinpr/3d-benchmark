"""
Declarative method specification — the single entry point for third-party methods.

Why this exists
---------------
Every formal campaign so far ran through a bespoke ``benchmark/scripts/rerun_<method>_*.py``
runner (58 such scripts as of 2026-09-18). That works for the methods we adapted by hand,
but it means an outside author cannot plug their reconstruction method into this benchmark
without first writing a runner. This module replaces "write a runner" with "write a
``method_spec.yaml``".

What it enforces (the project rules, as code rather than as prose)
------------------------------------------------------------------
* ``.claude/rules/data-leakage.md`` — the rendered command is checked against the task's
  hold-out slice labels and ground-truth path. A spec that would hand a method its
  hold-out data is rejected before the method starts, not audited afterwards.
* ``.claude/rules/scientific-scope.md`` — capability flags decide task eligibility. A
  method that cannot legally run a task yields ``not_applicable_task`` with a reason
  string; it never yields a zero.
* ``agent_control/EXPERIMENT_RESOURCE_POLICY.md`` §5 — a spec without a declared T_max
  basis is invalid, so no run can start without a completion-time ceiling.
* ``agent_control/OPEN_ISSUES.md`` #1 — the capability audit becomes a machine-checked
  form (``templates/capability_audit.yaml``) instead of seven prose questions.

This module holds the reusable logic; ``benchmark/scripts/run_method.py`` is the thin CLI
(AGENTS.md §1.6: reusable logic in ``.py``, visible evidence in ``.ipynb``).

Schema version: 1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import re
import shlex

import yaml


SPEC_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Capability vocabulary
# ---------------------------------------------------------------------------

CAPABILITY_FLAGS: Dict[str, bool] = {
    # Does the method output an expression matrix for the target slice?
    "predicts_expression": True,
    # Does it generate the target point cloud itself (free-coordinate eligible)?
    "generates_coordinates": False,
    # Does it require a registered histology image / WSI per slice?
    "needs_image": False,
    # Does it require a matched scRNA-seq reference?
    "needs_scrna": False,
    # Does the method insist on aligning the slices with its own aligner, instead of
    # using the coordinates the dataset already carries? The aligner is part of the
    # DATASET, not the method (PLAN.md §12.2.6): if two methods align the same data
    # differently, a comparison between them is partly a comparison between two
    # aligners. True here is therefore an eligibility question, not a footnote.
    "needs_external_alignment": False,
}
"""Capability flags with their defaults. Unknown flags are rejected by the loader."""

PROTOCOLS: Set[str] = {"fixed_coordinate", "free_coordinate"}

ENV_KINDS: Set[str] = {"conda", "venv", "none"}

# Placeholders the runner will substitute. Anything else in a command template is a
# typo and must fail loudly rather than expand to an empty string.
ALLOWED_PLACEHOLDERS: Set[str] = {
    "python",        # interpreter for the declared environment
    "method_dir",    # third-party code directory (methods/<name>, never modified)
    "input_h5ad",    # visible-only input, staged by the runner
    "query_npz",     # fixed_coordinate only: target x/y query points
    "out_dir",       # per-run output directory
    "pred_h5ad",     # where the method must write its standardized prediction
    "seed",
    "n_threads",
    "gpu_id",
}

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


# ---------------------------------------------------------------------------
# Failure / eligibility vocabulary (AGENTS.md §8.2)
# ---------------------------------------------------------------------------

STATUS_NOT_APPLICABLE_TASK = "not_applicable_task"
STATUS_ELIGIBLE = "eligible"


class SpecError(ValueError):
    """Raised when a method spec is malformed. Never downgraded to a warning."""


@dataclass
class Verdict:
    """Outcome of a capability audit against one task."""

    status: str
    reasons: List[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return self.status == STATUS_ELIGIBLE

    def report(self) -> str:
        head = "ELIGIBLE" if self.eligible else self.status.upper()
        if not self.reasons:
            return head
        return head + "\n" + "\n".join(f"  - {r}" for r in self.reasons)


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

@dataclass
class MethodSpec:
    """A validated declarative method specification."""

    name: str
    capabilities: Dict[str, bool]
    env_kind: str
    env_ref: Optional[str]
    python: str
    command: str
    method_dir: Optional[str]
    cwd: Optional[str]
    t_max_basis: Dict[str, Any]
    resources: Dict[str, Any]
    supported_platforms: Optional[List[str]]
    supported_resolutions: Optional[List[str]]
    supported_protocols: List[str]
    min_visible_slices: int
    upstream: Dict[str, Any]
    raw: Dict[str, Any]

    # -- capability audit ---------------------------------------------------

    def audit(self, task: Dict[str, Any]) -> Verdict:
        """Decide whether this method may legally run ``task``.

        Returns ``not_applicable_task`` with reasons instead of letting an
        incompatible method produce a number that would later be read as a score.
        """
        reasons: List[str] = []

        protocol = task.get("protocol") or (task.get("evaluation") or {}).get(
            "coordinate_protocol"
        )
        if protocol and protocol not in self.supported_protocols:
            reasons.append(
                f"task protocol '{protocol}' not in supported_protocols "
                f"{self.supported_protocols}"
            )

        if protocol == "free_coordinate" and not self.capabilities["generates_coordinates"]:
            reasons.append(
                "free_coordinate task requires generates_coordinates=true "
                "(a method that only fills expression at given coordinates cannot "
                "be scored on geometry)"
            )

        if not self.capabilities["predicts_expression"] and _task_scores_expression(task):
            reasons.append(
                "task evaluates expression recovery but predicts_expression=false"
            )

        platform = _task_platform(task)
        if platform and self.supported_platforms is not None:
            if platform not in self.supported_platforms:
                reasons.append(
                    f"dataset platform '{platform}' not in supported_platforms "
                    f"{self.supported_platforms}"
                )

        resolution = _task_resolution(task)
        if resolution and self.supported_resolutions is not None:
            if resolution not in self.supported_resolutions:
                reasons.append(
                    f"dataset resolution '{resolution}' not in supported_resolutions "
                    f"{self.supported_resolutions}"
                )

        n_visible = (task.get("split") or {}).get("n_visible")
        if n_visible is not None and n_visible < self.min_visible_slices:
            reasons.append(
                f"task provides {n_visible} visible slices, method declares "
                f"min_visible_slices={self.min_visible_slices}"
            )

        if self.capabilities["needs_image"] and not _task_has_image(task):
            reasons.append("method needs_image=true but task declares no image modality")

        if self.capabilities["needs_scrna"] and not _task_has_scrna(task):
            reasons.append("method needs_scrna=true but task declares no scRNA reference")

        if reasons:
            return Verdict(STATUS_NOT_APPLICABLE_TASK, reasons)
        return Verdict(STATUS_ELIGIBLE)

    # -- command rendering --------------------------------------------------

    def render_command(self, context: Dict[str, Any]) -> List[str]:
        """Render ``command`` into an argv list.

        Unknown placeholders are an error (a silent empty expansion would change what
        the method is run with, which is exactly the kind of drift this benchmark
        cannot tolerate).
        """
        used = set(_PLACEHOLDER_RE.findall(self.command))
        unknown = used - ALLOWED_PLACEHOLDERS
        if unknown:
            raise SpecError(
                f"[{self.name}] command uses unknown placeholder(s): "
                f"{sorted(unknown)}; allowed: {sorted(ALLOWED_PLACEHOLDERS)}"
            )
        missing = used - set(context)
        if missing:
            raise SpecError(
                f"[{self.name}] command needs placeholder(s) {sorted(missing)} "
                "but the run context does not provide them"
            )
        rendered = self.command
        for key in used:
            rendered = rendered.replace("{" + key + "}", str(context[key]))
        return shlex.split(rendered)


# ---------------------------------------------------------------------------
# Task-side helpers (kept tolerant: task yamls predate this module)
# ---------------------------------------------------------------------------

def _task_platform(task: Dict[str, Any]) -> Optional[str]:
    return (task.get("dataset") or {}).get("platform")


def _task_resolution(task: Dict[str, Any]) -> Optional[str]:
    return (task.get("dataset") or {}).get("resolution")


def _task_has_image(task: Dict[str, Any]) -> bool:
    mods = (task.get("dataset") or {}).get("modalities") or []
    return any(str(m).lower() in {"he", "h&e", "image", "wsi"} for m in mods)


def _task_has_scrna(task: Dict[str, Any]) -> bool:
    mods = (task.get("dataset") or {}).get("modalities") or []
    return any(str(m).lower() in {"scrna", "scrna-seq", "scrnaseq"} for m in mods)


def _task_scores_expression(task: Dict[str, Any]) -> bool:
    """A task scores expression unless it explicitly says it is geometry-only."""
    ev = task.get("evaluation") or {}
    return not bool(ev.get("geometry_only", False))


def holdout_labels(task: Dict[str, Any]) -> List[str]:
    return list((task.get("split") or {}).get("holdout_slices") or [])


# ---------------------------------------------------------------------------
# Leakage gate
# ---------------------------------------------------------------------------

def assert_command_leakage_free(
    argv: List[str],
    task: Dict[str, Any],
    allow_query_coords: Optional[bool] = None,
) -> None:
    """Refuse a command that would hand the method hold-out data.

    Checks, per ``.claude/rules/data-leakage.md``:

    1. no hold-out slice label appears anywhere in the argv;
    2. the task's raw dataset path (which contains hold-out expression) is not passed —
       methods read the runner-staged visible-only input instead;
    3. under ``free_coordinate``, no query-coordinate file is passed at all, since the
       target XY is part of what must be predicted.

    Raises ``SpecError`` on violation. Silence means the argv passed all three.
    """
    joined = " ".join(argv)

    for label in holdout_labels(task):
        if label and label in joined:
            raise SpecError(
                f"leakage gate: hold-out slice label '{label}' appears in the method "
                f"command. Hold-out identity/expression must not reach the method."
            )

    raw_path = (task.get("dataset") or {}).get("path")
    if raw_path and str(raw_path) in joined:
        raise SpecError(
            "leakage gate: the full dataset path is passed to the method "
            f"('{raw_path}'). It contains hold-out expression; pass the staged "
            "visible-only input ({input_h5ad}) instead."
        )

    protocol = task.get("protocol") or (task.get("evaluation") or {}).get(
        "coordinate_protocol"
    )
    if allow_query_coords is None:
        allow_query_coords = protocol == "fixed_coordinate"
    if not allow_query_coords and any("query" in a for a in argv):
        raise SpecError(
            "leakage gate: a query-coordinate argument is passed under protocol "
            f"'{protocol}'. Free-coordinate tasks hide the target XY — the method "
            "must generate it."
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL = {"schema_version", "method", "capabilities", "environment",
                       "execution", "resources"}


def load_spec(path: str | Path) -> MethodSpec:
    """Load and validate a ``method_spec.yaml``.

    Every validation failure raises ``SpecError``; nothing is defaulted silently
    except the capability flags listed in ``CAPABILITY_FLAGS``.
    """
    path = Path(path)
    if not path.exists():
        raise SpecError(f"method spec not found: {path}")

    with path.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise SpecError(f"{path}: spec must be a YAML mapping")

    missing = _REQUIRED_TOP_LEVEL - set(raw)
    if missing:
        raise SpecError(f"{path}: missing required section(s) {sorted(missing)}")

    version = raw["schema_version"]
    if version != SPEC_SCHEMA_VERSION:
        raise SpecError(
            f"{path}: schema_version {version!r} unsupported "
            f"(this module speaks {SPEC_SCHEMA_VERSION})"
        )

    method = raw["method"] or {}
    name = method.get("name")
    if not name:
        raise SpecError(f"{path}: method.name is required")

    # -- capabilities: known keys only, booleans only -----------------------
    caps_in = raw["capabilities"] or {}
    unknown = set(caps_in) - set(CAPABILITY_FLAGS) - {
        "supported_platforms", "supported_resolutions", "supported_protocols",
        "min_visible_slices",
    }
    if unknown:
        raise SpecError(
            f"{path}: unknown capability key(s) {sorted(unknown)}; "
            f"known flags: {sorted(CAPABILITY_FLAGS)}"
        )
    caps = dict(CAPABILITY_FLAGS)
    for key in CAPABILITY_FLAGS:
        if key in caps_in:
            val = caps_in[key]
            if not isinstance(val, bool):
                raise SpecError(
                    f"{path}: capabilities.{key} must be a boolean, got {val!r}"
                )
            caps[key] = val

    supported_protocols = caps_in.get("supported_protocols") or ["fixed_coordinate"]
    bad = set(supported_protocols) - PROTOCOLS
    if bad:
        raise SpecError(
            f"{path}: unknown protocol(s) {sorted(bad)}; known: {sorted(PROTOCOLS)}"
        )
    if "free_coordinate" in supported_protocols and not caps["generates_coordinates"]:
        raise SpecError(
            f"{path}: supported_protocols includes 'free_coordinate' but "
            "capabilities.generates_coordinates is false — a method that does not "
            "generate coordinates cannot be scored on geometry."
        )
    if not caps["predicts_expression"] and not caps["generates_coordinates"]:
        raise SpecError(
            f"{path}: the method declares neither predicts_expression nor "
            "generates_coordinates — there is nothing this benchmark could score. "
            "Structure/embedding-only methods need a separate track "
            "(agent_control/OPEN_ISSUES.md #2)."
        )

    # -- environment --------------------------------------------------------
    env = raw["environment"] or {}
    env_kind = env.get("kind")
    if env_kind not in ENV_KINDS:
        raise SpecError(
            f"{path}: environment.kind must be one of {sorted(ENV_KINDS)}, "
            f"got {env_kind!r}"
        )
    env_ref = env.get("name") or env.get("path")
    if env_kind in {"conda", "venv"} and not env_ref:
        raise SpecError(
            f"{path}: environment.kind='{env_kind}' requires environment.name "
            "(conda env name) or environment.path (venv prefix)"
        )

    # -- execution ----------------------------------------------------------
    execution = raw["execution"] or {}
    command = execution.get("command")
    if not command:
        raise SpecError(f"{path}: execution.command is required")
    if "{pred_h5ad}" not in command and "{out_dir}" not in command:
        raise SpecError(
            f"{path}: execution.command must pass {{pred_h5ad}} or {{out_dir}} so the "
            "runner knows where the standardized prediction is written"
        )

    # -- T_max basis (EXPERIMENT_RESOURCE_POLICY §5) ------------------------
    resources = raw["resources"] or {}
    t_max = resources.get("t_max") or {}
    needed = {"measured_sec_per_step", "n_steps"}
    if not needed <= set(t_max) and "expected_wall_sec" not in t_max:
        raise SpecError(
            f"{path}: resources.t_max must declare either "
            "{measured_sec_per_step, n_steps} (preferred: probe-measured) or "
            "expected_wall_sec. Every run needs a completion-time ceiling before it "
            "starts (agent_control/EXPERIMENT_RESOURCE_POLICY.md §5)."
        )

    return MethodSpec(
        name=name,
        capabilities=caps,
        env_kind=env_kind,
        env_ref=env_ref,
        python=env.get("python", "python"),
        command=command,
        method_dir=execution.get("method_dir"),
        cwd=execution.get("cwd"),
        t_max_basis=t_max,
        resources=resources,
        supported_platforms=caps_in.get("supported_platforms"),
        supported_resolutions=caps_in.get("supported_resolutions"),
        supported_protocols=list(supported_protocols),
        min_visible_slices=int(caps_in.get("min_visible_slices", 2)),
        upstream=method.get("upstream") or {},
        raw=raw,
    )


def t_max_seconds(spec: MethodSpec, margin: float = 1.5, overhead_sec: float = 1800.0) -> float:
    """T_max = measured per-step time x steps x margin + setup overhead.

    The formula is the project's, not this module's
    (``agent_control/EXPERIMENT_RESOURCE_POLICY.md`` §5); it lives here so no runner
    has to re-derive it.
    """
    basis = spec.t_max_basis
    if "measured_sec_per_step" in basis and "n_steps" in basis:
        core = float(basis["measured_sec_per_step"]) * float(basis["n_steps"])
    else:
        core = float(basis["expected_wall_sec"])
    return core * margin + overhead_sec


def load_task(path: str | Path) -> Dict[str, Any]:
    """Load a frozen task yaml (``benchmark/configs/tasks/<family>/<task>.yaml``)."""
    path = Path(path)
    if not path.exists():
        raise SpecError(f"task file not found: {path}")
    with path.open() as fh:
        task = yaml.safe_load(fh)
    if not isinstance(task, dict):
        raise SpecError(f"{path}: task must be a YAML mapping")
    if "task_id" not in task:
        raise SpecError(f"{path}: task_id is required")
    return task
