"""
Base adapter framework for benchmark methods.

Each method adapter must:
1. Accept a standardized input AnnData (conforming to input schema)
2. Run the method (possibly via subprocess + conda env isolation)
3. Return a standardized output AnnData (conforming to output schema)

The base class handles schema validation, runtime tracking, and error recording.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import time
import json
import tempfile
import subprocess
import os
import warnings

import anndata as ad

from benchmark.src.utils.validate import validate_input, validate_output, ValidationResult


# ---------------------------------------------------------------------------
# Failure classification (PLAN.md §13.6)
# ---------------------------------------------------------------------------

class FailureType:
    F1_ENV = "F1_env"           # Environment/engineering failure (fixable)
    F2_INTERNAL = "F2_internal"  # Method intrinsic failure (OOM, non-convergence)
    F3_NOT_APPLICABLE = "F3_not_applicable"  # Input modality mismatch


class AdapterError(Exception):
    """Raised when an adapter encounters a failure."""

    def __init__(self, message: str, failure_type: str = FailureType.F2_INTERNAL,
                 details: Optional[Dict] = None):
        super().__init__(message)
        self.failure_type = failure_type
        self.details = details or {}


# ---------------------------------------------------------------------------
# Adapter result
# ---------------------------------------------------------------------------

class AdapterResult:
    """Container for adapter execution results.

    Always records the outcome, whether success or failure.
    Failures are data, not noise (PLAN.md A5 axiom).
    """

    def __init__(
        self,
        method_name: str,
        output_path: Optional[Path] = None,
        output_adata: Optional[ad.AnnData] = None,
        runtime_sec: float = 0.0,
        config: Optional[Dict] = None,
        validation_result: Optional[ValidationResult] = None,
        error: Optional[str] = None,
        failure_type: Optional[str] = None,
        extra_info: Optional[Dict] = None,
    ):
        self.method_name = method_name
        self.output_path = output_path
        self.output_adata = output_adata
        self.runtime_sec = runtime_sec
        self.config = config or {}
        self.validation_result = validation_result
        self.error = error
        self.failure_type = failure_type
        self.extra_info = extra_info or {}

    @property
    def success(self) -> bool:
        return self.error is None and self.output_adata is not None

    @property
    def status(self) -> str:
        if self.success:
            return "success"
        if self.failure_type == FailureType.F3_NOT_APPLICABLE:
            return "not_applicable"
        return "failed"

    def to_dict(self) -> Dict:
        return {
            "method": self.method_name,
            "status": self.status,
            "runtime_sec": self.runtime_sec,
            "config": self.config,
            "error": self.error,
            "failure_type": self.failure_type,
            "validation_errors": (
                self.validation_result.errors if self.validation_result else []
            ),
            **self.extra_info,
        }


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class BaseAdapter(ABC):
    """Abstract base class for method adapters.

    Subclasses implement run_impl() which does the actual work.
    The base class handles validation, timing, and error recording.
    """

    # Subclasses must set these
    method_name: str = "base"
    conda_env: Optional[str] = None  # conda environment name, if using subprocess
    method_dir: Optional[Path] = None  # path to method code

    def __init__(self, work_dir: Optional[Path] = None, verbose: bool = False):
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix=f"adapter_{self.method_name}_"))
        self.verbose = verbose

    def run(
        self,
        input_path: Path,
        config: Optional[Dict] = None,
        output_dir: Optional[Path] = None,
    ) -> AdapterResult:
        """Run the method adapter end-to-end.

        Args:
            input_path: Path to standardized input h5ad.
            config: Method-specific configuration.
            output_dir: Directory for output files.

        Returns:
            AdapterResult with output, runtime, and validation status.
        """
        config = config or {}
        output_dir = output_dir or self.work_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()

        try:
            # Load and validate input
            adata_input = self._load_input(input_path)

            # Run the method
            output_adata = self.run_impl(adata_input, config, output_dir)

            # Validate output
            result = validate_output(output_adata)

            runtime = time.time() - t0

            if not result.is_valid:
                warnings.warn(
                    f"Output schema validation failed for {self.method_name}:\n"
                    f"{result.report()}"
                )

            # Ensure runtime recorded
            output_adata.uns["runtime_sec"] = runtime

            # Save output
            output_path = output_dir / f"{self.method_name}_output.h5ad"
            output_adata.write_h5ad(output_path)

            return AdapterResult(
                method_name=self.method_name,
                output_path=output_path,
                output_adata=output_adata,
                runtime_sec=runtime,
                config=config,
                validation_result=result,
            )

        except AdapterError as e:
            runtime = time.time() - t0
            return AdapterResult(
                method_name=self.method_name,
                runtime_sec=runtime,
                config=config,
                error=str(e),
                failure_type=e.failure_type,
                extra_info=e.details,
            )

        except Exception as e:
            runtime = time.time() - t0
            return AdapterResult(
                method_name=self.method_name,
                runtime_sec=runtime,
                config=config,
                error=f"{type(e).__name__}: {e}",
                failure_type=FailureType.F2_INTERNAL,
            )

    def _load_input(self, input_path: Path) -> ad.AnnData:
        """Load and validate input AnnData."""
        if not input_path.exists():
            raise AdapterError(
                f"Input file not found: {input_path}",
                failure_type=FailureType.F1_ENV,
            )

        adata = ad.read_h5ad(input_path)
        result = validate_input(adata)
        if not result.is_valid:
            raise AdapterError(
                f"Input schema validation failed:\n{result.report()}",
                failure_type=FailureType.F1_ENV,
                details={"validation_errors": result.errors},
            )

        return adata

    @abstractmethod
    def run_impl(
        self,
        adata_input: ad.AnnData,
        config: Dict,
        output_dir: Path,
    ) -> ad.AnnData:
        """Implement the actual method execution.

        Args:
            adata_input: Validated input AnnData.
            config: Method configuration.
            output_dir: Directory for intermediate files.

        Returns:
            Output AnnData conforming to output schema.
        """
        ...


# ---------------------------------------------------------------------------
# Subprocess helper (for conda env isolation)
# ---------------------------------------------------------------------------

def assigned_gpu_indices():
    """The GPUs this process was explicitly GIVEN, or None when nothing is pinned.

    `CUDA_VISIBLE_DEVICES=3` means "you get physical device 3". A runner that then picks its
    own card by free memory silently overrides that assignment -- which is how a cell ended up
    on a card the queue had never allocated (2026-09-24: isoST ran on GPU 1 while the queue had
    assigned GPU 3; same class as SINTER3D landing on the busy GPU 0 on 2026-09-23). An explicit
    numeric assignment is authoritative; unset, empty, or a UUID/MIG form leaves the callers'
    existing free-memory choice untouched, so stand-alone runs behave exactly as before.
    """
    import os
    import re

    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not v:
        return None
    parts = [p.strip() for p in v.split(",")]
    if all(re.fullmatch(r"\d+", p) for p in parts):
        return [int(p) for p in parts]
    return None


def pick_from_assigned_or_free(free_pairs):
    """Choose among (index, free_mib) pairs, restricted to the assigned GPUs when there are any."""
    cand = assigned_gpu_indices()
    if cand:
        pool = [x for x in free_pairs if x[0] in cand]
        return (max(pool, key=lambda x: x[1]) if pool else (cand[0], 0))[0]
    if not free_pairs:
        return 0
    return max(free_pairs, key=lambda x: x[1])[0]


def run_in_conda_env(
    env_name: str,
    script: str,
    cwd: Optional[Path] = None,
    env_vars: Optional[Dict[str, str]] = None,
    timeout_sec: int = 3600,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """Run a Python script in an isolated conda environment.

    Uses `conda run -n <env> python <script>` with subprocess.

    Args:
        env_name: Conda environment name.
        script: Python script content or path to script.
        cwd: Working directory.
        env_vars: Additional environment variables.
        timeout_sec: Timeout in seconds.
        capture_output: If True, capture stdout/stderr.

    Returns:
        CompletedProcess result.

    Raises:
        AdapterError: on timeout or subprocess failure.
    """
    # Write script to temp file if it's inline content
    if "\n" in script or not Path(script).exists():
        tmp_path = Path(tempfile.mktemp(suffix=".py", prefix=f"adapter_{env_name}_"))
        tmp_path.write_text(script)
        script_path = str(tmp_path)
        cleanup = True
    else:
        script_path = script
        cleanup = False

    cmd = [
        "conda", "run", "-n", env_name,
        "python", script_path,
    ]

    extra_env = os.environ.copy()
    if env_vars:
        extra_env.update(env_vars)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=extra_env,
            capture_output=capture_output,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        if cleanup:
            tmp_path.unlink(missing_ok=True)
        raise AdapterError(
            f"Subprocess timed out after {timeout_sec}s in env '{env_name}'",
            failure_type=FailureType.F2_INTERNAL,
            details={"timeout_sec": timeout_sec, "stdout": e.stdout, "stderr": e.stderr},
        )
    finally:
        if cleanup:
            tmp_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise AdapterError(
            f"Subprocess failed with code {result.returncode} in env '{env_name}'\n"
            f"STDERR: {result.stderr[:500] if result.stderr else '(none)'}",
            failure_type=FailureType.F2_INTERNAL,
            details={"returncode": result.returncode, "stderr": result.stderr},
        )

    return result


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: Dict[str, type] = {}
"""Global registry mapping method_name → adapter class."""


def register_adapter(cls: type) -> type:
    """Decorator to register an adapter class."""
    ADAPTER_REGISTRY[cls.method_name] = cls
    return cls


def get_adapter(method_name: str) -> BaseAdapter:
    """Get an adapter instance by method name."""
    if method_name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown method: {method_name}. Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[method_name]()
