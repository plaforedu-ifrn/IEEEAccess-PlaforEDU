import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    coverage_error,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.preprocessing import MultiLabelBinarizer

try:
    import fcntl
except ImportError:  # pragma: no cover - the experiment target is Linux/macOS.
    fcntl = None


DEFAULT_SEED = 42
DEFAULT_K_VALUES = [1, 3, 5, 7, 10]
LABEL_COLUMN = "comp_name_filtered"
TEXT_COLUMN = "combinedText"


UNIT_SPEC_FIELDS = (
    "experiment",
    "notebook",
    "model",
    "framework",
    "representation",
    "coordinates",
    "assignment_seed",
    "model_seed",
    "model_config",
    "authoritative_labels",
    "train_course_ids",
    "eval_course_ids",
    "source_csv_hashes",
    "notebook_code_fingerprint",
    "pipeline_utils_hash",
    "package_versions",
    "feature_identity",
    "protocol_version",
)

BERT_IDENTITY_FIELDS = (
    "requested_model",
    "resolved_model_revision",
    "requested_tokenizer",
    "resolved_tokenizer_revision",
)

RAW_TEXT_PREPROCESSING_FIELDS = (
    "source_column",
    "autokeras_block_type",
    "fit_scope",
    "inner_validation_scope",
)


@dataclass(frozen=True)
class ExperimentAttempt:
    """One concrete execution of a deterministic experiment unit."""

    unit_key: str
    unit_dir: Path
    attempt_id: str
    path: Path
    action: str

    @property
    def should_run(self):
        return self.action != "skip"


def _json_normalize(value):
    """Return a deterministic, JSON-compatible representation of ``value``."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON mapping keys must be strings")
        return {
            key: _json_normalize(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    if isinstance(value, set):
        normalized = [_json_normalize(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
        )
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_normalize(value.item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-normalizable")


def canonical_payload_hash(payload):
    """Hash a payload after recursive JSON normalization."""
    normalized = _json_normalize(payload)
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalized_code_source(source):
    if isinstance(source, list):
        if any(not isinstance(part, str) for part in source):
            raise ValueError("Notebook code-cell source list must contain only strings")
        source = "".join(source)
    elif not isinstance(source, str):
        raise ValueError("Notebook code-cell source must be a string or list of strings")
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    if source.endswith("\n"):
        source = source[:-1]
    return source


def notebook_code_fingerprint(notebook_path):
    """Hash only normalized code-cell sources from a serialized notebook."""
    path = Path(notebook_path)
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read notebook JSON: {path}") from exc
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Notebook is missing a cells list")
    code_sources = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("Notebook cells must be mappings")
        if cell.get("cell_type") == "code":
            if "source" not in cell:
                raise ValueError("Notebook code cell is missing source")
            code_sources.append(_normalized_code_source(cell["source"]))
    return canonical_payload_hash(code_sources)


def _missing_fields(payload, fields):
    return [field for field in fields if field not in payload]


def _validate_nonempty_mapping(payload, field):
    value = payload[field]
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a non-empty mapping")


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_unit_spec(unit_spec):
    if not isinstance(unit_spec, Mapping):
        raise ValueError("unit_spec must be a mapping")
    missing = _missing_fields(unit_spec, UNIT_SPEC_FIELDS)
    if missing:
        raise ValueError(f"Missing unit-spec fields: {', '.join(missing)}")

    for field in ("experiment", "notebook", "model", "framework", "representation", "protocol_version"):
        if not isinstance(unit_spec[field], str) or not unit_spec[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    for field in ("assignment_seed", "model_seed"):
        if isinstance(unit_spec[field], bool) or not isinstance(unit_spec[field], int):
            raise ValueError(f"{field} must be an integer")
    for field in ("model_config", "source_csv_hashes", "package_versions", "feature_identity"):
        _validate_nonempty_mapping(unit_spec, field)
    for field in ("authoritative_labels", "train_course_ids", "eval_course_ids"):
        value = unit_spec[field]
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"{field} must be a non-empty ordered sequence")
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"{field} must contain non-empty strings")
        if len(value) != len(set(value)):
            raise ValueError(f"{field} values must be unique")
    if set(unit_spec["train_course_ids"]) & set(unit_spec["eval_course_ids"]):
        raise ValueError("train_course_ids and eval_course_ids must be disjoint")
    for field in ("notebook_code_fingerprint", "pipeline_utils_hash"):
        if not _is_sha256(unit_spec[field]):
            raise ValueError(f"{field} must be a SHA-256 hexadecimal digest")
    if any(not isinstance(path, str) or not path for path in unit_spec["source_csv_hashes"]):
        raise ValueError("source_csv_hashes keys must be non-empty strings")
    if any(not _is_sha256(value) for value in unit_spec["source_csv_hashes"].values()):
        raise ValueError("source_csv_hashes values must be SHA-256 hexadecimal digests")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        for name, version in unit_spec["package_versions"].items()
    ):
        raise ValueError("package_versions must map package names to non-empty versions")

    coordinates = unit_spec["coordinates"]
    if not isinstance(coordinates, Mapping):
        raise ValueError("coordinates must be a mapping")
    missing_coordinates = _missing_fields(coordinates, ("fold", "training_size", "repetition"))
    if missing_coordinates:
        raise ValueError(
            "coordinates must contain fold, training_size, and repetition; missing "
            + ", ".join(missing_coordinates)
        )
    has_fold = coordinates["fold"] is not None
    has_curve = coordinates["training_size"] is not None or coordinates["repetition"] is not None
    if not has_fold and not has_curve:
        raise ValueError("coordinates require fold or training_size and repetition")
    if has_fold and has_curve:
        raise ValueError("coordinates must specify either fold or training_size and repetition")
    if has_curve and (coordinates["training_size"] is None or coordinates["repetition"] is None):
        raise ValueError("training_size and repetition must both be specified")
    coordinate_fields = ("fold",) if has_fold else ("training_size", "repetition")
    if any(isinstance(coordinates[field], bool) or not isinstance(coordinates[field], int) for field in coordinate_fields):
        raise ValueError("fold, training_size, and repetition coordinates must be integers")
    if has_fold and coordinates["fold"] < 0:
        raise ValueError("fold must be non-negative")
    if has_curve and coordinates["training_size"] <= 0:
        raise ValueError("training_size must be positive")
    if has_curve and coordinates["repetition"] < 0:
        raise ValueError("repetition must be non-negative")

    feature_identity = unit_spec["feature_identity"]
    kind = feature_identity.get("kind")
    if not isinstance(kind, str):
        raise ValueError("feature_identity.kind must be specified")
    if unit_spec["representation"].lower() != kind.lower():
        raise ValueError("representation must match feature_identity.kind")
    if kind.lower() in {"tfidf", "word2vec"}:
        if not isinstance(feature_identity.get("preprocessing"), Mapping) or not feature_identity["preprocessing"]:
            raise ValueError(f"{kind} feature_identity requires preprocessing")
    elif kind.lower() == "bert":
        missing_bert = _missing_fields(feature_identity, BERT_IDENTITY_FIELDS + ("embedding_cache_identity",))
        if missing_bert:
            raise ValueError(f"Missing BERT feature identity fields: {', '.join(missing_bert)}")
        empty_bert = [
            field
            for field in BERT_IDENTITY_FIELDS
            if not isinstance(feature_identity[field], str) or not feature_identity[field].strip()
        ]
        if empty_bert:
            raise ValueError(f"Empty BERT feature identity fields: {', '.join(empty_bert)}")
        if not isinstance(feature_identity["embedding_cache_identity"], Mapping) or not feature_identity["embedding_cache_identity"]:
            raise ValueError("embedding_cache_identity must be a non-empty mapping")
    elif kind.lower() == "raw_text":
        preprocessing = feature_identity.get("preprocessing")
        if not isinstance(preprocessing, Mapping) or not preprocessing:
            raise ValueError("raw_text feature_identity requires non-empty preprocessing")
        expected_fields = set(RAW_TEXT_PREPROCESSING_FIELDS)
        actual_fields = set(preprocessing)
        if actual_fields != expected_fields:
            missing_fields = sorted(expected_fields - actual_fields)
            extra_fields = sorted(actual_fields - expected_fields)
            details = []
            if missing_fields:
                details.append("missing " + ", ".join(missing_fields))
            if extra_fields:
                details.append("unexpected " + ", ".join(extra_fields))
            raise ValueError("raw_text preprocessing fields: " + "; ".join(details))
        empty_fields = [
            field
            for field in RAW_TEXT_PREPROCESSING_FIELDS
            if not isinstance(preprocessing[field], str) or not preprocessing[field].strip()
        ]
        if empty_fields:
            raise ValueError(
                "raw_text preprocessing fields must be non-empty strings: "
                + ", ".join(empty_fields)
            )
    else:
        raise ValueError("feature_identity.kind must be tfidf, word2vec, bert, or raw_text")

    # Ensure the entire payload can be represented canonically before returning it.
    _json_normalize(unit_spec)
    return unit_spec


def experiment_unit_key(unit_spec):
    """Validate and hash the complete scientific identity of one experiment unit."""
    return canonical_payload_hash(_validate_unit_spec(unit_spec))


def collect_run_provenance(
    package_names,
    *,
    device,
    available_hardware,
    bert_identity,
    embedding_cache_identity,
):
    """Collect run metadata without importing optional ML frameworks."""
    if isinstance(package_names, str) or not isinstance(package_names, (list, tuple, set)):
        raise ValueError("package_names must be a sequence of distribution names")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string")
    if not isinstance(available_hardware, Mapping) or not available_hardware:
        raise ValueError("available_hardware must be a non-empty mapping")
    if not isinstance(bert_identity, Mapping):
        raise ValueError("bert_identity must be a mapping")
    missing_bert = _missing_fields(bert_identity, BERT_IDENTITY_FIELDS)
    if missing_bert:
        raise ValueError(f"Missing BERT provenance fields: {', '.join(missing_bert)}")
    empty_bert = [
        field
        for field in BERT_IDENTITY_FIELDS
        if not isinstance(bert_identity[field], str) or not bert_identity[field].strip()
    ]
    if empty_bert:
        raise ValueError(f"Empty BERT provenance fields: {', '.join(empty_bert)}")
    if not isinstance(embedding_cache_identity, Mapping) or not embedding_cache_identity:
        raise ValueError("embedding_cache_identity must be a non-empty mapping")

    package_versions = {}
    for name in sorted(set(package_names)):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("package_names entries must be non-empty strings")
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None

    provenance = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "package_versions": package_versions,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "device": device,
        "available_hardware": dict(available_hardware),
        "bert_identity": {field: bert_identity[field] for field in BERT_IDENTITY_FIELDS},
        "embedding_cache_identity": dict(embedding_cache_identity),
    }
    _json_normalize(provenance)
    return provenance


def authoritative_label_classes(processed_df, label_column=LABEL_COLUMN, expected_n_labels=None):
    if label_column not in processed_df.columns:
        raise ValueError(f"Missing label column: {label_column}")
    classes = sorted({label for labels in processed_df[label_column] for label in parse_label_list(labels)})
    if expected_n_labels is not None and len(classes) != expected_n_labels:
        raise ValueError(f"Expected {expected_n_labels} labels, found {len(classes)}")
    return classes


def fixed_label_matrix(label_lists, classes):
    classes = list(classes)
    index = {label: idx for idx, label in enumerate(classes)}
    unknown = sorted({label for labels in label_lists for label in parse_label_list(labels) if label not in index})
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return labels_to_matrix([parse_label_list(labels) for labels in label_lists], classes)


def validate_dataset_identity(processed_df, train_df, test_df, course_id_column="courseId"):
    frames = {"processed": processed_df, "train": train_df, "test": test_df}
    id_sets = {}
    for name, frame in frames.items():
        if course_id_column not in frame.columns:
            raise ValueError(f"{name} is missing {course_id_column}")
        ids = frame[course_id_column].astype(str)
        if ids.duplicated().any():
            raise ValueError(f"Duplicate course identifiers in {name}")
        id_sets[name] = set(ids)
    overlap = id_sets["train"] & id_sets["test"]
    if overlap:
        raise ValueError(f"Train/test overlap: {sorted(overlap)[:5]}")
    if id_sets["train"] | id_sets["test"] != id_sets["processed"]:
        raise ValueError("Train/test union does not equal processed course identifiers")
    return {f"n_{name}": len(values) for name, values in id_sets.items()}


def allocate_nominal_budget(total_seconds, n_units):
    if total_seconds <= 0 or n_units <= 0:
        raise ValueError("Budget and unit count must be positive")
    per_unit = float(total_seconds) / int(n_units)
    return {
        "total_seconds": float(total_seconds),
        "n_units": int(n_units),
        "per_unit_seconds": per_unit,
        "allocated_seconds": per_unit * int(n_units),
    }


def normalize_binary_probability_columns(probability_outputs, classes_by_output):
    if len(probability_outputs) != len(classes_by_output):
        raise ValueError("Probability outputs and class metadata differ in length")
    columns = []
    n_samples = None
    for values, classes in zip(probability_outputs, classes_by_output):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if n_samples is None:
            n_samples = values.shape[0]
        if values.shape[0] != n_samples:
            raise ValueError("Probability outputs have inconsistent row counts")
        classes = list(classes)
        if 1 in classes:
            columns.append(values[:, classes.index(1)])
        else:
            columns.append(np.zeros(n_samples, dtype=float))
    return np.column_stack(columns)


def partition_label_columns(y):
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError("Label matrix must be two-dimensional")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Label matrix must contain only binary values")

    active_indices = []
    constants = {}
    for index in range(y.shape[1]):
        values = np.unique(y[:, index])
        if len(values) == 1:
            constants[index] = int(values[0])
        else:
            active_indices.append(index)
    return active_indices, constants


def expand_active_scores(scores, n_labels, active_indices, constants):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("Active score matrix must be two-dimensional")

    active_indices = list(active_indices)
    constants = dict(constants)
    if scores.shape[1] != len(active_indices):
        raise ValueError("Score width must match the number of active columns")
    if any(value not in (0, 1) for value in constants.values()):
        raise ValueError("Constant label values must be binary")

    indices = active_indices + list(constants)
    if (
        len(indices) != len(set(indices))
        or set(indices) != set(range(n_labels))
    ):
        raise ValueError("Active and constant indices must partition all label columns")

    expanded = np.empty((scores.shape[0], n_labels), dtype=float)
    expanded[:, active_indices] = scores
    for index, value in constants.items():
        expanded[:, index] = value
    return expanded


class SafeClassifierChain:
    """Classifier chain that keeps fixed columns when a target is single-class."""

    def __init__(self, base_estimator, order="random", random_state=42, threshold=0.5):
        self.base_estimator = base_estimator
        self.order = order
        self.random_state = random_state
        self.threshold = threshold

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        n_labels = y.shape[1]
        if isinstance(self.order, str):
            if self.order != "random":
                raise ValueError("order must be 'random' or an explicit permutation")
            self.order_ = np.random.RandomState(self.random_state).permutation(n_labels)
        else:
            self.order_ = np.asarray(self.order, dtype=int)
        if sorted(self.order_.tolist()) != list(range(n_labels)):
            raise ValueError("order must be a permutation of label indices")
        self.estimators_ = []
        augmented = X
        for label_idx in self.order_:
            target = y[:, label_idx]
            unique = np.unique(target)
            if len(unique) == 1:
                estimator = {"constant": float(unique[0])}
            else:
                estimator = clone(self.base_estimator)
                estimator.fit(augmented, target)
            self.estimators_.append(estimator)
            augmented = np.column_stack([augmented, target])
        self.n_labels_in_ = n_labels
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        augmented = X
        scores = np.zeros((len(X), self.n_labels_in_), dtype=float)
        for label_idx, estimator in zip(self.order_, self.estimators_):
            if isinstance(estimator, dict):
                positive = np.full(len(X), estimator["constant"], dtype=float)
            else:
                values = estimator.predict_proba(augmented)
                positive = normalize_binary_probability_columns([values], [estimator.classes_])[:, 0]
            scores[:, label_idx] = positive
            augmented = np.column_stack([augmented, (positive >= self.threshold).astype(int)])
        return scores


def aggregate_ecc_scores(score_matrices):
    if not score_matrices:
        raise ValueError("At least one score matrix is required")
    arrays = [np.asarray(scores, dtype=float) for scores in score_matrices]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("All ECC score matrices must have the same shape")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("ECC score matrices must contain finite values")
    return np.mean(arrays, axis=0)


def classify_crossover(sizes, means, lowers, uppers, reference):
    rows = sorted(zip(sizes, means, lowers, uppers), key=lambda row: row[0])
    if not rows or any(not np.isfinite(value) for row in rows for value in row[1:]):
        return {"status": "indeterminate", "reason": "incomplete_or_nonfinite"}
    means_sorted = [row[1] for row in rows]
    if any(b < a for a, b in zip(means_sorted, means_sorted[1:])):
        return {"status": "indeterminate", "reason": "non_monotonic"}
    positions = []
    for _, _, lower, upper in rows:
        if upper < reference:
            positions.append("below")
        elif lower > reference:
            positions.append("above")
        else:
            positions.append("overlap")
    if "overlap" in positions:
        return {"status": "indeterminate", "reason": "interval_overlaps_reference"}
    if "below" in positions and "above" in positions:
        return {"status": "observed", "reason": "bounded_transition"}
    return {"status": "not_observed", "reason": f"all_{positions[0]}"}


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


_ATTEMPT_PATTERN = re.compile(r"^attempt-(\d{4,})$")
_UNIT_STATE_LOCKS = {}
_UNIT_STATE_LOCKS_GUARD = threading.Lock()
_COMPLETED_FIELDS = (
    "completion_state",
    "unit_key",
    "attempt_id",
    "artifacts",
    "provenance",
    "active_label_indices",
    "constant_labels",
    "completed_at",
)
_PROVENANCE_FIELDS = (
    "python_version",
    "python_implementation",
    "python_executable",
    "package_versions",
    "os",
    "device",
    "available_hardware",
    "bert_identity",
    "embedding_cache_identity",
)


@contextmanager
def _unit_state_lock(unit_dir):
    if fcntl is None:
        raise RuntimeError(
            "Inter-process attempt-ledger locking requires fcntl on this platform"
        )
    key = str(Path(unit_dir).resolve())
    with _UNIT_STATE_LOCKS_GUARD:
        thread_lock = _UNIT_STATE_LOCKS.setdefault(key, threading.Lock())
    with thread_lock:
        lock_path = Path(unit_dir) / ".attempt-ledger.lock"
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(lock_path), flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _read_json_mapping(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _path_sha256(path):
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        raise ValueError(f"Artifact does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def _artifact_measure(path):
    path = Path(path)
    if path.is_dir():
        return "recursive_files", sum(1 for item in path.rglob("*") if item.is_file())
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid structured artifact: {path}") from exc
        if not isinstance(payload, (list, Mapping)):
            raise ValueError(f"Invalid structured artifact: {path}")
        return "json_items", len(payload)
    if suffix in {".csv", ".tsv"}:
        try:
            separator = "\t" if suffix == ".tsv" else ","
            return "rows", len(pd.read_csv(path, sep=separator))
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
            raise ValueError(f"Invalid structured artifact: {path}") from exc
    if suffix == ".npy":
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            return "rows", int(array.shape[0]) if array.ndim else 1
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid structured artifact: {path}") from exc
    if path.is_file():
        return "byte_size", path.stat().st_size
    raise ValueError(f"Artifact does not exist: {path}")


def _relative_artifact_path(path, base_dir):
    path = Path(path)
    if not path.is_absolute():
        path = Path(base_dir) / path
    try:
        relative = path.resolve().relative_to(Path(base_dir).resolve())
    except (OSError, ValueError) as exc:
        raise ValueError("Artifacts must be inside the attempt/checkpoint directory") from exc
    if relative == Path("."):
        raise ValueError("An artifact must not be the whole attempt/checkpoint directory")
    return path, relative.as_posix()


def _valid_provenance(provenance):
    if not isinstance(provenance, Mapping) or _missing_fields(provenance, _PROVENANCE_FIELDS):
        return False
    string_fields = ("python_version", "python_implementation", "python_executable", "device")
    if any(
        not isinstance(provenance[field], str) or not provenance[field].strip()
        for field in string_fields
    ):
        return False
    mapping_fields = (
        "package_versions",
        "os",
        "available_hardware",
        "bert_identity",
        "embedding_cache_identity",
    )
    if any(not isinstance(provenance[field], Mapping) or not provenance[field] for field in mapping_fields):
        return False
    os_fields = ("system", "release", "version", "machine")
    if _missing_fields(provenance["os"], os_fields) or any(
        not isinstance(provenance["os"][field], str) or not provenance["os"][field].strip()
        for field in os_fields
    ):
        return False
    if _missing_fields(provenance["bert_identity"], BERT_IDENTITY_FIELDS) or any(
        not isinstance(provenance["bert_identity"][field], str)
        or not provenance["bert_identity"][field].strip()
        for field in BERT_IDENTITY_FIELDS
    ):
        return False
    cache_key = provenance["embedding_cache_identity"].get("cache_key")
    if not isinstance(cache_key, str) or not cache_key.strip():
        return False
    if any(
        not isinstance(name, str)
        or not name.strip()
        or (version is not None and (not isinstance(version, str) or not version.strip()))
        for name, version in provenance["package_versions"].items()
    ):
        return False
    try:
        canonical_payload_hash(provenance)
    except (TypeError, ValueError):
        return False
    return True


def _valid_label_partition(active_label_indices, constant_labels):
    if not isinstance(active_label_indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in active_label_indices
    ):
        return False
    if active_label_indices != sorted(set(active_label_indices)):
        return False
    if not isinstance(constant_labels, list):
        return False
    constant_indices = []
    for item in constant_labels:
        if not isinstance(item, Mapping) or set(item) != {"index", "value"}:
            return False
        index, value = item["index"], item["value"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or value not in (0, 1):
            return False
        constant_indices.append(index)
    if constant_indices != sorted(set(constant_indices)):
        return False
    all_indices = active_label_indices + constant_indices
    return bool(all_indices) and len(all_indices) == len(set(all_indices)) and set(all_indices) == set(
        range(max(all_indices) + 1)
    )


def _validated_artifacts(base_dir, artifacts):
    if not isinstance(artifacts, Mapping) or not artifacts:
        return False
    for name, record in artifacts.items():
        if not isinstance(name, str) or not name or not isinstance(record, Mapping):
            return False
        if set(record) != {"path", "sha256", "count_kind", "count"}:
            return False
        if (
            not isinstance(record["path"], str)
            or not _is_sha256(record["sha256"])
            or record["count_kind"] not in {"json_items", "rows", "byte_size", "recursive_files"}
        ):
            return False
        if isinstance(record["count"], bool) or not isinstance(record["count"], int) or record["count"] < 0:
            return False
        try:
            artifact_path, relative = _relative_artifact_path(record["path"], base_dir)
            if relative != Path(record["path"]).as_posix():
                return False
            if _path_sha256(artifact_path) != record["sha256"]:
                return False
        except (OSError, ValueError):
            return False
        try:
            observed_kind, observed_count = _artifact_measure(artifact_path)
        except (OSError, ValueError):
            return False
        if observed_kind != record["count_kind"] or observed_count != record["count"]:
            return False
    return True


def validate_completed_attempt(attempt_path, *, expected_unit_key=None):
    """Return a valid completion manifest, or ``None`` for any incomplete/invalid attempt."""
    attempt_path = Path(attempt_path)
    if (attempt_path / "failed.json").exists():
        return None
    manifest = _read_json_mapping(attempt_path / "completed.json")
    if manifest is None or _missing_fields(manifest, _COMPLETED_FIELDS):
        return None
    if set(manifest) != set(_COMPLETED_FIELDS) or manifest["completion_state"] != "completed":
        return None
    if not _is_sha256(manifest["unit_key"]) or (
        expected_unit_key is not None and manifest["unit_key"] != expected_unit_key
    ):
        return None
    if manifest["attempt_id"] != attempt_path.name or not _ATTEMPT_PATTERN.fullmatch(attempt_path.name):
        return None
    if not isinstance(manifest["completed_at"], str) or not manifest["completed_at"]:
        return None
    if not _valid_provenance(manifest["provenance"]):
        return None
    if not _valid_label_partition(manifest["active_label_indices"], manifest["constant_labels"]):
        return None
    if not _validated_artifacts(attempt_path, manifest["artifacts"]):
        return None
    return dict(manifest)


def selected_attempt_path(unit_dir):
    """Resolve only the explicitly selected and currently valid completed attempt."""
    unit_dir = Path(unit_dir)
    if unit_dir.is_symlink() or not unit_dir.is_dir():
        return None
    pointer_path = unit_dir / "selected_attempt.json"
    pointer = _read_json_mapping(pointer_path)
    if pointer is None or set(pointer) != {"unit_key", "attempt_id"}:
        return None
    if pointer["unit_key"] != unit_dir.name:
        return None
    attempt_path = _safe_attempt_path(unit_dir, pointer["attempt_id"])
    if attempt_path is None:
        return None
    if validate_completed_attempt(attempt_path, expected_unit_key=pointer["unit_key"]) is None:
        return None
    return attempt_path


def _safe_attempt_path(unit_dir, attempt_id):
    unit_dir = Path(unit_dir)
    if (
        not isinstance(attempt_id, str)
        or Path(attempt_id).name != attempt_id
        or not _ATTEMPT_PATTERN.fullmatch(attempt_id)
    ):
        return None
    path = unit_dir / attempt_id
    if path.is_symlink() or not path.is_dir():
        return None
    try:
        if path.resolve().parent != unit_dir.resolve():
            return None
    except OSError:
        return None
    return path


def _attempt_from_pointer(unit_dir, pointer_name, unit_key):
    pointer = _read_json_mapping(Path(unit_dir) / pointer_name)
    if pointer is None or set(pointer) != {"unit_key", "attempt_id"}:
        return None
    if pointer["unit_key"] != unit_key:
        return None
    return _safe_attempt_path(unit_dir, pointer["attempt_id"])


def _next_attempt_id(unit_dir):
    numbers = []
    for path in Path(unit_dir).glob("attempt-*"):
        match = _ATTEMPT_PATTERN.fullmatch(path.name)
        if match:
            numbers.append(int(match.group(1)))
    return f"attempt-{max(numbers, default=0) + 1:04d}"


def prepare_experiment_attempt(base_dir, unit_spec, *, force_rerun=False):
    """Skip a valid selection, resume a compatible active attempt, or allocate a new one."""
    unit_key = experiment_unit_key(unit_spec)
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    unit_dir = base_dir / unit_key
    if unit_dir.is_symlink():
        raise ValueError("Experiment unit directory must not be a symlink")
    unit_dir.mkdir(exist_ok=True)
    if (
        unit_dir.is_symlink()
        or not unit_dir.is_dir()
        or unit_dir.resolve().parent != base_dir.resolve()
    ):
        raise ValueError("Experiment unit directory must be a direct child of the ledger base")

    with _unit_state_lock(unit_dir):
        if not force_rerun:
            selected = selected_attempt_path(unit_dir)
            if selected is not None:
                return ExperimentAttempt(unit_key, unit_dir, selected.name, selected, "skip")
            active = _attempt_from_pointer(unit_dir, "active_attempt.json", unit_key)
            if active is not None and not (active / "failed.json").exists() and not (
                active / "completed.json"
            ).exists():
                return ExperimentAttempt(unit_key, unit_dir, active.name, active, "resume")

        while True:
            attempt_id = _next_attempt_id(unit_dir)
            attempt_path = unit_dir / attempt_id
            try:
                attempt_path.mkdir()
            except FileExistsError:
                continue
            break
        atomic_write_json(attempt_path / "unit_spec.json", _json_normalize(unit_spec))
        atomic_write_json(
            unit_dir / "active_attempt.json",
            {"unit_key": unit_key, "attempt_id": attempt_id},
        )
        return ExperimentAttempt(unit_key, unit_dir, attempt_id, attempt_path, "new")


def _artifact_manifest(base_dir, artifact_paths, artifact_counts):
    if not isinstance(artifact_paths, Mapping) or not artifact_paths:
        raise ValueError("artifact_paths must be a non-empty mapping")
    if not isinstance(artifact_counts, Mapping) or set(artifact_counts) != set(artifact_paths):
        raise ValueError("artifact_counts must match artifact_paths")
    records = {}
    for name, original_path in sorted(artifact_paths.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("Artifact names must be non-empty strings")
        path, relative = _relative_artifact_path(original_path, base_dir)
        if not path.exists():
            raise ValueError(f"Artifact does not exist: {path}")
        count = artifact_counts[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Artifact counts must be non-negative integers")
        count_kind, observed_count = _artifact_measure(path)
        if observed_count != count:
            raise ValueError(f"Artifact count does not match contents: {name}")
        records[name] = {
            "path": relative,
            "sha256": _path_sha256(path),
            "count_kind": count_kind,
            "count": count,
        }
    return records


def complete_experiment_attempt(
    attempt,
    *,
    artifact_paths,
    artifact_counts,
    provenance,
    active_label_indices,
    constant_labels,
):
    """Validate and atomically publish a successful attempt before selecting it."""
    if not isinstance(attempt, ExperimentAttempt) or not attempt.should_run:
        raise ValueError("A runnable ExperimentAttempt is required")
    with _unit_state_lock(attempt.unit_dir):
        if (attempt.path / "completed.json").exists():
            raise ValueError("Attempt is already completed")
        if (attempt.path / "failed.json").exists():
            raise ValueError("A failed attempt cannot later be completed")
        active_pointer = _attempt_from_pointer(
            attempt.unit_dir, "active_attempt.json", attempt.unit_key
        )
        if active_pointer != attempt.path:
            raise ValueError("Attempt is no longer active")
        if not _valid_provenance(provenance):
            raise ValueError("provenance is incomplete or invalid")
        if not isinstance(constant_labels, Mapping):
            raise ValueError("constant_labels must map label indices to binary values")
        constants = [
            {"index": index, "value": value}
            for index, value in sorted(constant_labels.items())
        ]
        active = list(active_label_indices)
        if not _valid_label_partition(active, constants):
            raise ValueError("Active and constant labels must form an ordered, complete partition")
        manifest = {
            "completion_state": "completed",
            "unit_key": attempt.unit_key,
            "attempt_id": attempt.attempt_id,
            "artifacts": _artifact_manifest(attempt.path, artifact_paths, artifact_counts),
            "provenance": _json_normalize(provenance),
            "active_label_indices": active,
            "constant_labels": constants,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(attempt.path / "completed.json", manifest)
        if validate_completed_attempt(attempt.path, expected_unit_key=attempt.unit_key) is None:
            raise ValueError("Attempt completion manifest failed validation")
        atomic_write_json(
            attempt.unit_dir / "selected_attempt.json",
            {"unit_key": attempt.unit_key, "attempt_id": attempt.attempt_id},
        )
        (attempt.unit_dir / "active_attempt.json").unlink(missing_ok=True)
        return manifest


def write_attempt_failure(attempt, error):
    """Atomically mark an attempt failed and make it ineligible for implicit resume."""
    if not isinstance(attempt, ExperimentAttempt):
        raise ValueError("attempt must be an ExperimentAttempt")
    with _unit_state_lock(attempt.unit_dir):
        if (attempt.path / "completed.json").exists():
            raise ValueError("A completed attempt cannot later become failed")
        if (attempt.path / "failed.json").exists():
            raise ValueError("Attempt is already failed")
        active_pointer = _attempt_from_pointer(
            attempt.unit_dir, "active_attempt.json", attempt.unit_key
        )
        if active_pointer != attempt.path:
            raise ValueError("Attempt is no longer active")
        failure = {
            "completion_state": "failed",
            "unit_key": attempt.unit_key,
            "attempt_id": attempt.attempt_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(attempt.path / "failed.json", failure)
        (attempt.unit_dir / "active_attempt.json").unlink(missing_ok=True)
        return failure


def write_label_checkpoint(
    attempt,
    *,
    label_index,
    label_name,
    configuration,
    observed_classes,
    artifact_paths,
):
    """Write a final label-specific completion marker after predictor artifacts exist."""
    if not isinstance(attempt, ExperimentAttempt) or not attempt.should_run:
        raise ValueError("A runnable ExperimentAttempt is required")
    if isinstance(label_index, bool) or not isinstance(label_index, int) or label_index < 0:
        raise ValueError("label_index must be a non-negative integer")
    if not isinstance(label_name, str) or not label_name:
        raise ValueError("label_name must be a non-empty string")
    if not isinstance(configuration, Mapping) or not configuration:
        raise ValueError("configuration must be a non-empty mapping")
    classes = list(observed_classes)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("observed_classes must be non-empty and unique")
    if not isinstance(artifact_paths, Mapping) or not artifact_paths:
        raise ValueError("artifact_paths must be a non-empty mapping")
    with _unit_state_lock(attempt.unit_dir):
        if (attempt.path / "completed.json").exists() or (attempt.path / "failed.json").exists():
            raise ValueError("A terminal attempt cannot accept label checkpoints")
        active_pointer = _attempt_from_pointer(
            attempt.unit_dir, "active_attempt.json", attempt.unit_key
        )
        if active_pointer != attempt.path:
            raise ValueError("Attempt is no longer active")
        resolved = [_relative_artifact_path(path, attempt.path)[0] for path in artifact_paths.values()]
        checkpoint_dirs = {path.parent for path in resolved}
        if len(checkpoint_dirs) != 1:
            raise ValueError("Label artifacts must share one checkpoint directory")
        checkpoint_dir = checkpoint_dirs.pop()
        expected_checkpoint_dir = attempt.path / "labels" / f"label-{label_index:03d}"
        try:
            location_matches = (
                not checkpoint_dir.is_symlink()
                and not checkpoint_dir.parent.is_symlink()
                and checkpoint_dir.resolve() == expected_checkpoint_dir.resolve()
                and checkpoint_dir.resolve().is_relative_to(attempt.path.resolve())
            )
        except OSError:
            location_matches = False
        if not location_matches:
            raise ValueError("Label artifacts must use the label-specific child directory")
        records = {}
        for name, path in sorted(zip(artifact_paths, resolved)):
            if not path.exists():
                raise ValueError(f"Label artifact does not exist: {path}")
            records[name] = {
                "path": path.relative_to(checkpoint_dir).as_posix(),
                "sha256": _path_sha256(path),
            }
        marker = {
            "completion_state": "completed",
            "unit_key": attempt.unit_key,
            "label_index": label_index,
            "label_name": label_name,
            "configuration": _json_normalize(configuration),
            "observed_classes": _json_normalize(classes),
            "artifacts": records,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(checkpoint_dir / "completed.json", marker)
        return checkpoint_dir / "completed.json"


def validate_label_checkpoint(
    checkpoint_dir,
    *,
    unit_key,
    label_index,
    label_name,
    configuration,
    observed_classes,
    expected_artifacts,
    loader,
):
    """Return a matching loadable checkpoint marker path, else ``None`` so callers refit."""
    checkpoint_dir = Path(checkpoint_dir)
    attempt_dir = checkpoint_dir.parent.parent
    try:
        resolved_within_attempt = checkpoint_dir.resolve().is_relative_to(attempt_dir.resolve())
    except OSError:
        resolved_within_attempt = False
    if (
        checkpoint_dir.is_symlink()
        or checkpoint_dir.parent.is_symlink()
        or attempt_dir.is_symlink()
        or attempt_dir.parent.is_symlink()
        or not resolved_within_attempt
        or checkpoint_dir.name != f"label-{label_index:03d}"
        or checkpoint_dir.parent.name != "labels"
        or not _ATTEMPT_PATTERN.fullmatch(attempt_dir.name)
        or attempt_dir.parent.name != unit_key
    ):
        return None
    marker_path = checkpoint_dir / "completed.json"
    marker = _read_json_mapping(marker_path)
    required = {
        "completion_state",
        "unit_key",
        "label_index",
        "label_name",
        "configuration",
        "observed_classes",
        "artifacts",
        "completed_at",
    }
    if marker is None or set(marker) != required or marker["completion_state"] != "completed":
        return None
    if (
        marker["unit_key"] != unit_key
        or marker["label_index"] != label_index
        or marker["label_name"] != label_name
        or marker["configuration"] != _json_normalize(configuration)
        or marker["observed_classes"] != _json_normalize(list(observed_classes))
        or not isinstance(marker["completed_at"], str)
        or not marker["completed_at"]
    ):
        return None
    artifacts = marker["artifacts"]
    if not isinstance(artifacts, Mapping) or not artifacts:
        return None
    if not isinstance(expected_artifacts, Mapping) or not expected_artifacts:
        return None
    normalized_expected = {}
    for name, expected_path in expected_artifacts.items():
        if not isinstance(name, str) or not name or not isinstance(expected_path, (str, Path)):
            return None
        try:
            _, relative = _relative_artifact_path(expected_path, checkpoint_dir)
        except (OSError, ValueError):
            return None
        if relative != Path(expected_path).as_posix():
            return None
        normalized_expected[name] = relative
    marker_mapping = {
        name: record.get("path") if isinstance(record, Mapping) else None
        for name, record in artifacts.items()
    }
    if marker_mapping != normalized_expected:
        return None
    paths = []
    for name, record in artifacts.items():
        if (
            not isinstance(name, str)
            or not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or not isinstance(record["path"], str)
            or not _is_sha256(record["sha256"])
        ):
            return None
        try:
            path, relative = _relative_artifact_path(record["path"], checkpoint_dir)
            if relative != Path(record["path"]).as_posix() or _path_sha256(path) != record["sha256"]:
                return None
        except (OSError, ValueError):
            return None
        paths.append(path)
    if not callable(loader):
        return None
    load_target = paths[0] if len(paths) == 1 else checkpoint_dir
    try:
        loaded = loader(load_target)
    except Exception:
        return None
    if loaded is None:
        return None
    return marker_path


def cache_identity(source_path, course_ids, model_id, max_length, batch_size, expected_shape):
    identity = {
        "source_path": str(Path(source_path).resolve()),
        "source_sha256": file_sha256(source_path),
        "ordered_course_ids_sha256": hashlib.sha256(
            json.dumps([str(value) for value in course_ids], ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "model_id": str(model_id),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "expected_shape": [int(value) for value in expected_shape],
    }
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    identity["cache_key"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return identity


def atomic_write_csv(path, frame, **to_csv_kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, **to_csv_kwargs)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def multilabel_stratified_folds(y, n_splits=5, seed=DEFAULT_SEED):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ImportError as exc:
        raise ImportError(
            "Install iterative-stratification to run stratified cross-validation"
        ) from exc
    y = np.asarray(y, dtype=int)
    if n_splits < 2 or n_splits > len(y):
        raise ValueError("n_splits must be between 2 and the number of samples")
    splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(np.zeros(len(y)), y)]


def nested_multilabel_stratified_subsets(y, sizes, seed=DEFAULT_SEED):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as exc:
        raise ImportError(
            "Install iterative-stratification to build stratified training subsets"
        ) from exc
    y = np.asarray(y, dtype=int)
    requested = sorted({int(size) for size in sizes}, reverse=True)
    if not requested or requested[0] > len(y) or requested[-1] <= 0:
        raise ValueError("Subset sizes must be positive and no larger than the dataset")
    current = np.arange(len(y))
    result = {len(y): current.copy()} if len(y) in requested else {}
    for step, size in enumerate(requested):
        if size == len(current):
            result[size] = current.copy()
            continue
        if size > len(current):
            continue
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, train_size=size, random_state=seed + step
        )
        selected_relative, _ = next(splitter.split(np.zeros(len(current)), y[current]))
        current = current[selected_relative]
        result[size] = np.sort(current)
    missing = set(requested) - set(result)
    if missing:
        raise ValueError(f"Could not construct nested subset sizes: {sorted(missing)}")
    return result


_PROTOCOL_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PROTOCOL_VERSION = "protocol-v1"
_PROTOCOL_ARTIFACTS = {
    "fold_assignments": "fold_assignments_v1.csv",
    "training_subsets": "training_subsets_v1.csv",
}
_PROTOCOL_MANIFEST_FIELDS = {
    "protocol_key",
    "protocol_version",
    "source_csv_hashes",
    "course_ids",
    "train_course_ids",
    "test_course_ids",
    "authoritative_labels",
    "assignment_seed",
    "model_seed",
    "cv_model_seeds",
    "subset_seeds",
    "curve_model_seeds",
    "n_splits",
    "training_sizes",
    "repetitions",
    "expected_counts",
    "artifacts",
}
_PROTOCOL_IDENTITY_FIELDS = _PROTOCOL_MANIFEST_FIELDS - {"protocol_key", "artifacts"}
_PROTOCOL_COUNT_FIELDS = {"courses", "train_courses", "test_courses", "labels"}


def _plain_integer(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))


def _valid_ordered_strings(values):
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, str) and bool(value.strip()) for value in values)
        and len(values) == len(set(values))
    )


def _valid_protocol_identity(identity):
    if not isinstance(identity, Mapping) or set(identity) != _PROTOCOL_IDENTITY_FIELDS:
        return False
    if not isinstance(identity["protocol_version"], str) or not identity["protocol_version"].strip():
        return False
    source_hashes = identity["source_csv_hashes"]
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != {"processed.csv", "train.csv", "test.csv"}
        or any(not _is_sha256(value) for value in source_hashes.values())
    ):
        return False
    sequence_fields = (
        "course_ids",
        "train_course_ids",
        "test_course_ids",
        "authoritative_labels",
    )
    if any(not _valid_ordered_strings(identity[field]) for field in sequence_fields):
        return False
    course_ids = identity["course_ids"]
    train_ids = identity["train_course_ids"]
    test_ids = identity["test_course_ids"]
    if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != set(course_ids):
        return False
    train_set = set(train_ids)
    test_set = set(test_ids)
    if (
        [course_id for course_id in course_ids if course_id in train_set] != train_ids
        or [course_id for course_id in course_ids if course_id in test_set] != test_ids
    ):
        return False
    integer_fields = ("assignment_seed", "model_seed", "n_splits", "repetitions")
    if any(not _plain_integer(identity[field]) for field in integer_fields):
        return False
    if (
        identity["assignment_seed"] < 0
        or identity["model_seed"] < 0
        or identity["n_splits"] < 2
        or identity["n_splits"] > len(course_ids)
        or identity["repetitions"] <= 0
    ):
        return False
    sizes = identity["training_sizes"]
    if (
        not isinstance(sizes, list)
        or not sizes
        or any(not _plain_integer(size) for size in sizes)
        or any(size <= 0 or size > len(train_ids) for size in sizes)
        or sizes != sorted(set(sizes), reverse=True)
        or sizes[0] != len(train_ids)
    ):
        return False
    counts = identity["expected_counts"]
    expected_counts = {
        "courses": len(course_ids),
        "train_courses": len(train_ids),
        "test_courses": len(test_ids),
        "labels": len(identity["authoritative_labels"]),
    }
    if (
        not isinstance(counts, Mapping)
        or set(counts) != _PROTOCOL_COUNT_FIELDS
        or any(not _plain_integer(value) for value in counts.values())
        or dict(counts) != expected_counts
    ):
        return False
    expected_cv_seeds = [identity["model_seed"] + fold for fold in range(identity["n_splits"])]
    expected_curve_seeds = [
        identity["model_seed"] + repetition for repetition in range(identity["repetitions"])
    ]
    seed_sequence_fields = ("cv_model_seeds", "subset_seeds", "curve_model_seeds")
    if any(
        not isinstance(identity[field], list)
        or any(not _plain_integer(value) for value in identity[field])
        for field in seed_sequence_fields
    ):
        return False
    return (
        identity["cv_model_seeds"] == expected_cv_seeds
        and identity["subset_seeds"] == expected_curve_seeds
        and identity["curve_model_seeds"] == expected_curve_seeds
    )


def _safe_regular_child(path, parent):
    path = Path(path)
    parent = Path(parent)
    if path.is_symlink() or not path.is_file() or parent.is_symlink():
        return False
    try:
        return path.resolve(strict=True).parent == parent.resolve(strict=True)
    except OSError:
        return False


def _integral_csv_series(series):
    converted = []
    for value in series.tolist():
        if _plain_integer(value):
            converted.append(int(value))
        elif isinstance(value, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
            converted.append(int(value))
        else:
            return None
    return pd.Series(converted, index=series.index, dtype=object)


def _current_pointer_exists(protocol_dir):
    pointer_path = Path(protocol_dir) / "current_protocol.json"
    return pointer_path.exists() or pointer_path.is_symlink()


def _protocol_identity(
    processed_df,
    train_df,
    test_df,
    authoritative_labels,
    source_csv_hashes,
    assignment_seed,
    model_seed,
    n_splits,
    training_sizes,
    repetitions,
    expected_counts,
    protocol_version,
):
    validate_dataset_identity(processed_df, train_df, test_df)
    course_ids = processed_df["courseId"].astype(str).tolist()
    train_course_ids = train_df["courseId"].astype(str).tolist()
    test_course_ids = test_df["courseId"].astype(str).tolist()
    labels = list(authoritative_labels)
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("authoritative_labels must contain non-empty strings")
    if len(labels) != len(set(labels)):
        raise ValueError("authoritative_labels must be unique and ordered")
    if not isinstance(source_csv_hashes, Mapping) or set(source_csv_hashes) != {
        "processed.csv",
        "train.csv",
        "test.csv",
    }:
        raise ValueError("source_csv_hashes must identify processed.csv, train.csv, and test.csv")
    if any(not _is_sha256(value) for value in source_csv_hashes.values()):
        raise ValueError("source_csv_hashes values must be SHA-256 digests")
    integer_values = {
        "assignment_seed": assignment_seed,
        "model_seed": model_seed,
        "n_splits": n_splits,
        "repetitions": repetitions,
    }
    if any(not _plain_integer(value) for value in integer_values.values()):
        raise ValueError("Protocol seeds, n_splits, and repetitions must be integers")
    assignment_seed = int(assignment_seed)
    model_seed = int(model_seed)
    n_splits = int(n_splits)
    repetitions = int(repetitions)
    if assignment_seed < 0 or model_seed < 0 or n_splits < 2 or n_splits > len(course_ids) or repetitions <= 0:
        raise ValueError("Protocol requires valid n_splits and positive repetitions")
    raw_sizes = list(training_sizes)
    if (
        not raw_sizes
        or any(not _plain_integer(size) for size in raw_sizes)
    ):
        raise ValueError("training_sizes must contain integers")
    sizes = [int(size) for size in raw_sizes]
    if (
        any(size <= 0 or size > len(train_course_ids) for size in sizes)
        or len(sizes) != len(set(sizes))
        or sizes != sorted(sizes, reverse=True)
        or sizes[0] != len(train_course_ids)
    ):
        raise ValueError("training_sizes must be unique descending sizes beginning with full training")
    counts = {
        "courses": len(course_ids),
        "train_courses": len(train_course_ids),
        "test_courses": len(test_course_ids),
        "labels": len(labels),
    }
    if (
        not isinstance(expected_counts, Mapping)
        or set(expected_counts) != _PROTOCOL_COUNT_FIELDS
        or any(not _plain_integer(value) for value in expected_counts.values())
        or dict(expected_counts) != counts
    ):
        raise ValueError(f"Protocol expected-count mismatch: expected {counts}")
    if not isinstance(protocol_version, str) or not protocol_version:
        raise ValueError("protocol_version must be a non-empty string")
    identity = {
        "protocol_version": protocol_version,
        "source_csv_hashes": dict(sorted(source_csv_hashes.items())),
        "course_ids": course_ids,
        "train_course_ids": train_course_ids,
        "test_course_ids": test_course_ids,
        "authoritative_labels": labels,
        "assignment_seed": assignment_seed,
        "model_seed": model_seed,
        "cv_model_seeds": [model_seed + fold for fold in range(n_splits)],
        "subset_seeds": [model_seed + repetition for repetition in range(repetitions)],
        "curve_model_seeds": [model_seed + repetition for repetition in range(repetitions)],
        "n_splits": n_splits,
        "training_sizes": sizes,
        "repetitions": repetitions,
        "expected_counts": counts,
    }
    if not _valid_protocol_identity(identity):
        raise ValueError("Protocol identity is invalid")
    return identity


def _protocol_key(identity):
    return canonical_payload_hash(identity)


def _valid_protocol_assignments(manifest, folds, subsets):
    if list(folds.columns) != ["courseId", "fold", "split_seed"]:
        return False
    if len(folds) != manifest["expected_counts"]["courses"]:
        return False
    if folds["courseId"].astype(str).tolist() != manifest["course_ids"]:
        return False
    if folds["courseId"].duplicated().any():
        return False
    fold_values = _integral_csv_series(folds["fold"])
    split_seeds = _integral_csv_series(folds["split_seed"])
    if fold_values is None or split_seeds is None:
        return False
    if set(fold_values) != set(range(manifest["n_splits"])):
        return False
    if (fold_values.value_counts() <= 0).any() or set(split_seeds) != {manifest["assignment_seed"]}:
        return False

    if list(subsets.columns) != ["courseId", "repetition", "training_size", "subset_seed"]:
        return False
    expected_rows = manifest["repetitions"] * sum(manifest["training_sizes"])
    if len(subsets) != expected_rows:
        return False
    repetitions = _integral_csv_series(subsets["repetition"])
    training_sizes = _integral_csv_series(subsets["training_size"])
    subset_seeds = _integral_csv_series(subsets["subset_seed"])
    if repetitions is None or training_sizes is None or subset_seeds is None:
        return False
    if set(repetitions) != set(range(manifest["repetitions"])):
        return False
    if set(training_sizes) != set(manifest["training_sizes"]):
        return False
    train_ids = set(manifest["train_course_ids"])
    train_positions = {
        course_id: position for position, course_id in enumerate(manifest["train_course_ids"])
    }
    full_membership = None
    expected_rows = []
    for repetition in range(manifest["repetitions"]):
        prior = None
        repetition_rows = subsets.loc[repetitions == repetition]
        if set(subset_seeds.loc[repetition_rows.index]) != {manifest["subset_seeds"][repetition]}:
            return False
        for size in manifest["training_sizes"]:
            group = repetition_rows.loc[training_sizes.loc[repetition_rows.index] == size]
            ids = group["courseId"].astype(str).tolist()
            membership = set(ids)
            if len(ids) != size or len(membership) != size or not membership <= train_ids:
                return False
            canonical_ids = sorted(ids, key=train_positions.__getitem__)
            if ids != canonical_ids:
                return False
            expected_rows.extend(
                (course_id, repetition, size, manifest["subset_seeds"][repetition])
                for course_id in canonical_ids
            )
            if prior is not None and not membership < prior:
                return False
            prior = membership
            if size == len(train_ids):
                if membership != train_ids:
                    return False
                if full_membership is None:
                    full_membership = ids
                elif ids != full_membership:
                    return False
    observed_rows = list(
        zip(
            subsets["courseId"].astype(str),
            repetitions,
            training_sizes,
            subset_seeds,
        )
    )
    return (
        set(subset_seeds) == set(manifest["subset_seeds"])
        and observed_rows == expected_rows
    )


def validate_protocol_bundle(bundle_dir, *, expected_identity=None):
    """Load a completed immutable protocol bundle, or return ``None`` if invalid."""
    bundle_dir = Path(bundle_dir)
    bundles_dir = bundle_dir.parent
    protocol_dir = bundles_dir.parent
    if (
        bundle_dir.is_symlink()
        or bundles_dir.is_symlink()
        or protocol_dir.is_symlink()
        or not bundle_dir.is_dir()
        or not _PROTOCOL_KEY_PATTERN.fullmatch(bundle_dir.name)
        or bundles_dir.name != "bundles"
    ):
        return None
    try:
        resolved_bundle = bundle_dir.resolve()
        resolved_bundles = bundles_dir.resolve()
        resolved_protocol = protocol_dir.resolve()
        if (
            resolved_bundle.parent != resolved_bundles
            or resolved_bundles.parent != resolved_protocol
        ):
            return None
    except OSError:
        return None
    completed_path = bundle_dir / "completed.json"
    manifest_path = bundle_dir / "protocol_manifest_v1.json"
    if not _safe_regular_child(completed_path, bundle_dir) or not _safe_regular_child(
        manifest_path, bundle_dir
    ):
        return None
    completed = _read_json_mapping(completed_path)
    if completed is None or set(completed) != {
        "completion_state",
        "protocol_key",
        "manifest_sha256",
    }:
        return None
    if (
        completed["completion_state"] != "completed"
        or completed["protocol_key"] != bundle_dir.name
        or not _is_sha256(completed["manifest_sha256"])
    ):
        return None
    try:
        if file_sha256(manifest_path) != completed["manifest_sha256"]:
            return None
    except OSError:
        return None
    manifest = _read_json_mapping(manifest_path)
    if manifest is None or set(manifest) != _PROTOCOL_MANIFEST_FIELDS:
        return None
    identity = {key: manifest[key] for key in manifest if key not in {"protocol_key", "artifacts"}}
    if not _valid_protocol_identity(identity):
        return None
    try:
        identity_key = _protocol_key(identity)
        normalized_expected = (
            _json_normalize(expected_identity) if expected_identity is not None else None
        )
    except (TypeError, ValueError):
        return None
    if (
        manifest["protocol_key"] != bundle_dir.name
        or identity_key != bundle_dir.name
        or (normalized_expected is not None and identity != normalized_expected)
    ):
        return None
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(_PROTOCOL_ARTIFACTS):
        return None
    frames = {}
    for name, filename in _PROTOCOL_ARTIFACTS.items():
        record = artifacts[name]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256", "rows"}
            or record["path"] != filename
            or not _is_sha256(record["sha256"])
            or isinstance(record["rows"], bool)
            or not isinstance(record["rows"], int)
        ):
            return None
        path = bundle_dir / filename
        integer_columns = (
            ("fold", "split_seed")
            if name == "fold_assignments"
            else ("repetition", "training_size", "subset_seed")
        )
        csv_types = {"courseId": "string", **{column: "string" for column in integer_columns}}
        try:
            if not _safe_regular_child(path, bundle_dir) or file_sha256(path) != record["sha256"]:
                return None
            frame = pd.read_csv(path, dtype=csv_types)
        except (OSError, ValueError, pd.errors.ParserError):
            return None
        if len(frame) != record["rows"]:
            return None
        for column in integer_columns:
            parsed = _integral_csv_series(frame[column])
            if parsed is None:
                return None
            frame[column] = parsed
        frames[name] = frame
    try:
        valid_assignments = _valid_protocol_assignments(
            manifest, frames["fold_assignments"], frames["training_subsets"]
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not valid_assignments:
        return None
    return {
        "path": bundle_dir,
        "manifest": dict(manifest),
        "fold_assignments": frames["fold_assignments"],
        "training_subsets": frames["training_subsets"],
    }


def load_current_protocol_bundle(protocol_dir, *, expected_identity=None):
    """Resolve only the atomically selected, valid protocol bundle."""
    protocol_dir = Path(protocol_dir)
    pointer_path = protocol_dir / "current_protocol.json"
    if protocol_dir.is_symlink() or not _safe_regular_child(pointer_path, protocol_dir):
        return None
    pointer = _read_json_mapping(pointer_path)
    if pointer is None or set(pointer) != {"protocol_key"}:
        return None
    key = pointer["protocol_key"]
    if not isinstance(key, str) or not _PROTOCOL_KEY_PATTERN.fullmatch(key):
        return None
    bundles_dir = protocol_dir / "bundles"
    bundle_dir = bundles_dir / key
    try:
        if bundles_dir.is_symlink() or bundle_dir.resolve().parent != bundles_dir.resolve():
            return None
    except OSError:
        return None
    return validate_protocol_bundle(bundle_dir, expected_identity=expected_identity)


def publish_protocol_bundle(
    protocol_dir,
    *,
    processed_df,
    train_df,
    test_df,
    y,
    authoritative_labels,
    source_csv_hashes,
    assignment_seed=DEFAULT_SEED,
    model_seed=DEFAULT_SEED,
    n_splits=5,
    training_sizes=(270, 135, 68),
    repetitions=3,
    expected_counts=None,
    protocol_version=_PROTOCOL_VERSION,
):
    """Create, validate, and atomically select one immutable protocol bundle."""
    if expected_counts is None:
        expected_counts = {
            "courses": len(processed_df),
            "train_courses": len(train_df),
            "test_courses": len(test_df),
            "labels": len(authoritative_labels),
        }
    identity = _protocol_identity(
        processed_df,
        train_df,
        test_df,
        authoritative_labels,
        source_csv_hashes,
        assignment_seed,
        model_seed,
        n_splits,
        training_sizes,
        repetitions,
        expected_counts,
        protocol_version,
    )
    raw_y = np.asarray(y)
    if raw_y.shape != (len(processed_df), len(authoritative_labels)) or not np.isin(
        raw_y, [0, 1]
    ).all():
        raise ValueError("y must be a binary matrix matching courses and authoritative labels")
    y = raw_y.astype(int)
    protocol_key = _protocol_key(identity)
    protocol_dir = Path(protocol_dir)
    bundles_dir = protocol_dir / "bundles"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    if protocol_dir.is_symlink():
        raise ValueError("protocol_dir must not be a symlink")
    bundles_dir.mkdir(exist_ok=True)
    if bundles_dir.is_symlink():
        raise ValueError("protocol bundles directory must not be a symlink")

    with _unit_state_lock(protocol_dir):
        selected = load_current_protocol_bundle(protocol_dir)
        if selected is not None:
            selected_identity = {
                key: selected["manifest"][key]
                for key in selected["manifest"]
                if key not in {"protocol_key", "artifacts"}
            }
            if selected_identity != identity:
                raise ValueError("Current protocol identity mismatch")
            return selected["path"]
        if _current_pointer_exists(protocol_dir):
            raise ValueError("Existing current protocol pointer is corrupt or unsafe")
        bundle_dir = bundles_dir / protocol_key
        existing = validate_protocol_bundle(bundle_dir, expected_identity=identity)
        if bundle_dir.exists() and existing is None:
            raise ValueError("Existing protocol-key bundle is incomplete or corrupt")
        if existing is None:
            staging = Path(tempfile.mkdtemp(prefix=f".staging-{protocol_key}-", dir=bundles_dir))
            folds = multilabel_stratified_folds(y, n_splits=n_splits, seed=assignment_seed)
            assignments = np.full(len(y), -1, dtype=int)
            for fold, (_, eval_indices) in enumerate(folds):
                eval_indices = np.asarray(eval_indices, dtype=int)
                if (
                    len(eval_indices) == 0
                    or (eval_indices < 0).any()
                    or (eval_indices >= len(y)).any()
                    or (assignments[eval_indices] != -1).any()
                ):
                    raise ValueError("Fold assignments must cover each course exactly once")
                assignments[eval_indices] = fold
            if (assignments < 0).any():
                raise ValueError("Fold assignments must cover each course exactly once")
            fold_frame = pd.DataFrame(
                {
                    "courseId": identity["course_ids"],
                    "fold": assignments,
                    "split_seed": assignment_seed,
                }
            )
            train_positions = {value: index for index, value in enumerate(identity["course_ids"])}
            train_y = y[[train_positions[value] for value in identity["train_course_ids"]]]
            subset_rows = []
            for repetition, subset_seed in enumerate(identity["subset_seeds"]):
                subsets = nested_multilabel_stratified_subsets(
                    train_y, identity["training_sizes"], seed=subset_seed
                )
                for size in identity["training_sizes"]:
                    indices = np.sort(np.asarray(subsets[size], dtype=int))
                    for index in indices:
                        subset_rows.append(
                            {
                                "courseId": identity["train_course_ids"][index],
                                "repetition": repetition,
                                "training_size": size,
                                "subset_seed": subset_seed,
                            }
                        )
            subset_frame = pd.DataFrame(
                subset_rows,
                columns=["courseId", "repetition", "training_size", "subset_seed"],
            )
            if not _valid_protocol_assignments(identity, fold_frame, subset_frame):
                raise ValueError("Generated protocol assignments failed validation")
            atomic_write_csv(staging / _PROTOCOL_ARTIFACTS["fold_assignments"], fold_frame)
            atomic_write_csv(staging / _PROTOCOL_ARTIFACTS["training_subsets"], subset_frame)
            artifacts = {}
            for name, filename in _PROTOCOL_ARTIFACTS.items():
                path = staging / filename
                frame = fold_frame if name == "fold_assignments" else subset_frame
                artifacts[name] = {"path": filename, "sha256": file_sha256(path), "rows": len(frame)}
            manifest = {
                "protocol_key": protocol_key,
                **identity,
                "artifacts": artifacts,
            }
            manifest_path = staging / "protocol_manifest_v1.json"
            atomic_write_json(manifest_path, manifest)
            atomic_write_json(
                staging / "completed.json",
                {
                    "completion_state": "completed",
                    "protocol_key": protocol_key,
                    "manifest_sha256": file_sha256(manifest_path),
                },
            )
            staging.replace(bundle_dir)
            existing = validate_protocol_bundle(bundle_dir, expected_identity=identity)
            if existing is None:
                raise ValueError("Published protocol bundle failed validation")
        atomic_write_json(protocol_dir / "current_protocol.json", {"protocol_key": protocol_key})
        return bundle_dir


def ensure_experiment_protocol(protocol_dir, **protocol_kwargs):
    """Load the exact requested protocol, creating it only when no valid selection exists."""
    expected_counts = protocol_kwargs.get("expected_counts")
    if expected_counts is None:
        expected_counts = {
            "courses": len(protocol_kwargs["processed_df"]),
            "train_courses": len(protocol_kwargs["train_df"]),
            "test_courses": len(protocol_kwargs["test_df"]),
            "labels": len(protocol_kwargs["authoritative_labels"]),
        }
    identity = _protocol_identity(
        protocol_kwargs["processed_df"],
        protocol_kwargs["train_df"],
        protocol_kwargs["test_df"],
        protocol_kwargs["authoritative_labels"],
        protocol_kwargs["source_csv_hashes"],
        protocol_kwargs.get("assignment_seed", DEFAULT_SEED),
        protocol_kwargs.get("model_seed", DEFAULT_SEED),
        protocol_kwargs.get("n_splits", 5),
        protocol_kwargs.get("training_sizes", (270, 135, 68)),
        protocol_kwargs.get("repetitions", 3),
        expected_counts,
        protocol_kwargs.get("protocol_version", _PROTOCOL_VERSION),
    )
    selected = load_current_protocol_bundle(protocol_dir)
    if selected is not None:
        selected_identity = {
            key: selected["manifest"][key]
            for key in selected["manifest"]
            if key not in {"protocol_key", "artifacts"}
        }
        if selected_identity != identity:
            raise ValueError("Current protocol identity mismatch")
        return selected
    if _current_pointer_exists(protocol_dir):
        raise ValueError("Existing current protocol pointer is corrupt or unsafe")
    publish_kwargs = dict(protocol_kwargs)
    publish_kwargs["expected_counts"] = expected_counts
    publish_protocol_bundle(protocol_dir, **publish_kwargs)
    selected = load_current_protocol_bundle(protocol_dir, expected_identity=identity)
    if selected is None:
        raise ValueError("Protocol publication did not produce a valid selection")
    return selected


def label_support_summary(y, classes):
    y = np.asarray(y, dtype=int)
    if y.shape[1] != len(classes):
        raise ValueError("Target width does not match class list")
    support = y.sum(axis=0)
    return pd.DataFrame({"label": list(classes), "support": support.astype(int), "missing": support == 0})


def paired_bootstrap_difference(values_a, values_b, metric, n_resamples=10000, seed=DEFAULT_SEED):
    """Bootstrap paired rows and return percentile CI for metric(A)-metric(B)."""
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    if a.shape[0] != b.shape[0] or a.shape[0] == 0:
        raise ValueError("Paired bootstrap inputs must have the same nonzero row count")
    rng = np.random.default_rng(seed)
    differences = np.empty(int(n_resamples), dtype=float)
    for idx in range(int(n_resamples)):
        sample = rng.integers(0, a.shape[0], size=a.shape[0])
        differences[idx] = float(metric(a[sample]) - metric(b[sample]))
    return {
        "estimate": float(metric(a) - metric(b)),
        "lower": float(np.percentile(differences, 2.5)),
        "upper": float(np.percentile(differences, 97.5)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def paired_multilabel_bootstrap_differences(
    y_true,
    scores_a,
    scores_b,
    k_values=DEFAULT_K_VALUES,
    n_resamples=10000,
    seed=DEFAULT_SEED,
    chunk_size=250,
):
    """Efficient paired course-level bootstrap for the pre-specified Top-k metrics.

    Every estimate and interval is metric(A) - metric(B). For Hamming Loss,
    therefore, a negative difference favors A; all other metrics are
    higher-is-better. Resampling is aligned by row and performed in bounded
    chunks to avoid allocating a 10,000 x n_samples x n_labels tensor.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores_a = normalize_score_matrix(scores_a, y_true.shape[1])
    scores_b = normalize_score_matrix(scores_b, y_true.shape[1])
    if y_true.ndim != 2 or scores_a.shape != y_true.shape or scores_b.shape != y_true.shape:
        raise ValueError("Truth and both score matrices must have the same two-dimensional shape")
    if len(y_true) == 0 or n_resamples <= 0 or chunk_size <= 0:
        raise ValueError("Bootstrap requires samples, positive resamples, and a positive chunk size")

    rng = np.random.default_rng(seed)
    rows = []
    for k in k_values:
        if k <= 0 or k > y_true.shape[1]:
            raise ValueError("Each k must be between 1 and the number of labels")
        pred_a, _ = top_k_prediction_matrix(scores_a, k)
        pred_b, _ = top_k_prediction_matrix(scores_b, k)

        def components(pred):
            truth = y_true.astype(bool)
            predicted = pred.astype(bool)
            tp = np.logical_and(truth, predicted).astype(np.int16)
            fp = np.logical_and(~truth, predicted).astype(np.int16)
            fn = np.logical_and(truth, ~predicted).astype(np.int16)
            hits = tp.sum(axis=1)
            true_counts = truth.sum(axis=1)
            return {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mismatches": np.not_equal(truth, predicted).sum(axis=1),
                "precision_at_k": hits / float(k),
                "recall_at_k": np.divide(
                    hits, true_counts, out=np.zeros(len(hits), dtype=float), where=true_counts != 0
                ),
                "partial_hit_at_k": (hits > 0).astype(float),
            }

        components_a = components(pred_a)
        components_b = components(pred_b)

        def summarize(parts, indices=None):
            if indices is None:
                tp = parts["tp"].sum(axis=0, keepdims=True)
                fp = parts["fp"].sum(axis=0, keepdims=True)
                fn = parts["fn"].sum(axis=0, keepdims=True)
                divisor = len(y_true) * y_true.shape[1]
                scalar = lambda name: np.array([parts[name].mean()])
            else:
                tp = parts["tp"][indices].sum(axis=1)
                fp = parts["fp"][indices].sum(axis=1)
                fn = parts["fn"][indices].sum(axis=1)
                divisor = indices.shape[1] * y_true.shape[1]
                scalar = lambda name: parts[name][indices].mean(axis=1)
            denominator = 2 * tp + fp + fn
            per_label_f1 = np.divide(
                2 * tp, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0
            )
            tp_total = tp.sum(axis=1)
            fp_total = fp.sum(axis=1)
            fn_total = fn.sum(axis=1)
            micro_denominator = 2 * tp_total + fp_total + fn_total
            return {
                "micro_f1": np.divide(
                    2 * tp_total,
                    micro_denominator,
                    out=np.zeros_like(tp_total, dtype=float),
                    where=micro_denominator != 0,
                ),
                "macro_f1": per_label_f1.mean(axis=1),
                "hamming_loss": (
                    np.array([parts["mismatches"].sum() / divisor])
                    if indices is None
                    else parts["mismatches"][indices].sum(axis=1) / divisor
                ),
                "precision_at_k": scalar("precision_at_k"),
                "recall_at_k": scalar("recall_at_k"),
                "partial_hit_at_k": scalar("partial_hit_at_k"),
            }

        point_a = summarize(components_a)
        point_b = summarize(components_b)
        distributions = {name: [] for name in point_a}
        remaining = int(n_resamples)
        while remaining:
            count = min(int(chunk_size), remaining)
            indices = rng.integers(0, len(y_true), size=(count, len(y_true)))
            boot_a = summarize(components_a, indices)
            boot_b = summarize(components_b, indices)
            for metric in distributions:
                distributions[metric].append(boot_a[metric] - boot_b[metric])
            remaining -= count
        for metric, chunks in distributions.items():
            differences = np.concatenate(chunks)
            rows.append(
                {
                    "metric": metric,
                    "k": int(k),
                    "estimate": float(point_a[metric][0] - point_b[metric][0]),
                    "lower": float(np.percentile(differences, 2.5)),
                    "upper": float(np.percentile(differences, 97.5)),
                    "n_resamples": int(n_resamples),
                    "seed": int(seed),
                    "difference": "A_minus_B",
                    "favorable_direction": "lower" if metric == "hamming_loss" else "higher",
                    "multiplicity_adjustment": "none_exploratory",
                }
            )
    return pd.DataFrame(rows)


def _unique_sorted(values):
    return sorted({v for v in values if pd.notna(v)})


def parse_label_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        return list(parsed)
    return list(value)


def serialize_label_list(value):
    return json.dumps(list(value), ensure_ascii=False)


def process_raw_dataset(raw_df, min_course_count=5):
    required = [
        "courseId",
        "courseName",
        "courseDescription",
        "comp_id",
        "comp_name",
        "comp_description",
    ]
    missing = [col for col in required if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = raw_df.dropna(subset=["courseName", "courseDescription", "comp_name"]).copy()
    df = df.drop_duplicates()

    grouped = (
        df.groupby("courseId", as_index=False)
        .agg(
            {
                "courseName": "first",
                "courseDescription": "first",
                "courseHours": "first",
                "comp_id": _unique_sorted,
                "comp_name": _unique_sorted,
                "comp_description": _unique_sorted,
                "cat_id": _unique_sorted,
                "cat_name": _unique_sorted,
                "tax_id": _unique_sorted,
                "tax_name": _unique_sorted,
            }
        )
        .sort_values("courseId")
        .reset_index(drop=True)
    )

    counts = grouped["comp_name"].explode().value_counts()
    valid_labels = sorted(counts[counts >= min_course_count].index.tolist())
    removed_labels = sorted(counts[counts < min_course_count].index.tolist())

    grouped[LABEL_COLUMN] = grouped["comp_name"].apply(
        lambda labels: [label for label in labels if label in valid_labels]
    )
    processed = grouped[grouped[LABEL_COLUMN].map(len) > 0].copy()
    processed[TEXT_COLUMN] = (
        processed["courseName"].fillna("").astype(str).str.strip()
        + " "
        + processed["courseDescription"].fillna("").astype(str).str.strip()
    ).str.strip()

    metadata = {
        "min_course_count": min_course_count,
        "n_raw_rows": int(len(raw_df)),
        "n_rows_after_dropna": int(len(df)),
        "n_courses_before_filtering": int(len(grouped)),
        "n_courses_after_filtering": int(len(processed)),
        "n_labels_before_filtering": int(len(counts)),
        "n_labels_after_filtering": int(len(valid_labels)),
        "valid_labels": valid_labels,
        "removed_labels": removed_labels,
    }
    return processed.reset_index(drop=True), metadata


def save_processed_csv(processed_df, path):
    output = processed_df.copy()
    for col in output.columns:
        if output[col].map(lambda x: isinstance(x, list)).any():
            output[col] = output[col].apply(serialize_label_list)
    output.to_csv(path, index=False)


def load_processed_csv(path):
    df = pd.read_csv(path)
    list_columns = [
        "comp_id",
        "comp_name",
        "comp_description",
        "cat_id",
        "cat_name",
        "tax_id",
        "tax_name",
        LABEL_COLUMN,
    ]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_label_list)
    return df


def split_processed_dataset(processed_df, test_size=0.2, seed=DEFAULT_SEED):
    train_df, test_df = train_test_split(
        processed_df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    train_df = train_df.sort_values("courseId").reset_index(drop=True)
    test_df = test_df.sort_values("courseId").reset_index(drop=True)
    metadata = {
        "test_size": test_size,
        "seed": seed,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    return train_df, test_df, metadata


def save_split_csv(df, path):
    save_processed_csv(df, path)


def load_split_csv(path):
    return load_processed_csv(path)


def prepare_multilabel_targets(train_df, test_df, label_column=LABEL_COLUMN):
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_df[label_column])
    y_test = mlb.transform(test_df[label_column])
    return y_train, y_test, list(mlb.classes_), mlb


def labels_to_matrix(label_lists, classes):
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.zeros((len(label_lists), len(classes)), dtype=int)
    for row_idx, labels in enumerate(label_lists):
        for label in labels:
            if label in class_to_idx:
                y[row_idx, class_to_idx[label]] = 1
    return y


def normalize_score_matrix(y_score, n_labels):
    if isinstance(y_score, list):
        y_score = np.column_stack(
            [score[:, 1] if score.ndim == 2 and score.shape[1] > 1 else score.ravel() for score in y_score]
        )
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)
    if y_score.shape[1] != n_labels:
        raise ValueError(f"Score matrix has {y_score.shape[1]} labels, expected {n_labels}.")
    return y_score


def top_k_prediction_matrix(y_score, k):
    y_score = np.asarray(y_score, dtype=float)
    k = min(k, y_score.shape[1])
    top_indices = np.argsort(y_score, axis=1)[:, -k:]
    y_pred = np.zeros_like(y_score, dtype=int)
    for row_idx, indices in enumerate(top_indices):
        y_pred[row_idx, indices] = 1
    return y_pred, top_indices


def compute_multilabel_metrics(y_true, y_score, labels, method, k_values=DEFAULT_K_VALUES):
    y_true = np.asarray(y_true, dtype=int)
    y_score = normalize_score_matrix(y_score, y_true.shape[1])
    labels = list(labels)

    metric_rows = []
    prediction_rows = []
    for k in k_values:
        if k > y_true.shape[1]:
            continue
        y_pred, top_indices = top_k_prediction_matrix(y_score, k)
        hits = np.logical_and(y_true, y_pred).sum(axis=1)
        true_counts = y_true.sum(axis=1)

        precision_at_k = hits.sum() / (len(y_true) * k)
        recall_at_k = np.divide(
            hits,
            true_counts,
            out=np.zeros_like(hits, dtype=float),
            where=true_counts != 0,
        ).mean()
        partial_hit_at_k = (hits > 0).mean()

        try:
            lrap = label_ranking_average_precision_score(y_true, y_score)
        except ValueError:
            lrap = np.nan

        try:
            cov_error = coverage_error(y_true, y_score)
        except ValueError:
            cov_error = np.nan

        metric_rows.append(
            {
                "method": method,
                "k": k,
                "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "hamming_loss": hamming_loss(y_true, y_pred),
                "subset_accuracy": accuracy_score(y_true, y_pred),
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "partial_hit_at_k": partial_hit_at_k,
                "lrap": lrap,
                "coverage_error": cov_error,
                "n_samples": int(y_true.shape[0]),
                "n_labels": int(y_true.shape[1]),
            }
        )

        for row_idx, indices in enumerate(top_indices):
            ranked = list(reversed(indices.tolist()))
            prediction_rows.append(
                {
                    "method": method,
                    "k": k,
                    "sample_index": row_idx,
                    "predicted_labels": [labels[idx] for idx in ranked],
                    "true_labels": [labels[idx] for idx in np.where(y_true[row_idx] == 1)[0]],
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def compute_ranked_label_metrics(y_true, ranked_predictions, labels, method, k_values=DEFAULT_K_VALUES):
    y_true = np.asarray(y_true, dtype=int)
    labels = list(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    metric_rows = []
    prediction_rows = []
    for k in k_values:
        if k > len(labels):
            continue

        y_pred = np.zeros_like(y_true, dtype=int)
        normalized_rankings = []
        for row_idx, ranking in enumerate(ranked_predictions):
            deduped = []
            for label in ranking:
                if label in label_to_idx and label not in deduped:
                    deduped.append(label)
            selected = deduped[:k]
            normalized_rankings.append(selected)
            for label in selected:
                y_pred[row_idx, label_to_idx[label]] = 1

        hits = np.logical_and(y_true, y_pred).sum(axis=1)
        true_counts = y_true.sum(axis=1)
        precision_at_k = hits.sum() / (len(y_true) * k)
        recall_at_k = np.divide(
            hits,
            true_counts,
            out=np.zeros_like(hits, dtype=float),
            where=true_counts != 0,
        ).mean()
        partial_hit_at_k = (hits > 0).mean()

        metric_rows.append(
            {
                "method": method,
                "k": k,
                "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "hamming_loss": hamming_loss(y_true, y_pred),
                "subset_accuracy": accuracy_score(y_true, y_pred),
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "partial_hit_at_k": partial_hit_at_k,
                "lrap": np.nan,
                "coverage_error": np.nan,
                "n_samples": int(y_true.shape[0]),
                "n_labels": int(y_true.shape[1]),
            }
        )

        for row_idx, selected in enumerate(normalized_rankings):
            prediction_rows.append(
                {
                    "method": method,
                    "k": k,
                    "sample_index": row_idx,
                    "predicted_labels": selected,
                    "true_labels": [labels[idx] for idx in np.where(y_true[row_idx] == 1)[0]],
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def build_frequency_baseline_scores(y_train, n_samples):
    frequencies = np.asarray(y_train, dtype=float).mean(axis=0)
    return np.tile(frequencies, (n_samples, 1))


def build_random_distribution_baseline_scores(y_train, n_samples, seed=DEFAULT_SEED):
    frequencies = np.asarray(y_train, dtype=float).mean(axis=0)
    rng = np.random.default_rng(seed)
    return rng.random((n_samples, len(frequencies))) * frequencies.reshape(1, -1)


def evaluate_baselines(y_train, y_test, labels, k_values=DEFAULT_K_VALUES, seed=DEFAULT_SEED):
    rows = []
    predictions = []
    timings = []
    baselines = {
        "baseline_frequency_topk": lambda: build_frequency_baseline_scores(y_train, len(y_test)),
        "baseline_random_distribution": lambda: build_random_distribution_baseline_scores(
            y_train, len(y_test), seed=seed
        ),
    }

    for method, builder in baselines.items():
        train_start = time.perf_counter()
        train_time = time.perf_counter() - train_start
        infer_start = time.perf_counter()
        scores = builder()
        infer_time = time.perf_counter() - infer_start
        metric_df, pred_df = compute_multilabel_metrics(y_test, scores, labels, method, k_values)
        rows.append(metric_df)
        predictions.append(pred_df)
        timings.append(
            {
                "method": method,
                "train_seconds": train_time,
                "inference_seconds": infer_time,
                "inference_seconds_per_sample": infer_time / max(1, len(y_test)),
            }
        )

    return pd.concat(rows, ignore_index=True), pd.concat(predictions, ignore_index=True), pd.DataFrame(timings)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def export_experiment_artifacts(
    results_dir,
    method,
    metrics_df,
    predictions_df,
    timing_rows,
    config,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / f"{method}_metrics.csv"
    sensitivity_path = results_dir / f"{method}_sensitivity_by_k.csv"
    predictions_path = results_dir / f"{method}_predictions.csv"
    timing_path = results_dir / f"{method}_timing.csv"
    config_path = results_dir / f"{method}_config.json"

    metrics_df[metrics_df["k"] == 7].to_csv(metrics_path, index=False)
    metrics_df.to_csv(sensitivity_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    pd.DataFrame(timing_rows).to_csv(timing_path, index=False)

    enriched_config = dict(config)
    enriched_config["run_datetime_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(config_path, enriched_config)

    return {
        "metrics": str(metrics_path),
        "sensitivity": str(sensitivity_path),
        "predictions": str(predictions_path),
        "timing": str(timing_path),
        "config": str(config_path),
    }
