"""
Tests for training/registry.py — model artifact loading/versioning. Required
by the task spec ("Model loading/versioning"). No live Postgres is reachable
in this environment, so these use a mocked Session (same pattern as
tests/test_ingest_classification.py) to verify the queries/inserts are built
correctly, rather than a real round-trip.
"""
from unittest.mock import MagicMock

from training import registry
from models import ModelArtifact


def test_register_model_generates_a_version_when_none_given():
    session = MagicMock()
    artifact = registry.register_model(
        session,
        model_key="private_traintest",
        artifact_path="/models/private_traintest/x_model.pkl",
    )
    assert isinstance(artifact, ModelArtifact)
    assert artifact.version  # non-empty, auto-generated
    assert artifact.model_key == "private_traintest"
    session.add.assert_called_once_with(artifact)
    session.commit.assert_called_once()


def test_register_model_uses_explicit_version_when_given():
    session = MagicMock()
    artifact = registry.register_model(
        session,
        model_key="dealer_traintest",
        artifact_path="/models/dealer_traintest/x_model.pkl",
        version="20260101-000000",
    )
    assert artifact.version == "20260101-000000"


def test_register_model_is_baseline_flag_defaults_false():
    session = MagicMock()
    artifact = registry.register_model(
        session, model_key="private_traintest", artifact_path="/x.pkl",
    )
    assert artifact.is_baseline is False


def test_register_model_baseline_flag_can_be_set_true():
    session = MagicMock()
    artifact = registry.register_model(
        session, model_key="private_csv", artifact_path="/x.pkl", is_baseline=True,
    )
    assert artifact.is_baseline is True


def test_register_model_defaults_mutable_fields_to_empty_not_none():
    session = MagicMock()
    artifact = registry.register_model(
        session, model_key="private_traintest", artifact_path="/x.pkl",
    )
    assert artifact.feature_list == []
    assert artifact.model_config == {}
    assert artifact.metrics == {}


def test_register_model_passes_through_metrics_and_feature_list():
    session = MagicMock()
    artifact = registry.register_model(
        session,
        model_key="salvage_traintest",
        artifact_path="/x.pkl",
        feature_list=["year", "make", "damage_primary"],
        metrics={"mae": 1200.0, "r2": 0.8},
        dataset_fingerprint="abc123",
    )
    assert artifact.feature_list == ["year", "make", "damage_primary"]
    assert artifact.metrics == {"mae": 1200.0, "r2": 0.8}
    assert artifact.dataset_fingerprint == "abc123"


def test_get_latest_queries_ordered_by_trained_at_desc_limit_one():
    session = MagicMock()
    session.execute.return_value.scalars.return_value.first.return_value = "latest_artifact"
    result = registry.get_latest(session, "private_traintest")
    assert result == "latest_artifact"
    compiled = str(session.execute.call_args[0][0])
    assert "ORDER BY" in compiled
    assert "LIMIT" in compiled


def test_get_latest_returns_none_when_no_versions_exist():
    session = MagicMock()
    session.execute.return_value.scalars.return_value.first.return_value = None
    result = registry.get_latest(session, "salvage_traintest")
    assert result is None


def test_list_versions_queries_all_matching_model_key_newest_first():
    session = MagicMock()
    session.execute.return_value.scalars.return_value.all.return_value = ["v3", "v2", "v1"]
    result = registry.list_versions(session, "dealer_traintest")
    assert result == ["v3", "v2", "v1"]
    where_clause = session.execute.call_args[0][0]
    compiled = str(where_clause.compile(compile_kwargs={"literal_binds": True}))
    assert "'dealer_traintest'" in compiled
