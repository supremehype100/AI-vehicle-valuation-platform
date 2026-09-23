"""
Docker configuration tests ("Docker configuration where practical" per the
task spec). Docker itself isn't installed in this environment, so this
validates what's actually checkable without it: the compose files are valid
YAML, define the expected services, never auto-run a job on `up`, and the GPU
reservation / CPU-fallback override are structurally present.
"""
import yaml
import pytest


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_docker_compose_is_valid_yaml():
    compose = _load("docker-compose.yml")
    assert isinstance(compose, dict)


def test_docker_compose_defines_expected_services():
    compose = _load("docker-compose.yml")
    assert set(compose["services"].keys()) == {"db", "ollama", "app"}


def test_db_service_uses_postgres_with_persistent_volume():
    compose = _load("docker-compose.yml")
    db = compose["services"]["db"]
    assert db["image"].startswith("postgres")
    volume_targets = [v.split(":")[-1] for v in db.get("volumes", [])]
    assert "/var/lib/postgresql/data" in volume_targets


def test_ollama_service_requests_gpu_reservation():
    compose = _load("docker-compose.yml")
    ollama = compose["services"]["ollama"]
    devices = ollama["deploy"]["resources"]["reservations"]["devices"]
    assert any(d.get("driver") == "nvidia" for d in devices)


def test_app_service_default_command_is_idle_not_a_job():
    """`docker compose up` must never auto-start a scrape/train/backfill job."""
    compose = _load("docker-compose.yml")
    app = compose["services"]["app"]
    command = app.get("command")
    assert command is not None
    joined = " ".join(command) if isinstance(command, list) else str(command)
    for job_keyword in ("scrape", "train", "backfill"):
        assert job_keyword not in joined.lower()


def test_app_service_points_at_db_and_ollama_service_names():
    compose = _load("docker-compose.yml")
    env = compose["services"]["app"]["environment"]
    assert "db:5432" in env["DATABASE_URL"]
    assert env["OLLAMA_HOST"] == "http://ollama:11434"


def test_cpu_only_override_file_is_valid_yaml_and_touches_ollama():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        class _IgnoreUnknownTagsLoader(yaml.SafeLoader):
            pass

        def _ignore_unknown(loader, tag_suffix, node):
            return None

        _IgnoreUnknownTagsLoader.add_multi_constructor("!", _ignore_unknown)
        with open("docker-compose.cpu-only.yml", encoding="utf-8") as f:
            override = yaml.load(f, Loader=_IgnoreUnknownTagsLoader)
    assert "ollama" in override.get("services", {})


def test_dockerignore_excludes_private_and_generated_paths():
    with open(".dockerignore", encoding="utf-8") as f:
        content = f.read()
    for path in ("fb_session", "fb_chrome_profile_fresh", ".csv", ".git"):
        assert path in content, f"{path} missing from .dockerignore"


def test_env_example_documents_every_config_var():
    import config as cfg_module
    with open(".env.example", encoding="utf-8") as f:
        env_example = f.read()

    checked = 0
    for name in ("DATABASE_URL", "MODEL_DIR", "OLLAMA_HOST", "QWEN_MODEL",
                 "FB_LISTING_DELAY_MIN_SECONDS", "FB_LISTING_DELAY_MAX_SECONDS"):
        assert hasattr(cfg_module, name), f"config.py no longer defines {name}"
        assert name in env_example, f"{name} missing from .env.example"
        checked += 1
    assert checked == 6
