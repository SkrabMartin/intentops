"""Public architecture and source links match the repository, not local artifacts."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_architecture_real_paths_and_local_only_note():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = readme.split("## Repository Architecture\n", 1)[1].split("\n## ", 1)[0]
    tree = re.search(r"```text\n(.*?)\n```", architecture, re.DOTALL)[1]
    assert len(tree.splitlines()) <= 80
    for relative in (
        "app/api.py",
        "app/inference.py",
        "app/schemas.py",
        "app/request_limits.py",
        "app/local_http.py",
        "ml/dataset.py",
        "ml/data_config.py",
        "ml/train.py",
        "ml/evaluate.py",
        "ml/model_support.py",
        "ml/calibration.py",
        "ml/calibrate.py",
        "ml/review_policy.py",
        "dashboard/streamlit_app.py",
        "dashboard/batch_ui.py",
        "dashboard/csv_workflow.py",
        "scripts/benchmark_serving.py",
        "notebooks/exploration.ipynb",
        "artifacts/.gitkeep",
        ".github/workflows/tests.yml",
        "Dockerfile",
        "pyproject.toml",
        "compose.yaml",
        "compose.model.yaml",
        "compose.policy.yaml",
        "docker-constraints.txt",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "docs/batch.md",
        "docs/benchmarks.md",
    ):
        assert (ROOT / relative).is_file(), relative
        assert Path(relative).name in tree, relative
    assert "**Local-only storage:**" in architecture and "10.13%" in architecture
    assert "10.00%" in architecture and "No paid API" in architecture
    paths_only = "\n".join(line.split("#", 1)[0] for line in tree.splitlines())
    assert "policies/" not in paths_only and "data/" not in paths_only
    assert "current.json" not in tree and ".joblib" not in tree


def test_public_readme_and_new_document_links_resolve():
    for relative in ("README.md", "docs/batch.md", "docs/benchmarks.md"):
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        assert "C:\\Users\\" not in text and "PHASE_5_REPORT" not in text
        assert "placeholder must be replaced" not in text
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            assert (path.parent / target.split("#", 1)[0]).is_file(), target
