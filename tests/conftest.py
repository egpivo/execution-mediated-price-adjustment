"""Makes the KEPT test files (test_rmt_core.py, test_support_ladder.py,
copied verbatim from work/jfqa_response_map_transport(_support)/code/) importable
without editing their `from rmt_core import ...` / `from support_core import ...`
statements, which used flat same-directory imports in the original layout.

src/research_core/methods/rmt.py is rmt_core.py renamed for package hygiene;
support_ladder.py is support_core.py renamed likewise (see
CODE_CORE_EXTRACTION_PLAN.md KEEP table). Aliasing sys.modules here means the
test files themselves stay byte-for-byte identical to the originals.
"""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_METHODS = _ROOT / "src" / "research_core" / "methods"
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_METHODS))
sys.path.insert(0, str(_SCRIPTS))

import rmt as _rmt_core  # noqa: E402
import support_ladder as _support_core  # noqa: E402

sys.modules.setdefault("rmt_core", _rmt_core)
sys.modules.setdefault("support_core", _support_core)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# run_rmt_analysis.py -> scripts/run_rmt.py; run_support.py -> scripts/run_support_ladder.py
sys.modules.setdefault("run_rmt_analysis", _load("run_rmt_analysis", _SCRIPTS / "run_rmt.py"))
sys.modules.setdefault("run_support", _load("run_support", _SCRIPTS / "run_support_ladder.py"))
