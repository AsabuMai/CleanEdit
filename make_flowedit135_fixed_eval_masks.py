from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
GENERATOR = PROJ / "make_flowedit118_fixed_eval_masks.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("flowedit_fixed_eval_masks", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_generator()
    module.MANIFEST = PROJ / "data/flowedit_compatible_135/manifest.json"
    module.OUT_DIR = PROJ / "data/flowedit_compatible_135/eval_masks"
    module.OVERLAY_DIR = module.OUT_DIR / "overlays"
    module.AUDIT_CSV = module.OUT_DIR / "_audit.csv"
    module.AUDIT_JSON = module.OUT_DIR / "_audit.json"
    module.CONTACT = module.OUT_DIR / "_contact_sheet.jpg"
    module.main()


if __name__ == "__main__":
    main()
