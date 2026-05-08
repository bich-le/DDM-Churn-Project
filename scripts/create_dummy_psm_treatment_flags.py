"""Create a dummy PSM treatment-flag file for dry-run testing only.

This script is intentionally temporary. It lets M5 test the propensity-score
pipeline before M4 delivers the real treatment flags.

Run from project root:
    python scripts/create_dummy_psm_treatment_flags.py --config config/paths.yaml --overwrite

Important:
- Do NOT use the dummy output for reporting or final PSM analysis.
- Replace models/psm_inputs/psm_treatment_flags.csv with the real M4 file before final handoff.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.utils import ensure_project_structure, find_project_root, load_config, resolve_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create dummy psm_treatment_flags.csv for dry-run testing."
    )
    parser.add_argument("--config", default="config/paths.yaml")
    parser.add_argument(
        "--treated-rate",
        type=float,
        default=0.48,
        help="Dummy treated share. Use only for dry-run testing.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing psm_treatment_flags.csv if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    try:
        project_root = find_project_root()
    except FileNotFoundError:
        # Some GitHub-ready zips do not include the Data/ folder.
        # In that case, this script still runs from the repository root inferred from its own location.
        project_root = PROJECT_ROOT_FOR_IMPORT

    config = load_config(config_path=args.config, project_root=project_root)
    paths = resolve_project_paths(config, project_root)
    ensure_project_structure(paths)

    id_col = config.get("modeling", {}).get("id_col", "household_key")
    cut_off_day = int(config.get("psm", {}).get("treatment_cutoff_day", config.get("modeling", {}).get("cut_off_day", 651)))
    feature_path = paths["feature_table_csv"]
    treatment_path = paths.get("psm_treatment_flags_csv", paths["project_root"] / "models" / "psm_inputs" / "psm_treatment_flags.csv")
    treatment_path.parent.mkdir(parents=True, exist_ok=True)

    if treatment_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Treatment flag file already exists: {treatment_path}. "
            "Use --overwrite only if this is still a dummy dry-run file."
        )

    features = pd.read_csv(feature_path, usecols=[id_col])
    features[id_col] = features[id_col].astype(int)
    features = features.drop_duplicates(id_col).reset_index(drop=True)

    rng = np.random.default_rng(args.random_state)
    dummy = pd.DataFrame(
        {
            id_col: features[id_col],
            "is_treated": rng.binomial(1, args.treated_rate, size=len(features)),
            "treatment_source": "DUMMY_PLACEHOLDER_DO_NOT_REPORT",
            "treatment_cutoff_day": cut_off_day,
        }
    )
    dummy.to_csv(treatment_path, index=False)

    counts = dummy["is_treated"].value_counts(dropna=False).to_dict()
    share = dummy["is_treated"].mean()
    print(f"[DRY-RUN ONLY] Wrote dummy treatment flags: {treatment_path}")
    print(f"Rows: {len(dummy)} | Treated share: {share:.3f} | Counts: {counts}")
    print("Replace this file with M4's real treatment flags before final PSM handoff.")


if __name__ == "__main__":
    main()
