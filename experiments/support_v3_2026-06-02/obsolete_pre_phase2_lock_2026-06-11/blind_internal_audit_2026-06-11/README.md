# Blind Internal Audit Package (Core-5)

Internal visual audit only; do not call this a user study.

Scope: strict Core-5 Table 1 outputs, 5 tasks x 3 seeds x 4 paper-facing methods.

Rater workflow:

1. Open `rater_XX/rater_XX_index.html`.
2. Fill the matching `rater_XX/rater_XX_sheet.csv`.
3. Score each anonymous candidate independently.

Rubric:

- `edit_correct_1_5`: target edit is present and visually convincing.
- `relation_correct_1_5`: edit is in the requested relation/location.
- `source_preservation_1_5`: source identity, layout, and background are preserved.
- `locality_1_5`: change is localized to the intended edit region.
- `artifact_severity_1_5`: 1 is clean/best, 5 is severe/worst.
- `overall_1_5`: overall usefulness for the requested local edit.
- `failure_type`: one of none, semantic_miss, relation_error, under_edit, over_edit, identity_drift, background_drift, locality_leak, artifact, texture_failure, recolor_failure, ambiguous.

Method names are hidden in rater files. Keep `blind_internal_audit_method_key_private.csv` private until all rater sheets are frozen.
