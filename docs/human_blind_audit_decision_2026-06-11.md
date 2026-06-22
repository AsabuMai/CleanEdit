# Human Blind Audit Decision 2026-06-11

## Decision

Do not require human blind audit before Phase2 paper closure.

## Current State

The Phase2 blind audit package is complete and proxy-filled:

```text
experiments/support_v3_2026-06-02/blind_internal_audit_phase2_t1_t5_2026-06-11/
```

The current scores are Codex metric-guided proxy ratings, not human ratings.
They are useful for internal sanity checking and summary plumbing.

## Why Not Require Human Ratings Now

- The paper's core claim is preservation-first, and the quantitative evidence
  already directly measures non-edit-region preservation with outside-mask L1
  and source consistency.
- The proxy audit trend agrees with the quantitative preservation story.
- Adding human ratings now is optional strengthening, not a blocker for the
  conservative Phase2 claim.
- Human ratings would be necessary only if the paper wants to make a visual
  preference or user-study-style claim.

## Trigger To Collect Human Ratings Later

Collect human ratings if:

- the paper text wants to say "visual audit confirms" without proxy caveats;
- an advisor or reviewer asks for independent visual judgment;
- proxy ratings and qualitative inspection appear inconsistent;
- the final submission needs a stronger qualitative validation section.

## If Human Ratings Are Collected

Restore the blank sheets:

```text
rater_01/rater_01_sheet_blank_before_proxy.csv
rater_02/rater_02_sheet_blank_before_proxy.csv
rater_03/rater_03_sheet_blank_before_proxy.csv
```

Then collect ratings, replace the proxy-filled rater sheets, and rerun:

```bash
.venv/bin/python scripts/summarize_blind_internal_audit.py
```
