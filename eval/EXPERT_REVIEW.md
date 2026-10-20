# Expert review of FlyBot Sentinel decisions (10 minutes)

For an AgriHub agronomist. You will judge 50 decisions the agent made on fruit-fly trap photos. You do not need to run anything:
a spreadsheet and the photos are enough.

## What you review

The agent watches a trap camera. For each photo it either does nothing (`no_op`), asks the camera to take another photo
(`capture`, e.g. because the photo is blurry or something covers the trap), asks a person to look (`request_human_review`),
or proposes an action for a person to approve (`propose_treatment`, never executed without a person). Your question for each
row is only this:

> With this photo and these numbers, is what the agent decided what a careful agronomist would also have done?

Not "is the fly count exactly right" (nobody has counted these photos), and not "is the code good". Judge the decision.

## Files

| File | What it is |
|---|---|
| `docs/evidence/eval/expert_review.csv` | the sheet: 55 rows (50 to review + 5 worked examples `EX-1..EX-5`) |
| `docs/evidence/eval/expert_review_sheet.html` | the same rows with each photo next to the decision (open in a browser; easier to read than the CSV) |
| `docs/evidence/eval/expert_review/<id>.jpg` | the photo of each row (the row's `evidence_link` is the same photo, or its S3 key) |

Open the CSV in Excel or Google Sheets (UTF-8). Do not change the first eight columns or the row order.

## The columns

`id`, `source` (where the decision comes from), `camera`, `captured_at`, `evidence_link`, `opencv_summary` (what the image
analysis measured: counts, quality issues, whether the trap is in view), `agent_decision` (action and rule), `agent_reason`
(the agent's own words).

Fill only the last two:

- `expert_verdict`: exactly one of
  - `correct`: I agree with the decision;
  - `incorrect`: I would have done something else (wrong action, missed a problem, or a needless alarm / action);
  - `unsure`: the photo and numbers do not let me decide.
- `expert_note`: optional for `correct`; **write one line for every `incorrect`** (what should it have done, or what is wrong
  in the photo). Start the note with `reviewer=<your name>;` if you can (it is how we know it is not an example).

Leave a row empty if you skip it.

## What each source means (so you know how to read it)

- `aws_live`: real photos from two EZVIZ trap cameras in production, decided by the live system today. `capture` rows are
  recapture requests (the camera could not actually be reached, so only the decision counts, not the result).
- `replay_30d_review`: real photos from the last 30 days, decisions computed afterwards on the same photos
  (`request_human_review` when two counters disagree on the fly count).
- `staged_proposal`: real photos but the **counts were rewritten to simulate a fly surge** (no real surge has happened yet,
  the traps are empty since 7 Oct). Judge whether the decision is right *given those counts*. A `propose_treatment` row here is
  a proposal awaiting a person, not a spray.
- `failure_injection`: a real photo that we **damaged on purpose** (blur, glare, dark, hand over the trap, text written on the
  picture, camera offline...). `opencv_summary` says what was done. Judge the agent's reaction to that situation.
- `EX-1..EX-5`: already filled by the agent that built the sheet (`reviewer=agent-example`) to show the format. They are
  **not** scored. Do not copy their verdicts: judge your 50 yourself.

## Rules of thumb (not rules)

- A blurry, glared or dark photo: asking for one more photo is right; counting it as is, is not.
- Something covering the trap (hand, animal, cloth): do not count the photo; one more photo is right.
- No trap in view: a person should know (a recapture will not help).
- A big difference between two counters on a clear photo: asking a person to look is right, but a needless review of a photo
  where the counts are both low and close is `incorrect`.
- A surge confirmed by the neighbouring traps: a proposal that waits for a person is right; any action without a person is not.
- When in doubt between `incorrect` and `unsure`: `unsure`, with a note.

## Timing

About 12 seconds per row. If you only have 5 minutes, do the rows in order and stop: the score works on whatever you filled
(it reports how many rows it used).

## When you are done

Save the CSV (same name, UTF-8, comma separated) and send it back. Then:

```
uv run python -m eval.score_expert_review      # policy-correct rate, by source and by decision, with a 95 % interval
uv run python -m eval.run_eval --render-only   # puts it in eval/RESULTS.md section 6
```

The score is `correct / (correct + incorrect)`; `unsure` rows are counted separately and also shown as a worst case (every
`unsure` counted as incorrect). Rows marked `reviewer=agent-example` are never scored. With 0 verdicts the report says "0
verdicts, not measured": nothing is filled in on your behalf.

## The online review page (how the advisor answered)

The project's agronomy advisor, MSc Trương Quốc Ánh (Institute of Agricultural Science for Southern Vietnam), reviews in
Vietnamese, without the CSV: a phone page on FlyBot (`flybot/client/public/review/<token>/`, no account, not indexed) shows
the same 50 rows with the photo, the measurements, the decision and the reason in plain Vietnamese, the original English
text one tap away, and the 5 examples apart. The English `opencv_summary`, `agent_decision` and `agent_reason` in the CSV
are the source of that presentation. The page saves as he goes (`GET/POST /api/public/expert-review/<token>`); "send" also
emails the answers to the operator.

The sheet was built on 14-17 Oct 2026 from the earlier Sentinel pipeline. Its decision names map to the current Lambda's
FlyBot actions: `capture` -> `capture_now`, `request_human_review` -> `review_request`, `propose_treatment` -> a
`review_request` of kind `treatment` (a proposal waiting for a person), `no_op` -> nothing sent.

Bring the answers into the sheet:

```
uv run python -m eval.import_expert_review --url https://flybot.agrhub.com/api/public/expert-review/<token> --translations notes_en.json
```

- `expert_verdict`: the verdict code as he gave it (`correct` / `incorrect` / `unsure`, no language).
- `expert_note`: `reviewer=Trương Quốc Ánh;` and then his note **verbatim, in Vietnamese**.
- `expert_note_en`: an English translation of that note, **made afterwards by the team, not written by the reviewer**
  (`notes_en.json`, `{row id: text}`). Reports quote this column and point to `expert_note` for his own words.

The import refuses an `incorrect` without a note, a note without a translation, and (without `--force`) a row that already
has a verdict; it never touches the `EX-*` rows.

## Notes for the people building the sheet

- Built by `eval/expert_review.py` (seeded stratified sample: 15 live AWS, 12 replay, 8 staged, 15 failure injection). Rebuilding
  refuses to overwrite a sheet that already has verdicts.
- Real photos are from fixed trap cameras (no people are expected in them). The only hand in the sheet is a synthetic one.
  Before the photos went on the review page, the YuNet face detector (`lambda/vision/sentinel/vision/models/`) found no
  face in any of the 55.
- One reviewer gives a policy-correct rate, not inter-rater agreement; a second reviewer on the same sheet would add that.
