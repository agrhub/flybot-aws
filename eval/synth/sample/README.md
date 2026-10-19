# SIMULATED DATA: demo-30d

**SIMULATED DATA: synthetic composite (real empty trap photo + pasted fruit flies), not a field photo.**

Every photo here is a composite made by `eval/synth` (flybot-aws): a real photo of an *empty* trap with
fruit flies cut from a public dataset pasted in by OpenCV. None of it was captured in the field. The label is in
the file name (`SYNTH_`), the JPEG metadata (EXIF ImageDescription, XMP `flybot:synthetic="true"`, comment),
a visible `SIMULATED` stamp, and `synthetic: true` on every manifest row.

- Photos: 7 (demo-lure-a 2, demo-lure-b 1, demo-lure-c 1, demo-sticky-d 3), local dates 2026-09-25 .. 2026-10-15
- Damaged on purpose (`defect` set, with the action the policy expects): 2
- Seed 20261019, generator eval/synth of the commit that adds it, OpenCV 5.0.0
- Manifest: `manifest.json` (schema `eval/synth/manifest.schema.json`), `manifest.csv`

## Credits

- Flies: "fruit-fly" dataset by CamOS on Roboflow Universe, https://universe.roboflow.com/camos/fruit-fly-wmcqi/dataset/3, licensed CC BY 4.0. Crops from the test/valid splits.
- Backgrounds: Empty lure-trap photos from AGRHUB's orchard cameras (FlyBot), data/samples in this repository. No photo with people is used (data/samples/README.md).
- Sticky traps: the yellow card is synthetic too (no real sticky-card photo exists).
