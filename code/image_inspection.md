# Phase 5 — Image Inspection and Quality Assessment

## Components
- `code/image_quality.py`: deterministic quality measurements using Pillow only.
- `code/image_inspector.py`: per-image orchestration, structured observations, and error isolation.

## Quality signals
- Width/height and low-resolution flags.
- Grayscale mean brightness and contrast.
- Edge-energy variance as a rough low-detail/blur signal (not a calibrated blur classifier).
- Estimated share of near-white or near-black pixels as an exposure-clipping warning.
- Stable source hash from the existing normalizer for deduplication/caching in later phases.

## Design safeguards
- Input paths go through the existing dataset-root path validation and bounded image decoder.
- One broken/missing image becomes a structured observation; it does not abort inspection of other images in the same claim.
- Duplicate non-empty paths are processed once and output order follows input order.
- Quality warnings are not automatically treated as proof of fraud or damage.
- Semantic fields remain `unknown` and `relevance_confirmed=false`. Technical metrics cannot establish which object or part is pictured or whether damage is present; Phase 6 adds the vision provider that supplies those findings.

## Run tests
From the repository root:

    python -m unittest code.tests.test_image_quality -v

## Interpretation
`usable_for_review` means only that the image clears basic technical hard limits. It does not mean that it is relevant, authentic, or sufficient for the claim. Thresholds are explicit and injectable so they can be calibrated on validation data rather than silently tuned against the test set.
