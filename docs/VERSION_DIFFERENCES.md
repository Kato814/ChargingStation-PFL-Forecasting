# Version differences

## WITH_SPATIAL_V1.0

Purpose: exact reproduction archive for the audited final experiment.

- Input dimension: 539.
- Includes coordinates, POI descriptors and location-derived zone attributes.
- Uses the frozen fold-specific parameters from the final experiment.
- Reproduces the locked paper-result archive and is the authoritative record when exact historical reproduction is required.

## NO_SPATIAL_V1.0

Purpose: paper-input-aligned research variant for subsequent modification.

- Input dimension: 527.
- Excludes coordinates, POI descriptors and location-derived zone attributes.
- Uses the same fold-specific parameters as the frozen 539-D experiment so that feature removal can be isolated.
- The original validation-search provenance needed to re-select parameters is not available; this variant does not claim post-removal retuning.
- Neural Table-2 baseline predictions from the 539-D archive are not reused as matched-input results.

## Non-negotiable separation rule

Do not mix caches, sufficient statistics, OOF predictions, reference results or claims between the two variants. Changes should be developed inside one variant and ported explicitly only after numerical verification.
