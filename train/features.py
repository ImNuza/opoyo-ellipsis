"""Re-export of the shared feature code.

The implementation lives in ``shared.features`` so that the trainer and the
live edge path cannot drift. Kept here for the existing import sites.
"""

from shared.features import (  # noqa: F401
    FEATURE_NAMES,
    clip_of,
    crest_factor,
    peak_normalize,
    peak_window,
    spec_placeholder,
    vector,
)
