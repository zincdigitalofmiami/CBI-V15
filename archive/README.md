# Archive directory

This folder holds legacy tools, scripts, and scaffolding that are no longer part of the
active, supported toolchain, but are kept for historical reference. You indicated these
should be "pushed to external drive with the rest of the archives" — until that move
happens, everything is consolidated here.

Structure:

- tools/
  - augment/ (archived)
  - infrastructure/ (archived)
- legacy/
  - technical_indicators_incremental.py (legacy indicators script; superseded by SQL macros)
  - validation_schema.py (Pandera-based validation; replaced by Qodana + tests)
- Data/ (legacy capitalized data dir; replaced by data/)

Safe to delete at any time after you back up externally.
