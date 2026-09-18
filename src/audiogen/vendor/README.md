# Compatibility imports

Numerical implementations now live in the YuE engine fork's `yue2.profiles`.
`metal_norm.py` and `sampling.py` retain old import paths for diagnostics; they
only delegate to the engine. The engine bundles the source attribution and
licenses. Audiogen does not install or patch these operations at runtime.

The original license files are retained with the historical diagnostic sources.
