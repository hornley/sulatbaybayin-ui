# Compatibility shim: re-export BBoxDataset from the existing `script.dataset`
# so code that imports `src.detection.dataset` (legacy layout) keeps working.
try:
    from script.dataset import BBoxDataset
    __all__ = ['BBoxDataset']
except Exception:
    # If importing fails, raise a helpful error when attempted to import this shim
    raise
