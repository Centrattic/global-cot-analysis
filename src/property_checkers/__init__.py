from .base import PropertyChecker, PropertyCheckerBoolean, PropertyCheckerMulti
from .correctness import PropertyCheckerCorrectness
from .resampled import PropertyCheckerResampled
from .multi_algorithm import PropertyCheckerMultiAlgorithm

__all__ = [
    "PropertyChecker",
    "PropertyCheckerBoolean",
    "PropertyCheckerMulti",
    "PropertyCheckerCorrectness",
    "PropertyCheckerResampled",
    "PropertyCheckerMultiAlgorithm",
    "PROPERTY_CHECKER_REGISTRY",
]

PROPERTY_CHECKER_REGISTRY = {}
for _cls in [PropertyCheckerCorrectness, PropertyCheckerResampled, PropertyCheckerMultiAlgorithm]:
    if hasattr(_cls, "registry_name") and _cls.registry_name:
        PROPERTY_CHECKER_REGISTRY[_cls.registry_name] = _cls
