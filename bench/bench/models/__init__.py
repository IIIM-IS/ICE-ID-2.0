"""Model adapters for entity resolution."""

from .base import BaseModel
from .nars import NarsModel
from .fellegi_sunter import FellegiSunterModel
from .rules import RulesModel

try:
    from .ensemble import (
        XGBoostModel,
        LightGBMModel,
        RandomForestModel,
        GradientBoostingModel,
    )
except ImportError:
    XGBoostModel = None
    LightGBMModel = None
    RandomForestModel = None
    GradientBoostingModel = None

try:
    from .ditto_adapter import DittoModel
except ImportError:
    DittoModel = None

try:
    from .zeroer_adapter import ZeroERModel
except ImportError:
    ZeroERModel = None

try:
    from .anymatch_adapter import AnyMatchModel
except ImportError:
    AnyMatchModel = None

try:
    from .matchgpt_adapter import MatchGPTModel
except ImportError:
    MatchGPTModel = None

try:
    from .hiergat_adapter import HierGATModel
except ImportError:
    HierGATModel = None

from .opennars_adapter import OpenNARSModel
