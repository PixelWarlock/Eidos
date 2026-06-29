from eidos.regularizers.gram import GramReg, feature_gram
from eidos.regularizers.sigreg import SIGReg, WeakSIGReg

REGULARIZERS = {
    "gram": GramReg,
    "sigreg": SIGReg,
    "weak_sigreg": WeakSIGReg,
}

__all__ = ["REGULARIZERS", "GramReg", "SIGReg", "WeakSIGReg", "feature_gram"]
