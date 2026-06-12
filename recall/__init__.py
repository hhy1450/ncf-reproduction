from recall.base import RecallBase

try:
    from recall.cf import UserCF, ItemCF
except ImportError:
    pass

try:
    from recall.svd import SVDRecall
except ImportError:
    pass

try:
    from recall.nmf import NMFRecall
except ImportError:
    pass
