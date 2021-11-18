from deslib.dcs.ola import OLA
from deslib.dcs.lca import LCA
from deslib.dcs.mcb import MCB
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.a_priori import APriori


def create_selector2(method, pool_classifiers, DFP=False, knn=None, k=7):
    selector = None

    method = method.lower()

    if method == 'ola':
        selector = OLA
    elif method == 'lca':
        selector = LCA
    elif method == 'knorau':
        selector = KNORAU
    elif method == 'knorae':
        selector = KNORAE
    elif method == 'apriori':
        selector = APriori
    elif method == 'aposteriori':
        selector = APosteriori
    elif method == 'mcb':
        selector = MCB

    return selector(pool_classifiers, k=k, knne=(knn == "knne"), DFP=DFP)
