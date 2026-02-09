from .var_tests import KupicksPOF, HaasTBF, ChristoffersenTest, KupicksTest
from .es_tests import AcerbiSze1, AcerbiSze2


def generate_report(real, VaR, ES, alpha=0.05):
    return {
        "Kupicks POF": KupicksPOF(real, VaR, alpha=alpha, statistic=False),
        "Haas TBF": HaasTBF(real, VaR, alpha=alpha, statistic=False),
        "Christoffersen Test": ChristoffersenTest(real, VaR, alpha=alpha, statistic=False),
        "Kupicks Test": KupicksTest(real, VaR, alpha=alpha, statistic=False),
        "Acerbi Szekely 1": AcerbiSze1(real, VaR, ES),
        "Acerbi Szekely 2": AcerbiSze2(real, VaR, ES, alpha=alpha),
    }
