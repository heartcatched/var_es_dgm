from scipy.stats import chi2
import numpy as np
import torch


def KupicksPOF(real, VaR, alpha=0.05, statistic=True):
    diff = real - VaR

    M = diff[diff < 0].shape[0]
    T = real.shape[0]

    num = (1 - alpha) ** (T - M) * alpha ** (M)
    den = (1 - M / T) ** (T - M) * (M / T) ** M

    LR_POF = -2 * np.log(num / den)

    if statistic:
        return chi2.sf(LR_POF, df=1), LR_POF

    return chi2.sf(LR_POF, df=1)


def HaasTBF(real, VaR, alpha=0.05, statistic=True):
    diff = real - VaR
    idx = torch.where(diff < 0)[0]
    N = idx[1:] - idx[:-1]

    S = 0
    for N_i in N:
        num = (1 - alpha) ** (N_i - 1) * (alpha)
        den = 1 / N_i * (1 - 1 / N_i) ** (N_i - 1)
        S += np.log(num / den)
    LR_TUFF = -2 * S

    if statistic:
        return chi2.sf(LR_TUFF, df=idx.shape[0] - 1), LR_TUFF

    return chi2.sf(LR_TUFF, df=idx.shape[0] - 1)

# Christoffersen Test (Independence and Proportions)
def ChristoffersenTest(real, VaR, alpha=0.05, statistic=True):
    diff = real - VaR
    N = (diff < 0).int()  
    
    N_lag = torch.roll(N, 1)
    N_lag[0] = 0  

    diff_N = N - N_lag
    P_0 = torch.sum(diff_N == 0) / diff_N.shape[0]
    P_1 = torch.sum(diff_N == 1) / diff_N.shape[0]

    LR = -2 * (P_0 * torch.log(P_0) + P_1 * torch.log(P_1))

    if statistic:
        return chi2.sf(LR.item(), df=1), LR.item()

    return chi2.sf(LR.item(), df=1)


# The "Kupicks" (Caldara et al., 2021)
def KupicksTest(real, VaR, alpha=0.05, statistic=True):
    diff = real - VaR
    M = diff[diff < 0].shape[0]
    T = real.shape[0]
    num = (1 - alpha) ** (T - M) * alpha ** (M)
    den = (1 - M / T) ** (T - M) * (M / T) ** M
    LR_Kupicks = -2 * np.log(num / den)

    if statistic:
        return chi2.sf(LR_Kupicks, df=1), LR_Kupicks

    return chi2.sf(LR_Kupicks, df=1)