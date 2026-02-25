import torch
import numpy as np


class HistoricalSimulation:
    def __init__(self, alpha=0.05) -> None:
        """
        Initializes the HistoricalSimulation class with a specified confidence level.

        Parameters
        ----------
        alpha : float, optional
            Confidence level for Value at Risk (VaR) and Expected Shortfall (ES) calculations,
            by default 0.05.
        """
        self.alpha = alpha

    def fit(self, *args, **kwargs):
        """
        Placeholder fit method for compatibility with other models.
        """
        pass

    def predict(self, context, **kwargs):
        """
        Predicts Value at Risk (VaR) and Expected Shortfall (ES) for univariate context.

        Parameters
        ----------
        context : torch.Tensor
            Input data tensor.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        tuple
            VaR and ES values.
        """
        if context.shape[-1] > 1:
            return self.predict_multivariate(context, **kwargs)
        context = context.flatten()
        VaR = torch.quantile(context, q=self.alpha)
        ES = context[torch.where(context <= VaR)[0]]
        ES = torch.sum(ES) / ES.shape[0]
        return VaR, ES

    def predict_multivariate(self, context, **kwargs):
        """
        Predicts multivariate VaR and ES based on individual VaR and ES aggregated along assets in a portfolio.

        Parameters
        ----------
        context : torch.Tensor
            TxL array, where L is the number of variables.
        **kwargs : dict
            Additional arguments, which may include:
            - scaler : Scaler object for inverse transforming the context.
            - R : Precomputed correlation matrix.

        Returns
        -------
        tuple
            VaR and ES values.
        """
        if "scaler" in kwargs:
            context = torch.tensor(
                kwargs["scaler"].inverse_transform(torch.squeeze(context))
            )

        # Estimating individual VaR and ES
        n_vars = context.shape[-1]
        VaRs = torch.quantile(context, q=self.alpha, dim=0) / n_vars

        ESs = torch.zeros(n_vars)
        for i in range(n_vars):
            context_i = context[:, i]
            VaR_i = VaRs[i]
            ES_i = context_i[torch.where(context_i <= VaR_i)[0]]
            ESs[i] = torch.sum(ES_i) / (ES_i.shape[0] * n_vars)

        # Estimating correlation matrix for VaR or taking provided
        if "R" in kwargs:
            R = kwargs["R"]
        else:
            R = torch.corrcoef(context.T)

        # Computing final VaR and ES
        VaR = -torch.sqrt(VaRs @ R.to(VaRs.device) @ VaRs.T)
        ES = torch.sum(ESs)

        return VaR, ES
