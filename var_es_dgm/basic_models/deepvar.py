import torch
import numpy as np

class DeepVaR:
    def __init__(self, model, scaler=None, alpha=0.05, n_samples=1000, device="cuda", unscale_predictions=False):
        """
        Универсальная реализация DeepVaR (статья: Fatouros et al.)
        
        model: Обученная модель DeepAR (target_dim = 1 или N)
        scaler: Объект StandardScaler. Нужен только если unscale_predictions=True
        alpha: Уровень значимости (например, 0.05 или 0.01)
        n_samples: Количество путей Монте-Карло (по умолчанию 1000)
        unscale_predictions: Если False, возвращает VaR/ES в масштабе StandardScaler (как в вашем ноутбуке).
        """
        self.model = model
        self.scaler = scaler
        self.alpha = alpha
        self.n_samples = n_samples
        self.device = device
        self.unscale_predictions = unscale_predictions
        self.target_dim = model.target_dim

    def predict(self, context, weights=None, corr_matrix=None):
        self.model.eval()
        
        # 1. Задаем дефолтные веса и корреляцию для Univariate случая
        if self.target_dim == 1:
            weights = torch.tensor([1.0], dtype=torch.float32, device=self.device)
            corr_matrix = torch.tensor([[1.0]], dtype=torch.float32, device=self.device)
        else:
            if weights is None or corr_matrix is None:
                raise ValueError("Для Multivariate случая необходимо передать weights и corr_matrix")
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            corr_matrix = torch.tensor(corr_matrix, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # 2. Сэмплирование (Algorithm 1, строка 17)
            samples_scaled = self.model.sample(context, n_samples=self.n_samples, prediction_length=1)
            
            # Приводим к виду [n_samples, target_dim]
            if samples_scaled.dim() == 3:
                samples_scaled = samples_scaled.squeeze(1)
            elif samples_scaled.dim() == 4:
                samples_scaled = samples_scaled.squeeze(1).squeeze(1)

        # 3. Управление масштабом
        if self.unscale_predictions and self.scaler is not None:
            samples_np = samples_scaled.cpu().numpy()
            samples_unscaled = self.scaler.inverse_transform(samples_np)
            samples = torch.tensor(samples_unscaled, dtype=torch.float32, device=self.device)
        else:
            # Оставляем данные масштабированными (то, что нужно для вашего ноутбука)
            samples = samples_scaled.to(self.device)

        # 4. Вычисляем квантили (Algorithm 1, строки 19-20)
        lower_q = torch.quantile(samples, self.alpha, dim=0)       # [target_dim]
        upper_q = torch.quantile(samples, 1 - self.alpha, dim=0)   # [target_dim]

        # 5. Применяем веса позиций (Algorithm 1, строки 21-28)
        V = torch.zeros_like(weights)
        for i in range(self.target_dim):
            if weights[i] < 0:
                V[i] = weights[i] * upper_q[i] 
            else:
                V[i] = weights[i] * lower_q[i]

        # 6. Матричное вычисление VaR портфеля (Algorithm 1, строка 29)
        V = V.unsqueeze(0) # [1, target_dim]
        var_p_squared = torch.matmul(torch.matmul(V, corr_matrix), V.t())
        
        # Берем со знаком минус, так как VaR традиционно отображается как отрицательное число
        VaR_p = -torch.sqrt(torch.abs(var_p_squared)).squeeze()

        # 7. Эмпирический расчет Expected Shortfall (ES)
        simulated_portfolio_returns = torch.matmul(samples, weights)
        empirical_var = torch.quantile(simulated_portfolio_returns, self.alpha)
        
        tail_losses = simulated_portfolio_returns[simulated_portfolio_returns <= empirical_var]
        if len(tail_losses) > 0:
            ES_p = tail_losses.mean()
        else:
            ES_p = empirical_var

        return VaR_p.item(), ES_p.item()