import torch
from torch import nn
from tqdm.auto import tqdm


class QuantileLoss(nn.Module):
    def __init__(self, alpha):
        super(QuantileLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predicted, target):
        loss = torch.max((self.alpha - 1) * (target - predicted), self.alpha * (target - predicted))
        return torch.mean(loss)

class DeepVaR(nn.Module):
    def __init__(self, target_dim, input_size, num_layers=2, hidden_size=64, dropout_rate=0.2, num_inference_steps=100):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, target_dim)
        )
        self.criterion = QuantileLoss(alpha=0.05)  # Используем QuantileLoss для оценки VaR
        self.num_inference_steps = num_inference_steps
        self.target_dim = target_dim

    def forward(self, x, prediction_length=1):
        batch_size = x.shape[0]
        _, (h, c) = self.lstm(x)
        next_sample = self.fc(h[-1]).reshape(batch_size, 1, -1)
        future_samples = next_sample

        if prediction_length > 1:
            for _ in tqdm(range(1, prediction_length), desc="Prediction step"):
                _, (h, c) = self.lstm(next_sample, (h, c))
                next_sample = self.fc(h[-1]).reshape(batch_size, 1, -1)
                future_samples = torch.cat((future_samples, next_sample), dim=1)
        return future_samples

    def sample(self, context, n_samples=500, prediction_length=1):
        batch_size = context.shape[0]
        samples = torch.zeros(n_samples, batch_size, self.target_dim, device=context.device)

        for i in range(n_samples):
            future_samples = self.forward(context, prediction_length=prediction_length).squeeze(1)
            samples[i] = future_samples

        return samples

    def fit(self, train_loader, optimizer, n_epochs=50, device="cpu", verbose=True):
        losses = list()
        self.train()
        with tqdm(range(n_epochs), desc="Epochs", disable=not verbose) as t:
            for _ in t:
                total_loss = 0
                for train, target in train_loader:
                    loss = self.loss(train.to(device), target.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.detach().cpu().item()
                t.set_postfix(total_loss=(total_loss / len(train_loader)))
                losses.append(total_loss / len(train_loader))
        return losses

    def loss(self, x_context, x_prediction):
        batch_size = x_context.shape[0]
        _, (h, _) = self.lstm(x_context)
        h_t_minus_1 = h[-1]
        model_output = self.fc(h_t_minus_1)
        return self.criterion(model_output, x_prediction)
