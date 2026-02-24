import torch
from torch import nn
from torch.distributions import Normal, StudentT
from tqdm.auto import tqdm

class DeepAR(nn.Module):
    def __init__(self, target_dim, input_size, num_layers=2, hidden_size=64, dropout_rate=0.2, dist_type='student'):
        """
        dist_type: 'normal' или 'student' 
        """
        super().__init__()
        self.target_dim = target_dim
        self.dist_type = dist_type
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(128, target_dim)
        self.fc_sigma = nn.Linear(128, target_dim)
        
        if self.dist_type == 'student':
            self.fc_nu = nn.Linear(128, target_dim) 
            
        self.softplus = nn.Softplus()

    def get_distribution(self, h_out):
        features = self.fc(h_out)
        
        mu = self.fc_mu(features)
        sigma = self.softplus(self.fc_sigma(features)) + 1e-6 
        
        if self.dist_type == 'normal':
            return Normal(loc=mu, scale=sigma)
            
        elif self.dist_type == 'student':
            nu = self.softplus(self.fc_nu(features)) + 2.0 
            return StudentT(df=nu, loc=mu, scale=sigma)

    def forward(self, x, return_dist=False):
        """
        Если return_dist=True, возвращает объект распределения (нужно для обучения).
        Если return_dist=False, сэмплирует случайное значение (нужно для функции estimate_var_es_torch).
        """
        _, (h, c) = self.lstm(x)
        h_last = h[-1] 
        dist = self.get_distribution(h_last)
        
        if return_dist:
            return dist
        else:
            return dist.sample()

    def loss(self, x_context, x_target):
        dist = self.forward(x_context, return_dist=True)
        
        if x_target.dim() == 3 and x_target.shape[1] == 1:
            x_target = x_target.squeeze(1)
            
        nll = -dist.log_prob(x_target) 
        return nll.mean()

    def sample(self, context, n_samples=500, prediction_length=1):
        self.eval() 
        batch_size = context.shape[0]
        context_repeated = context.repeat_interleave(n_samples, dim=0)
        
        samples = []
        current_input = context_repeated
        
        with torch.no_grad():
            out, (h, c) = self.lstm(current_input)
            for _ in range(prediction_length):
                h_last = h[-1]
                dist = self.get_distribution(h_last)
                next_sample = dist.sample() 
                samples.append(next_sample.view(batch_size, n_samples, 1, self.target_dim))
                if prediction_length > 1:
                    _, (h, c) = self.lstm(next_sample.unsqueeze(1), (h, c))
                    
        samples = torch.cat(samples, dim=2)
        samples = samples.permute(1, 0, 2, 3)
        if prediction_length == 1:
            samples = samples.squeeze(2)
            
        return samples

    def fit(self, train_loader, optimizer, n_epochs=50, device="cpu", verbose=True):
        self.to(device)
        losses = list()
        
        with tqdm(range(n_epochs), desc="Epochs", disable=not verbose) as t:
            for _ in t:
                self.train()
                total_loss = 0
                for train, target in train_loader:
                    train, target = train.to(device), target.to(device)
                    
                    loss = self.loss(train, target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                avg_loss = total_loss / len(train_loader)
                t.set_postfix(NLL_Loss=avg_loss)
                losses.append(avg_loss)
        return losses