import torch
from ..basic_models import HistoricalSimulation


def estimate_var_es_torch(model, test_obs, alpha=0.05, n_samples=500, device="cpu"):
    model.eval()
    model.to(device)
    batch = 500
    arr = torch.zeros((n_samples // batch) * batch)

    with torch.no_grad():
        for i in range(0, n_samples // batch):
            x = torch.cat([test_obs for _ in range(batch)])
            arr[i * batch : (i + 1) * batch] = (
                model.forward(x.to(device)).flatten().detach().cpu()
            )
            
    arr = arr.reshape(-1, 1)
    
    arr = arr.to(device, dtype=torch.float32) 
    
    est = HistoricalSimulation(alpha=alpha)

    return est.predict(arr)


def estimate_var_es_torch_multivariate(
    model, test_obs, scaler, R, alpha=0.05, n_samples=500, device="cpu"
):
    model.eval()
    model.to(device)
    batch = 500
    arr = torch.zeros((n_samples // batch) * batch, test_obs.shape[-1])

    with torch.no_grad():
        for i in range(0, n_samples // batch):
            x = torch.cat([test_obs for _ in range(batch)])
            arr[i * batch : (i + 1) * batch] = torch.squeeze(
                model.forward(x.to(device)).detach().cpu()
            )

    arr_unscaled = scaler.inverse_transform(arr)
    arr = torch.tensor(arr_unscaled, dtype=torch.float32, device=device)
    
    R = R.to(device=device, dtype=torch.float32)

    est = HistoricalSimulation(alpha=alpha)

    return est.predict(arr, R=R)