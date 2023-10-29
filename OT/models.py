import torch


class Reg(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, model_type: str, torch_dtype: torch.dtype,
                 bias: bool,
                 torch_device: torch.device, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        if model_type == 'linear':
            self.models = torch.nn.ModuleList(
                [torch.nn.Linear(in_features=in_dim, out_features=1, bias=bias, dtype=torch_dtype, device=torch_device)
                 for _ in range(out_dim)])
            # self.model = torch.nn.Linear(in_dim, out_dim)
        elif model_type == 'nonlinear':
            self.models = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(in_dim, hidden_dim),
                                                                   torch.nn.ReLU(),
                                                                   torch.nn.Linear(hidden_dim, 1))
                                               for _ in range(out_dim)])
        else:
            raise ValueError(f'invalid model_type {model_type}')

    def forward(self, x):
        y_hat_list = [model_(x) for model_ in self.models]
        y_hat_ = torch.cat(y_hat_list, dim=1)
        return y_hat_
