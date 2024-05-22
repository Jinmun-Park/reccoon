import numpy as np
import torch

class FieldAwareFactorizationMachineModel(torch.nn.Module):
    """
    @Reference:
        Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.
    """

    def __init__(self, field_dims, embed_dim: int):
        """
        :param field_dims: max(UserID), max(MovieID)
        :param embed_dim: k, where latent_vector [1,2,4] are roughly same
        """
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)

    def forward(self, x):
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))


class FeaturesLinear(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        """
        sigma(x_i, w_i)
        :param output_dim: 1
        """
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.longlong)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FieldAwareFactorizationMachine(torch.nn.Module):
    def __init__(self, field_dims, embed_dim: int):
        """
        sigma sigma ( (w_j1,f2 * w_j2,f1) * (x_j1 * x_j2) )

        @Reference:
            https://yngie-c.github.io/deep%20learning/2020/03/17/parameter_init/
            Glorot, X. & Bengio, Y. (2010). `Understanding the difficulty of training deep feedforward neural networks`
        """
        super().__init__()
        self.num_fields = len(field_dims)  # 2
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.longlong)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix