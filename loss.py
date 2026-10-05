import torch


class WeightedMSE:

    def __init__(self, reduction, outputs_points, device, model=None, l1_lambda=0.0):
        self.reduction = reduction
        outputs_idx = [point-1 for point in outputs_points]
        self.weights = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).to(device)
        self.weights = self.weights[outputs_idx]
        self.model = model
        self.l1_lambda = l1_lambda

    def __call__(self, input, target):

        squared_error = (input - target) ** 2 * self.weights
        
        loss = 0
        if self.reduction == 'mean':
            loss = squared_error.mean()
        elif self.reduction == 'sum':
            loss = squared_error.sum()
        else:
            return squared_error

        if self.model is not None and self.l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in self.model.parameters())
            loss += self.l1_lambda * l1_norm

        return loss

