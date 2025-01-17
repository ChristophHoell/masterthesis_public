import torch
import torch.nn as nn

def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))

def masked_l2(a, b, mask):
    """
    Calculates the l2 (MSE) loss between a and b given mask
    -> calculates the l2 difference between a nd b
    -> calculates the loss sum of diff * mask (removes masked elements)
    -> calculates average with respect to non zero mask elements (number of elements "visible")
    """

    l2_loss = lambda a, b: (a - b) ** 2

    loss = l2_loss(a, b)
    loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
    #n_entries = a.shape[1]
    #print(loss)
    n_entries = a.shape[2] * a.shape[3]
    non_zero_elements = sum_flat(mask) * n_entries
    #print(f"{sum_flat(mask)} - {n_entries}")
    mse_loss_val = loss / non_zero_elements
    return mse_loss_val



mse = nn.MSELoss()


a = torch.randn((8, 90, 5023, 3))
b = torch.randn((8, 90, 5023, 3))
m = torch.ones((8, 90, 1, 1))

print(mse(a, b))
print(masked_l2(a, b, m).mean())