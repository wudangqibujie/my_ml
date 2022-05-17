import torch
import numpy as np


data = [
    [1, 2],
    [3, 4]
]
x_data = torch.tensor(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

shape = (2, 2, 2, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(torch.is_tensor(x_np), torch.is_tensor(np_array), torch.is_tensor(zeros_tensor))
print(torch.is_storage(x_np))
print(x_np.storage())

print(rand_tensor)
print(rand_tensor.storage(), torch.is_storage(rand_tensor))
rand_tensor_shp = torch.reshape(rand_tensor, (4, 2))
print(rand_tensor_shp.storage())
print(id(rand_tensor.storage()), id(rand_tensor_shp.storage()))


print(torch.numel(rand_tensor))
tns_eye = torch.eye(4, 5)
print(tns_eye)
tns_lin = torch.linspace(1, 5, 5)
print(tns_lin)


tns_rnd_p = torch.randperm(5)
print(tns_rnd_p)

tns_0 = torch.zeros((4, 3, 2))
tns_1 = torch.ones((4, 3, 2))
tns_con = torch.concat([tns_0, tns_1], dim=0)
print(tns_con.shape)


tns_rng = torch.arange(0, 10)
print(tns_rng)


