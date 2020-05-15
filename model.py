import torch
import torch.nn.functional as F

class Polynomials(torch.nn.Module):
    def __init__(self, input_dim, output_dim, orders=None):
        super(Polynomials, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.orders = orders
        self.linear = torch.nn.Linear(self.input_dim * len(orders), self.output_dim, bias=False)

    def forward(self, input):
        # input: B x ...
        # output: B x self.out_dim
        bs = input.size(0)
        input = input.view(bs, -1)
        assert input.size(1) == self.input_dim
        x = []
        for order in self.orders:
            x.append(input ** order)
        x = torch.cat(x, dim=1) # B x (self.input_dim*len(orders))
        output = self.linear(x)
        return output

num_dim_projected = 3
def projection(x):
    return x[:, 0:num_dim_projected]

def get_model(num_dim):
    model = torch.nn.Sequential(
            torch.nn.Linear(num_dim+num_dim+1+1, 128, bias=False),
            # torch.nn.BatchNorm1d(300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 512, bias=False),
            # torch.nn.BatchNorm1d(300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 128, bias=False),
            # torch.nn.BatchNorm1d(300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, num_dim_projected * num_dim_projected, bias=False))
            # torch.nn.Linear(128, 1, bias=False))
    # model = Polynomials(num_dim+num_dim+1+1, 1, orders=range(6))
    def forward_hook(module, input, output):
        I = torch.eye(num_dim).view((1, num_dim_projected * num_dim_projected))
        I = I.repeat(output.size(0), 1, 1).cuda()
        I.requires_grad_(False)
        output = output.view(-1,1,1)
        output = output * I
        return output
    # model.register_forward_hook(forward_hook)
    def forward(input):
        output = model(input)
        # import ipdb; ipdb.set_trace()
        output = output.view(input.shape[0], num_dim_projected, num_dim_projected)
        output = output / (input[:, -2].unsqueeze(-1).unsqueeze(-1))
        return output
    return model, forward

# model.margin = torch.nn.Parameter(torch.Tensor(1))
# model.P = torch.nn.Parameter(torch.from_numpy(10*np.eye(2,2).astype('float32')).view(1,2,2))
# model.P = torch.nn.Parameter(torch.from_numpy(100*np.eye(2,2).astype('float32')).view(1,2,2))
# model.P = torch.nn.Parameter(torch.from_numpy(100*np.eye(2,2).astype('float32')).view(1,2,2))
# model.P = torch.nn.Parameter(torch.from_numpy(np.array([[-176.81919284, 306.34485252], [-306.34485252,  496.81919284]]).astype('float32')).view(1,2,2))
