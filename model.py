import torch
import torch.nn.functional as F

mult = None

def get_model(num_dim_input, num_dim_output, config):
    global mult
    model = torch.nn.Sequential(
            torch.nn.Linear(num_dim_input, 128, bias=False),
            # torch.nn.BatchNorm1d(300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 512, bias=False),
            # torch.nn.BatchNorm1d(300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 128, bias=False),
            # torch.nn.BatchNorm1d(300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, num_dim_output*num_dim_output, bias=False))
            # torch.nn.Linear(128, 1, bias=False))
    if hasattr(config, 'get_xt_scale'):
        scale = config.get_xt_scale()
        mult = torch.diag(torch.from_numpy(scale))
    else:
        mult = None

    # model = Polynomials(num_dim+num_dim+1+1, 1, orders=range(6))
    def forward_hook(module, input, output):
        I = torch.eye(num_dim).view((1, ))
        I = I.repeat(output.size(0), 1, 1).cuda()
        I.requires_grad_(False)
        output = output.view(-1,1,1)
        output = output * I
        return output
    # model.register_forward_hook(forward_hook)

    def forward(input):
        global mult
        output = model(input)
        # import ipdb; ipdb.set_trace()
        output = output.view(input.shape[0], num_dim_output, num_dim_output)
        if mult is not None:
            mult = mult.type(input.type())
            output = torch.matmul(output, mult)
        # output = output / (input[:, -2].unsqueeze(-1).unsqueeze(-1))
        return output
    return model, forward
