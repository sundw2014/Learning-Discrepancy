import torch
import torch.nn.functional as F

class Model(object):
    def __init__(self, num_dim_output):
        super(Model, self).__init__()
        self.num_dim_output = num_dim_output

    def __call__(self, x):
        x = x.cpu().detach().numpy()
        r = x[-2]
        t = x[-1]
        K = 1
        dt = t - self.t
        dt[dt < 0] = np.inf
        idx = dis.argmin()
        exp = (self.dt[:idx] * self.gammas[:idx]).sum() if idx > 0 else 0
        exp += (t - self.t[idx]) * self.gamms[idx]
        dis = r*K*np.exp(exp)
        return 1/dis*np.eye(self.num_dim_output)

    def load_state_dict(self, state_dict):
        self.gammas = state_dict[0]
        self.t = state_dict[1]
        self.dt = self.t[1:] - self.t[:-1]

    return model, forward

def get_model(num_dim_input, num_dim_output):
    model = Model(num_dim_output)
    return model, model
