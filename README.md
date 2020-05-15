### Requirements
To install the requirements:
```
pip3 install -r requirements.txt
```

### Usage
Training:
```
mkdir log
python3 main.py
```

Plotting:
```
python3 plot.py
```

### Algorithm
1. Training data: we first randomly sample some tracjectories starting from a hyperrectangle in [data.py](https://github.com/sundw2014/Learning-Discrepancy/blob/19d33cdb3d5e4a62c177b958cece7c3be337cd6c/data.py#L48). Then, for each pair of trajectories xi_1, xi_2 and each time step t, we construct a training sample {xi_1(0), xi_1(t), ||xi_1(0) - xi_2(0)||, t, xi_2(t)}.
2. Train the neural network using the training data. The neural network is a matrix-valued function P of {xi_1(0), xi_1(t), ||xi_1(0) - xi_2(0)||, t}. For each training sample, we require the inequality ||P(xi_1(0), xi_1(t), ||xi_1(0) - xi_2(0)||, t) \cdot (xi_2(t) - xi_1(t))|| \leq 1 to hold, i.e. xi_2(t) falls into an ellipsoid centered at xi_1(t).
3. After training. Given an initial set that is a ball centered at c with radius r, then the reachable set at time t is an ellipsoid determined by P(c, xi(t), r, t) centered at xi(t), where xi is the trajectory starting from c.
