def batchnorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None
    x_hat, batch_var, eps, gamma, xmu = cache
    N, D = x_hat.shape
    sqrtvar = np.sqrt(batch_var + eps)
    inv_var = 1. / sqrtvar
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    dxhat = dout * gamma
    dxmu_1 = dxhat * inv_var
    dinvvar = np.sum(dxhat * xmu, axis=0)
    dvar = 0.5 * dinvvar * (-1.) / ((sqrtvar) ** 3)
    dxmu_2 = 2 * xmu * (1./N) * np.ones((N, D)) * dvar
    dx2 = (1./N) * np.ones((N, D)) * (-1) * np.sum(dxmu_1 + dxmu_2, axis=0)
    dx = dxmu_1 + dxmu_2+ dx2
    return dx, dgamma, dbeta