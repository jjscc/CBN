import torch as th
import torch.nn as nn
from torch.autograd import Function as F
from . import functional

dtype = th.double
device = th.device('cpu')


class CholeskyDe(nn.Module):
    '''
      Cholesky Decomposition
    '''

    def forward(self, P):
        return functional.CholeskyDecomposition.apply(P)


class CholeskyBatchNormSPD(nn.Module):
    """
      Implement Riemannian batch normalization in the Cholesky manifold
    """

    def __init__(self, n):
        super(__class__, self).__init__()
        self.momentum = 0.1
        self.running_mean = th.eye(n, dtype=dtype, requires_grad=False)  ################################
        self.running_var = th.tensor(1.0)
        self.s = nn.Parameter(th.tensor(1.0), requires_grad=True)


    def forward(self, X):
        N, h, n, n = X.shape
        X_batched = X.permute(2, 3, 0, 1).contiguous().view(n, n, N * h, 1).permute(2, 3, 0, 1).contiguous()
        if self.training:
            mean = functional.choleskyBaryGeom(X_batched)
            var = functional.choleskyBaryGeom_var(X_batched, mean)
            with th.no_grad():
                self.running_mean.data = functional.cholesky_geodesic(self.running_mean, mean, self.momentum)
                self.running_var.data = functional.cholesky_geodesic_var(self.running_var, var, self.momentum)  # update running var
            X_centered = functional.cholesky_pt(N, h, X_batched, mean)
            X_scalling = functional.cholesky_pt_scal(N, X_centered, var, self.s)
        else:
            X_centered = functional.cholesky_pt(N, h, X_batched, self.running_mean)
            X_scalling = functional.cholesky_pt_scal(N, X_centered, self.running_var, self.s)  # batch scalling in the test stage
        return X_scalling.permute(2, 3, 0, 1).contiguous().view(n, n, N, h).permute(2, 3, 0, 1).contiguous()


class CholeskyPt(nn.Module):

    """
       Implement Cholesky manifold-valued bias operation
    """

    def __init__(self, n):
        super(__class__, self).__init__()
        self.weight = functional.CholeskyParameter(th.empty(n, n, dtype=dtype, device=device))
        functional.init_pt_parameter(self.weight) # for opti of bias on L_+


    def forward(self, X):
        N, h, n, n = X.shape
        X_N = functional.cholesky_pt_xx(N, h, X, self.weight)  # for opti of bias on L_+
        return X_N.permute(2, 3, 0, 1).contiguous().view(n, n, N, h).permute(2, 3, 0, 1).contiguous()


class CholeskyMu(nn.Module):
    '''
      The inverse map of Cholesky decomposition
    '''

    def forward(self, P):
        return functional.CholeskyMu(P)

class BiMap(nn.Module):
    """
    Input X: (batch_size,hi) SPD matrices of size (ni,ni)
    Output P: (batch_size,ho) of bilinearly mapped matrices of size (no,no)
    Stiefel parameter of size (ho,hi,ni,no)
    """

    def __init__(self, ho, hi, ni, no, bType):
        super(BiMap, self).__init__()
        self._W = functional.StiefelParameter(th.empty(ho, hi, ni, no, dtype=dtype, device=device))
        self._ho = ho
        self._hi = hi
        self._ni = ni
        self._no = no
        self.bType = bType
        functional.init_bimap_parameter(self._W, self.bType)

    def forward(self, X):
        return functional.bimap_channels(X, self._W)


class LogEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return functional.LogEig.apply(P)


class ReEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return functional.ReEig.apply(P)
