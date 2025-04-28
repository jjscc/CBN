from typing import Any

import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.autograd import Function as F


class StiefelParameter(nn.Parameter):
    """ Parameter constrained to the Stiefel manifold (for BiMap layers) """
    pass


def init_bimap_parameter(W, type):
    """ initializes a (ho,hi,ni,no) 4D-StiefelParameter"""
    ho, hi, ni, no = W.shape
    for i in range(ho):
        for j in range(hi):
            if type == 0:
                xx = ni
            else:
                xx = no
            v = th.empty(xx, xx, dtype=W.dtype, device=W.device).uniform_(0., 1.)
            vv = th.svd(v.matmul(v.t()))[0][:ni, :no]
            W.data[i, j] = vv


class SPDParameter(nn.Parameter):
    """ Parameter constrained to the SPD manifold (for ParNorm) """
    pass


class CholeskyParameter(nn.Parameter):
    """ Parameter constrained to Cholesky """
    pass


def bimap(X, W):
    '''
    Bilinear mapping function
    :param X: Input matrix of shape (batch_size,n_in,n_in)
    :param W: Stiefel parameter of shape (n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,n_out,n_out)
    '''
    # print(X)
    # print('-----')
    # print(W)
    return W.t().matmul(X).matmul(W)


def bimap_channels(X, W):
    '''
    Bilinear mapping function over multiple input and output channels
    :param X: Input matrix of shape (batch_size,channels_in,n_in,n_in)
    :param W: Stiefel parameter of shape (channels_out,channels_in,n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,channels_out,n_out,n_out)
    '''
    # Pi=th.zeros(X.shape[0],1,W.shape[-1],W.shape[-1],dtype=X.dtype,device=X.device)
    # for j in range(X.shape[1]):
    #     Pi=Pi+bimap(X,W[j])
    batch_size, channels_in, n_in, _ = X.shape
    channels_out, _, _, n_out = W.shape
    P = th.zeros(batch_size, channels_out, n_out, n_out, dtype=X.dtype, device=X.device)
    for co in range(channels_out):
        P[:, co, :, :] = sum([bimap(X[:, ci, :, :], W[co, ci, :, :]) for ci in range(channels_in)])
    return P


def modeig_forward(P, op, eig_mode='svd', param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size, channels, n, n = P.shape  # batch size,channel depth,dimension
    U, S = th.zeros_like(P, device=P.device), th.zeros(batch_size, channels, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if (eig_mode == 'eig'):
                s, U[i, j] = th.linalg.eig(P[i, j])
                # S[i, j] = s[:, 0]
                S[i, j] = s[:]
            elif (eig_mode == 'svd'):
                U[i, j], S[i, j], _ = th.svd(P[i, j])
    S_fn = op.fn(S, param)
    X = U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2, 3))
    return X, U, S, S_fn


def modeig_backward(dx, U, S, S_fn, op, param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    # if __debug__:
    #     import pydevd
    #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
    S_fn_deriv = BatchDiag(op.fn_deriv(S, param))
    SS = S[..., None].repeat(1, 1, 1, S.shape[-1])
    SS_fn = S_fn[..., None].repeat(1, 1, 1, S_fn.shape[-1])
    L = (SS_fn - SS_fn.transpose(2, 3)) / (SS - SS.transpose(2, 3))
    L[L == -np.inf] = 0
    L[L == np.inf] = 0
    L[th.isnan(L)] = 0
    L = L + S_fn_deriv
    dp = L * (U.transpose(2, 3).matmul(dx).matmul(U))
    dp = U.matmul(dp).matmul(U.transpose(2, 3))
    return dp


class LogEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Log_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Log_op)


class ReEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Re_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Re_op)


class SqmEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of square root eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqm_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqm_op)


class SqminvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse square root eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqminv_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqminv_op)


def add_id(P, alpha):
    '''
    Input P of shape (batch_size,1,n,n)
    Add Id
    '''
    for i in range(P.shape[0]):
        P[i] = P[i] + alpha * P[i].trace() * th.eye(P[i].shape[-1], dtype=P.dtype, device=P.device)
    return P


def CongrG(P, G, mode):
    """
    Input P: (batch_size,channels) SPD matrices of size (n,n) or single matrix (n,n)
    Input G: matrix (n,n) to do the congruence by
    Output PP: (batch_size,channels) of congruence by sqm(G) or sqminv(G) or single matrix (n,n)
    """
    if (mode == 'pos'):
        GG = SqmEig.apply(G[None, None, :, :])
    elif (mode == 'neg'):
        GG = SqminvEig.apply(G[None, None, :, :])
    PP = GG.matmul(P).matmul(GG)
    return PP


class Exp_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.exp(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return th.exp(S)


class ExpEig(F):
    """
    Input P: (batch_size,h) symmetric matrices of size (n,n)
    Output X: (batch_size,h) of exponential eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Exp_op, eig_mode='svd')
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Exp_op)


def BatchDiag(P):
    """
    Input P: (batch_size,channels) vectors of size (n)
    Output Q: (batch_size,channels) diagonal matrices of size (n,n)
    """
    batch_size, channels, n = P.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            Q[i, j] = P[i, j].diag()
    return Q


def Cholesky_LogG(x, X):
    """ Logarithmc mapping of x on the SPD manifold at X """
    # lx = th.cholesky(x, upper=False)
    batch_size, channels, n, n = x.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        dx = th.diag(x[i][0])
        dL = th.diag(X)
        L = X - th.diag(dL)
        K = x[i][0] - th.diag(dx)
        Q[i][0] = K - L + th.diag(dL * th.log(1 / dL * dx))
    return Q


def Cholesky_ExpG(x, X):
    """ Exp mapping of x on the SPD manifold at X """
    batch_size, channels, n, n = x.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=x.dtype, device=x.device)
    # lx = th.cholesky(x, upper=False)
    for i in range(x.shape[0]):
        dx = th.diag(x[i][0])
        dL = th.diag(X)
        L = th.tril(X, -1)
        K = th.tril(x[i][0], -1)
        Q[i][0] = K + L + th.diag(dL * th.exp((1 / dL) * dx))
    return Q


def Cholesky_ExpG_Ins(x, X):
    """ Exp mapping of x on the SPD manifold at X """
    n, n = x.shape  # batch size,channel depth,dimension
    Q = th.zeros(n, n, dtype=x.dtype, device=x.device)
    dx = th.diag(x)
    dL = th.diag(X)
    K = th.tril(x, -1)
    L = th.tril(X, -1)
    Q = K + L + th.diag(dL * th.exp((1 / dL) * dx))
    return Q


def cholesky_karcher_step_xx(x):
    '''
    One step in the Karcher flow
    '''
    Q = th.zeros(x.shape[2], x.shape[2], dtype=x.dtype, device=x.device)
    P = th.zeros(x.shape[2], x.shape[2], dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        a = th.tril(x[i][0], -1)
        b = th.diag(th.log(th.diag(x[i][0])))
        Q = Q + a
        P = P + b
    G = Q / x.shape[0] + th.diag(th.exp(th.diag(P) / x.shape[0]))
    return G


def cholesky_dia(A):
    batch_size, channels, n, n = A.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=A.dtype, device=A.device)
    P = th.zeros(batch_size, channels, n, n, dtype=A.dtype, device=A.device)
    for i in range(batch_size):
        a = A[i][0]
        P[i][0] = th.diag(th.diag(a))
        Q[i][0] = a - P[i][0]
    return Q, P


def cholesky_geodesic(A, B, t):
    '''
    Geodesic from A to B at step t
    :param A: running mean
    :param B: mean
    :param t: momentum
    note1: th.diag(X) return diagonal matrix (nxn matrix return a 1xn vector)
    note2: X-th.diag(th.diag(X)) = th.tril(x,-1) return low triangular matrix
    :return:
    '''
    M = (1 - t) * (A - th.diag(th.diag(A))) + t * (B - th.diag(th.diag(B))) + th.diag(
        th.exp((1 - t) * th.log(th.diag(A)) + t * th.log(th.diag(B))))
    return M


def cholesky_geodesic_var(A, B, t):
    '''
    Geodesic from A to B at step t
    :param A: running mean
    :param B: mean
    :param t: momentum
    note1: th.diag(X) return diagonal matrix (nxn matrix return a 1xn vector)
    note2: X-th.diag(th.diag(X)) = th.tril(x,-1) return low triangular matrix
    :return:
    '''
    V = (1 - t) * A + t * B
    return V


def cholesky_pt(N, h, A, B):
    batch_size, channels, n, n = A.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=A.dtype, device=A.device)
    for i in range(N):
        a = A[i][0]
        Q[i][0] = th.tril(a, -1) - th.tril(B, -1) + th.diag((1 / th.diag(B)) * th.diag(a))
    return Q


def cholesky_pt_scal(N, A, B, c):  # here A is the centered samples, B is the variance
    batch_size, channels, n, n = A.shape  # batch size,channel depth,dimension
    S = th.zeros(batch_size, channels, n, n, dtype=A.dtype, device=A.device)
    for i in range(N):
        a = A[i][0]
        a_bar = (a / th.sqrt(B)) * c  # divide the variance and scaling in the tangent space
        S[i][0] = a_bar  # th.tril(a, -1) - th.tril(B, -1) + th.diag((1 / th.diag(B)) * th.diag(a))
    return S


def cholesky_pt_xx(N, h, A, B):
    batch_size, channels, n, n = A.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=A.dtype, device=A.device)
    for i in range(N):
        a = A[i][0]
        Q[i][0] = th.tril(a, -1) + th.tril(B, -1) + th.diag(th.diag(B) * th.diag(a))
    return Q


def cholesky_backward_mean(dx, X):  # speed up
    # 1. Cholesky decomposition
    downX, diaX = cholesky_dia(X)
    N, h, n, _ = downX.shape

    # 2. Logarithm and mean of diagonal elements
    logDiaX_mean = torch.mean(torch.log(diaX))

    # 3. Precompute exponentials and avoid redundant transpose
    exp_logDiaX_div_N = torch.exp(logDiaX_mean / N)

    # 4. Use torch.diagonal for faster extraction and computation
    downX_t = downX.transpose(2, 3)
    intermediate_result = downX_t.matmul(exp_logDiaX_div_N * dx)

    # 5. Use efficient element-wise operations
    diag_elements = torch.diagonal(1 / intermediate_result, dim1=-2, dim2=-1)

    # 6. Compute final result
    result = dx / N + diag_elements

    return result


def choleskyBaryGeom(x):
    '''
    x after cholesky decomposition
    '''
    k = 1
    for _ in range(k):
        G = cholesky_karcher_step_xx(x)
    return G


def choleskyBaryGeom_var(x, mean):
    '''
    x after cholesky decomposition
    compute the batch variance
    '''
    a1 = th.tril(x[:, 0, :, :], -1)
    a2 = th.tril(mean, -1)

    b1 = th.log(th.diagonal(x[:, 0, :, :], dim1=-2, dim2=-1))
    b2 = th.log(th.diagonal(mean, dim1=-2, dim2=-1))

    d1 = a1 - a2
    d2 = b1 - b2

    dist = th.norm(d1, p=2, dim=(-2, -1)).sum() + (d2 ** 2).sum()

    var = dist / x.shape[0]

    return var


def CholeskyDe(x):
    batch_size, channels, n, n = x.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        Q[i][0] = th.cholesky(x[i][0], upper=False)
    return Q


class CholeskyDecomposition(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        Q = CholeskyDe(P)
        ctx.save_for_backward(P)
        return Q

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        P = ctx.saved_variables[0]
        return cholesky_backward_de(dx, P)


def cholesky_backward_de(dx, Q, param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input Q: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    # if __debug__:
    #     import pydevd
    #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
    Q_inverse = th.inverse(Q)
    dp = Q_inverse.transpose(2, 3).matmul(th.tril(Q.transpose(2, 3).matmul(dx))).matmul(Q_inverse)
    return dp


def CholeskyMu(x):
    batch_size, channels, n, _ = x.shape
    assert x.shape[2] == x.shape[3], "The last two dimensions of x must be square matrices."

    Q = th.bmm(x.view(batch_size * channels, n, n), x.view(batch_size * channels, n, n).transpose(1, 2))

    Q = Q.view(batch_size, channels, n, n)

    return Q


class Log_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.log(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 1 / S


class Re_op():
    """ Log function and its derivative """
    _threshold = 1e-6

    @classmethod
    def fn(cls, S, param=None):
        return nn.Threshold(cls._threshold, cls._threshold)(S)

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (S > cls._threshold).double()


class Sqm_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 0.5 / th.sqrt(S)


class Sqminv_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return 1 / th.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return -0.5 / th.sqrt(S) ** 3


def init_pt_parameter(W):
    xx, xx = W.shape
    # v = th.empty(xx, xx, dtype=W.dtype, device=W.device).uniform_(0., 1.)
    # W.data = th.tril(v)
    th.eye(xx)


def clip(W):
    g = th.norm(W)
    thr = 10  # 10-->hdm05
    W = W * min(thr / g, 1)
    return W
