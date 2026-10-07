"""Coursework implementation retained from student/assisted source; see ATTRIBUTION.md."""
import math
import numpy as np
def leaky_relu(z, alpha=0.01):
    """
    Leaky ReLU activation.
    a = z if z > 0, else alpha * z
    """
    a = np.where(z > 0, z, alpha * z)
    cache = (z, alpha)
    return (a, cache)

def leaky_relu_backward(dA, cache):
    """
    Gradient of loss w.r.t. z through Leaky ReLU.
    """
    z, alpha = cache
    dz_mask = np.where(z > 0, 1, alpha)
    dZ = dA * dz_mask
    return dZ

def softmax(z):
    """Numerically stable row-wise softmax."""
    rowMax = np.max(z, axis=1, keepdims=True)
    zShifted = z - rowMax
    expZ = np.exp(zShifted)
    sumExpZ = np.sum(expZ, axis=1, keepdims=True)
    a = expZ / sumExpZ
    return a

def init_params(layer_dims):
    """
    He initialisation: W ~ N(0,1) * sqrt(2 / n_in),  b = zeros.

    Parameters
    ----------
    layer_dims : list[int]  e.g. [23, 128, 64, 4]

    Returns
    -------
    params : dict  {W1, b1, W2, b2, …}
             W[l] shape (layer_dims[l-1], layer_dims[l])
             b[l] shape (1, layer_dims[l])
    """
    if len(layer_dims)<2 or any(not isinstance(n,int) or n<=0 for n in layer_dims): raise ValueError("Invalid layer dimensions")
    params = {}
    L = len(layer_dims) - 1
    for l in range(1, L + 1):
        nIn = layer_dims[l - 1]
        nOut = layer_dims[l]
        heScale = np.sqrt(2 / nIn)
        params['W' + str(l)] = np.random.randn(nIn, nOut) * heScale
        params['b' + str(l)] = np.zeros((1, nOut))
    return params

def forward(X, params, training=True):
    """
    Full forward pass. Hidden layers -> ReLU. Output -> softmax.

    Returns
    -------
    y_hat  : (N, C)
    caches : list of (A_prev, W, b, relu_cache_or_Z) per layer
    """
    if X.ndim!=2 or not len(X) or not np.isfinite(X).all(): raise ValueError("Invalid feature matrix")
    caches = []
    A = X
    L = len(params) // 2
    for l in range(1, L):
        aPrev = A
        W = params['W' + str(l)]
        b = params['b' + str(l)]
        Z = np.dot(aPrev, W) + b
        A, reluCache = leaky_relu(Z)
        keep_prob = 0.8
        if training:
            mask = (np.random.rand(*A.shape) < keep_prob) / keep_prob
            A = A * mask
        else:
            mask = np.ones_like(A)
        caches.append((aPrev, W, b, reluCache, mask))
    aPrev = A
    W = params['W' + str(L)]
    b = params['b' + str(L)]
    Z = np.dot(aPrev, W) + b
    y_hat = softmax(Z)
    caches.append((aPrev, W, b, Z))
    return (y_hat, caches)

def cross_entropy_loss(y_hat, y):
    """Categorical cross-entropy averaged over the batch."""
    N = y_hat.shape[0]
    correctProbs = y_hat[range(N), y]
    correctProbs = np.clip(correctProbs, 1e-15, 1.0)
    loss = -np.sum(np.log(correctProbs)) / N
    return loss

def backward(y_hat, y, params, caches):
    """
    Full backprop via the chain rule.

    Output layer gradient (softmax + cross-entropy combined):
        dZ = y_hat.copy(); dZ[range(N), y] -= 1; dZ /= N

    Per layer (output → hidden, reversed):
        dW = A_prev.T @ dZ
        db = dZ.sum(axis=0, keepdims=True)
        dA_prev = dZ @ W.T
        dZ = relu_backward(dA_prev, relu_cache)   ← hidden layers only
    """
    grads = {}
    N = y_hat.shape[0]
    L = len(caches)
    dZ = y_hat.copy()
    dZ[range(N), y] -= 1
    dZ /= N
    aPrev, W, b, _ = caches[L - 1]
    grads['dW' + str(L)] = aPrev.T @ dZ
    grads['db' + str(L)] = dZ.sum(axis=0, keepdims=True)
    dAPrev = dZ @ W.T
    for l in reversed(range(1, L)):
        aPrev, W, b, reluCache, mask = caches[l - 1]
        dAPrev = dAPrev * mask
        dZ = leaky_relu_backward(dAPrev, reluCache)
        grads['dW' + str(l)] = aPrev.T @ dZ
        grads['db' + str(l)] = dZ.sum(axis=0, keepdims=True)
        dAPrev = dZ @ W.T
    return grads

def update_params(params, grads, lr):
    """θ ← θ − lr · ∇θ"""
    L = len(params) // 2
    for l in range(1, L + 1):
        params['W' + str(l)] -= lr * grads['dW' + str(l)]
        params['b' + str(l)] -= lr * grads['db' + str(l)]
    return params

def train(X, y, layer_dims, lr=0.01, epochs=200, batch_size=64, l2_lambda=0.0001, verbose=True):
    """
    Mini-batch SGD with L2 regularisation.

    Per mini-batch:
      1. forward()
      2. loss = cross_entropy + 0.5 * l2_lambda * sum||W||^2
      3. backward()
      4. add L2 gradient: dW += l2_lambda * W
      5. update_params()
    """
    if epochs<1 or batch_size<1 or not math.isfinite(lr) or lr<=0 or not math.isfinite(l2_lambda) or l2_lambda<0: raise ValueError("Invalid training settings")
    if len(X)!=len(y) or len(X)==0 or y.dtype.kind not in "iu" or y.min()<0 or y.max()>=layer_dims[-1]: raise ValueError("Invalid target labels")
    params = init_params(layer_dims)
    N = X.shape[0]
    history = {'loss': [], 'acc': []}
    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(N)
        X_s, y_s = (X[idx], y[idx])
        epoch_loss, n_batches = (0.0, 0)
        for start in range(0, N, batch_size):
            Xb = X_s[start:start + batch_size]
            yb = y_s[start:start + batch_size]
            yHat, caches = forward(Xb, params, training=True)
            batchLoss = cross_entropy_loss(yHat, yb)
            l2Loss = 0
            for l in range(1, len(params) // 2 + 1):
                l2Loss += np.sum(params['W' + str(l)] ** 2)
            batchLoss += 0.5 * l2_lambda * l2Loss
            grads = backward(yHat, yb, params, caches)
            for l in range(1, len(params) // 2 + 1):
                grads['dW' + str(l)] += l2_lambda * params['W' + str(l)]
            current_lr = lr * 0.98 ** (epoch // 50)
            params = update_params(params, grads, current_lr)
            epoch_loss += batchLoss
            n_batches += 1
        y_hat_full, _ = forward(X, params, training=False)
        acc = np.mean(np.argmax(y_hat_full, axis=1) == y)
        history['loss'].append(epoch_loss / max(1, n_batches))
        history['acc'].append(acc)
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch:>4}/{epochs}  loss={history['loss'][-1]:.4f}  acc={acc:.4f}")
    return (params, history)

def predict(X, params):
    y_hat, _ = forward(X, params, training=False)
    return np.argmax(y_hat, axis=1)
