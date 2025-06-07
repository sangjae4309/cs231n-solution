from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    n_sample = x.shape[0]
    x_resh = x.reshape(n_sample, -1)
    out = x_resh.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    dx = w.dot(dout.T).T.reshape(x.shape)
    x_resh = x.reshape(x.shape[0], -1)
    dw = dout.T.dot(x_resh).T
    db = dout.sum(axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    idx = np.where(x < 0)
    dout[idx] = 0
    dx = dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    N = x.shape[0]
    
    score = np.exp(x-x.max(axis=1, keepdims=True)) # for stable exponents
    score += 1e-12 # to avoid log(0)
    prb_score = score/score.sum(axis=1, keepdims=True)
    loss = -np.log(prb_score[range(N), y]).sum()/N

    prb_score[range(N), y] -= 1
    dx = prb_score/N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_norm = (x-mu)/np.sqrt(var + eps)
        
        out = gamma*x_norm + beta
        running_mean = (1-momentum)*running_mean + momentum*mu
        running_var = (1-momentum)*running_var + momentum*var

        cache = x, x_norm, gamma, var, mu, eps

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_norm = (x-running_mean)/np.sqrt(running_var + eps)

        out = gamma*x_norm + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x, x_norm, gamma, var, mu, eps = cache
    N = dout.shape[0]
    std = np.sqrt(var + eps)

    # easist one
    dgamma = np.sum(x_norm*dout, axis=0)
    dbeta = dout.sum(axis=0)

    # (core-part) dx

    dL_dxnorm = dout*gamma
    
    dL_dstd = np.sum(dL_dxnorm * -(x-mu), axis=0)/(std**2)
    dL_dvar = dL_dstd*0.5/std
    dL_dx_1 = dL_dxnorm/std + dL_dvar * 2*(x-mu) / N
    dL_dmu = -np.sum(dL_dx_1, axis=0)
    dL_dx_2 = dL_dmu / N

    dx = dL_dx_1 + dL_dx_2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    _, x_norm, gamma, var, _, eps = cache
    std = np.sqrt(var + eps)
    N = dout.shape[0]

    # easist one
    dgamma = np.sum(x_norm*dout, axis=0)
    dbeta = dout.sum(axis=0)

    #
    dL_dxnorm = dout*gamma
    dL_dx_1 = dL_dxnorm/std + np.sum(-dL_dxnorm*x_norm, axis=0)*x_norm/(N*std)
    dL_dx_2 = -np.sum(dL_dx_1, axis=0) / N
    
    dx = dL_dx_1 + dL_dx_2

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    mu = np.mean(x, axis=1)
    var = np.var(x, axis=1)
    x_norm = (x-mu[:, np.newaxis])/np.sqrt(var[:, np.newaxis] + eps)
        
    out = gamma*x_norm + beta

    cache = x, x_norm, gamma, var, mu, eps
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    _, x_norm, gamma, var, _, eps = cache
    std = np.sqrt(var + eps)[:, np.newaxis]
    D = dout.shape[1]

    # easist one
    dgamma = np.sum(x_norm*dout, axis=0)
    dbeta = dout.sum(axis=0)

    dL_dxnorm = dout*gamma
    dL_dx_1 = dL_dxnorm/std + np.sum(-dL_dxnorm*x_norm, axis=1)[:,np.newaxis]*x_norm/(D*std)
    dL_dx_2 = -np.sum(dL_dx_1, axis=1)[:,np.newaxis] / D
    dx = dL_dx_1 + dL_dx_2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.rand(*x.shape) < p
        out = x*mask / p
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask / dropout_param["p"]
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pad     = conv_param['pad']
    stride  = conv_param['stride']
    pad_dim = [(0,0), (0,0)] + [(pad, pad)] * 2

    x_pad = np.pad(x, pad_dim, mode='constant', constant_values=0)
    
    N, C, H, W = x_pad.shape
    F, CC, HH, WW = w.shape
    assert(C==CC)

    Hout = (H-HH)//stride+1
    Wout = (W-WW)//stride+1
    
    h_idx_s = np.arange(start=0, stop=H-HH+1, step=stride)
    h_idx_e = np.arange(start=HH, stop=H+1, step=stride)
    w_idx_s = np.arange(start=0, stop=W-WW+1, step=stride)
    w_idx_e = np.arange(start=WW, stop=W+1, step=stride)

    n_stride_w = len(w_idx_s)
    n_stride_h = len(h_idx_s)
    
    # x_pad im2col
    x_pad_im2col = np.zeros([N, Hout*Wout, HH*WW*C])
    for r in range(n_stride_h):
        for c in range(n_stride_w):
            x_pad_im2col[:,r*n_stride_w+c,:] = \
              x_pad[:,:,h_idx_s[r]:h_idx_e[r], w_idx_s[c]:w_idx_e[c]].reshape([N,HH*WW*C])
            
    # filter im2col
    w_im2col = w.reshape([F,HH*WW*C])
    conv_out = x_pad_im2col.dot(w_im2col.T) #[N, Hout*Wout, F]
    conv_out_swap = np.swapaxes(conv_out, 1, 2)
    
    out = conv_out_swap + b.reshape([1,F,1])
    out = out.reshape([N, F, Hout, Wout])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache

    pad     = conv_param['pad']
    stride  = conv_param['stride']
    pad_dim = [(0,0), (0,0)] + [(pad, pad)] * 2
    x_pad = np.pad(x, pad_dim, mode='constant', constant_values=0)

    N, C, H, W = x.shape
    _, _, H_pad, W_pad = x_pad.shape
    F, _, HH, WW = w.shape # [F, C, HH, WW]
    _, _, Hout, Wout = dout.shape # [N, F, Hout, Wout)]
  
    #1.calc db
    db = np.sum(dout, axis=(0,2,3))

    #2.calc dw
    Hout_pad = Hout+(stride-1)*(Hout-1)
    Wout_pad = Wout+(stride-1)*(Wout-1)
    dout_pad = np.zeros([N, F, Hout_pad, Wout_pad])
    dout_pad[:,:,::stride,::stride] = dout
    
    h_idx_s = np.arange(start=0, stop=H_pad-Hout_pad+1, step=1)
    h_idx_e = np.arange(start=Hout_pad, stop=H_pad+1, step=1)
    w_idx_s = np.arange(start=0, stop=W_pad-Wout_pad+1, step=1)
    w_idx_e = np.arange(start=Wout_pad, stop=W_pad+1, step=1)

    n_stride_w = len(w_idx_s)
    n_stride_h = len(h_idx_s)

    dw_x_im2col = np.zeros([C, HH*WW, Hout_pad*Wout_pad*N])
    for r in range(n_stride_h):
        for c in range(n_stride_w):
            dw_x_pad = x_pad[:,:,h_idx_s[r]:h_idx_e[r], w_idx_s[c]:w_idx_e[c]]  # [N, C, Hout, Wout]
            dw_x_pad = np.swapaxes(dw_x_pad, 0, 1) # [C, N, Hout_pad, Wout_pad]
            dw_x_pad = dw_x_pad.reshape([C,N*Hout_pad*Wout_pad])      
            dw_x_im2col[:,r*n_stride_w+c,:] = dw_x_pad
    dout_pad_im2col = np.swapaxes(dout_pad, 0, 1) # [F, N, Hout_pad, Wout_pad]
    dout_pad_im2col = dout_pad_im2col.reshape([F, N*Hout_pad*Wout_pad])
    dw = dw_x_im2col.dot(dout_pad_im2col.T) # [C, HH*WW, F]
    dw = np.swapaxes(dw, 1, 2) # [C, F, HH*WW]
    dw = np.swapaxes(dw, 0, 1) # [F, C, HH*WW]
    dw = dw.reshape([F, C, HH, WW])

    #3.calc dx
    dout_pad = np.pad(dout_pad, [(0,0), (0,0)] + [(1, 1)] * 2, mode='constant', constant_values=0)
    _, _, Hout_pad, Wout_pad = dout_pad.shape
    #with open("dout_pad.txt", "w") as f:
    #  with np.printoptions(threshold=np.inf, linewidth=np.inf, suppress=False):
    #    f.write(np.array2string(dout_pad[0]))
    dout_pad_im2col = np.zeros([N, H*W, HH*WW*F])

    h_idx_s = np.arange(start=0, stop=Hout_pad-HH+1, step=1)
    h_idx_e = np.arange(start=HH, stop=Hout_pad+1, step=1)
    w_idx_s = np.arange(start=0, stop=Wout_pad-WW+1, step=1)
    w_idx_e = np.arange(start=WW, stop=Wout_pad+1, step=1)

    n_stride_w = len(w_idx_s)
    n_stride_h = len(h_idx_s)

    for r in range(n_stride_h):
        for c in range(n_stride_w):
            dout_pad_pre = dout_pad[:, :,h_idx_s[r]:h_idx_e[r], w_idx_s[c]:w_idx_e[c]] # [N, F, HH, WW]
            dout_pad_pre = dout_pad_pre.reshape([N, HH*WW*F])
            dout_pad_im2col[:,r*n_stride_w+c,:] = dout_pad_pre
    #with open("dout_pad_im2col.txt", "w") as f:
    #  with np.printoptions(threshold=np.inf, linewidth=np.inf, suppress=False):
    #    f.write(np.array2string(dout_pad_im2col[0]))

    w_im2col_dx = np.swapaxes(w, 0, 1) # [C, F, HH, WW]
    w_im2col_dx = w_im2col_dx.reshape([C,F,HH*WW])
    w_im2col_dx = w_im2col_dx[:, :, ::-1] # reverse its order
    w_im2col_dx = w_im2col_dx.reshape([C,HH*WW*F])
    dx = dout_pad_im2col.dot(w_im2col_dx.T) # [N, H*W, HH*WW*F]*[HH*WW*F,C] --> [N, H*W, C]
    dx = np.swapaxes(dx, 1 ,2).reshape([N, C, H, W])
    #with open("dx.txt", "w") as f:
    #  with np.printoptions(threshold=np.inf, linewidth=np.inf, suppress=False):
    #    f.write(np.array2string(dx[0]))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    H_pool  = pool_param['pool_height']
    W_pool  = pool_param['pool_width']
    stride  = pool_param['stride']
    
    N, C, H, W = x.shape

    Hout = 1+int((H-H_pool)/stride)
    Wout = 1+int((W-W_pool)/stride)
    
    h_idx_s = np.arange(start=0, stop=H-H_pool+1, step=stride)
    h_idx_e = np.arange(start=H_pool, stop=H+1, step=stride)
    w_idx_s = np.arange(start=0, stop=W-W_pool+1, step=stride)
    w_idx_e = np.arange(start=W_pool, stop=W+1, step=stride)

    n_stride_w = len(w_idx_s)
    n_stride_h = len(h_idx_s)
    
    # x_pad im2col
    x_im2col = np.zeros([N, C, Hout*Wout, H_pool*W_pool])
    for r in range(n_stride_h):
        for c in range(n_stride_w):
            x_im2col[: , :, r*n_stride_w+c,:] = \
              x[:,:,h_idx_s[r]:h_idx_e[r], w_idx_s[c]:w_idx_e[c]].reshape(N,C,H_pool*W_pool)

    x_im2col_max_pool = x_im2col.max(axis=3) # max-pooling operation

    out = x_im2col_max_pool.reshape(N, C, Hout, Wout)

    # For back-propagation
    x_im2col_max_pool_idx = x_im2col.argmax(axis=3) # [N, C, Hout*Wout]
    
    max_map = np.zeros([N, C, H, W])
    for r in range(n_stride_h):
        for c in range(n_stride_w):
            tmp_map = np.zeros([N,C,H_pool*W_pool])
            idx = x_im2col_max_pool_idx[:,:, r*n_stride_w+c].flatten()
            tmp_map[:,:,idx] = 1
            max_map[:,:,h_idx_s[r]:h_idx_e[r], w_idx_s[c]:w_idx_e[c]] = tmp_map.reshape([N, C, H_pool, W_pool])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    
    H_pool  = pool_param['pool_height']
    W_pool  = pool_param['pool_width']
    stride  = pool_param['stride']
    
    N, C, H, W = x.shape
    _, _, Hout, Wout = dout.shape
    dx = np.zeros(x.shape)
    
    h_idx_s = np.arange(start=0, stop=H-H_pool+1, step=stride)
    h_idx_e = np.arange(start=H_pool, stop=H+1, step=stride)
    w_idx_s = np.arange(start=0, stop=W-W_pool+1, step=stride)
    w_idx_e = np.arange(start=W_pool, stop=W+1, step=stride)

    n_stride_w = len(w_idx_s)
    n_stride_h = len(h_idx_s)
    
    # x_pad im2col
    
    for r in range(n_stride_h):
        for c in range(n_stride_w):
              dx_im2col = np.zeros([N, C, H_pool*W_pool])
              x_pool_flat = x[:,:,h_idx_s[r]:h_idx_e[r], w_idx_s[c]:w_idx_e[c]].reshape(N, C, H_pool*W_pool)
              dout_pool_flat = dout[:,:,r,c].flatten()
              
              x_pool_flat_max_idx = x_pool_flat.argmax(axis=2).flatten() #[N, C,]
              i0, i1 = np.divmod(np.arange(N*C), C)
              dx_im2col[i0,i1,x_pool_flat_max_idx] = dout_pool_flat
              dx[:,:,h_idx_s[r]:h_idx_e[r], w_idx_s[c]:w_idx_e[c]] = dx_im2col.reshape([N, C, H_pool, W_pool])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    
    x_flat = x.transpose(0, 2, 3, 1).reshape([N*H*W, C])
    out_p, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)

    out = out_p.reshape([N, H, W, C]).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
  
    dout_flat = dout.transpose(0, 2, 3, 1).reshape([N*H*W, C])
    dx_p, dgamma, dbeta = batchnorm_backward_alt(dout_flat, cache)
    
    dx = dx_p.reshape([N , H, W, C]).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    N, C, H, W = x.shape
    ln_param = {"shape":(W, H, C, N), **gn_param}

    x_flat = x.reshape(N*G, -1)
    gamma_flat = np.tile(gamma, (N, 1, H, W)).reshape(N*G, -1)  # ([N*G, C/G*H*W])
    beta_flat = np.tile(beta, (N, 1, H, W)).reshape(N*G, -1) 

    out_flat, cache_flat = layernorm_forward(x_flat, gamma_flat, beta_flat, ln_param)

    out = out_flat.reshape([N, C, H, W])

    cache = (cache_flat, G)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    N, C, H, W = dout.shape
    cache_flat, G = cache
    _, x_norm_flat, _, _, _, _ = cache_flat

    dout_flat = dout.reshape(N*G, -1)

    dx_flat, _, _ = layernorm_backward(dout_flat, cache_flat)
    dx = dx_flat.reshape([N, C, H, W])

    dl_flat = (dout_flat*x_norm_flat).reshape([N, C, H, W])
    
    dgamma = dl_flat.sum(axis=(0,2,3), keepdims=True)
    dbeta = dout.sum(axis=(0,2,3), keepdims=True)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
