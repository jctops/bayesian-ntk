from neural_tangents.stax import *
from neural_tangents.stax import _supports_masking

def _randn(stddev=1e-2):
  """`jax.experimental.stax.randn` for implicitly-typed results."""
  def init(rng, shape):
    return stddev * random.normal(rng, shape)
  return init

def truncated_randn(stddev=1e-2):
  """`jax.experimental.stax.randn` for implicitly-typed results."""
  def init(rng, shape):
    return stddev * random.truncated_normal(rng, 0, 1000, shape)
  return init

@layer
@_supports_masking(remask_kernel=True)
def Dense(out_dim,
          W_std=1.,
          b_std=0.,
          W_init=_randn(1.0),
          b_init=_randn(1.0),
          parameterization='ntk',
          batch_axis=0,
          channel_axis=-1):
  r"""Layer constructor function for a dense (fully-connected) layer.

  Based on `jax.experimental.stax.Dense`. Has a similar API, apart from:

  `W_init` and `b_init` only change the behavior of the finite width network,
  and are not used by `kernel_fn`. In most cases, `W_std` and `b_std` should
  be used instead.

  Args:
    :parameterization: Either 'ntk' or 'standard'.

      Under ntk parameterization (https://arxiv.org/abs/1806.07572, page 3),
      weights and biases are initialized as :math:`W_{ij} \sim N(0,1)`,
      :math:`b_i \sim \mathcal{N}(0,1)`, and the finite width layer equation is
      :math:`z_i = \sigma_W / \sqrt{N} \sum_j W_{ij} x_j + \sigma_b b_i`.

      Under standard parameterization (https://arxiv.org/abs/2001.07301),
      weights and biases are initialized as :math:`W_{ij} \sim \matchal{N}(0,
      W_std^2/N)`,
      :math:`b_i \sim \mathcal{N}(0,\sigma_b^2)`, and the finite width layer
      equation is
      :math:`z_i = \sum_j W_ij x_j + b_i`.

    :batch_axis: integer, batch axis. Defaults to `0`, the leading axis.

    :channel_axis: integer, channel axis. Defaults to `-1`, the trailing axis.
      For `kernel_fn`, channel size is considered to be infinite.
      
      
  Returns:
    `(init_fn, apply_fn, kernel_fn)`.

  """
  # TODO: after experimentation, evaluate whether to change default
  # parameterization from "ntk" to "standard"

  parameterization = parameterization.lower()

  def ntk_init_fn(rng, input_shape):
    _channel_axis = channel_axis % len(input_shape)
    output_shape = (input_shape[:_channel_axis] + (out_dim,)
                    + input_shape[_channel_axis + 1:])
    k1, k2 = random.split(rng)
    W = W_init(k1, (input_shape[_channel_axis], out_dim))
    b = b_init(k2, (out_dim,))
    return output_shape, (W, b)

  def standard_init_fn(rng, input_shape):
    output_shape, (W, b) = ntk_init_fn(rng, input_shape)
    return output_shape, (W * W_std / np.sqrt(input_shape[channel_axis]),
                          b * b_std)

  if parameterization == 'ntk':
    init_fn = ntk_init_fn
  elif parameterization == 'standard':
    init_fn = standard_init_fn
  else:
    raise ValueError('Parameterization not supported: %s' % parameterization)

  def apply_fn(params, inputs, **kwargs):
    W, b = params
    prod = np.moveaxis(np.tensordot(W, inputs, (0, channel_axis)),
                       0, channel_axis)

    if parameterization == 'ntk':
      norm = W_std / np.sqrt(inputs.shape[channel_axis])
      outputs = norm * prod + b_std * b
    elif parameterization == 'standard':
      outputs = prod  + b
    else:
      raise ValueError('Parameterization not supported: %s' % parameterization)

    return outputs

  @_requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def kernel_fn(kernels):
    """Compute the transformed kernels after a dense layer."""
    cov1, nngp, cov2, ntk = \
      kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

    def fc(x):
      return _affine(x, W_std, b_std)

    if parameterization == 'ntk':
      cov1, nngp, cov2 = map(fc, (cov1, nngp, cov2))
      if ntk is not None:
        ntk = nngp + W_std**2 * ntk
    elif parameterization == 'standard':
      input_width = kernels.shape1[channel_axis]
      if ntk is not None:
        ntk = input_width * nngp + 1. + W_std**2 * ntk
      cov1, nngp, cov2 = map(fc, (cov1, nngp, cov2))

    return kernels.replace(cov1=cov1,
                           nngp=nngp,
                           cov2=cov2,
                           ntk=ntk,
                           is_gaussian=True,
                           is_input=False)

  def mask_fn(mask, input_shape):
    return np.all(mask, axis=channel_axis, keepdims=True)

  return init_fn, apply_fn, kernel_fn, mask_fn