
import dataclasses
from flax import nnx
import jax
import jax.numpy as jnp

_MAX_WAVELENGTH = 10000

@dataclasses.dataclass
class TransformerConfig:
    d: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dff: int = 2048
    dropout_rate: float = 0.1
    rope_max_wavelength: int = _MAX_WAVELENGTH
    rope_scale_factor: float = 1.0
    vocab_size: int = 0

    def k(self) -> int:
        """Calculate the key dimension."""
        return self.d // self.num_heads

def Rope(
    inputs: jax.Array,  # [B, L, H, D]
    positions: jax.Array,  # [B, L]
    max_wavelength: int,
    scale_factor: float,
) -> jax.Array: # [B, L, H, D]
  """Applies RoPE."""
  head_dim = inputs.shape[-1]
  fraction = 2.0 * jnp.arange(0, head_dim // 2) / head_dim
  wavelength = max_wavelength**fraction

  phase = (
      positions[..., jnp.newaxis] / wavelength[jnp.newaxis, jnp.newaxis, :]
  )
  phase = phase[..., jnp.newaxis, :]
  if scale_factor < 1.0:
    raise ValueError(f'scale_factor must be >= 1.0, got {scale_factor}')
  phase /= scale_factor

  sin = jnp.sin(phase)
  cos = jnp.cos(phase)

  # https://arxiv.org/pdf/2104.09864 uses odd / even inputs. There's nothing special
  # about each dimension, and we have permutation invariance.
  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = first_half * sin + second_half * cos
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)

class MultiHeadAttention(nnx.Module):
    def __init__(self, d: int, h: int, k: int, v: int, rope_max_wavelength: int, rope_scale_factor: float, rngs: nnx.Rngs):
        self.P_q = nnx.Param(jax.random.normal(rngs(), (h, d, k)))
        self.P_k = nnx.Param(jax.random.normal(rngs(), (h, d, k)))
        self.P_v = nnx.Param(jax.random.normal(rngs(), (h, d, v)))
        self.P_o = nnx.Param(jax.random.normal(rngs(), (h, d, v)))
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scale_factor = rope_scale_factor
    
    def __call__(self, query: jax.Array, q_positions: jax.Array,
                 key: jax.Array, k_positions: jax.Array,
                 value: jax.Array, mask: jax.Array = None) -> jax.Array:

        q = jnp.einsum('bnd,hdk->bnhk', query, self.P_q)
        k = jnp.einsum('bmd,hdk->bmhk', key, self.P_k)
        v = jnp.einsum('bmd,hdv->bmhv', value, self.P_v)

        q = RoPE(q, q_positions, max_wavelength=self.rope_max_wavelength, scale_factor=self.rope_scale_factor)
        k = RoPE(k, k_positions, max_wavelength=self.rope_max_wavelength, scale_factor=self.rope_scale_factor)

        logits = jnp.einsum('bnhk,bmhk->bhnm', q, k)
        if mask is not None:
            logits = jnp.where(mask, logits, -jnp.inf)
        weights = nnx.softmax(logits, axis=-1)

        v = jnp.einsum('bhnm,bmhv->bnhv', weights, v)

        return jnp.einsum('bnhv,hdv->bnd', v, self.P_o)

class EncoderLayer(nnx.Module):
    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.mha = MultiHeadAttention(config.d, config.num_heads, config.k(), config.k(), config.rope_max_wavelength, config.rope_scale_factor, rngs)
        self.linear1 = nnx.Linear(config.d, config.dff, rngs)
        self.linear2 = nnx.Linear(config.dff, config.d, rngs)
        self.layernorm1 = nnx.LayerNorm(config.d, rngs=rngs)
        self.layernorm2 = nnx.LayerNorm(config.d, rngs=rngs)

        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
    
    def __call__(self, x: jax.Array, positions: jax.Array) -> jax.Array:
        x = self.layernorm1(x + self.dropout1(self.mha(x, positions, x, positions, x)))
        x = self.linear2(nnx.relu(self.linear1(x)))
        x = self.layernorm2(x + self.dropout2(x))
        return x
    
class DecoderLayer(nnx.Module):
    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.mha1 = MultiHeadAttention(config.d, config.num_heads, config.k(), config.k(), config.rope_max_wavelength, config.rope_scale_factor, rngs)
        self.mha2 = MultiHeadAttention(config.d, config.num_heads, config.k(), config.k(), config.rope_max_wavelength, config.rope_scale_factor, rngs)
        self.linear1 = nnx.Linear(config.d, config.dff, rngs)
        self.linear2 = nnx.Linear(config.dff, config.d, rngs)
        self.layernorm1 = nnx.LayerNorm(config.d, rngs=rngs)
        self.layernorm2 = nnx.LayerNorm(config.d, rngs=rngs)
        self.layernorm3 = nnx.LayerNorm(config.d, rngs=rngs)

        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, positions: jax.Array,
                    encoder_output: jax.Array, encoder_positions: jax.Array) -> jax.Array:
        causal_mask = jnp.tril(jnp.ones((x.shape[1], x.shape[1]), dtype=jnp.int32))
        x = self.layernorm1(x + self.dropout1(self.mha1(x, positions, x, positions, x, causal_mask)))
        x = self.layernorm2(x + self.dropout2(self.mha2(x, positions, encoder_output, encoder_positions, encoder_output)))
        x = self.linear2(nnx.relu(self.linear1(x)))
        x = self.layernorm3(x + self.dropout3(x))
        return x

class Encoder(nnx.Module):
    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.layers = nnx.vmap(lambda key: EncoderLayer(config, nnx.Rngs(key)), in_axes=0, out_axes=0)(jax.random.split(rngs(), config.num_layers))

    def __call__(self, x: jax.Array, positions: jax.Array) -> jax.Array:
        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        def _apply_encoder_layer(encoder_layer, args):
            x, positions = args
            return (encoder_layer(x, positions), positions)
        x = self.dropout(x)
        x, _ = _apply_encoder_layer(self.layers, (x, positions))
        return x

class Decoder(nnx.Module):
    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.layers = nnx.vmap(lambda key: DecoderLayer(config, nnx.Rngs(key)), in_axes=0, out_axes=0)(jax.random.split(rngs(), config.num_layers))

    def __call__(self, x: jax.Array, x_positions: jax.Array,
                 encoder_output: jax.Array, encoder_positions: jax.Array) -> jax.Array:
        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        def _apply_decoder_layer(decoder_layer, args):
            x, x_positions, encoder_output, encoder_positions = args
            return (decoder_layer(x, x_positions, encoder_output, encoder_positions), x_positions, encoder_output, encoder_positions)
        x = self.dropout(x)
        x, _, _, _ = _apply_decoder_layer(self.layers, (x, x_positions, encoder_output, encoder_positions))
        return x

class Transformer(nnx.Module):
    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.embeddings = nnx.Embed(config.vocab_size, config.d, rngs)
        self.encoder = Encoder(config, rngs)
        self.decoder = Decoder(config, rngs)
        self.output_layer = nnx.Linear(config.d, config.vocab_size, rngs)

    def __call__(self, inputs: jax.Array,
                 targets: jax.Array) -> jax.Array:
        x = self.embeddings(inputs)
        x_positions = jnp.repeat(jnp.arange(inputs.shape[-1])[jnp.newaxis, :],
                                 inputs.shape[-2], axis=0)
        encoder_output = self.encoder(x, x_positions)

        y = self.embeddings(targets)
        y_positions = jnp.repeat(jnp.arange(targets.shape[-1])[jnp.newaxis, :],
                                 targets.shape[-2], axis=0)
        decoder_output = self.decoder(y, y_positions, encoder_output, x_positions)
        return self.output_layer(decoder_output)  # Final output logits