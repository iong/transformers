#!/usr/bin/env python3
from absl import flags
from absl import app
from flax import nnx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

import datasets
import transformer

_NUM_TRAINS_STEPS = flags.DEFINE_integer(
    'num_train_steps', 10000, 'Number of training steps to run.'
)
_NUM_EVAL_STEPS = flags.DEFINE_integer(
    'num_eval_steps', 100, 'Number of evaluation steps to run.'
)

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 1e-3, 'Initial learning rate for the optimizer.'
)

_EVAL_PERIOD = flags.DEFINE_integer(
    'eval_period', 5, 'Number of training steps between evaluations.'
)
_CHECKPOINT_PERIOD = flags.DEFINE_integer(
    'checkpoint_period', 5, 'Number of training steps between checkpoints.'
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir', '/tmp/checkpoints', 'Directory to save checkpoints.'
)

def numpy_to_jax(pytree):
    """Convert a PyTree of NumPy arrays to JAX arrays."""
    return jax.tree.map(lambda x: jnp.array(x), pytree)

def loss_fn(model: transformer.EncoderDecoder, batch):
    logits = model(batch['inputs'], batch['targets'][:, :-1])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['targets'][:, 1:]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model: transformer.EncoderDecoder, optimizer: nnx.Optimizer,
               metrics: nnx.MultiMetric, batch):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=0)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['targets'][:, 1:])
    optimizer.update(grads)


@nnx.jit
def eval_step(model: transformer.EncoderDecoder, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['targets'][:, 1:])


def main() -> None: 
    train_ds, test_ds = datasets.create_train_and_test()

    ckpt_dir = ocp.test_utils.erase_and_create_empty(_CHECKPOINT_DIR.value)
    checkpointer = ocp.StandardCheckpointer()

    rngs = nnx.Rngs(0)
    encdec = transformer.EncoderDecoder(
        transformer.TransformerConfig(
            vocab_size=10000
        ),
        rngs=rngs
    )
    optimizer = nnx.Optimizer(encdec, optax.adamw(_LEARNING_RATE.value))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss')
    )
    eval_metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss')
    )

    metrics_history = {
        'step': [],
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }

    for step, batch in enumerate(train_ds.take(_NUM_TRAINS_STEPS.value).as_numpy_iterator()):
        batch = numpy_to_jax(batch)
        train_step(encdec, optimizer, metrics, numpy_to_jax(batch))

        if step % _CHECKPOINT_PERIOD.value == 0:
            _, state = nnx.split(encdec)
            checkpointer.save(ckpt_dir / f"state_{step:06d}", state)

        if step > 0 and step % _EVAL_PERIOD.value == 0:
            metrics_history['step'].append(step)
            for metric_name, metric_value in metrics.compute().items():
                metrics_history[f'train_{metric_name}'].append(metric_value)
            metrics.reset()
   
            for eval_batch in test_ds.take(_NUM_EVAL_STEPS.value):
                eval_step(encdec, eval_metrics, numpy_to_jax(eval_batch))
            for metric_name, metric_value in eval_metrics.compute().items():
                metrics_history[f'test_{metric_name}'].append(metric_value)
            eval_metrics.reset()

            print(f'Step {step}, Train Metrics: {metrics}, Eval Metrics: {eval_metrics}')

if __name__ == '__main__':
    app.run(main)