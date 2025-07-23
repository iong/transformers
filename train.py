#!/usr/bin/env python3
from absl import flags
from absl import app
from flax import nnx
import jax
import jax.numpy as jnp
import sentencepiece as spm

import functools
import tensorflow as tf
import tensorflow_datasets as tfds

import transformer

_TOKENIZER = flags.DEFINE_string('tokenizer', '',
                                 'Path to the SentencePiece tokenizer model.')

def pad_and_trim_sequence(tokens, bos_id, length):
    tokens = tf.concat([[bos_id], tokens], axis=0)[:length]
    paddings = tf.pad(length - tf.shape(tokens), [[1, 0]], constant_values=0)
    tokens  = tf.pad(tokens, paddings[tf.newaxis, :], constant_values=0)
    return tokens

def pad_and_trim(example, input_length, target_length, bos_id=None):
    example['inputs'] = pad_and_trim_sequence(
        example['inputs'], bos_id, input_length)
    example['targets'] = pad_and_trim_sequence(
        example['targets'], bos_id, target_length)
    return example

def tokenize(example, input_key='inputs', target_key='targets', tokenizer=None):
    return {
        'inputs': tokenizer.encode_tf(example[input_key]),
        'targets': tokenizer.encode_tf(example[target_key]),
        'inputs_pretokenized': example[input_key],
        'targets_pretokenized': example[target_key]
    }

def main() -> None:
     builder = tfds.builder("wmt19_translate/de-en")
     if not builder.is_prepared():
         builder.download_and_prepare()
    
    train_ds = (
        builder
        .as_dataset(split='train', shuffle_files=True)
        .shuffle(1024)
        )
    rngs = nnx.Rngs(0)
    encdec = transformer.EncoderDecoder(
        transformer.TransformerConfig(
            vocab_size=10000
        ),
        rngs=rngs
    )

    key, subk1, subk2 = jax.random.split(rngs(), 3)
    print(encdec(jax.random.randint(subk1, (2, 10), 0, 10000),
        jax.random.randint(subk1, (2, 10), 0, 10000)))

if __name__ == '__main__':
    app.run(main)