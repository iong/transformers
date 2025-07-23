

from absl import flags

import dataclasses
import functools
import sentencepiece as spm
import tensorflow as tf
import tensorflow_datasets as tfds

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

class Preprocess:
    def __init__(self, input_key: str, target_key: str, input_length: int, target_length: int, tokenizer_path: str):
        self._input_key = input_key
        self._target_key = target_key
        self._input_length = input_length
        self._target_length = target_length
        self._tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    
    def tokenize(self, example):
        return {
            'inputs': self._tokenizer.encode_tf(example[self._input_key]),
            'targets': self._tokenizer.encode_tf(example[self._target_key]),
            'inputs_pretokenized': example[self._input_key],
            'targets_pretokenized': example[self._target_key]
        }

    def __call__(self, example):
        x = self.tokenize(example)
        return pad_and_trim(
            x, self._input_length, self._target_length,
            bos_id=self._tokenizer.bos_id())

def create_train_and_test(
        tfds_name: str =  "wmt19_translate/de-en",
        input_key: str = 'de',
        target_key: str = 'en',
        input_length: int = 128,
        target_length: int = 128,
        batch_size: int = 64
):
    builder = tfds.builder(tfds_name)
    if not builder.is_prepared():
         builder.download_and_prepare()
    
    preprocess = Preprocess(
        input_key=input_key,
        target_key=target_key,
        input_length=input_length,
        target_length=target_length,
        tokenizer_path=_TOKENIZER.value
    )

    train_ds = (
        builder
        .as_dataset(split='train', shuffle_files=True)
        .shuffle(1024)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        builder
        .as_dataset(split='test')
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds