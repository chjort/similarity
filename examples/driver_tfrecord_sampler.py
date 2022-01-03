import os
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import tensorflow as tf
from absl import logging

from tests.samplers.test_tfrecord_samplers import deserialization_fn


def TFRecordDatasetSampler(
    shard_path: str,
    deserialization_fn: Callable,
    example_per_class: int = 2,
    batch_size: int = 32,
    shards_per_cycle: int = None,
    compression: Optional[str] = None,
    parallelism: int = tf.data.AUTOTUNE,
    async_cycle: bool = False,
    prefetch_size: Optional[int] = None,
    shard_suffix: str = "*.tfrec",
    num_repeat: int = -1,
) -> tf.data.Dataset:
    shards_list = [
        i.decode()
        for i in tf.io.matching_files(os.path.join(shard_path, shard_suffix))
        .numpy()
        .tolist()
    ]
    logging.debug(f"found {shards_list}")
    total_shards = len(shards_list)
    logging.info(f"found {total_shards} shards")

    if not prefetch_size:
        prefetch_size = 10

    # how many shard to iterate over in parallels.
    cycle_length = shards_per_cycle if shards_per_cycle else total_shards
    # how many threads to use when fetching inputs from the cycle shards
    num_parallel_calls = cycle_length if async_cycle else 1

    with tf.device("/cpu:0"):
        # shuffle the shard order
        ds = tf.data.Dataset.from_tensor_slices(shards_list)

        # shuffle shard order
        ds = ds.shuffle(total_shards)

        # repeat shard order
        ds = ds.repeat(count=num_repeat)

        # This is the tricky part, we are using the interleave function to
        # do the sampling as requested by the user. This is not the
        # standard use of the function or an obvious way to do it but
        # its by far the faster and more compatible way to do so
        # we are favoring for once those factors over readability
        # deterministic=False is not an error, it is what allows us to
        # create random batch
        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(
                x,
                compression_type=compression,
            )
            .shuffle(100)
            .take(example_per_class),  # noqa
            cycle_length=cycle_length,
            block_length=example_per_class,
            num_parallel_calls=num_parallel_calls,
            deterministic=False,
        )
        ds = ds.map(deserialization_fn, num_parallel_calls=parallelism)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(prefetch_size)
        print("Parallel calls:", num_parallel_calls)
        print()
        return ds


tmpdir = Path("tfrecord_files")

# create_data(tmpdir)

sampler = TFRecordDatasetSampler(
    tmpdir,
    deserialization_fn=deserialization_fn,
    shards_per_cycle=5,
    example_per_class=6,
    async_cycle=True,
    batch_size=30,
    num_repeat=-1,
)

for i, (x, y) in sampler.enumerate():
    counts_x = Counter(x.numpy())
    if not all(c == 6 for c in counts_x.values()):
        print(i.numpy())
        print(*counts_x.items(), sep="\n")
        break
