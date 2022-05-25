import tensorflow as tf
import copy


tf.logging.set_verbosity(tf.logging.DEBUG)
max_seq_length = 128
max_predictions_per_seq = 20
batch_size = 16


def map_func(example):
    feature_map = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }
    parsed_example = tf.parse_single_example(example, features=feature_map)
    return (parsed_example, [])


class Bert:
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_onehot_embedding=False,
                 scope=None
                 ):
        config = copy.deepcopy(config)
        # if not is_training:
            # config.hidden


dataset = tf.data.TFRecordDataset(["tf_examples.tfrecord"])
dataset = dataset.map(map_func).shuffle(10).batch(batch_size).repeat(2)
dataset = dataset.make_one_shot_iterator().get_next()


if __name__ == '__main__':
    sess = tf.Session()
    while True:
        try:
            batch_data, _ = sess.run(dataset)
            print(batch_data.keys())
            print(batch_data["input_ids"].shape)
            print(batch_data["input_mask"].shape)
            print(batch_data["masked_lm_ids"].shape)
            print(batch_data["masked_lm_positions"].shape)
            print(batch_data["masked_lm_weights"].shape)
            print(batch_data["next_sentence_labels"].shape)
            print(batch_data["segment_ids"].shape)
        except tf.errors.OutOfRangeError:
            break