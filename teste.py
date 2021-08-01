import tensorflow as tf
import json
import argparse
from models.dataset import Dataset
from models.model import Model


class RTVSRGAN(Model):
    def __init__(self, args):
        super().__init__(args)
        self._prediction_offset = self._scale_factor * 4

    def get_data(self):
        data_batch, initializer = self.dataset.get_data()

        lr_batch = tf.cast(data_batch['lr1'], tf.float32) / 255.0
        hr_batch = tf.cast(data_batch['hr'], tf.float32) / 255.0

        return [lr_batch, hr_batch], initializer

    def get_placeholder(self):
        input_ph = tf.placeholder(tf.float32, shape=[1, None, None, 1], name="x")

        return [input_ph]

    def load_model(self, data_batch):
        lr_batch = data_batch[0]

        with tf.variable_scope('espcn'):
            if not self._using_dataset:
                lr_batch = tf.pad(lr_batch, [[0, 0], [4, 4], [4, 4], [0, 0]], 'SYMMETRIC')
            net = tf.layers.conv2d(lr_batch, 64, 5, activation=tf.nn.tanh, padding='valid', name='conv1',
                                   kernel_initializer=tf.keras.initializers.he_normal())
            net = tf.layers.conv2d(net, 32, 3, activation=tf.nn.tanh, padding='valid', name='conv2',
                                   kernel_initializer=tf.keras.initializers.he_normal())
            net = tf.layers.conv2d(net, self._scale_factor ** 2, 3, activation=tf.nn.sigmoid, padding='valid',
                                   name='conv3', kernel_initializer=tf.keras.initializers.he_normal())
            predicted_batch = tf.depth_to_space(net, self._scale_factor, name='prediction')

        espcn_variables = tf.trainable_variables(scope='espcn')
        for variable in espcn_variables:
            if 'conv3' in variable.name:
                self.lr_multipliers[variable.name] = 0.1
            else:
                self.lr_multipliers[variable.name] = 1.0

        if self._using_dataset:
            tf.summary.image('Low_resolution', data_batch[0][:, 4:-4, 4:-4], max_outputs=self._save_num)
            tf.summary.image('High_resolution',
                             data_batch[1][:, self._prediction_offset:-self._prediction_offset,
                                           self._prediction_offset:-self._prediction_offset],
                             max_outputs=self._save_num)
            tf.summary.image('High_resolution_prediction', predicted_batch, max_outputs=self._save_num)

        return predicted_batch

    def get_loss(self, data_batch, predicted_batch):
        loss = tf.losses.mean_squared_error(
            data_batch[1][:,
                          self._prediction_offset:-self._prediction_offset,
                          self._prediction_offset:-self._prediction_offset],
            predicted_batch)

        tf.summary.scalar('MSE', loss)
        tf.summary.scalar('PSNR', tf.reduce_mean(tf.image.psnr(
                                                     data_batch[1][:,
                                                                   self._prediction_offset:-self._prediction_offset,
                                                                   self._prediction_offset:-self._prediction_offset],
                                                     predicted_batch,
                                                     max_val=1.0)))
        tf.summary.scalar('SSIM', tf.reduce_mean(tf.image.ssim(
                                                     data_batch[1][:,
                                                                   self._prediction_offset:-self._prediction_offset,
                                                                   self._prediction_offset:-self._prediction_offset],
                                                     predicted_batch,
                                                     max_val=1.0)))

        return loss

    def calculate_metrics(self, data_batch, predicted_batch):
        diff = data_batch[1][:, self._prediction_offset:-self._prediction_offset,
               self._prediction_offset:-self._prediction_offset] - predicted_batch
        diff_sqr = tf.square(diff)

        mse = ('MSE', tf.reduce_mean(diff_sqr, axis=[1, 2, 3]))
        psnr = ('PSNR', tf.squeeze(tf.image.psnr(
                                       data_batch[1][:,
                                                     self._prediction_offset:-self._prediction_offset,
                                                     self._prediction_offset:-self._prediction_offset],
                                       predicted_batch,
                                       max_val=1.0)))
        ssim = ('SSIM', tf.squeeze(tf.image.ssim(
                                       data_batch[1][:,
                                                     self._prediction_offset:-self._prediction_offset,
                                                     self._prediction_offset:-self._prediction_offset],
                                       predicted_batch,
                                       max_val=1.0)))

        return [mse, psnr, ssim]


TRAINING_LOGDIR='logdir/espcn_batch_32_lr_1e-3_decay_adam/train'
EVAL_LOGDIR='logdir/espcn_batch_32_lr_1e-3_decay_adam/test'
TRAINING_DATASET_PATH='datasets/train_div2k/dataset.tfrecords'
TRAINING_DATASET_INFO_PATH='datasets/train_div2k/dataset_info.txt'
TESTING_DATASET_PATH='datasets/test_div2k/dataset.tfrecords'
TESTING_DATASET_INFO_PATH='datasets/test_div2k/dataset_info.txt'

MODEL='rtvsrgan'
BATCH_SIZE=32
OPTIMIZER='adam'
LEARNING_RATE=1e-3
USE_LR_DECAY_FLAG='--use_lr_decay'
LEARNING_DECAY_RATE=0.1
LEARNING_DECAY_EPOCHS=30
STAIRCASE_LR_DECAY_FLAG='--staircase_lr_decay'
STEPS_PER_LOG=1000
NUM_EPOCHS=100
EPOCHS_PER_EVAL=1
EPOCHS_PER_SAVE=1
SHUFFLE_BUFFER_SIZE=100000



def get_arguments():
    parser = argparse.ArgumentParser(description='train one of the models for image and video super-resolution')
    parser.add_argument('--model', type=str, default=MODEL, choices=['rtvsrgan','srcnn', 'espcn', 'vespcn', 'vsrnet'],
                        help='What model to train')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of images in batch')
    parser.add_argument('--dataset_path', type=str, default=TRAINING_DATASET_PATH,
                        help='Path to the dataset')
    parser.add_argument('--dataset_info_path', type=str, default=TRAINING_DATASET_INFO_PATH,
                        help='Path to the dataset info')
    parser.add_argument('--ckpt_path', default=None,
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--shuffle_buffer_size', type=int, default=SHUFFLE_BUFFER_SIZE,
                        help='Buffer size used for shuffling examples in dataset')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, choices=['adam', 'momentum', 'sgd'],
                        help='What optimizer to use for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate used for training')
    parser.add_argument('--use_lr_decay', action='store_true',
                        help='Whether to apply exponential decay to the learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=LEARNING_DECAY_RATE,
                        help='Learning rate decay rate used in exponential decay')
    parser.add_argument('--lr_decay_epochs', type=int, default=LEARNING_DECAY_EPOCHS,
                        help='Number of epochs before full decay rate tick used in exponential decay')
    parser.add_argument('--staircase_lr_decay', action='store_true',
                        help='Whether to decay the learning rate at discrete intervals')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--save_num', type=int, default=NUM_EPOCHS,
                        help='How many images to write to summary')
    parser.add_argument('--steps_per_log', type=int, default=STEPS_PER_LOG,
                        help='How often to save summaries')
    parser.add_argument('--epochs_per_save', type=int, default=EPOCHS_PER_SAVE,
                        help='How often to save checkpoints')
    parser.add_argument('--use_mc', action='store_true',
                        help='Whether to use motion compensation in video super resolution model')
    parser.add_argument('--mc_independent', action='store_true',
                        help='Whether to train motion compensation network independent from super resolution network')
    parser.add_argument('--logdir', type=str, default=TRAINING_LOGDIR,
                        help='Where to save checkpoints and summaries')

    return parser.parse_args()




def main():
     
    args = get_arguments()
    print(args)

    if args.model == 'rtvsrgan':
        model = RTVSRGAN(args)
    

    data_batch, data_initializer = model.get_data()

    print(data_batch)

main()
