#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .model import Model


class SRESNETP(Model):
    def __init__(self, args):
        super().__init__(args)
        self._prediction_offset = self._scale_factor * 4
        print("prediction_offset: {}".format(self._prediction_offset))

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

        with tf.variable_scope('rtvsrgan'):
            if not self._using_dataset:
                lr_batch = tf.pad(lr_batch, [[0, 0], [4, 4], [4, 4], [0, 0]], 'SYMMETRIC')
            print("lr_batch: {}".format(lr_batch.shape))
            #[?, 36, 36, 64]
            net = tf.keras.layers.Conv2D( 64, 3, padding='same',strides=(1, 1), name='conv1',
                                   kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
                                   mode='fan_in', distribution='truncated_normal', seed=None))(lr_batch)
            net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
            net1 = net
            print("net1: {}".format(net1.shape))
            #[?, 36, 36, 64]
            net = tf.keras.layers.Conv2D(64, 3, padding='same',strides=(1, 1), name='conv2',
                                   kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
                                   mode='fan_in', distribution='truncated_normal', seed=None))(net)
            net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
            net2 = net
            print("net2: {}".format(net2.shape)) 
            #[?, 36, 36, 128]
            net = tf.concat([net1, net],axis=3)

            #[?, 36, 36, 64]
            net = tf.keras.layers.Conv2D(64, 3, padding='same',strides=(1, 1), name='conv3',
                                   kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
                                   mode='fan_in', distribution='truncated_normal', seed=None))(net)
            net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
            net3 = net
            print("net3: {}".format(net3.shape))
            
            net1 = tf.keras.layers.Lambda(lambda x: x * 0.2)(net1)
            net2 = tf.keras.layers.Lambda(lambda x: x * 0.2)(net2)
            net3 = tf.keras.layers.Lambda(lambda x: x * 0.6)(net3)
            
            #net = tf.concat([net1, net2,net3],axis=3)
            net = tf.keras.layers.add([net1, net2,net3])
            print("net4: {}".format(net.shape))

            net = tf.keras.layers.Conv2D( self._scale_factor ** 2, 3, activation=tf.nn.tanh, padding='same',strides=(1, 1),
                                   name='conv4', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
                                   mode='fan_in', distribution='truncated_normal', seed=None))(net)
            predicted_batch = tf.depth_to_space(net, self._scale_factor, name='prediction')
            print("predicted_batch: {}".format(predicted_batch.shape))                                            

        rtvsrgan_variables = tf.trainable_variables(scope='rtvsrgan')
        for variable in rtvsrgan_variables:
            if 'conv4' in variable.name:
                self.lr_multipliers[variable.name] = 0.1
            else:
                self.lr_multipliers[variable.name] = 1.0

        if self._using_dataset:
            tf.summary.image('Low_resolution', data_batch[0][:, 0:-4, 4:-4], max_outputs=self._save_num)
            tf.summary.image('High_resolution',
                             data_batch[1][:, self._prediction_offset:-self._prediction_offset,
                                           self._prediction_offset:-self._prediction_offset],
                             max_outputs=self._save_num)
            tf.summary.image('High_resolution_prediction', predicted_batch, max_outputs=self._save_num)

        return predicted_batch[:,self._prediction_offset:-self._prediction_offset,self._prediction_offset:-self._prediction_offset]

    def get_loss(self, data_batch, predicted_batch):
        print("data_batch: {}".format(data_batch[1].shape))
        print("predicted_batch: {}".format(predicted_batch.shape))
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
