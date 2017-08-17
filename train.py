import tensorflow as tf
from c_gan_model import CGAN
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 100, 'batch_size: default:100')

tf.flags.DEFINE_integer('image_size', 28, 'image_size: default:28')
tf.flags.DEFINE_integer('image_channel', 1, 'image_channel: dfault:1')
tf.flags.DEFINE_integer('z_dim', 100, 'z_dim: default:100')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'learning rate: default:2e-4')
tf.flags.DEFINE_integer('ngf', 512, 'number of gen filters in first conv layer')
tf.flags.DEFINE_integer('ndf', 64, 'number of dis filters in first conv layer')
tf.flags.DEFINE_integer('ngfc', 784, 'fully connected dimension for generator')
tf.flags.DEFINE_integer('ndfc', 784, 'fully connected dimension for discriminator')
tf.flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data', 'Directory for storing input data')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20170602-1936), default=None')
tf.flags.DEFINE_integer('max_num_steps', 2000, 'Number of steps to train')
tf.flags.DEFINE_integer('num_steps_run', 200, 'Number of steps to run per this script call')


def main(_):
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.load_model
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    print(mnist.train.images.shape)
    assert mnist.train.images.shape == (55000, 784)
    assert mnist.test.images.shape == (10000, 784)

    graph = tf.Graph()
    with graph.as_default():
        cgan = CGAN(
            batch_size=FLAGS.batch_size,
            y_dim=10,
            z_dim=FLAGS.z_dim,
            input_img_size=FLAGS.image_size,
            output_img_size=FLAGS.image_size,
            num_channels=FLAGS.image_channel,
            ngf=FLAGS.ngf,
            ndf=FLAGS.ndf,
            ngfc=FLAGS.ngfc,
            ndfc=FLAGS.ndfc,
            learning_rate=FLAGS.learning_rate
        )
        input_images = cgan.input_placeholder
        z = cgan.input_placeholder
        labels = cgan.y_placeholder
        g_optimizer, d_optimizer = cgan.g_optimizer, cgan.d_optimizer
        g_loss, d_loss_real, d_loss_fake = cgan.g_loss, cgan.d_loss_real, cgan.d_loss_fake
        generated_imgs = cgan.g_output

        summary_op = cgan.summary_op
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    # exit()
    #    saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        """
        batch = mnist.train.next_batch(50)
        x: batch[0], y_: batch[1]
        """
        if FLAGS.load_model is not None:
            # TODO
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        rel_step = 0
        try:
            while rel_step < FLAGS.num_steps_run and step < FLAGS.max_num_steps:
                batch = mnist.train.next_batch(FLAGS.batch_size)
                x = batch[0].reshape([-1, FLAGS.image_size, FLAGS.image_size, FLAGS.image_channel])
                y_ = batch[1]
                z = np.random.uniform(-1, 1, [FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
                _, _, g_loss_val, d_loss_real_val, d_loss_fake_val, summary = sess.run(
                    fetches=[g_optimizer, d_optimizer, g_loss, d_loss_real, d_loss_fake, summary_op],
                    feed_dict={
                        cgan.input_placeholder: x,
                        cgan.y_placeholder: y_,
                        cgan.z_placeholder: z
                    }
                )
                train_writer.add_summary(summary)
                train_writer.flush()
                print(step)

                if step % 10 == 0:
                    print('-----------Step %d:-------------' % step)
                    print('  G_loss   : {}'.format(g_loss_val))
                    print('  D(G(Z))_loss : {}'.format(d_loss_fake_val))
                    print('  D(X)_loss : {}'.format(d_loss_real_val))
                    print('-----------Sample img----------')
                    y_samples = np.zeros(shape=[0, 10])
                    for _ in range(10):
                        y_samples = np.vstack([y_samples, np.eye(10)])
                    z_samples = np.random.uniform(-1, 1, [10*10, FLAGS.z_dim]).astype(np.float32)

                    g_z = sess.run(
                        fetches=generated_imgs,
                        feed_dict={
                            cgan.y_placeholder: y_samples,
                            cgan.z_placeholder: z_samples
                        }
                    )
                    fig = _plot(g_z)
                    plt.savefig('{}/{}.png'.format(checkpoints_dir, str(step).zfill(3)), bbox_inches='tight')
                    plt.close(fig)

                if step % 150 == 0:
                    print("SAVED")
                    # save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    # print("Model saved in file: %s" % save_path)
                rel_step += 1
                step += 1
        except KeyboardInterrupt:
            print("Manually stopped.")
        finally:
            print("Done")
            # save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            # print("Model saved in file: %s" % save_path)


def _plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig

if __name__ == '__main__':
    tf.app.run()
