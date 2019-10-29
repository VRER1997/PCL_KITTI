# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 下午8:04
# @Author  : Gao Xiaosa

import os
import tensorflow as tf

from dataloader import DataLoader
from model import MODEL


dataset_dir = "object_cloud"
log_dir = os.path.join("./log")
save_model_dir = os.path.join("./save_model")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_model_dir, exist_ok=True)

batch_size = 10
lr = 0.001
max_epoch = 100


def main(_):

    with DataLoader(object_dir=os.path.join(dataset_dir, 'training'), queue_size=30, is_testset=False, require_shuffle=True, \
                    batch_size=batch_size, use_multi_process_num=0) as train_loader, \
        DataLoader(object_dir=os.path.join(dataset_dir, 'testing'), queue_size=30, is_testset=False,\
                   batch_size=batch_size, use_multi_process_num=0) as valid_loader:
        with tf.Session() as sess:
            model = MODEL(
                batch_size=batch_size,
                learning_rate=lr
            )

            if tf.train.get_checkpoint_state(save_model_dir):
                model.saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))
            else:
                tf.global_variables_initializer().run()

            iter_per_epoch = int(len(train_loader) / batch_size)

            # summary_interval = 5
            save_model_interval = int(iter_per_epoch / 2)
            validate_interval = 60
            # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

            while model.epoch.eval() < max_epoch:
                is_summary, is_validate = False, False
                iter = model.global_step.eval()
                # if not iter % summary_interval:
                #     is_summary = True
                if not iter % validate_interval:
                    is_validate = True
                if not iter % save_model_interval:
                    model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'),
                                     global_step=model.global_step)
                if not iter % iter_per_epoch:
                    sess.run(model.epoch_add_op)
                    print('train {} epoch, total: {}'.format(model.epoch.eval(), max_epoch))
                # [cls_loss, acc_rate]
                ret = model.train_step(sess, train_loader.load(), train=True, summary=is_summary)
                print('train: {}/{} @ epoch:{}/{}  cls_loss:{} acc_rate:{}'.format(iter, iter_per_epoch*max_epoch,
                      model.epoch.eval(), max_epoch, ret[1], ret[2]))
                
                # if is_summary:
                #     summary_writer.add_summary(ret[-1], iter)

                if is_validate:
                    ret = model.validate_step(sess, valid_loader.load(), summary=False)
                    print('validate: {}/{} @ epoch:{}/{}  cls_loss:{} acc_rate:{}'.format(iter, iter_per_epoch * max_epoch,
                                                                                       model.epoch.eval(), max_epoch,
                                                                                       ret[0], ret[1]))

            print('train done. total epoch:{} iter:{}'.format(
                model.epoch.eval(), model.global_step.eval()))

            model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)


if __name__ == '__main__':
    tf.app.run(main)