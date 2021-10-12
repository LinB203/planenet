import keras
from keras import backend as K


def exponent(global_epoch,
             learning_rate_base,
             decay_rate,
             min_learn_rate=0,
             ):
    learning_rate = learning_rate_base * pow(decay_rate, global_epoch)
    learning_rate = max(learning_rate, min_learn_rate)
    print(learning_rate)
    return learning_rate

class ExponentDecayScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """
    def __init__(self,
                 learning_rate_base,
                 decay_rate,
                 global_epoch_init=0,
                 min_learn_rate=0,
                 verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 全局初始化epoch
        self.global_epoch = global_epoch_init

        self.decay_rate = decay_rate
        # 参数显示
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

    def on_epoch_end(self, epochs, logs=None):
        self.global_epoch = self.global_epoch + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    # 更新学习率
    def on_epoch_begin(self, batch, logs=None):
        lr = exponent(global_epoch=self.global_epoch,
                      learning_rate_base=self.learning_rate_base,
                      decay_rate=self.decay_rate,
                      min_learn_rate=self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_epoch + 1, lr))


