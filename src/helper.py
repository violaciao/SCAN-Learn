from config import *


# This function changes the teacher forcing ratio over the training model.
def teacher_forcing_decay(num_iters):
    if num_iters < 5000:
        return 0.8
    else:
        return 0.5


# This function changes the learning rate over the training model.
def exp_lr_scheduler(optimizer, iter, init_lr=BASE_LR, lr_decay_iter=ITER_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_iter iters."""
    lr = init_lr * (DECAY_WEIGHT**(iter // lr_decay_iter))

    if iter % lr_decay_iter == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer