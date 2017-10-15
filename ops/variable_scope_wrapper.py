# variable_scope_wrapper.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import variable_scope


GLOBAL_MASK = {}


def set_mask(mask):
    global GLOBAL_MASK
    GLOBAL_MASK = mask


def get_variable(name, shape=None, dtype=None, initializer=None,
                 regularizer=None, trainable=True):
    var = variable_scope.get_variable(name, shape, dtype, initializer,
                                      regularizer, trainable)
    name = var.name

    if name not in GLOBAL_MASK:
        return var

    new_name = name + "_mask"
    mask_var = variable_scope.get_variable(new_name, shape, dtype, None,
                                           None, False)
    mask_var.set_value(GLOBAL_MASK[name])

    return var * mask_var
