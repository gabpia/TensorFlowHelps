import tensorflow as tf

# smothing factor for making the loss functions smooth
dice_smooth = 1
# jaccard_smooth = 1


def dice_coef(y_true, y_pred):
    label_intersection = tf.reduce_sum(y_pred * y_true)
    label_sum = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true)
    return (2 * label_intersection + dice_smooth)/(label_sum + dice_smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
