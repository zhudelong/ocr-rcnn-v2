import tensorflow as tf

def native_crop_and_resize(image, boxes, crop_size, scope=None):
  """Same as `matmul_crop_and_resize` but uses tf.image.crop_and_resize."""
  def get_box_inds(proposals):
    proposals_shape = proposals.get_shape().as_list()
    if any(dim is None for dim in proposals_shape):
      proposals_shape = tf.shape(proposals)
    ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
    multiplier = tf.expand_dims(
        tf.range(start=0, limit=proposals_shape[0]), 1)
    return tf.reshape(ones_mat * multiplier, [-1])

  with tf.name_scope(scope, 'CropAndResize'):
    cropped_regions = tf.image.crop_and_resize(
        image, tf.reshape(boxes, [-1] + boxes.shape.as_list()[2:]),
        get_box_inds(boxes), crop_size)
    final_shape = tf.concat([tf.shape(boxes)[:2],
                             tf.shape(cropped_regions)[1:]], axis=0)
    return tf.reshape(cropped_regions, final_shape)