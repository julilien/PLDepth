import tensorflow as tf
import tensorflow.keras.backend as keras_backend

from pldepth.data.depth_utils import get_depth_relation_tf


def get_dl_relation(y_pred, theta, dtype=tf.float32):
    a0 = y_pred[:, 0]
    a1 = y_pred[:, 1]
    a01 = theta * tf.sqrt(a0 * a1)
    alternative_sum = a0 + a1 + a01 + keras_backend.epsilon()

    p_a_b = a0 / alternative_sum
    p_b_a = a1 / alternative_sum
    p_eq = a01 / alternative_sum

    max_idx = tf.argmax([p_a_b, p_b_a, p_eq])
    relations = tf.where(tf.equal(max_idx, 0), tf.constant(1, dtype),
                         tf.where(tf.equal(max_idx, 1), tf.constant(-1, dtype),
                                  tf.constant(0, dtype)))

    return tf.cast(relations, tf.int8)


def rsme(y_true, y_pred):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))


class RMSE(tf.keras.metrics.Metric):
    def __init__(self, name="rmse", **kwargs):
        super(RMSE, self).__init__(name=name, **kwargs)
        self.rmse = self.add_weight(name="rmse", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred):
        self.count.assign_add(1)

        rsme_res = rsme(y_true, y_pred)

        return self.rmse.assign_add(rsme_res)

    def result(self):
        return tf.math.divide_no_nan(self.rmse, self.count)


class WKDR(tf.keras.metrics.Metric):
    def __init__(self, gt_threshold, pred_threshold=None, invert_relations=False, davidson_luce=False,
                 inner_dtype=tf.float32, name="wkdr", **kwargs):
        super(WKDR, self).__init__(name=name, **kwargs)

        self.gt_threshold = gt_threshold
        if pred_threshold is None:
            self.pred_threshold = gt_threshold
        else:
            self.pred_threshold = pred_threshold

        self.inner_dtype = inner_dtype

        self.wkdr = self.add_weight(name="wkdr", initializer="zeros", dtype=self.inner_dtype)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=self.inner_dtype)

        self.invert_relations = invert_relations

        self.davidson_luce = davidson_luce
        self.dl_thetas = None

    @staticmethod
    def process_shape(y_true, y_pred):
        if len(y_pred.shape) == 1:
            y_pred = tf.reshape(y_pred, [1, 2])

        if len(y_true.shape) == 1:
            y_true = tf.reshape(y_true, [-1, 1])
        else:
            y_true = tf.reshape(y_true, [-1, 2])

        assert y_true.shape[0] == y_pred.shape[0], "The batch size must match!"

        return y_true, y_pred

    def update_state(self, y_true, y_pred):
        y_true, y_pred = self.process_shape(y_true, y_pred)

        self.count.assign_add(y_true.shape[0])

        if not self.davidson_luce:
            relation = get_depth_relation_tf(y_pred, self.pred_threshold, self.inner_dtype)
        else:
            relation = get_dl_relation(y_pred, self.dl_thetas[0], self.inner_dtype)

        if len(y_true.shape) > 1:
            y_true = get_depth_relation_tf(y_true, self.gt_threshold, self.inner_dtype)
        if self.invert_relations:
            y_true *= -1

        wkdr_res = tf.reduce_sum(tf.cast(tf.not_equal(y_true, relation), self.inner_dtype))

        return self.wkdr.assign_add(wkdr_res)

    def result(self):
        if self.count > 0:
            return tf.cast(self.wkdr / self.count, dtype=self.inner_dtype)
        else:
            return 0.

    def set_dl_theta_parameters(self, thetas):
        assert self.davidson_luce, "In order to use Davidson-Luce theta parameters, it must be activated explicitly."
        self.dl_thetas = thetas


class WKDR_eq(WKDR):
    def __init__(self, gt_threshold, pred_threshold=None, invert_relations=False, davidson_luce=False,
                 inner_dtype=tf.float32, name="wkdr_eq", **kwargs):
        super(WKDR_eq, self).__init__(gt_threshold, pred_threshold, invert_relations, davidson_luce, inner_dtype,
                                      name=name, **kwargs)

    def update_state(self, y_true, y_pred):
        y_true, y_pred = self.process_shape(y_true, y_pred)

        if not self.davidson_luce:
            relation = get_depth_relation_tf(y_pred, self.pred_threshold, self.inner_dtype)
        else:
            relation = get_dl_relation(y_pred, self.dl_thetas[0], self.inner_dtype)

        if len(y_true.shape) > 1:
            y_true = get_depth_relation_tf(y_true, self.gt_threshold, self.inner_dtype)

        if self.invert_relations:
            y_true *= -1

        y_true_mask = (tf.cast(y_true, tf.int8) == tf.constant(0, tf.int8))
        new_count = tf.reduce_sum(tf.cast(y_true_mask, self.inner_dtype))

        self.count.assign_add(new_count)

        wkdr_res = tf.reduce_sum(
            tf.cast(tf.not_equal(relation[y_true_mask], tf.constant(0, dtype=tf.int8)), dtype=self.inner_dtype))

        return self.wkdr.assign_add(wkdr_res)


class WKDR_neq(WKDR):
    def __init__(self, gt_threshold, pred_threshold=None, invert_relations=False, davidson_luce=False,
                 inner_dtype=tf.float32, name="wkdr_neq", **kwargs):
        super(WKDR_neq, self).__init__(gt_threshold, pred_threshold, invert_relations, davidson_luce, inner_dtype,
                                       name=name, **kwargs)

    def update_state(self, y_true, y_pred):
        y_true, y_pred = self.process_shape(y_true, y_pred)

        if not self.davidson_luce:
            relation = get_depth_relation_tf(y_pred, self.pred_threshold, self.inner_dtype)
        else:
            relation = get_dl_relation(y_pred, self.dl_thetas[0], self.inner_dtype)

        if len(y_true.shape) > 1:
            y_true = get_depth_relation_tf(y_true, self.gt_threshold, self.inner_dtype)

        if self.invert_relations:
            y_true *= -1

        y_true_mask = (tf.cast(y_true, tf.int8) != tf.constant(0, tf.int8))
        new_count = tf.reduce_sum(tf.cast(y_true_mask, self.inner_dtype))

        self.count.assign_add(new_count)

        wkdr_res = tf.reduce_sum(
            tf.cast(tf.not_equal(relation[y_true_mask], tf.cast(y_true[y_true_mask], dtype=tf.int8)),
                    dtype=self.inner_dtype))

        return self.wkdr.assign_add(wkdr_res)


class WKDR_PL(tf.keras.metrics.Metric):
    def __init__(self, threshold, delta, invert_relations=False, inner_dtype=tf.float32, name="wkdr_pl", **kwargs):
        super(WKDR_PL, self).__init__(name=name, **kwargs)

        self.threshold = threshold
        self.delta = delta
        self.inner_dtype = inner_dtype

        self.wkdr_pl = self.add_weight(name="wkdr_pl", initializer="zeros", dtype=self.inner_dtype)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=self.inner_dtype)

        self.invert_relations = invert_relations

    def update_state(self, y_true, y_pred):
        y_true, y_pred = WKDR.process_shape(y_true, y_pred)

        self.count.assign_add(y_true.shape[0])

        relation = get_dl_relation(y_pred, self.delta, self.inner_dtype)

        if len(y_true.shape) > 1:
            y_true = get_depth_relation_tf(y_true, self.threshold, self.inner_dtype)
        if self.invert_relations:
            y_true *= -1

        wkdr_res = tf.reduce_sum(tf.cast(tf.not_equal(y_true, relation), self.inner_dtype))

        return self.wkdr_pl.assign_add(wkdr_res)

    def result(self):
        if self.count > 0:
            return tf.cast(self.wkdr_pl / self.count, dtype=self.inner_dtype)
        else:
            return 0.


class WKDR_eq_PL(WKDR_PL):
    def __init__(self, threshold, delta, invert_relations=False, inner_dtype=tf.float32, name="wkdr_eq_pl", **kwargs):
        super(WKDR_eq_PL, self).__init__(threshold, delta, invert_relations, inner_dtype, name=name, **kwargs)

    def update_state(self, y_true, y_pred):
        y_true, y_pred = WKDR.process_shape(y_true, y_pred)

        relation = get_dl_relation(y_pred, self.delta, self.inner_dtype)

        if len(y_true.shape) > 1:
            y_true = get_depth_relation_tf(y_true, self.threshold, self.inner_dtype)

        if self.invert_relations:
            y_true *= -1

        y_true_mask = (tf.cast(y_true, tf.int8) == tf.constant(0, tf.int8))
        new_count = tf.reduce_sum(tf.cast(y_true_mask, self.inner_dtype))

        self.count.assign_add(new_count)

        wkdr_res = tf.reduce_sum(
            tf.cast(tf.not_equal(relation[y_true_mask], tf.constant(0, dtype=tf.int8)), dtype=self.inner_dtype))

        return self.wkdr_pl.assign_add(wkdr_res)


class WKDR_neq_PL(WKDR_PL):
    def __init__(self, threshold, delta, invert_relations=False, inner_dtype=tf.float32, name="wkdr_neq_pl", **kwargs):
        super(WKDR_neq_PL, self).__init__(threshold, delta, invert_relations, inner_dtype, name=name, **kwargs)

    def update_state(self, y_true, y_pred):
        y_true, y_pred = WKDR.process_shape(y_true, y_pred)

        relation = get_dl_relation(y_pred, self.delta, self.inner_dtype)

        if len(y_true.shape) > 1:
            y_true = get_depth_relation_tf(y_true, self.threshold, self.inner_dtype)

        if self.invert_relations:
            y_true *= -1

        y_true_mask = (tf.cast(y_true, tf.int8) != tf.constant(0, tf.int8))
        new_count = tf.reduce_sum(tf.cast(y_true_mask, self.inner_dtype))

        self.count.assign_add(new_count)

        wkdr_res = tf.reduce_sum(
            tf.cast(tf.not_equal(relation[y_true_mask], tf.cast(y_true[y_true_mask], dtype=tf.int8)),
                    dtype=self.inner_dtype))

        return self.wkdr_pl.assign_add(wkdr_res)
