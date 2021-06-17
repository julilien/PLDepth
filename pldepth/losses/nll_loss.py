import tensorflow as tf
from tensorflow_ranking.python.losses_impl import _ListwiseLoss
from tensorflow_ranking.python.losses_impl import ListMLELoss
from tensorflow_ranking.python.keras.losses import _RankingLoss

from pldepth.data.depth_utils import prepare_fully_fledged_loss_input
from pldepth.util.str_literals import LOSS_IMPL_STR


class NegativeLogLikelihoodLoss(_RankingLoss):
    def __init__(self, ranking_size, reduction=tf.losses.Reduction.AUTO, name=None, lambda_weight=None):
        super(NegativeLogLikelihoodLoss, self).__init__(reduction, name)
        internal_loss = ListMLELoss(name=LOSS_IMPL_STR.format(name), lambda_weight=lambda_weight)

        self._loss = MetaBatchListMLELoss(ranking_size, name=LOSS_IMPL_STR.format(name), lambda_weight=lambda_weight,
                                          internal_loss=internal_loss)


class MetaBatchListMLELoss(_ListwiseLoss):
    def __init__(self, ranking_size, internal_loss, name, lambda_weight=None, params=None):
        super(MetaBatchListMLELoss, self).__init__(name, lambda_weight, params)
        self.internal_loss = internal_loss
        self.ranking_size = ranking_size

    def compute_unreduced_loss(self, labels, logits):
        transformed_labels = tf.reshape(labels, [-1, self.ranking_size])
        transformed_logits = tf.reshape(logits, [-1, self.ranking_size])

        return self.internal_loss.compute_unreduced_loss(transformed_labels, transformed_logits)


class HourglassNegativeLogLikelihood(_RankingLoss):
    def __init__(self, ranking_size, batch_size, reduction=tf.losses.Reduction.AUTO, name=None,
                 lambda_weight=None, debug=False):
        super(HourglassNegativeLogLikelihood, self).__init__(reduction, name)
        internal_loss = ListMLELoss(name=LOSS_IMPL_STR.format(name), lambda_weight=lambda_weight)

        self._loss = FullyFledgedMetaBatchListMLELoss(ranking_size, batch_size, name=LOSS_IMPL_STR.format(name),
                                                      lambda_weight=lambda_weight,
                                                      internal_loss=internal_loss, debug=debug)


class FullyFledgedMetaBatchListMLELoss(_ListwiseLoss):
    def __init__(self, ranking_size, batch_size, internal_loss, name, lambda_weight=None, params=None, debug=False):
        super(FullyFledgedMetaBatchListMLELoss, self).__init__(name, lambda_weight, params)
        self.internal_loss = internal_loss
        self.ranking_size = ranking_size
        self.batch_size = batch_size
        self.debug = debug

    def compute_unreduced_loss(self, labels, logits):
        """

        :param labels: Rankings (assumed shape: (batch_size, rankings_per_image, ranking_size, 2)
        :param logits: Predicted depth maps (assumed shape: (batch_size, height, width))
        :return: Returns the unreduced loss
        """
        # Make batch size explicit, since the number of rankings per image can vary between training and validation
        selected_depths, reshaped_labels = prepare_fully_fledged_loss_input(labels, logits, self.batch_size,
                                                                            self.ranking_size, self.debug)

        return self.internal_loss.compute_unreduced_loss(reshaped_labels, selected_depths)
