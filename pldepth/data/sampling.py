import abc
import numpy as np

from pldepth.data.depth_utils import get_depth_relation


class SamplingStrategy(object):
    def __init__(self, model_params):
        self.num_points_per_sample = model_params.get_parameter('ranking_size')

    @abc.abstractmethod
    def sample_points(self, image, gt):
        """

        :param image: Reshaped image (to reduce computational effort)
        :param gt: Full-size ground truth (to preserve precision)
        :return: Returns point in the space of the reshaped image (shape: (number_points, 2)
        """
        pass

    @abc.abstractmethod
    def sample_points_batch(self, image, gt, batch_size, batch_size_factor=1.5):
        pass

    @property
    def num_points_per_sample(self):
        return self._num_points_per_sample

    @num_points_per_sample.setter
    def num_points_per_sample(self, value):
        self._num_points_per_sample = value

    @staticmethod
    def calculate_depth_differences(depth_values):
        loc_idxs = np.argsort(depth_values)[::-1]
        sorted_depths = depth_values[loc_idxs]

        tmp_dist = 0
        for j in range(len(sorted_depths) - 1):
            depth_diff = abs(sorted_depths[j] - sorted_depths[j + 1])
            tmp_dist += depth_diff
        return tmp_dist

    def __str__(self):
        return "{}(num_points_per_sample={})".format(self.__class__.__name__, self._num_points_per_sample)


class RandomSamplingStrategy(SamplingStrategy):
    def __init__(self, model_params):
        super(RandomSamplingStrategy, self).__init__(model_params)
        self.threshold = 0.03
        self.downscaling_factor = model_params.get_parameter("downscaling_factor")

    def initialize_buffers(self, batch_size, batch_size_factor):
        result_matrix = np.zeros([int(batch_size * batch_size_factor), self._num_points_per_sample, 2],
                                 dtype=np.float32)
        gts_buffer = np.zeros(result_matrix.shape[1])
        point_buffer = np.zeros([self._num_points_per_sample])

        # Dists of the rankings used to sort afterwards
        dists = np.zeros(result_matrix.shape[0])

        return result_matrix, gts_buffer, point_buffer, dists

    @staticmethod
    def sample_single_point(image, gt, gts_buffer, index):
        sample = np.array([np.random.randint(image.shape[0]), np.random.randint(image.shape[1])], dtype=np.int)
        sample_gt = gt[sample[0], sample[1]]

        # Skip close elements
        dists = np.abs(gts_buffer[:index] - sample_gt)

        return sample, sample_gt, dists

    def sample_points(self, image, gt):
        result = np.zeros([self._num_points_per_sample, 2], dtype=np.int)
        gts = np.zeros(self.num_points_per_sample)

        for i in range(result.shape[0]):
            sample, sample_gt, dists = self.sample_single_point(image, gt, gts, index=i)

            while (i == 1 and dists <= self.threshold) or (i > 1 and (dists.min() <= self.threshold).any()):
                sample, sample_gt, dists = self.sample_single_point(image, gt, gts, index=i)

            result[i] = sample
            gts[i] = sample_gt
        return result

    def sample_points_batch(self, image, gt, batch_size, batch_size_factor=1.5):
        result_matrix, gts_buffer, point_buffer, dists = self.initialize_buffers(batch_size, batch_size_factor)

        for i in range(result_matrix.shape[0]):
            for j in range(result_matrix.shape[1]):
                sample = np.array([np.random.randint(image.shape[0]), np.random.randint(image.shape[1])], dtype=np.int)
                gts_buffer[j] = gt[sample[0], sample[1]]
                point_buffer[j] = sample[0] * image.shape[1] + sample[1]

            result_matrix[i, :, 0] = point_buffer
            result_matrix[i, :, 1] = gts_buffer
            dists[i] = self.calculate_depth_differences(gts_buffer)

        # Return only first places
        return result_matrix[np.argsort(dists)[::-1]][:batch_size]


class MaskedRandomSamplingStrategy(RandomSamplingStrategy):
    def __init__(self, model_params):
        super().__init__(model_params)

    @staticmethod
    def sample_single_masked_ranking(image, gt, point_buffer, gts_buffer, mask_points, x_scale, y_scale):
        for j in range(point_buffer.shape[0]):
            sel_point_idx = np.random.randint(mask_points[0].shape[0])

            sample = np.array([int(mask_points[0][sel_point_idx] * x_scale),
                               int(mask_points[1][sel_point_idx] * y_scale)])

            gts_buffer[j] = gt[sample[0], sample[1]]
            point_buffer[j] = sample[0] * image.shape[1] + sample[1]

        loc_idxs = np.argsort(gts_buffer)[::-1]
        return point_buffer[loc_idxs], gts_buffer[loc_idxs]

    @staticmethod
    def determine_x_y_scales(image, mask):
        x_scale = image.shape[0] / mask.shape[0]
        y_scale = image.shape[1] / mask.shape[1]

        return x_scale, y_scale

    def sample_masked_rankings(self, image, mask, gt, batch_size, batch_size_factor=1.5):
        x_scale, y_scale = self.determine_x_y_scales(image, mask)
        result_matrix, gts_buffer, point_buffer, dists = self.initialize_buffers(batch_size, batch_size_factor)

        mask_points = np.where(mask > 0)

        for i in range(result_matrix.shape[0]):
            point_buffer, gts_buffer = self.sample_single_masked_ranking(image, gt, point_buffer, gts_buffer,
                                                                         mask_points,
                                                                         x_scale, y_scale)

            result_matrix[i, :, 0] = point_buffer
            result_matrix[i, :, 1] = gts_buffer

        return result_matrix, dists

    def sample_masked_point_batch(self, image, mask, gt, batch_size, batch_size_factor=1.5):
        result_matrix, dists = self.sample_masked_rankings(image, mask, gt, batch_size, batch_size_factor)

        # Calculate distances
        for i in range(result_matrix.shape[0]):
            gts_buffer = result_matrix[i, :, 1]
            tmp_dist = 0
            for j in range(len(gts_buffer) - 1):
                depth_diff = abs(gts_buffer[j] - gts_buffer[j + 1])
                tmp_dist += depth_diff
            dists[i] = tmp_dist

        return result_matrix[np.argsort(dists)[::-1]][:batch_size]


class ThresholdedMaskedRandomSamplingStrategy(MaskedRandomSamplingStrategy):
    """
    This class provides a sampling mechanism that is thresholded to explicitly exclude rankings including
    "equally-distant" points. Since the Plackett-Luce model is not inherently supporting equal relations, this
    mechanism aims to cope with this by abstention.
    """

    def __init__(self, model_params, threshold=0.03, equality_penalty=-1000):
        """

        :param model_params:
        :param threshold: Threshold used to filter "equally-distant" elements. Defaults to 0.03 as used in Xian et
        al., 2020
        """
        super().__init__(model_params)
        self.threshold = threshold
        self.equality_penalty = equality_penalty

    def sample_masked_point_batch(self, image, mask, gt, batch_size, batch_size_factor=1.5):
        result_matrix, dists = self.sample_masked_rankings(image, mask, gt, batch_size, batch_size_factor)

        # Calculate distances
        for i in range(result_matrix.shape[0]):
            gts_buffer = result_matrix[i, :, 1]
            tmp_dist = 0
            for j in range(len(gts_buffer) - 1):
                depth_diff = abs(gts_buffer[j] - gts_buffer[j + 1])

                # Penalize equal relations
                if get_depth_relation(gts_buffer[j], gts_buffer[j + 1], self.threshold) == 0:
                    tmp_dist += self.equality_penalty

                tmp_dist += depth_diff
            dists[i] = tmp_dist

        # Return only first places
        return result_matrix[np.argsort(dists)[::-1]][:batch_size]

    def __str__(self):
        return "{}(num_points_per_sample={}, threshold={})".format(self.__class__.__name__, self._num_points_per_sample,
                                                                   self.threshold)
