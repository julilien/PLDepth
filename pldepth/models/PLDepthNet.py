from pldepth.models.pl_hourglass import EffNetFullyFledged
from pldepth.models.redweb import ReDWebNetTFVersion
from pldepth.models.models_meta import ModelType


def get_pl_depth_net(model_params, input_shape):
    model_type = model_params.get_parameter("model_type")

    if model_type == ModelType.FULLY_FLEDGED_EFFNET:
        model, preprocess_fn = EffNetFullyFledged.get_model_and_normalization(input_shape,
                                                                              model_params.get_parameter(
                                                                                  "ranking_size"),
                                                                              model_params.get_parameter("loss_type"))
    elif model_type == ModelType.FULLY_FLEDGED_REDWEB:
        model, preprocess_fn = ReDWebNetTFVersion.get_model_and_normalization(input_shape,
                                                                              model_params.get_parameter(
                                                                                  "ranking_size"),
                                                                              model_params.get_parameter("loss_type"))
    else:
        raise ValueError("Unknown model type: {}".format(model_params.get_parameter("model_type")))
    return model, preprocess_fn
