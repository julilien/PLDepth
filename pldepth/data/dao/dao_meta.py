from pldepth.data.dao.diode import DIODETFDataAccessObject
from pldepth.data.dao.hr_wsi import HRWSITFDataAccessObject
from pldepth.data.dao.ibims import IbimsTFDataAccessObject
from pldepth.data.dao.sintel import SintelTFDataAccessObject
from pldepth.data.dao.tum import TUMTFDataAccessObject
from pldepth.data.io_utils import Dataset


def get_dao_for_dataset_type(dataset_type, config, model_input_shape, seed=0):
    if dataset_type == Dataset.IBIMS:
        dao = IbimsTFDataAccessObject(config["DATA"]["IBIMS_ROOT_PATH"], model_input_shape)
    elif dataset_type == Dataset.DIODE:
        dao = DIODETFDataAccessObject(config["DATA"]["DIODE_ROOT_PATH"], model_input_shape)
    elif dataset_type == Dataset.SINTEL:
        dao = SintelTFDataAccessObject(config["DATA"]["SINTEL_ROOT_PATH"], model_input_shape)
    elif dataset_type == Dataset.TUM:
        dao = TUMTFDataAccessObject(config["DATA"]["TUM_ROOT_PATH"], model_input_shape)
    elif dataset_type == Dataset.HR_WSI:
        dao = HRWSITFDataAccessObject(config["DATA"]["HR_WSI_ROOT_PATH"], model_input_shape, seed)
    else:
        raise NotImplementedError("Model evaluation currently does not support dataset type '{}'.".format(dataset_type))
    return dao
