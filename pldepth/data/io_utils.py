from pldepth.models.models_meta import StringEnum


class Dataset(StringEnum):
    HR_WSI = "HR-WSI"
    IBIMS = "IBIMS"
    SINTEL = "SINTEL"
    DIODE = "DIODE"
    TUM = "TUM"


def get_dataset_type_by_name(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == Dataset.HR_WSI.value.lower() or dataset_name == "hr_wsi":
        return Dataset.HR_WSI
    elif dataset_name == Dataset.IBIMS.value.lower():
        return Dataset.IBIMS
    elif dataset_name == Dataset.SINTEL.value.lower():
        return Dataset.SINTEL
    elif dataset_name == Dataset.DIODE.value.lower():
        return Dataset.DIODE
    elif dataset_name == Dataset.TUM.value.lower():
        return Dataset.TUM
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))
