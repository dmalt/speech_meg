from .compute_mfccs_conf import *  # noqa
from .find_muscle_ics_conf import ica_comp_filt, mi_thresh  # noqa
from .mi_config import bands  # noqa
from .paths import *  # noqa

maxfilt_config: dict = {"t_window": "auto"}

ica_config: dict = {
    "random_state": 28,
    "n_components": 0.99,
    "decim": 3,
    "annot_rej": True,
}

audio_ch = "MISC008"

resamp_freq = 500
