import sys
sys.path.append("SPAM/spams-python")
import spams
import numpy as np
from utils import stain_utils as ut


def get_stain_matrix(I, threshold=0.8, lamda=0.1):
    """
    Get 2x3 stain matrix. First row H and second row E
    :param I:
    :param threshold:
    :param lamda:
    :return:
    """
    mask = ut.notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]
    dictionary = spams.trainDL(OD.T, K=2, lambda1=lamda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False).T
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    dictionary = ut.normalize_rows(dictionary)
    return dictionary


class normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.stain_matrix_target = None

    def fit(self, target):
        target = ut.standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)

    def target_stains(self):
        return ut.OD_to_RGB(self.stain_matrix_target)

    def transform(self, I):
        I = ut.standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        return (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(I.shape))).astype(
            np.uint8)

    def hematoxylin(self, I):
        I = ut.standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)
        return H