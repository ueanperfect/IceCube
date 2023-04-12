import numpy as np
import torch
import torch.nn as nn


class basic_loss(nn.Module):
    def forward(self,az_true_zen_true, az_pred_zen_pred):
        az_true,zen_true = az_true_zen_true[:,0],az_true_zen_true[:,1]
        az_pred,zen_pred = az_pred_zen_pred[:,0],az_pred_zen_pred[:,1]
        # if not (torch.all(torch.isfinite(az_true)) and
        #         torch.all(torch.isfinite(zen_true)) and
        #         torch.all(torch.isfinite(az_pred)) and
        #         torch.all(torch.isfinite(zen_pred))):
        #     raise ValueError("All arguments must be finite")

        # pre-compute all sine and cosine values
        sa1 = torch.sin(az_true)
        ca1 = torch.cos(az_true)
        sz1 = torch.sin(zen_true)
        cz1 = torch.cos(zen_true)

        sa2 = torch.sin(az_pred)
        ca2 = torch.cos(az_pred)
        sz2 = torch.sin(zen_pred)
        cz2 = torch.cos(zen_pred)

        # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
        scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

        # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
        # that might otherwise occure from the finite precision of the sine and cosine functions
        scalar_prod = torch.clip(scalar_prod, -1, 1)

        # convert back to an angle (in radian)
        return torch.mean(torch.abs(torch.arccos(scalar_prod)))


def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    '''
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two input vectors

    Parameters:
    -----------

    az_true : float (or array thereof)
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian
    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian

    Returns:
    --------

    dist : float
        mean over the angular distance(s) in radian
    '''

    if not (np.all(np.isfinite(az_true)) and
            np.all(np.isfinite(zen_true)) and
            np.all(np.isfinite(az_pred)) and
            np.all(np.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")

    # pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)

    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod = np.clip(scalar_prod, -1, 1)

    # convert back to an angle (in radian)
    return np.average(np.abs(np.arccos(scalar_prod)))
