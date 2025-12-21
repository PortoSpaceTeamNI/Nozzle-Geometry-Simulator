import numpy as np
import math

def _unique_x_with_mean_y(x, y, tol=1e-12):
    """
    Remove duplicates in x by binning values that are very close (<= tol)
    and averaging y inside each bin.
    Returns x_u, y_u strictly increasing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return x, y

    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    # bin close x
    x_u = [x[0]]
    y_acc = [y[0]]
    n_acc = [1]

    for xi, yi in zip(x[1:], y[1:]):
        if abs(xi - x_u[-1]) <= tol:
            y_acc[-1] += yi
            n_acc[-1] += 1
        else:
            x_u.append(xi)
            y_acc.append(yi)
            n_acc.append(1)

    x_u = np.array(x_u, dtype=float)
    y_u = np.array([s/n for s, n in zip(y_acc, n_acc)], dtype=float)
    return x_u, y_u


def turning_metric_phi(x_all, r_all, x_throat, only_diverging=True, resample_N=800):
    """
    Phi = ∫ |dθ/dx| dx, with θ = atan(dr/dx)

    Fixes gradient warnings by:
      1) sorting + removing duplicate x (averaging r)
      2) resampling to a uniform x grid (optional but recommended)

    Returns:
      Phi (radians), dbg dict
    """
    x = np.asarray(x_all, dtype=float)
    r = np.asarray(r_all, dtype=float)

    # sort + unique x
    x, r = _unique_x_with_mean_y(x, r, tol=1e-10)

    if only_diverging:
        m = x >= float(x_throat)
        x, r = x[m], r[m]

    if x.size < 5:
        return 0.0, {"x": x, "theta": np.array([]), "dtheta_dx": np.array([])}

    # resample to uniform x for stable derivatives
    if resample_N is not None and resample_N >= 50:
        x_min, x_max = float(x[0]), float(x[-1])
        if x_max - x_min > 1e-9:
            x_uniform = np.linspace(x_min, x_max, int(resample_N))
            r_uniform = np.interp(x_uniform, x, r)
            x, r = x_uniform, r_uniform

    # derivatives
    drdx = np.gradient(r, x)
    theta = np.arctan(drdx)
    dtheta_dx = np.gradient(theta, x)

    Phi = float(np.trapezoid(np.abs(dtheta_dx), x))
    return Phi, {"x": x, "theta": theta, "dtheta_dx": dtheta_dx}


def calibrate_k_from_reference(Phi_ref, eta_turn_ref=0.99):
    Phi_ref = max(float(Phi_ref), 1e-12)
    eta_turn_ref = float(eta_turn_ref)
    if not (0.0 < eta_turn_ref < 1.0):
        raise ValueError("eta_turn_ref must be between 0 and 1 (e.g., 0.99).")
    return float(-math.log(eta_turn_ref) / Phi_ref)


def eta_turn_from_phi(Phi, k):
    Phi = max(float(Phi), 0.0)
    k = max(float(k), 0.0)
    return float(math.exp(-k * Phi))

