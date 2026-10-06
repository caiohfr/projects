"""EcoDrive Component Estimator — reference prototype v1.0

Frozen Gate-2 reference implementation for Estimator Contract v1.0.
Holdout validation must not tune thresholds or physics in this file.
Not production integration.

Units:
- force: N
- speed: km/h
- quadratic coefficients: N/(km/h)^2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy.optimize import lsq_linear

LBF_TO_N = 4.4482216152605
MPH_TO_KPH = 1.609344
RHO_AIR = 1.225
ROLLING_K = 0.00436


@dataclass(frozen=True)
class ABC:
    A: float
    B: float
    C: float

    def force(self, v_kph):
        v = np.asarray(v_kph, dtype=float)
        return self.A + self.B * v + self.C * v**2


def epa_to_si(A_lbf: float, B_lbf_per_mph: float, C_lbf_per_mph2: float) -> ABC:
    return ABC(
        A=A_lbf * LBF_TO_N,
        B=B_lbf_per_mph * LBF_TO_N / MPH_TO_KPH,
        C=C_lbf_per_mph2 * LBF_TO_N / MPH_TO_KPH**2,
    )


def aero_c_from_cda(cda_m2: float, rho: float = RHO_AIR) -> float:
    return rho * cda_m2 / 25.92


def cda_from_aero_c(c_aero: float, rho: float = RHO_AIR) -> float:
    return c_aero * 25.92 / rho


def normalized_condition_number(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=0)
    if np.any(norms == 0):
        return np.inf
    Xn = X / norms
    s = np.linalg.svd(Xn, compute_uv=False)
    if s[-1] <= np.finfo(float).eps:
        return np.inf
    return float(s[0] / s[-1])


def moskalik_full_target_set(
    target: ABC,
    dyno_set: ABC,
    rolling_k: float = ROLLING_K,
    rho: float = RHO_AIR,
):
    """Closed-form Moskalik-derived full Target+Set decomposition."""
    L0 = target.A - dyno_set.A
    L1 = target.B - dyno_set.B
    L2 = target.C - dyno_set.C

    T2 = L2
    T1 = -159.0 * T2
    T0 = -36.9 * T1 + 22.2

    RT0 = L0 - T0
    RT1 = L1 - T1

    A1 = (dyno_set.B + RT1 - rolling_k * (dyno_set.A + RT0)) / (
        1.0 + 27.5 * rolling_k
    )
    A0 = -27.5 * A1

    RD0 = dyno_set.A - A0
    RD1 = dyno_set.B - A1

    R0 = RD0 + RT0
    R1 = RD1 + RT1

    pure_aero_C = dyno_set.C + A1 / 133.0
    cda = cda_from_aero_c(pure_aero_C, rho=rho)

    return {
        "CdA_m2": cda,
        "R0_N": R0,
        "R1_N_per_kph": R1,
        "T0_N": T0,
        "T1_N_per_kph": T1,
        "T2_N_per_kph2": T2,
        "pure_aero_C": pure_aero_C,
    }


def moskalik_target_only_structured(
    target: ABC,
    rolling_k: float = ROLLING_K,
    rho: float = RHO_AIR,
    cda_bounds=(0.0, np.inf),
    r0_bounds=(0.0, np.inf),
):
    """Three-DOF bounded Target-only structural estimator.

    Unknowns:
        R0, T1, CdA

    Structural relations:
        R1 = k*R0
        T2 = -T1/159
        T0 = -36.9*T1 + 22.2
    """
    X = np.array([
        [1.0, -36.9, 0.0],
        [rolling_k, 1.0, 0.0],
        [0.0, -1.0 / 159.0, rho / 25.92],
    ])
    y = np.array([
        target.A - 22.2,
        target.B,
        target.C,
    ])

    lower = np.array([r0_bounds[0], -np.inf, cda_bounds[0]])
    upper = np.array([r0_bounds[1], np.inf, cda_bounds[1]])

    fit = lsq_linear(X, y, bounds=(lower, upper))
    R0, T1, cda = fit.x

    T2 = -T1 / 159.0
    T0 = -36.9 * T1 + 22.2
    R1 = rolling_k * R0

    return {
        "CdA_m2": float(cda),
        "R0_N": float(R0),
        "R1_N_per_kph": float(R1),
        "T0_N": float(T0),
        "T1_N_per_kph": float(T1),
        "T2_N_per_kph2": float(T2),
        "solver_cost": float(fit.cost),
        "scaled_condition_number": normalized_condition_number(X),
    }


def fixed_gear_target_set_2wd(
    target: ABC,
    dyno_set: ABC,
    q_roll: float,
    rolling_k: float = ROLLING_K,
    rho: float = RHO_AIR,
    v_min: float = 24.0,
    v_max: float = 105.0,
    n: int = 200,
):
    """Constrained 2WD Target+Set fixed-gear estimator.

    Parameters:
        CdA, R0, K0, Kb, Kc >= 0

    q_roll must be frozen from evidence/approved assumption before fitting.
    """
    if not (0.0 <= q_roll <= 1.0):
        raise ValueError("q_roll must be within [0, 1]")

    v = np.linspace(v_min, v_max, n)
    Ft = target.force(v)
    D = dyno_set.force(v)
    L = Ft - D
    roll_basis = 1.0 + rolling_k * v

    X_D = np.column_stack([
        (rho / 25.92) * v**2,
        (1.0 - q_roll) * roll_basis,
        np.zeros_like(v),
        np.zeros_like(v),
        np.zeros_like(v),
    ])
    X_L = np.column_stack([
        np.zeros_like(v),
        q_roll * roll_basis,
        np.ones_like(v),
        v ** (2.0 / 3.0),
        v**2,
    ])

    X = np.vstack([X_D, X_L])
    y = np.concatenate([D, L])

    fit = lsq_linear(X, y, bounds=(0.0, np.inf))
    CdA, R0, K0, Kb, Kc = fit.x

    D_hat = X_D @ fit.x
    L_hat = X_L @ fit.x
    Ft_hat = D_hat + L_hat

    target_rmse = float(np.sqrt(np.mean((Ft_hat - Ft) ** 2)))
    target_nrmse = target_rmse / float(np.mean(np.abs(Ft)))

    return {
        "CdA_m2": float(CdA),
        "R0_N": float(R0),
        "R1_N_per_kph": float(rolling_k * R0),
        "K0": float(K0),
        "Kb": float(Kb),
        "Kc": float(Kc),
        "target_RMSE_N": target_rmse,
        "target_NRMSE": float(target_nrmse),
        "set_RMSE_N": float(np.sqrt(np.mean((D_hat - D) ** 2))),
        "vehicle_loss_RMSE_N": float(np.sqrt(np.mean((L_hat - L) ** 2))),
        "scaled_condition_number": normalized_condition_number(X),
        "source_information_rank": 6,
        "free_parameter_count": 5,
        "q_roll": float(q_roll),
    }


def dht_aggregated_target_set_2wd(
    target: ABC,
    dyno_set: ABC,
    q_roll: float,
    rolling_k: float = ROLLING_K,
    rho: float = RHO_AIR,
    v_min: float = 24.0,
    v_max: float = 105.0,
    n: int = 200,
):
    """Aggregate DHT estimator. No internal DHT parameter split."""
    if not (0.0 <= q_roll <= 1.0):
        raise ValueError("q_roll must be within [0, 1]")

    v = np.linspace(v_min, v_max, n)
    Ft = target.force(v)
    D = dyno_set.force(v)
    L = Ft - D
    roll_basis = 1.0 + rolling_k * v

    X_D = np.column_stack([
        (rho / 25.92) * v**2,
        (1.0 - q_roll) * roll_basis,
    ])
    fit = lsq_linear(X_D, D, bounds=(0.0, np.inf))
    CdA, R0 = fit.x

    R = R0 * roll_basis
    F_dht = L - q_roll * R

    return {
        "CdA_m2": float(CdA),
        "R0_N": float(R0),
        "R1_N_per_kph": float(rolling_k * R0),
        "speed_kph": v,
        "F_DHT_agg_N": F_dht,
        "set_RMSE_N": float(np.sqrt(np.mean((X_D @ fit.x - D) ** 2))),
        "scaled_condition_number": normalized_condition_number(X_D),
        "source_information_rank": 6,
        "free_parameter_count": 2,
        "q_roll": float(q_roll),
    }


def project_curve_to_abc(v_kph: Iterable[float], force_N: Iterable[float]):
    """Uniform-weight OLS quadratic projection."""
    v = np.asarray(v_kph, dtype=float)
    f = np.asarray(force_N, dtype=float)
    X = np.column_stack([np.ones_like(v), v, v**2])
    coeff, *_ = np.linalg.lstsq(X, f, rcond=None)
    pred = X @ coeff
    rmse = float(np.sqrt(np.mean((pred - f) ** 2)))
    mean_abs = float(np.mean(np.abs(f)))
    return {
        "A": float(coeff[0]),
        "B": float(coeff[1]),
        "C": float(coeff[2]),
        "RMSE_N": rmse,
        "NRMSE": rmse / mean_abs if mean_abs > 0 else np.nan,
        "max_abs_error_N": float(np.max(np.abs(pred - f))),
    }


def rrc_eq(R0_N: float, R1_N_per_kph: float, etw_lb: float, v_kph: float) -> float:
    load_N = etw_lb * LBF_TO_N
    return (R0_N + R1_N_per_kph * v_kph) / (load_N / 1000.0)



# ---------------------------------------------------------------------------
# v1.0 frozen QA instrumentation
# ---------------------------------------------------------------------------

def matrix_rank_scaled(X: np.ndarray) -> int:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=0)
    if np.any(norms == 0):
        return int(np.linalg.matrix_rank(X))
    return int(np.linalg.matrix_rank(X / norms))

def bound_diagnostics(values, lower, upper, atol=1e-8):
    values=np.asarray(values,dtype=float); lower=np.asarray(lower,dtype=float); upper=np.asarray(upper,dtype=float)
    out=[]
    for i,v in enumerate(values):
        lo=lower[i]; hi=upper[i]
        dlo=float(v-lo) if np.isfinite(lo) else np.inf
        dhi=float(hi-v) if np.isfinite(hi) else np.inf
        out.append({"at_lower_bound": bool(np.isfinite(lo) and dlo <= atol), "at_upper_bound": bool(np.isfinite(hi) and dhi <= atol), "distance_to_bound": float(min(dlo,dhi))})
    return out

def reconstruction_metrics(observed, predicted):
    observed=np.asarray(observed,dtype=float); predicted=np.asarray(predicted,dtype=float)
    err=predicted-observed; mean_abs=float(np.mean(np.abs(observed)))
    rmse=float(np.sqrt(np.mean(err**2))); max_abs=float(np.max(np.abs(err)))
    return {"RMSE":rmse,"NRMSE":rmse/mean_abs if mean_abs>0 else np.nan,"max_abs_error":max_abs,"max_abs_error_norm":max_abs/mean_abs if mean_abs>0 else np.nan}

if __name__ == "__main__":
    # Regression: Carnival exact EPA state from pilot.
    carnival_target = epa_to_si(40.823, -0.00861, 0.028783)
    carnival_set = epa_to_si(13.005, 0.13425, 0.026741)

    full = moskalik_full_target_set(carnival_target, carnival_set)
    target_only = moskalik_target_only_structured(carnival_target)

    print("Carnival full:", full)
    print("Carnival target-only:", target_only)
    print("Carnival target-only RRC80:",
          rrc_eq(target_only["R0_N"], target_only["R1_N_per_kph"], 5250, 80.0))

    # Fixed-gear stress example: Mirai state used in Gate-1.
    mirai_target = epa_to_si(32.199, 0.14993, 0.018787)
    mirai_set = epa_to_si(22.739, -0.25055, 0.022761)
    for q in (0.40, 0.50, 0.60):
        print("Mirai fixed-gear q=", q, fixed_gear_target_set_2wd(
            mirai_target, mirai_set, q_roll=q
        ))


# ---------------------------------------------------------------------------
# Gate-2 v0.2 QA helpers
# ---------------------------------------------------------------------------

def classify_target_nrmse(nrmse: float) -> str:
    if nrmse <= 0.02:
        return "PASS"
    if nrmse <= 0.05:
        return "CONDITIONAL"
    return "REJECTED_MODEL"


def classify_max_error_norm(value: float) -> str:
    if value <= 0.05:
        return "PASS"
    if value <= 0.10:
        return "CONDITIONAL"
    return "REJECTED_MODEL"


def near_zero(value: float, atol: float = 1e-8) -> bool:
    return abs(float(value)) <= atol


def fixed_gear_bound_flags(result: dict) -> list[str]:
    flags = []
    if near_zero(result.get("R0_N", 0.0)):
        flags.append("ROLLING_COLLAPSED_TO_ZERO")
    if all(near_zero(result.get(k, 0.0)) for k in ("K0", "Kb", "Kc")):
        flags.append("DRIVETRAIN_COLLAPSED_TO_ZERO")
    elif any(near_zero(result.get(k, 0.0)) for k in ("K0", "Kb", "Kc")):
        flags.append("DRIVETRAIN_PARAMETER_AT_BOUND")
    return flags
