import numpy as np
from .models import TirePrior, RoadLoadABC

def c_aero(rho, Cx, Af_m2):
    """Return aerodynamic 'C' term in N/(kph^2)."""
    return 0.5 * rho * Cx * Af_m2 * (1/3.6)**2

def _f_rr_axle(v_kph, P_kpa, N_newton, epsP, epsN, k0, k1, k2):
    # Rolling resistance law: pressure^epsP * load^epsN * (k0 + k1*v + k2*v^2)
    return (P_kpa**epsP) * (N_newton**epsN) * (k0 + k1*v_kph + k2*(v_kph**2))

def compose_abc_from_blocks(scn):
    """Build A/B/C from tires, aero and parasitics."""
    g = 9.81
    Nf = scn.frac_front * scn.inertia.mass_test_kg * g
    Nr = (1 - scn.frac_front) * scn.inertia.mass_test_kg * g
    tf = scn.tires_front or TirePrior()
    tr = scn.tires_rear or tf

    # RR force at v=0,1,2 kph
    A_rr = _f_rr_axle(0, scn.pressure_kpa, Nf, tf.eps_pressure, tf.eps_load, tf.k0, tf.k1, tf.k2) \
         + _f_rr_axle(0, scn.pressure_kpa, Nr, tr.eps_pressure, tr.eps_load, tr.k0, tr.k1, tr.k2)

    f1 = _f_rr_axle(1, scn.pressure_kpa, Nf, tf.eps_pressure, tf.eps_load, tf.k0, tf.k1, tf.k2) \
       + _f_rr_axle(1, scn.pressure_kpa, Nr, tr.eps_pressure, tr.eps_load, tr.k0, tr.k1, tr.k2)
    B_rr = f1 - A_rr

    f2 = _f_rr_axle(2, scn.pressure_kpa, Nf, tf.eps_pressure, tf.eps_load, tf.k0, tf.k1, tf.k2) \
       + _f_rr_axle(2, scn.pressure_kpa, Nr, tr.eps_pressure, tr.eps_load, tr.k0, tr.k1, tr.k2)
    C_rr = (f2 - (A_rr + B_rr*2.0)) / (2.0**2)

    C_a = c_aero(scn.aero.rho, scn.aero.Cx, scn.aero.Af_m2)

    A = A_rr + scn.parasitic.A_par
    B = B_rr + scn.parasitic.B_par
    C = C_rr + C_a + scn.parasitic.C_par
    return RoadLoadABC(A, B, C)

def f_road(v_kph, abc):
    return abc.A + abc.B*v_kph + abc.C*(v_kph**2)

def power_patch(F_N, m_test_kg, v_kph, a_ms2):
    v_ms = v_kph / 3.6
    return (F_N + m_test_kg * a_ms2) * v_ms  # [W]
