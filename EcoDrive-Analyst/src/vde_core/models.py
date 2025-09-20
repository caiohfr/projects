# src/vde_core/models.py

class TirePrior:
    def __init__(self, eps_pressure=-0.5, eps_load=1.0, k0=0.08, k1=2.5e-4, k2=-2.3e-7):
        self.eps_pressure = eps_pressure
        self.eps_load = eps_load
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2

class VehicleAero:
    def __init__(self, rho=1.20, Cx=0.30, Af_m2=2.20):
        self.rho = rho
        self.Cx = Cx
        self.Af_m2 = Af_m2

class RoadLoadABC:
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C

class Parasitic:
    def __init__(self, A_par=0.0, B_par=0.0, C_par=0.0):
        self.A_par = A_par
        self.B_par = B_par
        self.C_par = C_par

class InertiaSpec:
    def __init__(self, mass_test_kg=1500.0, source="User"):
        self.mass_test_kg = mass_test_kg
        self.source = source  # e.g. "EPA", "WLTP", "User"

class Scenario:
    def __init__(self, mode="BASELINE", roadload=None, tires_front=None, tires_rear=None,
                 pressure_kpa=230.0, frac_front=0.5, aero=None, parasitic=None, inertia=None):
        self.mode = mode                  # "BASELINE" or "SEMI_PARAM"
        self.roadload = roadload          # RoadLoadABC
        self.tires_front = tires_front    # TirePrior
        self.tires_rear = tires_rear      # TirePrior
        self.pressure_kpa = pressure_kpa
        self.frac_front = frac_front
        self.aero = aero or VehicleAero()
        self.parasitic = parasitic or Parasitic()
        self.inertia = inertia or InertiaSpec()
