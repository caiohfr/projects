"""
RoadLoad domain models for EcoDrive Analyzer.

This module contains the data structures used by the RoadLoad Engine.
It intentionally avoids Streamlit, database access, and VDE-cycle calculations.

Responsibility:
    - represent baseline road-load data
    - represent scenario/component changes
    - represent equivalent A/B/C output
    - provide a simple RoadLoadModel for force/power calculations

Processing functions such as normalize, resolve, apply, and synthesize should live in:
    src/roadload/engine.py

Adapters for Streamlit/DB inputs should live in:
    src/roadload/adapters.py
"""

from typing import Optional, Dict, Any, List


class BaselineInput:
    """
    Raw baseline input for a RoadLoad scenario.

    This may come from manual inputs, database rows, or imported files.
    Values can be partially missing if the engine is allowed to resolve them
    from a baseline record later.
    """

    def __init__(
        self,
        baseline_id: Optional[int] = None,
        A: Optional[float] = None,
        B: Optional[float] = None,
        C: Optional[float] = None,
        mass_kg: Optional[float] = None,
        legislation: Optional[str] = None,
        category: Optional[str] = None,
        source: str = "unknown",
    ):
        self.baseline_id = baseline_id
        self.A = A
        self.B = B
        self.C = C
        self.mass_kg = mass_kg
        self.legislation = legislation
        self.category = category
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "mass_kg": self.mass_kg,
            "legislation": self.legislation,
            "category": self.category,
            "source": self.source,
        }

    def __repr__(self) -> str:
        return f"BaselineInput({self.to_dict()})"


class OperatingModifiers:
    """
    Operating or scenario-level modifiers.

    These are not component-specific. They describe changes in how the vehicle
    is tested or configured, such as payload/mass changes.
    """

    def __init__(
        self,
        delta_mass_kg: float = 0.0,
        trailer: bool = False,
        target_legislation: Optional[str] = None,
    ):
        self.delta_mass_kg = delta_mass_kg
        self.trailer = trailer
        self.target_legislation = target_legislation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_mass_kg": self.delta_mass_kg,
            "trailer": self.trailer,
            "target_legislation": self.target_legislation,
        }

    def __repr__(self) -> str:
        return f"OperatingModifiers({self.to_dict()})"


class ComponentChange:
    """
    Represents a change applied to one road-load component.

    Supported modes expected by the engine:
        - delta_abc: add A/B/C deltas
        - delta_cda: convert delta CdA into delta C
        - improve: apply percentage improvement

    The engine decides how each mode is interpreted.
    """

    VALID_MODES = {"delta_abc", "delta_cda", "improve"}

    def __init__(
        self,
        mode: str = "delta_abc",
        A: float = 0.0,
        B: float = 0.0,
        C: float = 0.0,
        delta_cda_m2: float = 0.0,
        improve_pct: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.mode = mode
        self.A = A
        self.B = B
        self.C = C
        self.delta_cda_m2 = delta_cda_m2
        self.improve_pct = improve_pct
        self.meta = meta if meta is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "delta_cda_m2": self.delta_cda_m2,
            "improve_pct": self.improve_pct,
            "meta": self.meta,
        }

    def __repr__(self) -> str:
        return f"ComponentChange({self.to_dict()})"


class ComponentChanges:
    """
    Collection of component-level changes for a RoadLoad scenario.

    Any attribute can be None when that component is not changed.
    """

    def __init__(
        self,
        tire: Optional[ComponentChange] = None,
        aero: Optional[ComponentChange] = None,
        transmission: Optional[ComponentChange] = None,
        axle: Optional[ComponentChange] = None,
        brakes: Optional[ComponentChange] = None,
        hubs: Optional[ComponentChange] = None,
        parasitic: Optional[ComponentChange] = None,
    ):
        self.tire = tire
        self.aero = aero
        self.transmission = transmission
        self.axle = axle
        self.brakes = brakes
        self.hubs = hubs
        self.parasitic = parasitic

    def items(self):
        """
        Returns component changes as (name, change) pairs.
        This is useful for engine loops.
        """
        return [
            ("tire", self.tire),
            ("aero", self.aero),
            ("transmission", self.transmission),
            ("axle", self.axle),
            ("brakes", self.brakes),
            ("hubs", self.hubs),
            ("parasitic", self.parasitic),
        ]

    def active_items(self):
        """
        Returns only components that have a defined change.
        """
        return [(name, change) for name, change in self.items() if change is not None]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tire": self.tire.to_dict() if self.tire else None,
            "aero": self.aero.to_dict() if self.aero else None,
            "transmission": self.transmission.to_dict() if self.transmission else None,
            "axle": self.axle.to_dict() if self.axle else None,
            "brakes": self.brakes.to_dict() if self.brakes else None,
            "hubs": self.hubs.to_dict() if self.hubs else None,
            "parasitic": self.parasitic.to_dict() if self.parasitic else None,
        }

    def __repr__(self) -> str:
        return f"ComponentChanges({self.to_dict()})"


class ResolutionOptions:
    """
    Options that control how incomplete requests are resolved.
    """

    def __init__(
        self,
        allow_estimation: bool = False,
        use_defaults: bool = True,
        inherit_from_baseline: bool = True,
    ):
        self.allow_estimation = allow_estimation
        self.use_defaults = use_defaults
        self.inherit_from_baseline = inherit_from_baseline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_estimation": self.allow_estimation,
            "use_defaults": self.use_defaults,
            "inherit_from_baseline": self.inherit_from_baseline,
        }

    def __repr__(self) -> str:
        return f"ResolutionOptions({self.to_dict()})"


class RoadLoadRequest:
    """
    Main input object for the RoadLoad Engine.

    This object is the contract between UI/adapters and the engine.
    """

    def __init__(
        self,
        baseline: BaselineInput,
        operating: Optional[OperatingModifiers] = None,
        components: Optional[ComponentChanges] = None,
        options: Optional[ResolutionOptions] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.baseline = baseline
        self.operating = operating if operating is not None else OperatingModifiers()
        self.components = components if components is not None else ComponentChanges()
        self.options = options if options is not None else ResolutionOptions()
        self.extra = extra if extra is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "operating": self.operating.to_dict(),
            "components": self.components.to_dict(),
            "options": self.options.to_dict(),
            "extra": self.extra,
        }

    def __repr__(self) -> str:
        return f"RoadLoadRequest({self.to_dict()})"


class ResolvedBaseline:
    """
    Baseline after validation/resolution.

    Unlike BaselineInput, this object should always have A/B/C/mass defined.
    """

    def __init__(
        self,
        A: float,
        B: float,
        C: float,
        mass_kg: float,
        legislation: Optional[str] = None,
        category: Optional[str] = None,
        source_map: Optional[Dict[str, str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.A = A
        self.B = B
        self.C = C
        self.mass_kg = mass_kg
        self.legislation = legislation
        self.category = category
        self.source_map = source_map if source_map is not None else {}
        self.warnings = warnings if warnings is not None else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "mass_kg": self.mass_kg,
            "legislation": self.legislation,
            "category": self.category,
            "source_map": self.source_map,
            "warnings": self.warnings,
        }

    def __repr__(self) -> str:
        return f"ResolvedBaseline({self.to_dict()})"


class RoadLoadComponent:
    """
    A single road-load contribution represented by A/B/C coefficients.

    In the first integration sprint this may simply be roadload_total.
    Later it can represent tire, aero, brakes, transmission, hubs, etc.
    """

    def __init__(
        self,
        name: str,
        A: float = 0.0,
        B: float = 0.0,
        C: float = 0.0,
        source: str = "unknown",
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.A = A
        self.B = B
        self.C = C
        self.source = source
        self.meta = meta if meta is not None else {}

    def copy(self, name: Optional[str] = None) -> "RoadLoadComponent":
        return RoadLoadComponent(
            name=name if name is not None else self.name,
            A=self.A,
            B=self.B,
            C=self.C,
            source=self.source,
            meta=dict(self.meta),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "source": self.source,
            "meta": self.meta,
        }

    def __repr__(self) -> str:
        return f"RoadLoadComponent({self.to_dict()})"


class ComponentSet:
    """
    Container for RoadLoadComponent objects.
    """

    def __init__(self, components: Optional[Dict[str, RoadLoadComponent]] = None):
        self.components = components if components is not None else {}

    def add(self, component: RoadLoadComponent):
        self.components[component.name] = component

    def get(self, name: str) -> Optional[RoadLoadComponent]:
        return self.components.get(name)

    def names(self) -> List[str]:
        return list(self.components.keys())

    def total_abc(self) -> Dict[str, float]:
        """
        Sums all components and returns equivalent A/B/C.
        """
        total_A = 0.0
        total_B = 0.0
        total_C = 0.0

        for component in self.components.values():
            total_A += component.A
            total_B += component.B
            total_C += component.C

        return {"A": total_A, "B": total_B, "C": total_C}

    def as_table(self) -> List[Dict[str, Any]]:
        return [component.to_dict() for component in self.components.values()]

    def to_dict(self) -> Dict[str, Any]:
        return {name: component.to_dict() for name, component in self.components.items()}

    def __repr__(self) -> str:
        return f"ComponentSet({self.as_table()})"


class EquivalentABC:
    """
    Final equivalent road-load result produced by the RoadLoad Engine.

    This object should be used as the input bridge to the VDE Core.
    """

    def __init__(
        self,
        A: float,
        B: float,
        C: float,
        mass_kg: float,
        component_table: Optional[List[Dict[str, Any]]] = None,
        warnings: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.A = A
        self.B = B
        self.C = C
        self.mass_kg = mass_kg
        self.component_table = component_table if component_table is not None else []
        self.warnings = warnings if warnings is not None else []
        self.meta = meta if meta is not None else {}

    def to_roadload_model(self) -> "RoadLoadModel":
        return RoadLoadModel(
            A=self.A,
            B=self.B,
            C=self.C,
            mass_kg=self.mass_kg,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "mass_kg": self.mass_kg,
            "component_table": self.component_table,
            "warnings": self.warnings,
            "meta": self.meta,
        }

    def __repr__(self) -> str:
        return f"EquivalentABC({self.to_dict()})"


class RoadLoadModel:
    """
    Operational road-load model.

    Force convention:
        F_road(v) = A + B*v + C*v^2

    where v is expected in kph because the A/B/C coefficients are usually
    stored in N, N/kph, and N/kph^2 in this project.
    """

    def __init__(
        self,
        A: float,
        B: float,
        C: float,
        mass_kg: float,
        name: str = "roadload_model",
    ):
        self.A = A
        self.B = B
        self.C = C
        self.mass_kg = mass_kg
        self.name = name

    def force(self, v_kph):
        """
        Computes road-load force in N.

        Supports scalar values and numpy/pandas array-like values.
        """
        return self.A + self.B * v_kph + self.C * (v_kph ** 2)

    def power(self, v_kph):
        """
        Computes road-load power in W.

        v_kph is converted internally to m/s.
        """
        v_mps = v_kph / 3.6
        return self.force(v_kph) * v_mps

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "mass_kg": self.mass_kg,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.summary()

    def __repr__(self) -> str:
        return f"RoadLoadModel({self.summary()})"

