from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..contracts import ResearchStatus, TechnicalResearchRequest
from ..core.validation import public_request_fields


TRANSMISSION_TARGET_FIELDS = (
    "transmission_hardware_designation",
    "transmission_family",
    "transmission_manufacturer",
    "transmission_supplier",
    "transmission_marketing_description",
    "transmission_type_normalized",
    "gears",
    "gear_ratios",
    "final_drive_ratio",
    "drive_variant",
    "application_years",
    "drag_coefficient_cd",
    "frontal_area_m2",
    "drag_area_cda_m2",
    "tire_size_front",
    "tire_size_rear",
    "tire_size_general",
)


@dataclass(frozen=True)
class TransmissionResearchProfile:
    domain: str = "TRANSMISSION"
    identity_field: str = "transmission_hardware_designation"

    def build_queries(self, request: TechnicalResearchRequest, *, round_number: int) -> Sequence[str]:
        fields = public_request_fields(request)
        vehicle = " ".join(
            str(fields.get(name, ""))
            for name in ("model_year", "make", "model", "trim", "engine")
            if fields.get(name) not in (None, "")
        )
        make_model = " ".join(
            str(fields.get(name, ""))
            for name in ("make", "model")
            if fields.get(name) not in (None, "")
        )
        if round_number <= 1:
            queries = [
                f'site:press.bmwgroup.com {vehicle} technical data transmission',
                f'site:press.bmwgroup.com {make_model} specifications PDF transmission',
                f'site:zf.com {make_model} transmission technical',
                f'site:bmwgroup.com {vehicle} technical data transmission PDF',
                f'{vehicle} gearbox hardware designation technical specifications PDF',
            ]
        elif round_number == 2:
            # This query must be inside the configured five-query window.
            queries = [
                f'site:bmwtechinfo.bmwgroup.com "{make_model}" transmission',
                f'site:press.bmwgroup.com "{vehicle}" attachment technical data',
                f'site:press.bmwgroup.com "{make_model}" transmission gear ratios PDF',
                f'"{vehicle}" transmission supplier gearbox code',
                f'"{make_model}" transmission final drive tire size Cd technical data',
            ]
        else:
            queries = [
                f'site:bmwtechinfo.bmwgroup.com "{vehicle}" training transmission',
                f'site:press.bmwgroup.com "{vehicle}" transmission final drive PDF',
                f'site:zf.com "{make_model}" gear ratios transmission PDF',
                f'"{vehicle}" gearbox code gear ratios service manual',
                f'"{make_model}" technical specifications transmission supplier Cd tires',
            ]
        return tuple(dict.fromkeys(query.strip() for query in queries if query.strip()))

    def evidence_is_sufficient(self, status: str) -> bool:
        return status in {ResearchStatus.SUPPORTED.value, ResearchStatus.CONFLICTING_EVIDENCE.value}
