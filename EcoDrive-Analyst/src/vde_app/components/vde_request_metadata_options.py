from __future__ import annotations

from copy import deepcopy

from src.vde_core.vde_setup_service import db_list_makes


METADATA_INHERIT_OPTION = "(inherit)"
METADATA_CUSTOM_MAKE_OPTION = "OTHER (TYPE MANUALLY)"

_EPA_CATEGORIES = [
    "UNKNOWN",
    "TWO SEATERS",
    "MINICOMPACT CARS",
    "SUBCOMPACT CARS",
    "COMPACT CARS",
    "MIDSIZE CARS",
    "LARGE CARS",
    "SMALL STATION WAGONS",
    "MIDSIZE STATION WAGONS",
    "SMALL SUVS",
    "STANDARD SUVS",
    "MINIVANS",
    "VANS",
    "SMALL PICKUP TRUCKS",
    "STANDARD PICKUP TRUCKS",
]
_WLTP_CATEGORIES = [
    "CLASS 1 (<850 KG)",
    "CLASS 2 (850-1220 KG)",
    "CLASS 3 (>1220 KG)",
]
_DEFAULT_MAKES = [
    "TOYOTA",
    "HONDA",
    "NISSAN",
    "MITSUBISHI",
    "MAZDA",
    "SUBARU",
    "HYUNDAI",
    "KIA",
    "VOLKSWAGEN",
    "AUDI",
    "BMW",
    "MERCEDES-BENZ",
    "PORSCHE",
    "PEUGEOT",
    "RENAULT",
    "CITROEN",
    "FIAT",
    "ALFA ROMEO",
    "VOLVO",
    "JAGUAR",
    "LAND ROVER",
    "SKODA",
    "SEAT",
    "OPEL",
    "FORD",
    "CHEVROLET",
    "DODGE",
    "CHRYSLER",
    "JEEP",
    "RAM",
    "CADILLAC",
    "BUICK",
    "GMC",
    "LINCOLN",
    "TESLA",
    "SUZUKI",
    "MINI",
    "SMART",
    "LEXUS",
    "INFINITI",
    "ACURA",
]
_CHOICE_OPTIONS = {
    "legislation": ["WLTP", "EPA", "ABNT (Brazil)"],
    "electrification": ["ICE", "HEV", "PHEV", "BEV"],
    "transmission_type": ["AT", "AMT", "CVT", "MT", "OT"],
    "drive_type": ["FWD", "RWD", "AWD", "4WD"],
    "fuel_type": ["GASOLINE", "ETHANOL", "FLEX", "DIESEL", "ELECTRIC", "CNG", "LPG", "HYDROGEN"],
}
_TEXT_FIELDS = {"name", "model", "description"}


def metadata_category_options(legislation: str) -> list[str]:
    if str(legislation or "").strip().upper() == "EPA":
        return list(_EPA_CATEGORIES)
    return list(_WLTP_CATEGORIES)


def metadata_choice_options(field_name: str, *, legislation: str, current_value: str = "") -> list[str] | None:
    field_key = str(field_name or "").strip()
    if field_key == "category":
        options = metadata_category_options(legislation)
    else:
        options = deepcopy(_CHOICE_OPTIONS.get(field_key))
    if options is None:
        return None
    value = _normalized_choice_value(field_key, current_value)
    if value and value not in options:
        options.append(value)
    return options


def metadata_make_options(*, legislation: str, category: str, current_value: str = "") -> list[str]:
    options = []
    try:
        options.extend(str(value or "").strip().upper() for value in db_list_makes(legislation, category))
    except Exception:
        pass
    options.extend(_DEFAULT_MAKES)
    value = str(current_value or "").strip().upper()
    if value:
        options.append(value)
    options.append(METADATA_CUSTOM_MAKE_OPTION)
    return [item for item in dict.fromkeys(item for item in options if item)]


def metadata_field_spec(field_name: str, *, legislation: str, category: str, current_value: str = "") -> dict:
    field_key = str(field_name or "").strip()
    if field_key == "make":
        options = [METADATA_INHERIT_OPTION] + metadata_make_options(
            legislation=legislation,
            category=category,
            current_value=current_value,
        )
        return {
            "widget": "select",
            "options": options,
            "allow_custom": True,
            "custom_option": METADATA_CUSTOM_MAKE_OPTION,
        }
    if field_key == "model_year":
        return {"widget": "text", "input_mode": "numeric"}
    if field_key in _TEXT_FIELDS:
        return {"widget": "text"}
    options = metadata_choice_options(field_key, legislation=legislation, current_value=current_value)
    if options is not None:
        return {"widget": "select", "options": [METADATA_INHERIT_OPTION] + options}
    return {"widget": "text"}


def metadata_override_value(field_name: str, raw_value, *, custom_value=None):
    field_key = str(field_name or "").strip()
    if field_key == "make" and raw_value == METADATA_CUSTOM_MAKE_OPTION:
        return str(custom_value or "").strip().upper()
    if raw_value == METADATA_INHERIT_OPTION:
        return ""
    if field_key == "model_year":
        text = str(raw_value or "").strip()
        return text
    if field_key in {"make", "category", "electrification", "transmission_type", "drive_type", "fuel_type"}:
        return _normalized_choice_value(field_key, raw_value)
    return str(raw_value or "").strip()


def _normalized_choice_value(field_name: str, value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if str(field_name or "").strip() == "legislation" and text.upper() == "ABNT (BRAZIL)":
        return "ABNT (Brazil)"
    if str(field_name or "").strip() == "legislation":
        return text.upper()
    return text.upper()


__all__ = [
    "METADATA_CUSTOM_MAKE_OPTION",
    "METADATA_INHERIT_OPTION",
    "metadata_category_options",
    "metadata_choice_options",
    "metadata_field_spec",
    "metadata_make_options",
    "metadata_override_value",
]
