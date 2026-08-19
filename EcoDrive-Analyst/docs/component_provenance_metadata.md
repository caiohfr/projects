# Component Provenance Metadata

Component records may carry optional provenance metadata that describes what the component ABC represents before the value is consumed by a VDE scenario. The metadata is descriptive and audit-oriented; it does not change the existing roadload or VDE math.

## Procedure Boundary

EcoDrive consumes representative component records after they have been validated outside the application. The expected flow is:

Historical PL Excel -> PL analysis notebooks -> validated comparable populations -> representative component ABC -> EcoDrive Component DB -> VDE Request Lookup -> ABC delta -> Roadload / VDE

The EcoDrive package does not import the historical Excel file and does not depend on the PL analysis workspace. It stores the selected component reference and the provenance fields that explain the boundary of that reference.

## Stored Fields

The optional fields are:

- `component_type`
- `component_position`
- `driveline_architecture`
- `physical_boundary`
- `configuration_from`
- `configuration_to`
- `test_condition_type`
- `test_method`
- `hardware_reference`
- `source_reference`
- `net_bridge_eligible`

Records that do not include these fields remain valid. Missing values are normalized to blank strings.

## Methodology Notes

The PL methodology reference, `Parasitic_Loss_Testing_Methodology(1).md`, defines the upstream expectation that comparable populations should use consistent physical boundaries and test-condition interpretation. EcoDrive keeps only concise metadata needed to audit the consumed reference.

Brake records can distinguish `BRAKE_BASELINE_AS_RECEIVED` from `BRAKE_STANDARD`. Transmission records can mark whether the source boundary is eligible for TOTAL-to-NET bridging with `net_bridge_eligible`. Axle and hub records can distinguish `AXLE`, `HUB`, and `AXLE_HUB_COMBINED`, including position. Generic parasitic records represent `OTHER_RESIDUAL_COMPONENT_LOSSES` and should not be used for known transmission, brake, axle, or hub boundaries.

The resolver continues to apply component changes through the existing ABC delta rule:

`ABC_TOTAL_result = ABC_TOTAL_source + Delta_Component_ABC`

The provenance metadata is saved in component snapshots so a saved VDE request remains reproducible even if the live component DB record is changed later.
