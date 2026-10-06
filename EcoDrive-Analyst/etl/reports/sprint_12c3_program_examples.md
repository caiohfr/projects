# Sprint 12C.3 — Program Consolidation Examples

These are deterministic, manually inspectable audit examples. Names and technical descriptors come from the local EPA source; no web/RAG/LLM generation identity was used.

## EX-01 — STABLE_ARCHITECTURE_2020_2026

Source key: `ACURA|RDX AWD`

```text
ACURA RDX AWD
2020  models=RDX AWD | configs=1 VDEs=1 runs=2 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2021  models=RDX AWD | configs=1 VDEs=1 runs=2 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2022  models=RDX AWD | configs=1 VDEs=1 runs=2 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2023  models=RDX AWD | configs=1 VDEs=1 runs=2 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2024  models=RDX AWD | configs=1 VDEs=1 runs=2 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2025  models=RDX AWD | configs=1 VDEs=1 runs=2 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2026  models=RDX AWD | configs=1 VDEs=1 runs=2 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
-------------------------
All six contiguous transitions satisfy the safe technical + persistent EPA identity rule.
```

Evidence: All six contiguous transitions satisfy the safe technical + persistent EPA identity rule.

## EX-02 — STABLE_ARCHITECTURE_2020_2026

Source key: `ACURA|RDX AWD A SPEC`

```text
ACURA RDX AWD A-SPEC
2020  models=RDX AWD A SPEC | configs=1 VDEs=1 runs=3 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2021  models=RDX AWD A SPEC | configs=2 VDEs=2 runs=6 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2022  models=RDX AWD A SPEC | configs=2 VDEs=2 runs=6 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2023  models=RDX AWD A SPEC | configs=2 VDEs=2 runs=6 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2024  models=RDX AWD A SPEC | configs=1 VDEs=1 runs=3 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2025  models=RDX AWD A SPEC | configs=1 VDEs=1 runs=3 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2026  models=RDX AWD A SPEC | configs=1 VDEs=1 runs=3 | engine=KYF1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
-------------------------
All six contiguous transitions satisfy the safe technical + persistent EPA identity rule.
```

Evidence: All six contiguous transitions satisfy the safe technical + persistent EPA identity rule.

## EX-03 — STABLE_ARCHITECTURE_2020_2026

Source key: `ALFA ROMEO|GIULIA`

```text
Alfa Romeo Giulia
2020  models=GIULIA | configs=2 VDEs=2 runs=6 | engine=AA 100 | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=TIER 2 CERT GASOLINE
2021  models=GIULIA | configs=2 VDEs=2 runs=6 | engine=AA 100 | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=TIER 2 CERT GASOLINE
2022  models=GIULIA | configs=2 VDEs=2 runs=6 | engine=AA 100 | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=TIER 2 CERT GASOLINE
2023  models=GIULIA | configs=2 VDEs=2 runs=6 | engine=AA 100 | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=TIER 2 CERT GASOLINE
2024  models=GIULIA | configs=2 VDEs=2 runs=6 | engine=AA 100 | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=TIER 2 CERT GASOLINE
2025  models=GIULIA | configs=1 VDEs=1 runs=4 | engine=AA 100 | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=TIER 2 CERT GASOLINE
2026  models=GIULIA | configs=1 VDEs=1 runs=4 | engine=AA 100 | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=TIER 2 CERT GASOLINE
-------------------------
All six contiguous transitions satisfy the safe technical + persistent EPA identity rule.
```

Evidence: All six contiguous transitions satisfy the safe technical + persistent EPA identity rule.

## EX-04 — OBVIOUS_TECHNICAL_GENERATION_BREAK

Source key: `PREVIEW-BOUNDARY-469AE04681C1E4E2`

```text
AUDI A6 Allroad
2020  models=A6 ALLROAD | configs=1 VDEs=2 runs=5 | engine=DLZA | trans=AUTOMATED MANUAL SELECTABLE E G AUTOMATED MANUAL WITH PADDLES | drive=ALL WHEEL DRIVE | fuel=COLD CO PREMIUM TIER 2;TIER 2 CERT GASOLINE
2021  models=A6 ALLROAD | configs=1 VDEs=2 runs=5 | engine=DLZA | trans=AUTOMATED MANUAL SELECTABLE E G AUTOMATED MANUAL WITH PADDLES | drive=ALL WHEEL DRIVE | fuel=COLD CO PREMIUM TIER 2;TIER 2 CERT GASOLINE
2024  models=A6 ALLROAD | configs=3 VDEs=3 runs=6 | engine=DLZA | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2025  models=A6 ALLROAD | configs=3 VDEs=3 runs=6 | engine=DLZA | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2026  models=A6 ALLROAD | configs=1 VDEs=2 runs=5 | engine=DLZA | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=COLD CO PREMIUM TIER 2;TIER 2 CERT GASOLINE
-------------------------
2021->2024: No exact architecture continuity; 3 populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed.
```

Evidence: 2021->2024: No exact architecture continuity; 3 populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed.

## EX-05 — OBVIOUS_TECHNICAL_GENERATION_BREAK

Source key: `PREVIEW-BOUNDARY-3C311FD39E6D8C9D`

```text
AUDI Q3
2020  models=Q3 | configs=1 VDEs=2 runs=5 | engine=DHHA | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=COLD CO PREMIUM TIER 2;TIER 2 CERT GASOLINE
2021  models=Q3 | configs=1 VDEs=1 runs=2 | engine=DHHA | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2023  models=Q3 | configs=2 VDEs=2 runs=4 | engine=DSNA;DSPA | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2024  models=Q3 | configs=2 VDEs=2 runs=4 | engine=DSNA;DSPA | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2026  models=Q3 | configs=1 VDEs=2 runs=5 | engine=DRND | trans=AUTOMATED MANUAL SELECTABLE E G AUTOMATED MANUAL WITH PADDLES | drive=ALL WHEEL DRIVE | fuel=COLD CO PREMIUM TIER 2;TIER 2 CERT GASOLINE
-------------------------
2024->2026: No exact architecture continuity; 3 populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed.
```

Evidence: 2024->2026: No exact architecture continuity; 3 populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed.

## EX-06 — OBVIOUS_TECHNICAL_GENERATION_BREAK

Source key: `PREVIEW-BOUNDARY-AA4156D51F14BAA7`

```text
AUDI S7
2020  models=S7 | configs=1 VDEs=2 runs=5 | engine=DKMB | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=COLD CO PREMIUM TIER 2;TIER 2 CERT GASOLINE
2021  models=S7 | configs=1 VDEs=2 runs=5 | engine=DKMB | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=COLD CO PREMIUM TIER 2;TIER 2 CERT GASOLINE
2022  models=S7 | configs=1 VDEs=2 runs=5 | engine=DKMB | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=COLD CO PREMIUM TIER 2;TIER 2 CERT GASOLINE
2023  models=S7 | configs=1 VDEs=1 runs=2 | engine=DKMB | trans=AUTOMATED MANUAL | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2024  models=S7 | configs=2 VDEs=2 runs=5 | engine=DKMB | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2025  models=S7 | configs=2 VDEs=2 runs=5 | engine=DKMB | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
-------------------------
2022->2023: No exact architecture continuity; 3 populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed.
```

Evidence: 2022->2023: No exact architecture continuity; 3 populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed.

## EX-07 — OBVIOUS_TECHNICAL_GENERATION_BREAK

Source key: `PREVIEW-BOUNDARY-D41E255EB9E3B79A`

```text
AUDI SQ7
2020  models=SQ7 | configs=1 VDEs=1 runs=2 | engine=DCUE | trans=AUTOMATED MANUAL | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2021  models=SQ7 | configs=1 VDEs=1 runs=2 | engine=DCUE | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2022  models=SQ7 | configs=1 VDEs=1 runs=2 | engine=DCUE | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2023  models=SQ7 | configs=1 VDEs=2 runs=5 | engine=DWRB | trans=AUTOMATED MANUAL | drive=ALL WHEEL DRIVE | fuel=COLD CO PREMIUM TIER 2;TIER 2 CERT GASOLINE
2024  models=SQ7 | configs=1 VDEs=1 runs=2 | engine=DWRB | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2025  models=SQ7 | configs=1 VDEs=1 runs=2 | engine=DWRB | trans=SEMI AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
-------------------------
2022->2023: No exact architecture continuity; 3 populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed.
```

Evidence: 2022->2023: No exact architecture continuity; 3 populated architecture domains reset. No OEM generation evidence is available, so this is not confirmed.

## EX-08 — MULTIPLE_CONFIGURATIONS_ONE_LIKELY_PROGRAM

Source key: `PREVIEW-SEMANTIC-PROGRAM-577652E6C4713FB0`

```text
Volkswagen ID.4 Pro
2021  models=ID 4 PRO | configs=4 VDEs=4 runs=8 | engine=EBJA | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=ELECTRICITY
2022  models=ID 4 PRO | configs=1 VDEs=1 runs=4 | engine=EBJA | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=ELECTRICITY
-------------------------
5 configurations remain distinct below one proposed Program parent.
```

Evidence: 5 configurations remain distinct below one proposed Program parent.

## EX-09 — MULTIPLE_CONFIGURATIONS_ONE_LIKELY_PROGRAM

Source key: `PREVIEW-SEMANTIC-PROGRAM-61ADF22479DB84C3`

```text
NISSAN PATHFINDER 4WD ROCK CREEK
2023  models=PATHFINDER 4WD ROCK CREEK | configs=1 VDEs=1 runs=3 | engine=AQ35DAA4 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2024  models=PATHFINDER 4WD ROCK CREEK | configs=1 VDEs=1 runs=3 | engine=AQ35DAA4 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2025  models=PATHFINDER 4WD ROCK CREEK | configs=1 VDEs=1 runs=3 | engine=AQ35DAA4 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2026  models=PATHFINDER 4WD ROCK CREEK | configs=1 VDEs=1 runs=3 | engine=AQ35DAA4 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
-------------------------
4 configurations remain distinct below one proposed Program parent.
```

Evidence: 4 configurations remain distinct below one proposed Program parent.

## EX-10 — MULTIPLE_CONFIGURATIONS_ONE_LIKELY_PROGRAM

Source key: `PREVIEW-SEMANTIC-PROGRAM-D24CE0D4FD30CAA3`

```text
Ford F150 PICKUP 2WD
2024  models=F150 PICKUP 2WD | configs=2 VDEs=2 runs=3 | engine=RTFDAVND0005;RTFDBMNC0004 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE REAR;PART TIME 4 WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2025  models=F150 PICKUP 2WD | configs=2 VDEs=2 runs=10 | engine=RTFDAVND0005;RTFDBMNC0004 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE REAR;PART TIME 4 WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2026  models=F150 PICKUP 2WD | configs=1 VDEs=1 runs=4 | engine=RTFDBMNC0004 | trans=SEMI AUTOMATIC | drive=PART TIME 4 WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
-------------------------
5 configurations remain distinct below one proposed Program parent.
```

Evidence: 5 configurations remain distinct below one proposed Program parent.

## EX-11 — MULTIPLE_CONFIGURATIONS_ONE_LIKELY_PROGRAM

Source key: `PREVIEW-SEMANTIC-PROGRAM-91C71314139A700B`

```text
Ferrari SF90 Spider
2022  models=SF90 SPIDER | configs=1 VDEs=1 runs=4 | engine=F154 FA | trans=AUTOMATED MANUAL | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2023  models=SF90 SPIDER | configs=1 VDEs=1 runs=4 | engine=F154 FA | trans=AUTOMATED MANUAL | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2024  models=SF90 SPIDER | configs=1 VDEs=1 runs=4 | engine=F154 FA | trans=AUTOMATED MANUAL | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
2025  models=SF90 SPIDER | configs=1 VDEs=1 runs=4 | engine=F154 FA | trans=AUTOMATED MANUAL | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
-------------------------
4 configurations remain distinct below one proposed Program parent.
```

Evidence: 4 configurations remain distinct below one proposed Program parent.

## EX-12 — COMMERCIAL_RENAME_OR_TRIM_NOMENCLATURE

Source key: `PREVIEW-RENAME-5F19C9EE3522AB3B`

```text
BMW i4 eDrive 35 Gran Coupe (18'' Wheels)
2025  models=I4 EDRIVE 35 GRAN COUPE 18 WHEELS | configs=1 VDEs=1 runs=2 | engine=HA0001N0G26S | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=ELECTRICITY
2026  models=I4 EDRIVE35 GRAN COUPE 18 WHEELS | configs=1 VDEs=1 runs=1 | engine=HA0001N1G26S | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=ELECTRICITY
-------------------------
Different commercial strings share an exact architecture signature in adjacent years; review only, never a deterministic merge.
```

Evidence: Different commercial strings share an exact architecture signature in adjacent years; review only, never a deterministic merge.

## EX-13 — COMMERCIAL_RENAME_OR_TRIM_NOMENCLATURE

Source key: `PREVIEW-RENAME-69A42B0EF395323D`

```text
BMW i4 eDrive 35 Gran Coupe (19'' Wheels)
2025  models=I4 EDRIVE 35 GRAN COUPE 19 WHEELS | configs=2 VDEs=2 runs=4 | engine=HA0001N0G26S | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=ELECTRICITY
2026  models=I4 EDRIVE35 GRAN COUPE 19 WHEELS | configs=2 VDEs=2 runs=2 | engine=HA0001N1G26S | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=ELECTRICITY
-------------------------
Different commercial strings share an exact architecture signature in adjacent years; review only, never a deterministic merge.
```

Evidence: Different commercial strings share an exact architecture signature in adjacent years; review only, never a deterministic merge.

## EX-14 — COMMERCIAL_RENAME_OR_TRIM_NOMENCLATURE

Source key: `PREVIEW-RENAME-C40415BE67CB167C`

```text
BMW i4 eDrive 40 Gran Coupe (18'' Wheels)
2025  models=I4 EDRIVE 40 GRAN COUPE 18 WHEELS | configs=1 VDEs=1 runs=2 | engine=HA0001N0G26S1 | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=ELECTRICITY
2026  models=I4 EDRIVE40 GRAN COUPE 18 WHEELS | configs=1 VDEs=1 runs=1 | engine=HA0001N1G26S1 | trans=AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=ELECTRICITY
-------------------------
Different commercial strings share an exact architecture signature in adjacent years; review only, never a deterministic merge.
```

Evidence: Different commercial strings share an exact architecture signature in adjacent years; review only, never a deterministic merge.

## EX-15 — SAME_COMMERCIAL_NAME_DISTINCT_GENERATIONS

Source key: `PREVIEW-BOUNDARY-011F21B466D91898`

```text
BUICK ENCLAVE FWD
2024  models=ENCLAVE FWD | configs=1 VDEs=1 runs=2 | engine=35 | trans=AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2026  models=ENCLAVE FWD | configs=1 VDEs=1 runs=2 | engine=1 | trans=AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
-------------------------
Same normalized name, but 3 architecture domains reset across the candidate boundary.
```

Evidence: Same normalized name, but 3 architecture domains reset across the candidate boundary.

## EX-16 — SAME_COMMERCIAL_NAME_DISTINCT_GENERATIONS

Source key: `PREVIEW-BOUNDARY-1DA4DE4F14F554C7`

```text
BUICK ENVISION FWD
2020  models=ENVISION FWD | configs=1 VDEs=1 runs=3 | engine=9 | trans=AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2021  models=ENVISION FWD | configs=1 VDEs=1 runs=2 | engine=3 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2022  models=ENVISION FWD | configs=1 VDEs=1 runs=2 | engine=3 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2023  models=ENVISION FWD | configs=2 VDEs=2 runs=4 | engine=3;7 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
-------------------------
Same normalized name, but 3 architecture domains reset across the candidate boundary.
```

Evidence: Same normalized name, but 3 architecture domains reset across the candidate boundary.

## EX-17 — SAME_COMMERCIAL_NAME_DISTINCT_GENERATIONS

Source key: `PREVIEW-BOUNDARY-9611AA4A56501B1A`

```text
Ford F150 Super Crew Cab 4x4
2020  models=F150 SUPER CREW CAB 4X4 | configs=1 VDEs=1 runs=4 | engine=JTFCAXNC25 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=E85 85 ETHANOL 15 EPA UNLEADED GASOLINE;TIER 2 CERT GASOLINE
2021  models=F150 SUPER CREW CAB 4X4 | configs=2 VDEs=2 runs=36 | engine=MTFDAVNC08;MTFDAVNS23 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=E85 85 ETHANOL 15 EPA UNLEADED GASOLINE;TIER 2 CERT GASOLINE
2022  models=F150 SUPER CREW CAB 4X4 | configs=2 VDEs=2 runs=30 | engine=MTFDAVNC08;MTFDAVNS23 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=E85 85 ETHANOL 15 EPA UNLEADED GASOLINE;TIER 2 CERT GASOLINE
2023  models=F150 SUPER CREW CAB 4X4 | configs=4 VDEs=4 runs=38 | engine=MTFDAVNB16;MTFDAVNS23;PTFDAVND06;PTFDAVNT06 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE REAR | fuel=E85 85 ETHANOL 15 EPA UNLEADED GASOLINE;TIER 2 CERT GASOLINE
-------------------------
Same normalized name, but 3 architecture domains reset across the candidate boundary.
```

Evidence: Same normalized name, but 3 architecture domains reset across the candidate boundary.

## EX-18 — SAME_COMMERCIAL_NAME_DISTINCT_GENERATIONS

Source key: `PREVIEW-BOUNDARY-1151D455FC486893`

```text
Ford MUSTANG MACH-E RWD EXTENDED
2024  models=MUSTANG MACH E RWD EXTENDED | configs=1 VDEs=1 runs=2 | engine=RCGWEHNH0002 | trans=AUTOMATIC | drive=4 WHEEL DRIVE | fuel=ELECTRICITY
2025  models=MUSTANG MACH E RWD EXTENDED | configs=1 VDEs=1 runs=1 | engine=SCGWEHNH00 | trans=CONTINUOUSLY VARIABLE | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
-------------------------
Same normalized name, but 3 architecture domains reset across the candidate boundary.
```

Evidence: Same normalized name, but 3 architecture domains reset across the candidate boundary.

## EX-19 — ONE_YEAR_ONLY_MODEL

Source key: `ACURA|MDX AWD A SPEC`

```text
ACURA MDX AWD A-spec
2020  models=MDX AWD A SPEC | configs=1 VDEs=1 runs=2 | engine=KBN1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
-------------------------
No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span.
```

Evidence: No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span.

## EX-20 — ONE_YEAR_ONLY_MODEL

Source key: `ACURA|RLX`

```text
ACURA RLX
2020  models=RLX | configs=1 VDEs=1 runs=2 | engine=J9P1A1 | trans=SEMI AUTOMATIC | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
-------------------------
No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span.
```

Evidence: No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span.

## EX-21 — ONE_YEAR_ONLY_MODEL

Source key: `ACURA|RLX HYBRID`

```text
ACURA RLX HYBRID
2020  models=RLX HYBRID | configs=1 VDEs=1 runs=2 | engine=J9S1D1 | trans=AUTOMATED MANUAL SELECTABLE E G AUTOMATED MANUAL WITH PADDLES | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
-------------------------
No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span.
```

Evidence: No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span.

## EX-22 — ONE_YEAR_ONLY_MODEL

Source key: `ACURA|TLX`

```text
ACURA TLX
2020  models=TLX | configs=4 VDEs=4 runs=8 | engine=JDF1D1 | trans=AUTOMATED MANUAL SELECTABLE E G AUTOMATED MANUAL WITH PADDLES | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
-------------------------
No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span.
```

Evidence: No cross-year evidence exists; retain the source-scoped fallback and do not infer a generation span.

## EX-23 — MODEL_YEAR_GAP

Source key: `PREVIEW-BOUNDARY-53001CCE79DA57A1`

```text
ACURA ZDX AWD
2024  models=ZDX AWD | configs=1 VDEs=1 runs=2 | engine=1 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
2026  models=ZDX AWD | configs=1 VDEs=1 runs=1 | engine=1 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
-------------------------
1 missing model year(s); status=UNRESOLVED, consolidation=UNRESOLVED.
```

Evidence: 1 missing model year(s); status=UNRESOLVED, consolidation=UNRESOLVED.

## EX-24 — MODEL_YEAR_GAP

Source key: `PREVIEW-BOUNDARY-78A32580B13DFE39`

```text
ACURA ZDX AWD TYPE S
2024  models=ZDX AWD TYPE S | configs=1 VDEs=1 runs=2 | engine=1 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
2026  models=ZDX AWD TYPE S | configs=1 VDEs=1 runs=1 | engine=1 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
-------------------------
1 missing model year(s); status=UNRESOLVED, consolidation=UNRESOLVED.
```

Evidence: 1 missing model year(s); status=UNRESOLVED, consolidation=UNRESOLVED.

## EX-25 — MODEL_YEAR_GAP

Source key: `PREVIEW-BOUNDARY-3C891EEFED0C008B`

```text
ACURA ZDX RWD
2024  models=ZDX RWD | configs=1 VDEs=1 runs=2 | engine=1 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
2026  models=ZDX RWD | configs=1 VDEs=1 runs=2 | engine=1 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
-------------------------
1 missing model year(s); status=UNRESOLVED, consolidation=UNRESOLVED.
```

Evidence: 1 missing model year(s); status=UNRESOLVED, consolidation=UNRESOLVED.

## EX-26 — POWERTRAIN_FAMILY_BEV

Source key: `ACURA|ZDX AWD`

```text
ACURA ZDX AWD
2024  models=ZDX AWD | configs=1 VDEs=1 runs=2 | engine=1 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
2026  models=ZDX AWD | configs=1 VDEs=1 runs=1 | engine=1 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
-------------------------
Audit classification=BEV; naming signals are context only and do not establish Program identity.
```

Evidence: Audit classification=BEV; naming signals are context only and do not establish Program identity.

## EX-27 — POWERTRAIN_FAMILY_ICE

Source key: `ACURA|ADX AWD`

```text
ACURA ADX AWD
2025  models=ADX AWD | configs=2 VDEs=2 runs=4 | engine=SVJ2C1 | trans=SELECTABLE CONTINUOUSLY VARIABLE E G CVT WITH PADDLES | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
2026  models=ADX AWD | configs=2 VDEs=2 runs=4 | engine=SVJ2C1 | trans=SELECTABLE CONTINUOUSLY VARIABLE E G CVT WITH PADDLES | drive=2 WHEEL DRIVE FRONT | fuel=TIER 2 CERT GASOLINE
-------------------------
Audit classification=ICE; naming signals are context only and do not establish Program identity.
```

Evidence: Audit classification=ICE; naming signals are context only and do not establish Program identity.

## EX-28 — POWERTRAIN_FAMILY_HEV_OR_PHEV_NAME_SIGNAL

Source key: `ACURA|RLX HYBRID`

```text
ACURA RLX HYBRID
2020  models=RLX HYBRID | configs=1 VDEs=1 runs=2 | engine=J9S1D1 | trans=AUTOMATED MANUAL SELECTABLE E G AUTOMATED MANUAL WITH PADDLES | drive=ALL WHEEL DRIVE | fuel=TIER 2 CERT GASOLINE
-------------------------
Audit classification=HEV_OR_PHEV_NAME_SIGNAL; naming signals are context only and do not establish Program identity.
```

Evidence: Audit classification=HEV_OR_PHEV_NAME_SIGNAL; naming signals are context only and do not establish Program identity.

## EX-29 — INSUFFICIENT_SOURCE_DATA

Source key: `2022|LUCID AIR DREAM P`

```text
2022 Lucid Air Dream P
2022  models=LUCID AIR DREAM P | configs=2 VDEs=2 runs=4 | engine=ZA2 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
2023  models=LUCID AIR DREAM P | configs=1 VDEs=1 runs=2 | engine=ZA2 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
-------------------------
Descriptor coverage or represented-make quality is insufficient; no automatic consolidation is allowed.
```

Evidence: Descriptor coverage or represented-make quality is insufficient; no automatic consolidation is allowed.

## EX-30 — INSUFFICIENT_SOURCE_DATA

Source key: `2022|LUCID AIR DREAM R`

```text
2022 Lucid Air Dream R
2022  models=LUCID AIR DREAM R | configs=2 VDEs=2 runs=4 | engine=ZA2 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
-------------------------
Descriptor coverage or represented-make quality is insufficient; no automatic consolidation is allowed.
```

Evidence: Descriptor coverage or represented-make quality is insufficient; no automatic consolidation is allowed.

## EX-31 — INSUFFICIENT_SOURCE_DATA

Source key: `2022|LUCID AIR GRAND TOURING`

```text
2022 Lucid Air Grand Touring
2022  models=LUCID AIR GRAND TOURING | configs=2 VDEs=2 runs=4 | engine=ZA2 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
2023  models=LUCID AIR GRAND TOURING | configs=2 VDEs=2 runs=4 | engine=ZA2 | trans=AUTOMATIC | drive=ALL WHEEL DRIVE | fuel=ELECTRICITY
-------------------------
Descriptor coverage or represented-make quality is insufficient; no automatic consolidation is allowed.
```

Evidence: Descriptor coverage or represented-make quality is insufficient; no automatic consolidation is allowed.
