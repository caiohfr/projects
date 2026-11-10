# Sprint 14A-03 — Contract & Dependency Audit · Handoff

**Date:** 2026-11-07  
**Branch/commit inspecionado:** `main@fca9a9ad189fe7e6930e987e29fee5717c0d532e`  
**Entrega:** documentação/proposta, sem alterações em Python/DB; não constitui encerramento do Architecture Gate.

## Evidência realizada

- **INSPECTION_ONLY:** `roadload/models.py`, `roadload/engine.py`, `roadload/tire_model.py`, `roadload/app_service.py`, `roadload/__init__.py`, `tire_roadload_service.py`, `vde_workflow_service.py`, `vde_request_resolver.py`, `vehicle_demand/{contracts,adapters,engine,physics,load_case}.py`, `comparison_report_service.py` e inspeções de busca para callers/testes.
- **TARGET_SPECIFIED:** arquitetura C+, contrato black-box de contribuição, responsabilidades, invariantes e plano de compatibilidade.
- **GAP:** inventário global AST/import graph completo, golden baselines, benchmark e execução de testes.

## Artefatos

- [`../architecture/roadload_dependency_audit_v0.1.md`](../architecture/roadload_dependency_audit_v0.1.md) — CURRENT, call edges conhecidos, integração incremental.
- [`../architecture/roadload_interface_contract_v0.1.md`](../architecture/roadload_interface_contract_v0.1.md) — TARGET draft (Tire → Roadload → Vehicle Demand).
- Baseline de referência existente: `../architecture/roadload_system_design_v0.1.md`, `../architecture/roadload_traceability_v0.1.md`, ADR-14A-005 e `SPRINT_14A_ROADLOAD_DESIGN_GATE.md`.

## Decisões arquiteturais recomendadas

1. Uma classe principal `RoadLoadModel`; contratos leves de contribuições, sem segunda engine universal.
2. `RoadLoadModel` sem DB e sem import de Vehicle Demand; service coordena e adapters traduzem contratos.
3. `Tire` fornece resultado resolvido; Roadload **não** recalcula SAE/ISO.
4. TOTAL/NET e VDE canônico preservados; residual assinado e `missing ≠ 0`.
5. API de mudanças de componente deve separar `reatribuir breakdown` vs `aplicar cenário físico` antes de congelar assinatura.

## Próximo incremento — 14A-04: baseline e medições

**Codex task outline (proposta de execução, sem rodar aqui):**

1. Inventariar automaticamente todas as referências/imports `RoadLoadModel`, `RoadLoadComponent`, `ComponentSet`, `EquivalentABC`, `run_roadload_scenario`, `_force_at`, `compute_vde_series`, funções TOTAL/NET e tire adapters com AST. Incluir chamadas via `__init__.py`, alias e dynamic refs marcadas.
2. Capturar fixtures golden **sintéticas ou públicas** a partir do comportamento atual, cobrindo VDE Setup v2.2 Preview, Roadload Engine, Load Case fixed/sweep/cycle, Vehicle Demand TOTAL/NET e Component Tire. Registrar exato commit/environment.
3. Executar suites existentes relacionadas, **reportando PASS/FAIL/SKIP por teste**; não confundir tests presentes com tests executados; não modificar física para forçar testes verdes.
4. Criar `scripts/benchmarks/bench_roadload_baseline.py` ou caminho já convencional de benchmarks. Medir `perf_counter`, tempo de CPU, `tracemalloc`, carga 1/10/100/1000, n=100..10_000, k=1..30, query counts SQLite; registrar máquina/versões/seed/variação de tempo/cache quente/frio.
5. Registrar outputs em `docs/architecture/roadload_benchmark_baseline_v0.1.md` (tabelas **reais**, não notas imaginadas). Nenhum DB write na execução de modelos; usar QA copy isolada quando houver teste de persistência.
6. Finalizar assinatura/mutabilidade dos métodos e tolerâncias; atualizar ADR/traceability apenas com evidência real.

## Gate

- `Documentation/Architecture`: **DRAFT READY FOR REVIEW**.
- `Caller audit`: **PARTIAL — AST PENDING**.
- `Golden/performance`: **NOT EXECUTED**.
- `Implementation`: **BLOCKED UNTIL GATE**, em respeito à Sprint 14 design-only e ao roadmap de Manual/Imported FuelCons.
