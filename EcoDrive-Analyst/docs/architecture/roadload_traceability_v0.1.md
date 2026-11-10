# EcoDrive — Roadload Requirement-to-Test Traceability v0.1

**Estado:** DRAFT / INSPECTION ONLY, 2026-11-07. Requisitos candidatos, não baselined nem comprovados. A matriz separa implementação existente de intenção TARGET; nomes de testes são indicações de cobertura a confirmar por execução e revisão de asserts.

## 1. Requisitos funcionais (FR)

| ID | Requisito verificável | Alocação preferida | Verificação planejada | Evidência atual |
|---|---|---|---|---|
| `RL-FR-01` | Avaliar `F=A+Bv+Cv²` em velocidades escalares e vetoriais, com unidades fixadas | `RoadLoadModel.force()` | Comparação analítica e arrays | `force` existente; INSPECTION ONLY |
| `RL-FR-02` | Avaliar `P=F*v/3.6` em W com v em km/h | `RoadLoadModel.power()` | Teste analítico, v=0 e v=100 | `power` existente; INSPECTION ONLY |
| `RL-FR-03` | Preservar ABC autoritativo e associar contribuições conhecidas sem reescrever TOTAL | Roadload Model + resolver | Teste baseline medido, componentes incompletos | `vehicle_demand/engine.py` mantém breakdown parcial; TARGET GAP |
| `RL-FR-04` | Montar ABC sintético somente de contribuições elegíveis, sem duplicação | RoadLoadModel + component resolvers | Golden build-up, overlaps, modos | `roadload/engine.py` e `vde_workflow_service.py` fazem build-up distintos; GAP de unificação |
| `RL-FR-05` | Gerar força por componente, frações físicas e residual assinado por velocidade | `breakdown()`, `fractions()`, `reconcile()` | Paridade `known + residual = total` | `vehicle_demand` calcula rolling/aero/residual; API RoadLoad GAP |
| `RL-FR-06` | Não confundir frações ABC com frações físicas por subsistema | `abc_terms()`, `fractions(kind=...)` | Teste A/B/C em v=100; asserts de rótulo | GAP |
| `RL-FR-07` | Propagar provenance, missing ≠ zero, `MODEL_BUILT` ≠ medido | DTO + service + model | Serialization/availability/edge tests | Contratos existentes parciais; TARGET GAP |
| `RL-FR-08` | Respeitar TOTAL/NET sem substituição silenciosa, e sem dupla contagem de trans/hubs/axle | Boundary resolver, application service | Teste TOTAL, NET ausente, overlaps | `test_vehicle_demand_engine.py` e contrato `comparison_report_service.py`; EXECUTION PENDING |
| `RL-FR-09` | Preservar VDE canônico a partir de ABC, massa e ciclo, incluindo inércia | `vehicle_demand/engine.py` e `vde_calc.py` | Golden parity | `test_vehicle_demand_engine.py`, `test_vehicle_demand_integration.py`; EXECUTION PENDING |
| `RL-FR-10` | Permitir cenários temporários sem persistir ou alterar baseline | Load Case service | No write and repeat determinism | `tests/test_vehicle_demand_load_case.py`; EXECUTION PENDING |

## 2. Restrições não funcionais (NFR)

| ID | Requisito / métrica | Alocação | Método | Situação |
|---|---|---|---|---|
| `RL-NFR-01` | `RoadLoadModel` sem imports/consultas de SQLite, Streamlit, Plotly | Domain boundary | Import inspection + fake repository | TARGET GATE |
| `RL-NFR-02` | Não aumentar consultas SQL desnecessariamente em `load_vehicle` e resoluções de pneus | Service + repositories | SQL query count antes/depois | BASELINE NOT MEASURED |
| `RL-NFR-03` | Vetorização e custo O(n) para curvas e O(nk) para breakdown, sob dimensões n/k | Model | Microbench + perf profiling | BASELINE NOT MEASURED |
| `RL-NFR-04` | Regressores de tempo, memória e latência medidos com protocolo reproduzível | Benchmark harness | p50/p95, wall/CPU, peak memory | BASELINE NOT MEASURED |
| `RL-NFR-05` | APIs legadas preservadas no ciclo de migração | Adapter/compatibility | Import smoke, antigas assinaturas | TARGET GATE |
| `RL-NFR-06` | Nenhuma alteração de física ou DB durante Sprint 14 design-only | Change control | Diff de código, inspeção | DOCUMENTS-ONLY |

## 3. Invariantes / testes existentes a reaproveitar

- `tests/test_vehicle_demand_engine.py` — conferir `CanonicalVdeReconciliationTests`, `TotalNetIndependenceTests`, `RoadloadClosureTests`, `NegativeResidualTests`.
- `tests/test_vehicle_demand_integration.py` — integração/ponderação ciclo (EPA); revisar se a expectativa golden ainda é válida.
- `tests/test_vehicle_demand_load_case.py` — coerência massa, trailer, grade, preservação de baseline, sem DB writes, repetibilidade; checar também reads.
- `tests/test_roadload_engine.py` — compatibilidade `run_roadload_scenario/EquivalentABC`.
- `tests/test_roadload_load_case_lab_ui.py` — UI smoke via `AppTest`; **AppTest não é smoke manual de navegador**.
- `tests/test_vde_request_resolver.py`, `tests/test_vde_workflow_service.py` e suites de persistência v2.2 — rotas de VDE Setup e NET.

Lista fundamentada na revisão e navegação do repositório; **teste existente não significa teste executado nem requisito totalmente coberto**. Executar e atribuir evidência a cada item na etapa de implementação.

## 4. Acceptance cases mínimos a construir

| Caso | Inputs | Resultado esperado |
|---|---|---|
| `AC-01` | ABC sintético (100, 0.5, 0.03), v=100 km/h | F=450 N; P=12500 W |
| `AC-02` | Mesmo ABC, v=[0,100] | Vetor F=[100,450] N e P=[0,12500] W |
| `AC-03` | Total 450 N a 100 km/h; componentes conhecidos somam 350 N | Residual=100 N, sem criar componente sintético nomeado |
| `AC-04` | Total 100 N; componentes conhecidos somam 110 N | Residual = -10 N e diagnóstico; sem clipping |
| `AC-05` | Componentes missing, zero explícito e intervalos com F_total≈0 | Ausente preservado; fração indisponível quando denominador inválido |
| `AC-06` | NET sem transmissão resolvida | NET indisponível, sem fallback para TOTAL |
| `AC-07` | Duas contribuições com fronteira sobreposta (ex.: trans + axle agregado em trans) | Falha explícita/aviso; proibir soma não justificada |
| `AC-08` | Loads de 1,10,100,1000 cenários | Performance e q consultas registradas; comparação reproduzível |
| `AC-09` | Baseline persistido + override temporário | Snapshot do DB inalterado; sem SQL writes pelo core |
| `AC-10` | Golden ciclo regulatório + TOTAL/NET | VDE idêntico à versão anterior dentro de tolerância acordada |

`AC-03/04` são casos físicos sintéticos de força num ponto; resultados não são dados observados. `AC-10` exige conjunto golden real autorizado e tolerâncias; não declarar 'byte-for-byte' sem caracterizar floating point, versões e ambiente.

## 5. Evidência e atualização da matriz

Usar estados: `NOT_SPECIFIED`, `SPECIFIED`, `TEST_EXISTS_NOT_RUN`, `INSPECTION_ONLY`, `EXECUTED_PASS`, `EXECUTED_FAIL`, `GAP`. Cada `EXECUTED_PASS` deve apontar a comando, commit, ambiente e log de execução; uma claim de verificação por inspeção não é 'pass'. Registrar mudanças no contrato e no ADR na mesma PR.

## 6. Evidência executada em 2026-11-09

Execução em `fca9a9ad189fe7e6930e987e29fee5717c0d532e`, documentada em [`roadload_14a04_execution_report_v0.1.md`](roadload_14a04_execution_report_v0.1.md):

| IDs | Estado após execução | Evidência |
|---|---|---|
| `RL-FR-01`, `RL-FR-02` | `EXECUTED_PASS` | golden analítico F/P e testes selecionados de Roadload |
| `RL-FR-03`, `RL-FR-05` | `GAP` parcial | residual assinado passa, mas a API TARGET de breakdown/reconcile não existe |
| `RL-FR-06`, `RL-FR-07` | `GAP` | contrato TARGET especificado, não implementado |
| `RL-FR-08` | `EXECUTED_PASS` nos caminhos atuais | TOTAL/NET e NET ausente nos testes selecionados e golden |
| `RL-FR-09` | `EXECUTED_PASS` | canonical VDE series, engine e integração selecionados |
| `RL-FR-10` | `EXECUTED_PASS` | Load Case sem write, determinístico e mass-only dynamic profile |
| `RL-NFR-02` | `GAP` | contagem SQL não medida |
| `RL-NFR-03`, `RL-NFR-04` | `GAP` parcial | microbenchmark capturado; n×k, RSS e UI não medidos |
| `RL-NFR-06` | `EXECUTED_PASS` | mudança limitada a docs e scripts de evidência |

A suíte selecionada terminou com 298 passes e 2 falhas preexistentes de seleção de repositório/fixture mock. A suíte completa não é reproduzível no checkout limpo sem artefatos históricos e bancos ETL não versionados. Portanto, a evidência acima não fecha o gate geral.
