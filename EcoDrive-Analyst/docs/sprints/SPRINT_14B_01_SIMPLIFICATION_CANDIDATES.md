# Sprint 14B-01 — Candidatos de simplificação seletiva

**Estado:** auditoria concluída; nenhuma alteração de produção realizada.
**Baseline:** `main@e7da21681205027348fcb7b66670f5ea8d8a01ec` (merge do PR #15).
**Branch da auditoria:** `docs/sprint14b01-simplification-audit`.
**Ambiente:** Windows, PowerShell, Python 3.11.9.
**Data da inspeção:** 2026-11-15.

## Escopo e regras preservadas

Foram inspecionadas as cinco fatias exigidas: ETL→DB; VDE Setup→Preview/Save; Quick/Load Case; Comparison; Powertrain Manual/L0→FuelCons. A matriz e os documentos de arquitetura foram usados somente como hipóteses. Esta auditoria não altera a numeração do roadmap: Sprint 14B continua sendo simplificação seletiva e Sprint 15 continua sendo UX.

Permaneceram fora de escopo: reorganização ampla em novos pacotes, mudança de schema, novas classes TARGET, Component Vector Search, alterações em `compute_vde_series`, TOTAL/NET, RRC, massa, L0 ou qualquer outra física. Preview, Quick Scenario, Load Case e L0 continuam sem persistência implícita; saves continuam explícitos.

## Mapa das cinco fatias

| Fatia | Entry point e fluxo observado | I/O, autoridade e persistência | Testes protetivos principais | Avaliação |
|---|---|---|---|---|
| ETL→DB | Scripts versionados em `etl/scripts/`, com destaque para `sprint_12e_migration_rehearsal.py`, `sprint_12e1_clean_rebuild_migration.py` e `sprint_12f12_full_population_consolidation.py` | Lê fontes e bancos explicitamente; escreve somente em staging protegido por `guard_output_path`; usa SQLite diretamente para transações, PRAGMAs, hashes e verificações de integridade. Não é o caminho de persistência interativa da aplicação | `etl/tests/test_sprint_12e_migration_rehearsal.py`, `test_sprint_12e1_clean_rebuild_migration.py`, `test_sprint_12f12_full_population_consolidation.py` | Código histórico/reprodutível. Há repetição, mas uma abstração prematura pode apagar diferenças de proteção entre fases |
| VDE Setup→Preview/Save | Canônico atual: `pages/VDE_Setup.py` → `vde_request_compact` → request resolver → `vde_workflow_service.build_vde_setup_preview`; persistência em `save_vde_setup_result` | `vde_workflow_service` resolve massa, ABC TOTAL/NET, preview, review e save explícito via repositories. O componente legado `vde_setup.py` ainda contém um segundo preview e renderizadores antigos, mas não é importado pela página canônica atual | `tests/test_vde_request_resolver.py`, `test_vde_workflow_service.py`, `test_vde_setup_service.py`, suites compact/persistence | O caminho canônico está protegido. A coexistência legado/canônico deve ser tratada por depreciação e golden, não por remoção imediata |
| Quick/Load Case | Comparison Quick: UI → `quick_scenario.comparison_adapter`/resolver. Load Case: `pages/Roadload_Load_Case_Lab.py` → componente → repositories → `vehicle_demand.load_case.calculate_roadload_load_case` → Vehicle Demand | Ambos são temporários. Load Case lê VDE/FuelCons para contexto, não escreve e reutiliza o resultado do Vehicle Demand. Massa, grade, trailer, pneu e ambiente permanecem no core | `tests/test_quick_scenario_resolver_parity.py`, `test_quick_scenario_vehicle_demand_integration.py`, `test_vehicle_demand_load_case.py`, `test_roadload_load_case_lab_ui.py` | Fronteiras físicas estão explícitas e a paridade é forte. Não há justificativa para mover física neste pacote |
| Comparison | `pages/Comparison_Report.py` → componente → `comparison_report_service` → repositories/DB + contratos de VDE, FuelCons e TOTAL/NET | Serviço reúne catálogo, lineage, boundaries, cálculo on-demand e comparação. Não persiste. O catálogo leve e o detalhado aplicam o mesmo conjunto de filtros, hoje por dois trechos parcialmente duplicados | `tests/test_comparison_report_service.py`, `test_comparison_report_page_smoke.py`, `test_comparison_report_8e_smoke_matrix.py` | Existe uma simplificação pequena, demonstrável e de baixo risco: usar o helper de filtros já existente no catálogo leve |
| Powertrain Manual/L0→FuelCons | `pages/Powertrain_Scenario.py` usa seleção/estado de `pwt_fuel_energy.py` e o workspace de System Scenario; a superfície legada permanece em Legacy Engineering Tools. Cálculo/salvamento canônico passa por `fuel_estimation.py` e repositories | `run_fuel_estimation` calcula sem persistir; `save_fuel_estimate_result` prepara payload comum e executa save explícito. O editor legado de tabela ainda monta updates e chama wrappers CRUD próprios | `tests/test_fuel_estimation.py`, `test_pwt_fuel_energy_service_pse_reference.py`, `test_powertrain_system_scenario_ui.py`, `test_system_scenario_*` | Há dois caminhos de edição/save. A convergência tem valor, mas precisa de testes de zero, payload e proveniência antes de qualquer mudança |

## Candidatos

Escala de evidência: **DIRECT_TESTED** = coberto por teste executado nesta auditoria; **TEST_PRESENT** = teste localizado, mas não executado aqui; **INSPECTION** = evidência direta de código; **GAP** = falta prova suficiente.

| ID | Fatia; arquivo/função/linha aproximada | Acoplamento ou duplicação demonstrável e causa | Benefício potencial | Risco e testes protetivos | Recomendação | Evidência |
|---|---|---|---|---|---|---|
| C01 | Comparison; `comparison_report_service.py::list_comparison_scenarios`, linhas 833–855; `_apply_scenario_filters`, 905–922 | O catálogo leve repete manualmente os mesmos quatro filtros que o helper declara centralizar para impedir drift; o catálogo detalhado já usa o helper | Uma única regra para `make`, `legislation`, `electrification` e `record_origin`; diff pequeno e legível | Baixo. Preservar SQL, ordem de parâmetros e ordenação. Proteger com `ScenarioCatalogTests` e `DetailedScenarioCatalogTests.test_filters_match_the_lightweight_catalog_exactly` | **SIMPLIFY_IN_PLACE** — prioridade 1 e menor PR viável | INSPECTION + DIRECT_TESTED |
| C02 | Comparison; `comparison_report_service.py`, linhas 28–40, 217–422 e 643–985 | O mesmo módulo reúne DB/catalog, lineage, boundary TOTAL/NET e montagem de dataset. Além disso, `vehicle_demand/adapters.py` importa o boundary resolver, criando uma direção de dependência sensível | Uma fronteira neutra poderia reduzir acoplamento e facilitar reuso | Alto: risco de ciclo de import, mudança de TOTAL/NET e perda de compatibilidade. Exige caller graph, testes de boundary, Vehicle Demand e Comparison completos | **DEFER** — não mover no primeiro PR | INSPECTION + documentação 14A |
| C03 | VDE Setup; `vde_setup_service.py::build_live_vde_preview`, linhas 485–558; `vde_workflow_service.py::build_vde_setup_preview`, 1357–1417; usos legados em `vde_setup.py`, ~16034–16286 | Existem dois envelopes de preview com semânticas e formatos distintos; o componente legado calcula ambos e contém fallback/preferência condicional | Reduzir duas fontes de apresentação e tornar paridade explícita | Alto: massa, fases, TOTAL/NET, save payload e UI legada. Exige golden por legislação e por origem de ABC antes de envolver produção | **WRAP** — prioridade 3, somente após congelar paridade | INSPECTION + DIRECT_TESTED no caminho canônico |
| C04 | VDE Setup UI; `vde_app/components/vde_setup.py` (17.142 linhas), renderizadores v2/v2.1/legacy, com callers em `docs/archive/pages` e testes | A página canônica atual importa `vde_request_compact`, não esse módulo; o arquivo grande permanece como compatibilidade de superfícies arquivadas e helpers testados | Uma política de depreciação poderia reduzir muito a superfície mantida | Alto: remoção pode degradar ferramenta legada e quebrar imports/testes. Primeiro provar ausência de caller runtime e definir retenção de arquivos arquivados | **DEFER** | INSPECTION |
| C05 | Quick Scenario; `quick_scenario/resolver.py`, linhas 38–45 e 92–108; `comparison_adapter.py`, linhas 23–34 | Resolver usa repositório e contratos de `comparison_report_service`; é uma fronteira de aplicação real, porém acopla cenário temporário ao serviço amplo de Comparison | Separar fetch/orquestração de resolução pura facilitaria testes e evitaria ciclos futuros | Médio/alto: source identity, FuelCons, TOTAL/NET, tire e paridade. Exige `test_quick_scenario_resolver*`, integration e Comparison | **DEFER** | INSPECTION + DIRECT_TESTED de paridade |
| C06 | Quick Scenario; `comparison_adapter.py::fetch_quick_source_rows_once`, linhas 86–94 | Wrapper apenas encaminha para `resolver.fetch_quick_source_rows`; o nome “once” descreve intenção do caller, não cache interno | Remover uma camada teria ganho mínimo | Baixo tecnicamente, mas pode quebrar import público sem ganho proporcional | **KEEP** como façade compatível | INSPECTION |
| C07 | Load Case; `roadload_load_case_lab.py::render_roadload_load_case_lab`, linhas 623–668 | A UI faz três leituras de repository e chama um único core; não replica física e não escreve. Um application service poderia agrupar as leituras, mas a orquestração atual é explícita | Pequena redução de imports da UI | Médio e benefício baixo. Proteger no-write, provenance de RRC, baseline parity e AppTest | **KEEP** por enquanto | INSPECTION + DIRECT_TESTED do core |
| C08 | Load Case; `vehicle_demand/load_case.py::_force_at`, linha ~705; Roadload force owners | Avaliação local de `A+Bv+Cv²` já foi marcada como possível duplicação na auditoria 14A | Um owner único de avaliação ABC pode reduzir duplicação matemática | Muito alto para um ganho pequeno: arquivo está na fronteira congelada de Vehicle Demand e contém overlays próprios. Exige todos os testes Load Case, Roadload e golden | **DEFER** | INSPECTION + DIRECT_TESTED do comportamento atual |
| C09 | Powertrain; `pwt_fuel_energy.py::render_fuelcons_table`, linhas 3424–3571; `fuel_estimation.py::save_fuel_estimate_result`, 577–616 | O editor legado monta update parcial na UI e chama wrappers CRUD, enquanto o fluxo de review/save usa o payload canônico. Nos gates `if value and value > 0`, zero não pode ser expresso pelo editor | Centralizar validação/payload/save e tornar a semântica de zero explícita, preservando proveniência | Médio/alto: é alteração comportamental e não deve ser mascarada como refactor. Reproduzir primeiro; testar zero vs ausente, allowed columns, insert/update/delete e AppTest | **WRAP** — prioridade 2, em spec/PR separado após reprodução | INSPECTION; teste direto específico é GAP |
| C10 | Powertrain; `pwt_fuel_energy_service.py`, linhas 33–92 e 278–313 | Diversos wrappers de uma linha sobre repositories mantêm a UI longe dos detalhes de DB; alguns parecem redundantes isoladamente | Remoção reduziria poucas linhas | Quebra de imports e testes sem benefício claro; façade é coerente com a fronteira UI/core | **KEEP** | INSPECTION + TEST_PRESENT |
| C11 | ETL→DB; `guard_output_path` em pelo menos cinco scripts (`12e`, `12e1`, `12e2`, `12f12`, `12f13`) | Estrutura repetida, mas cada fase possui conjuntos protegidos, raiz de staging e mensagem próprios | Helper parametrizado poderia uniformizar a proteção | Médio: scripts são evidência histórica e diferenças são relevantes. Qualquer extração exige testes por script e hashes/reprodutibilidade | **DEFER** | INSPECTION; guard isolado DIRECT_TESTED |
| C12 | ETL→DB; conexões/PRAGMAs/assinaturas em `sprint_12e1_clean_rebuild_migration.py` e `sprint_12f12_full_population_consolidation.py` | SQL direto e utilitários semelhantes aparecem em fases distintas, mas controlam transação, read-only, integridade e relatórios específicos | Uma biblioteca comum reduziria repetição aparente | Alto e baixo valor atual; abstração pode esconder ordem transacional e comprometer reprodutibilidade histórica | **KEEP** | INSPECTION; suíte histórica bloqueada por fixture ausente |

## Priorização

1. **C01 — filtros do catálogo de Comparison.** Melhor relação valor/custo/risco; nenhuma física, nenhum schema, nenhum efeito de persistência.
2. **C09 — convergência do editor legado de FuelCons.** Valor real, mas deve começar por teste de reprodução para zero/ausente e por uma decisão explícita de comportamento; não cabe no primeiro PR.
3. **C03 — previews legado e canônico do VDE Setup.** Pode reduzir complexidade relevante, porém somente após golden de massa, fases, TOTAL/NET e payload de save.

Os demais itens ficam `KEEP` quando a camada atual é uma façade útil ou complexidade necessária, e `DEFER` quando o risco de quebrar física, compatibilidade, histórico ou persistência supera o ganho demonstrado.

## Menor PR viável proposto para B-02

**Título sugerido:** `refactor(comparison): reuse canonical scenario filter builder`

**Mudança proposta:** fazer `list_comparison_scenarios()` chamar `_apply_scenario_filters()` já existente, mantendo seleção, aliases, parâmetros e `ORDER BY` byte-equivalentes. Não mover módulos, não alterar signatures, não tocar em TOTAL/NET, lineage, métricas, UI ou DB.

**Teste antes de editar:** ampliar `DetailedScenarioCatalogTests.test_filters_match_the_lightweight_catalog_exactly` para cobrir os quatro filtros isolados, uma combinação de filtros e filtros vazios; capturar os IDs e a ordem retornada. Depois aplicar a substituição e executar as mesmas provas.

**Comandos de aceite sugeridos para B-02:**

```powershell
python -m unittest -v tests.test_comparison_report_service
python -m unittest -v tests.test_comparison_report_page_smoke tests.test_comparison_report_8e_smoke_matrix
```

**Risco residual:** baixo, limitado a montagem/ordem dos parâmetros SQL. O teste de paridade leve×detalhado é o guard principal. Este B-01 não implementa B-02.

## Evidência executada nesta auditoria

Comandos relevantes realmente executados:

```powershell
git fetch origin --prune
git switch -c docs/sprint14b01-simplification-audit origin/main
git rev-parse HEAD
rg --files src pages etl tests
rg -n "^(def|class) |^    def " <arquivos inspecionados>
rg -n <callers, imports, DB access e testes> src pages etl tests docs
python --version
python -m pytest -q <suítes direcionadas>
python -m unittest -v <suítes direcionadas>
python -c <validação isolada de guard_output_path>
git status --short --branch
```

Resultados:

- Sincronização: `origin/main` avançou até `e7da2168`; branch nova criada a partir desse SHA.
- `pytest`: não executou testes porque o módulo não está instalado no Python disponível (`No module named pytest`). Nenhuma dependência foi instalada.
- `unittest`: **181 testes contabilizados; 180 PASS; 1 ERROR de ambiente/fixture**. O erro ocorreu no `setUpClass` de `Sprint12EMigrationRehearsalTest`, antes do teste solicitado, porque falta o artefato histórico não versionado `etl/data/processed/sprint_12e_migration_rehearsal/rehearsal_summary.json`.
- Validação isolada do guard ETL, sem depender do artefato: **2/2 PASS**; runtime DB e QA DB foram rejeitados como destinos de saída.
- As 180 provas executadas cobriram VDE Workflow, Quick Scenario parity, Load Case, Comparison, Fuel Estimation e referência PSE de Powertrain. Não houve falha nesses subsistemas.
- O worktree permaneceu sem alteração de produção; somente este documento foi criado.

## Gaps e stop conditions

- A suíte histórica ETL completa não é reproduzível neste checkout sem o `rehearsal_summary.json`. O artefato não foi regenerado porque isso ampliaria o escopo e poderia produzir outputs históricos.
- Não foi executado browser smoke real. AppTest localizado no repositório não equivale a verificação manual em navegador.
- Não foi executada a suíte completa do repositório; B-01 pediu inspeção e testes seguros/direcionados, não um fechamento geral de Sprint 14.
- C09 requer teste de reprodução antes de ser chamado de bug; nesta auditoria ele é uma divergência/hazard observada por inspeção.
- Esta auditoria não declara Sprint 14, 14A ou 14B concluída e não autoriza automaticamente nenhum candidato além do PR documental atual.
