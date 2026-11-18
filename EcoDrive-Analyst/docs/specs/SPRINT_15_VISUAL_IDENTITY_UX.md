# Sprint 15 — Identidade visual e continuidade de uso

Status **v0.1 para revisão**

Baseline inspecionado **`0f7a0f8`**, branch `sprint-15a-home`

Referência visual **EcoDrive Layout 8 Páginas · Protótipo v0.2**, nas fontes do projeto. A v0.3 será o roteiro visual da primeira entrega.

## Objetivo

Uma pessoa que chega ao EcoDrive deve conseguir iniciar uma proposta de VDE, entender o estado dos dados e abrir a ferramenta apropriada sem refazer seleções que o app já consegue reaproveitar. A mesma linguagem visual deve orientar os oito destinos do protótipo.

A sprint trabalha em Streamlit e adota os fluxos comprovados no [fechamento da Sprint 14](../sprints/SPRINT_14_CLOSURE_2026_11_15.md) e os [princípios de UX do projeto](../architecture/UX_PRINCIPLES.md). Aplicam-se os contratos de física e persistência atuais. A camada de apresentação reutiliza os resultados e a proveniência disponíveis, preservando zero explícito e mostrando ausência quando o dado não existe.

## Entregas

| Etapa | Superfícies | Entrega e critério de aceite |
| --- | --- | --- |
| **15A — Protótipo v0.3, Home e navegação** | Home e navegação global | Desenhar o primeiro acesso, com **New VDE proposal** em destaque, e a sessão em andamento, com **Continue analysis**. Implementar hero, cinco atalhos de trabalho e destinos de Methodology e Engineering Tools. A navegação deve abrir os destinos reais pelos mecanismos do Streamlit e preservar o acesso às páginas e ferramentas existentes. Active Context reflete a sessão atual e distingue baseline, proposta e referência FuelCons quando disponíveis. Recent Analysis só mostra itens recuperáveis, com estado vazio na ausência de uma fonte confiável. |
| **15B — Proposta e análise temporária** | VDE Setup e Roadload & Load Case Lab | Dar hierarquia visual ao baseline, matriz, inputs por domínio e preview/save. Explicitar o objeto aberto no Lab e manter cálculo temporário. Reusar a passagem atual do baseline persistido e demonstrar visualmente a opção de usar uma proposta temporária apenas se o spike da 15A comprovar que o contrato existente a representa com paridade. |
| **15C — Comparação e consumo** | Comparison Report e Powertrain Scenario | Manter os filtros, KPIs e diagnósticos existentes e trazer ciclo, massa de teste e base TOTAL/NET para perto dos resultados de Vehicle Demand, quando aplicáveis à seleção. Mostrar divergências de base ou ciclo quando detectáveis nos metadados existentes. No L0, destacar a seleção de referência FuelCons e a origem do PSE quando fornecida. Comparar resultados FuelCons pela consulta canônica dos registros persistidos. |
| **15D — Dados, metodologia e fechamento** | Database Management, Methodology & Models e Engineering Tools / Legacy | Demonstrar a sequência selecionar → editar/importar → preparar → revisar → confirmar no Database Management. Revisar o texto e as figuras de Methodology contra os contratos vigentes, em particular TOTAL/NET, e dar acesso às ferramentas legadas. Fechar com capturas em navegador, roteiro de demo e evidências por fluxo. |

## Fluxo clicável da v0.3

1. **Primeiro acesso.** Home → New VDE proposal → VDE Setup. Retomar a sessão existente deve ser uma ação distinta de iniciar outra. Qualquer descarte de rascunho exige ação explícita.
2. **Proposta.** Selecionar baseline → escolher hipótese de pneu → gerar preview → decidir se quer salvar a proposta, com status e proveniência visíveis. A revisão do preview deve poder ocorrer antes de qualquer save.
3. **Lab.** Abrir o baseline selecionado no Lab pela ação existente e identificar o objeto carregado. O atalho atual em [VDE Setup](../../pages/VDE_Setup.py) passa um `vde_id` salvo à página de destino. Usar o preview não salvo demandaria confirmar que o Lab aceita um caso equivalente por seus contratos atuais, inclusive componentes aplicados, base TOTAL/NET e ausência de gravação. Até essa verificação, o protótipo demonstra o baseline salvo como caminho implementável.
4. **Consumo.** Selecionar explicitamente um FuelCons salvo para entrar no L0. Se o roteiro precisar cadastrar um FuelCons, apontar o fluxo manual/importado hoje disponível em Engineering Tools e mostrar o save explícito. O cálculo do L0 aparece como resultado da sessão.
5. **Comparação.** Abrir o Comparison Report com registros FuelCons salvos que permitam demonstrar a diferença e seus drivers. Pode-se usar uma dupla sintética já preparada. Se houver preseleção por ID na implementação, validar o ID no banco ativo antes de aplicá-la. Caso contrário, deixar a seleção explícita no fluxo.

O protótipo usa dados marcados como demonstração e deve mostrar também o retorno à Home com sessão ativa. O app usa dados da sessão/banco ativo e rótulos fiéis ao estado de persistência. Trocar de banco ou perder a referência deve produzir um estado vazio acionável, sem reaplicar IDs de outro banco. O fluxo do L0 pode ser demonstrado ao lado da comparação dos FuelCons já salvos.

## Gate 15A

- Demonstrar o fluxo da v0.3, com as duas Home, navegando pelo menos até VDE Setup e Comparison e mostrando o ramo opcional do Lab e a seleção de referência no L0.
- Montar a Home no app com links reais, hierarquia próxima do HTML e tratamento de sessão vazia. Aproveitar tema e navegação nativos. Quando a aproximação visual exigir CSS, manter o CSS local a superfícies sob controle do app.
- Confirmar quais dados existentes podem alimentar Active Context e Recent Analysis. O histórico global por usuário depende de um contrato próprio e fica fora desta sprint. Registrar a fonte de cada campo exibido.
- Verificar como navegação e sidebar se comportam junto aos controles das páginas, em uma janela desktop larga e em uma janela estreita. Registrar as diferenças visuais aceitas para orientar as demais páginas.
- Fechar a decisão sobre o handoff temporário do VDE Setup para o Lab com evidência do contrato atual. Se depender de contrato novo, implementar a navegação para o baseline salvo e abrir a evolução separadamente.

## Aceite da sprint

| Evidência | Aceite |
| --- | --- |
| **AppTest direcionado** | Home abre com sessão vazia e ativa, renderiza os destinos esperados e trata contexto inválido. Rodar os testes existentes das páginas alteradas. Cobrir novos handoffs ou diferenças de estado quando os testes atuais não os verificarem. |
| **Navegador real** | Abrir os oito destinos pelo fluxo normal e percorrer o roteiro com dados sintéticos rotulados ou públicos. Registrar capturas em duas larguras, retorno à Home e estados vazios relevantes. Checar navegação, foco e legibilidade. |
| **Preservação dos resultados** | Verificar o preview/save de VDE, cálculo temporário do Lab, filtros e KPIs de Comparison, referência e resultado do L0 e revisão/commit da administração. Se houver mudança no caminho de dados, comparar os resultados da mesma fixture antes e depois. |
| **Fechamento** | Registrar cobertura por requisito como teste direto, evidência indireta, inspeção ou lacuna. Anotar resultados executados nesta sprint e eventuais limitações de verificação visual. |

Mudanças de física, esquema, persistência de resultados L0, autenticação, migração de frontend e deploy seguem fora do escopo. Mudanças de contrato necessárias para um handoff voltam para decisão antes de implementação.
