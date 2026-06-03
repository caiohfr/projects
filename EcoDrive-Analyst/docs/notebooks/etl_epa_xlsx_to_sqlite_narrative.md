# Narrativa do ETL - EPA XLSX -> SQLite

Objetivo: contar a historia do pipeline de ingestao de forma simples, para apresentacao interna, tracking e handoff.

Notebook alvo: `notebooks/etl_epa_xlsx_to_sqlite.ipynb`

## 1) Contexto
Recebemos dados de testes EPA (2020-2025), com diferentes niveis de limpeza e formatos.  
O papel deste ETL e transformar esses dados em registros consistentes no banco do EcoDrive.

## 2) Entrada (Extract)
O pipeline comeca lendo:
- base principal EPA (XLSX)
- tabelas auxiliares de classes/categorias
- arquivos de defaults e suporte

Resultado desta etapa:
- DataFrames brutos carregados em memoria para tratamento.

## 3) Padronizacao (Transform - Base)
Os dados brutos sao normalizados:
- tipos numericos e textos
- campos de identificacao (make/model/year)
- marcadores de powertrain e categoria
- convencoes de unidades do projeto

Resultado desta etapa:
- tabela canonical intermediaria (base para regras de negocio).

## 4) Regras de negocio (Transform - Enriquecimento)
Com a base padronizada:
- classificamos melhor os veiculos
- aplicamos funcoes de apoio do core
- calculamos campos derivados para VDE e consumo
- montamos estruturas finais para insercao

Resultado desta etapa:
- `df_vde_to_insert` e `df_fc` (ou equivalentes), prontos para carga.

## 5) Carga no banco (Load)
O ETL persiste os registros no SQLite:
- primeiro `vde_db`
- depois `fuelcons_db` vinculado ao `vde_id`

Resultado desta etapa:
- snapshots completos no banco, prontos para uso nas pages.

## 6) Conferencia (QA)
Apos carga:
- previews e contagens
- checks de campos obrigatorios
- comparacoes simples via graficos

Resultado desta etapa:
- confianca basica de que a carga ficou consistente.

## 7) Operacoes de manutencao (separadas)
Existem celulas utilitarias de manutencao (delete/update/limpeza de DB).  
Elas **nao** fazem parte do caminho principal de ETL e devem ficar separadas para evitar execucao acidental.

## 8) Ordem sugerida de execucao (sem refactor de codigo)
1. Config/paths/imports  
2. Extract  
3. Transform base  
4. Transform de negocio  
5. Preparacao dos dataframes finais  
6. Load (com `DRY_RUN=False` somente quando validado)  
7. QA  
8. Bloco de manutencao (opcional, isolado)

## 9) Mensagem executiva (pitch curto)
Este notebook transforma dados EPA heterogeneos em dados operacionais do EcoDrive, com rastreabilidade de etapas (extract, transform, load e QA), reduzindo retrabalho manual e acelerando a alimentacao do banco para analises e cenarios.

