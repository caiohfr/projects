# Canonical database environments

Normal application databases all use the same frozen canonical schema. The
path selects an environment instance; it does not select a different database
architecture.

| Role | Path | Purpose |
|---|---|---|
| STAGING | `data/db/staging/eco_drive_canonical_candidate.db` | Approved pipeline candidate and promotion source |
| QA | `data/db/eco_drive_qa.db` | Development, manual smoke, and disposable application writes |
| PROD | `data/db/eco_drive.db` | Official clean runtime database |
| ARCHIVE | `data/db/archive/` | Preserved pre-Sprint-12 databases; never a normal runtime target |

The controlled lifecycle is `source data -> ETL -> STAGING -> validation ->
QA -> manual smoke -> explicit PROD promotion`. QA changes are never promoted
automatically.

## Run the application against QA

From the repository root in PowerShell:

```powershell
$env:ECO_DRIVE_DB_PATH = "data/db/eco_drive_qa.db"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Remove the override to return to the production default:

```powershell
Remove-Item Env:ECO_DRIVE_DB_PATH
```

The Database Management and Comparison Report sidebars display the active DB
path. Comparison Report's **Switch to QA data** action selects the canonical QA
instance without regenerating or seeding it.

Synthetic historical test fixtures remain under `data/qa/`. They are isolated
test assets and are not part of the STAGING/QA/PROD promotion chain.
