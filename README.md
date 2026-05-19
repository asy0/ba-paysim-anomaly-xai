# PaySim-Anomalieerkennung (Isolation Forest + SHAP)

Repository mit dem Begleitcode zu einer Bachelorarbeit: unüberwachte Anomalieerkennung mit **Isolation Forest** auf dem **PaySim**-Datensatz, Erklärungen mit **SHAP** (TreeExplainer) sowie eine **Streamlit**-Oberfläche. Zusätzlich erzeugt `export_results.py` die für die Auswertung verwendeten Abbildungen und Tabellen batchweise.

---

## Voraussetzungen

- Python 3.10 oder höher
- Abhängigkeiten aus `requirements.txt`
- Datei `data/paysim.csv` (ohne diese Datei schlagen Datenladen und Training fehl)

---

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### Daten

Der Datensatz **PaySim** ist über Kaggle erhältlich: [PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1). Die CSV-Datei als `data/paysim.csv` ablegen. Der erwartete Pfad ist in `src/data/preprocessing.py` festgelegt.

---

## Nutzung

**Streamlit-Anwendung**

```bash
streamlit run app.py
```

In der Seitenleiste Modell trainieren oder gespeichertes Modell laden; Auswertung und Erklärungen über die Registerkarten.

**Ergebnisexport (Plots und Tabellen)**

```bash
python export_results.py
```

Ausgabe: `results/plots/` und `results/tables/`.

---

## Tests

Smoke-Test der Pipeline mit synthetischen Mini-Daten (ohne vollständige PaySim-CSV):

```bash
pytest tests/test_pipeline_smoke.py -q
```

---

## Verzeichnisstruktur

| Pfad | Funktion |
|------|----------|
| `app.py` | Einstiegspunkt Streamlit |
| `src/` | Aufbereitung, Modell, Metriken, SHAP, Visualisierung, UI |
| `export_results.py` | Reproduzierbarer Batch-Export |
| `tests/` | Tests |

---

## Methodische Kurznotizen

- **Training:** Isolation Forest ohne Nutzung der Labels im `fit`; Labels dienen der Auswertung und der schätzungsbasierten Wahl von `contamination` (siehe Code).
- **Scores:** Höhere Werte der `decision_function` entsprechen größerer Tendenz zu „normal“ im Sinne des Modells; niedrigere Werte stärkerer Anomalie-Tendenz. Für ROC wird mit dem Vorzeichen der Scores konsistent umgesetzt (siehe `src/metrics/evaluation.py` und `src/viz/plots.py`).
- **Reproduzierbarkeit:** Feste Zufallszahlen und feste Skriptparameter; identische Ergebniszahlen setzen dieselbe Datendatei und vergleichbare Paketversionen (insbesondere scikit-learn) voraus.

---

## Lizenz

Software: siehe Datei `LICENSE` im Repository, sofern vorhanden. **PaySim** unterliegt den Lizenz- und Nutzungsbedingungen der jeweiligen Quelle (Kaggle), unabhängig von der Lizenz dieses Codes.

---

**Fehlersuche:** Häufige Ursachen sind eine fehlende `data/paysim.csv` oder eine nicht aktivierte virtuelle Umgebung vor dem Aufruf von `pip` / `streamlit` / `python`.
