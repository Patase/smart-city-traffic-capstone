# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

Capstone project for the AI, Machine Learning and Data Science programme.

The project analyses approximately 48,000 hourly records of westbound I-94
traffic volume near Minneapolis–St Paul, together with weather conditions and
US federal holiday information, and develops it into an end-to-end traffic
intelligence solution across three connected parts.

**Author:** Patarachai Sereerat

## Dataset

`data/Metro_Interstate_Traffic_Volume.csv` — hourly traffic volume,
temperature, rainfall, snowfall, cloud cover, weather condition, date/time and
holiday flag, covering October 2012 to September 2018 (48,204 records).

## Repository structure

```
.
├── data/                        Raw dataset shared by all three parts
│
├── part1_data_analytics/        Part 1 — SQL, statistics, probability, Power BI
│   ├── traffic.sql              SQLite database dump of the loaded dataset
│   ├── sql/                     Task 1: load, verify, yearly and holiday trends
│   ├── statistics/              Tasks 2–3: descriptive stats, correlation,
│   │                            probability, and the Excel workbook
│   ├── powerbi/                 Task 4: Power BI dashboard (.pbix)
│   └── Part 1 Data Analytics Insights Report.docx
│
├── part2_python/                Part 2 — reproducible Python pipeline
│   ├── pipeline.py              Task 1: load, validate schema, clean data
│   ├── feature_engineering.py   Task 2: engineered features
│   ├── visualization.py         Task 3: Matplotlib figures
│   ├── cli_app.py               Task 4: command-line analytics application
│   ├── figures/                 Saved figures
│   ├── data/                    Cleaned and engineered datasets
│   ├── pipeline.log             Sample log output
│   ├── README.md                Run instructions and logging configuration
│   └── Part 2 summary.docx
│
└── part3_machine_learning/      Part 3 — machine learning and AI
    ├── prepare_part3_data.py    Shared data preparation and proxy label
    ├── classification.py        Task 1: proxy accident-risk classification
    ├── regression.py            Task 1: traffic volume regression
    ├── clustering.py            Task 2: K-means clustering
    ├── association_rules.py     Task 2: association rule mining
    ├── deep_learning.py         Task 3: neural network with SHAP
    ├── mlflow_tracking.py       Task 4: MLflow experiment tracking
    ├── recommend.py             Task 5: travel recommendation system
    ├── model_versioning.py      Task 6.1: model version history
    ├── deployment_model.py      Task 6.3: model packaging for deployment
    ├── api_app.py               Task 6.3: FastAPI deployment
    ├── monitor.py               Task 6.4/6.5: drift monitoring and alerting
    ├── results/                 Metrics, figures and result tables
    ├── mlflow.db                MLflow tracking store
    ├── part3_ml.log             Sample log output
    ├── README_Part3.md          Run instructions
    ├── Part 3 Final Report.docx
    └── Bias and Fairness report.docx
```

## Accident dataset note

No real accident dataset was provided for this capstone. For the Part 3
classification task, a documented **proxy** `high_risk` label was created from
High/Severe congestion combined with severe or low-visibility weather. The
classification results therefore describe this proxy condition and are not a
prediction of real accidents. See `part3_machine_learning/README_Part3.md` and
`Bias and Fairness report.docx` for the full explanation.

## Running the project

Each part has its own instructions:

- Part 2 — see `part2_python/README.md`
- Part 3 — see `part3_machine_learning/README_Part3.md`

Part 1 is run through SQLite (`part1_data_analytics/traffic.sql`), Excel, and
Power BI Desktop.

## Note on model files

Two trained model binaries (roughly 625 MB each) exceed GitHub's file size
limit and are therefore not included in this repository. They are regenerated
by running the Part 3 scripts, which save them to
`part3_machine_learning/models/`.
