# Smart City Traffic Intelligence

## Part 2 – Python Traffic Analytics Pipeline

This project is part of the **Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility** capstone.

Part 2 develops a reproducible Python workflow for cleaning traffic data, engineering features, creating visualisations and building a small command-line traffic analytics application.

The dataset contains approximately 48,000 hourly records of westbound I-94 traffic near Minneapolis–St Paul, together with temperature, weather and holiday information.

---

## Project Structure

```text
smart-city-traffic-capstone/
│
├── data/
│
├── part1_data_analytics/
│
└── part2_python/
    │
    ├── pipeline.py
    ├── feature_engineering.py
    ├── visualizations.py
    ├── cli_app.py
    ├── pipeline.log
    │
    ├── data/
    │   ├── Metro_Interstate_Traffic_Volume.csv
    │   ├── cleaned_traffic.csv
    │   └── engineered_traffic.csv
    │
    └── figures/
        ├── traffic_by_hour.png
        ├── weekday_vs_weekend.png
        └── temperature_vs_traffic.png
```

---

# Requirements

The scripts were developed using Python 3.11.

Required Python packages:

```text
pandas
numpy
matplotlib
```

The following Python modules are also used but are included with Python:

```text
logging
pathlib
argparse
sys
```

If required packages are missing, they can be installed using:

```bash
pip install pandas numpy matplotlib
```

---

# Running the Project

The scripts should be run in the following order:

```text
pipeline.py
      ↓
cleaned_traffic.csv
      ↓
feature_engineering.py
      ↓
engineered_traffic.csv
      ↓
visualizations.py
      ↓
figures
      ↓
cli_app.py
```

From the project root directory:

```bash
cd /Users/patarachaisereerat/smart-city-traffic-capstone
```

---

# Task 1 – Data Cleaning Pipeline

Run:

```bash
python part2_python/pipeline.py
```

The script loads the raw traffic dataset, validates the expected schema and performs data cleaning.

The following cleaning steps are included:

* Standardisation of categorical text values.
* Parsing and validation of `date_time`.
* Removal of exact duplicate rows.
* Detection of impossible temperature values at 0 K.
* Replacement of 0 K temperature values using the median temperature of the corresponding month.
* Detection of rainfall values above 9,000 mm.
* Replacement of rainfall values above 9,000 mm using the median valid rainfall value.

The cleaned dataset is saved as:

```text
part2_python/data/cleaned_traffic.csv
```

In the current run, the raw dataset contained 48,204 rows. After removing 17 exact duplicate rows, the cleaned dataset contained 48,187 rows.

Repeated timestamps are not automatically removed during the cleaning stage because different weather observations may occur for the same `date_time`.

---

# Task 2 – Feature Engineering

Run:

```bash
python part2_python/feature_engineering.py
```

The cleaned dataset from Task 1 is used to create features suitable for further analysis and machine learning.

## Time Features

The following time-based features are created:

```text
hour
day_of_week
is_weekend
```

`day_of_week` follows the Pandas convention:

```text
0 = Monday
1 = Tuesday
2 = Wednesday
3 = Thursday
4 = Friday
5 = Saturday
6 = Sunday
```

The weekend indicator is:

```text
0 = Weekday
1 = Weekend
```

## Cyclical Time Features

Hour and day of week are also represented using sine and cosine transformations:

```text
hour_sin
hour_cos
day_sin
day_cos
```

These features allow cyclical time relationships to be represented more appropriately. For example, hour 23 and hour 0 are close together in time even though their numerical values are far apart.

## Weather Features

Weather information is transformed using:

* A precipitation indicator.
* One-hot encoding of `weather_main`.

The precipitation indicator identifies weather categories such as:

```text
Rain
Snow
Drizzle
Thunderstorm
```

One-hot encoded weather columns are also created, such as:

```text
weather_Clear
weather_Clouds
weather_Rain
weather_Snow
```

## Numerical Scaling

Min-max scaling is applied to:

```text
temp
rain_1h
```

The resulting features are:

```text
temp_scaled
rain_scaled
```

The scaled values range approximately from 0 to 1.

## Congestion Category

A data-driven congestion category is created using the quartiles of `traffic_volume`.

```text
Traffic volume ≤ Q1       = Low
Q1 < Traffic volume ≤ Q2  = Moderate
Q2 < Traffic volume ≤ Q3  = High
Traffic volume > Q3       = Very High
```

The quartile thresholds are calculated from the dataset rather than using fixed traffic-volume cut-offs.

The engineered dataset is saved as:

```text
part2_python/data/engineered_traffic.csv
```

The current engineered dataset contains:

```text
48,187 rows
31 columns
```

---

# Task 3 – Traffic Visualisations

Run:

```bash
python part2_python/visualizations.py
```

Three Matplotlib visualisations are generated.

## 1. Average Traffic Volume by Hour

Output:

```text
part2_python/figures/traffic_by_hour.png
```

The current analysis shows that the highest average traffic volume occurs at approximately **16:00**, with an average of approximately **5,708.61 vehicles**.

This suggests a clear afternoon commuting peak.

## 2. Weekday vs Weekend Traffic

Output:

```text
part2_python/figures/weekday_vs_weekend.png
```

Current results:

```text
Average weekday traffic: 3557.44 vehicles
Average weekend traffic: 2623.93 vehicles
```

Average traffic is therefore higher on weekdays, suggesting that commuting and other weekday travel contribute substantially to traffic demand.

## 3. Temperature vs Traffic Volume

Output:

```text
part2_python/figures/temperature_vs_traffic.png
```

The Pearson correlation between temperature and traffic volume is approximately:

```text
r = 0.1395
```

This represents a weak positive relationship, suggesting that temperature alone has limited association with traffic volume.

For the traffic visualisations, repeated `date_time` records are excluded so that the same hourly traffic volume is not counted multiple times.

---

# Task 4 – Mini Traffic Analytics Application

The command-line application is:

```text
part2_python/cli_app.py
```

It provides three traffic analytics commands.

## Command 1 – Query Traffic by Date and Time

Example:

```bash
python part2_python/cli_app.py query 2017-06-01 08:00
```

The application returns available information for the selected date and time, including:

```text
Traffic volume
Weather condition
Congestion category
```

Invalid dates are handled without producing an unhandled traceback.

For example:

```bash
python part2_python/cli_app.py query hello 08:00
```

will return a clear error message explaining the expected date/time format.

---

## Command 2 – Identify High-Traffic Periods

Run:

```bash
python part2_python/cli_app.py high-traffic
```

The default traffic threshold is:

```text
traffic_volume > 5500
```

and the application displays the 10 highest traffic records meeting the criterion.

A different threshold and number of records can also be specified.

Example:

```bash
python part2_python/cli_app.py high-traffic --threshold 6000 --limit 5
```

---

## Command 3 – Compare Weekday and Weekend Traffic

Run:

```bash
python part2_python/cli_app.py compare-days
```

The application calculates and displays:

```text
Average weekday traffic
Average weekend traffic
Difference between the two averages
```

---

# Logging

Python's `logging` module is used throughout the project instead of `print()` statements for internal program status.

The log file is:

```text
part2_python/pipeline.log
```

Logs are written both to:

1. The console.
2. `pipeline.log`.

The logging format contains:

```text
Timestamp | Log level | Module name | Message
```

Example:

```text
2026-09-17 23:17:16,985 | INFO | feature_engineering | Dataset shape before feature engineering: 48187 rows, 9 columns.
```

## Logging Levels

### DEBUG

Used for detailed intermediate values that are mainly useful for troubleshooting.

For example:

```text
Traffic-volume quartile thresholds
```

DEBUG messages are not displayed during normal runs because the logger is normally configured at the INFO level.

### INFO

Used for normal and expected program events.

Examples include:

```text
Data successfully loaded
Schema validation completed
Features created
Figure saved
CLI command executed
Output file saved
```

### WARNING

Used for unexpected but recoverable data-quality issues or modifications.

Examples include:

```text
Duplicate rows removed
0 K temperature values imputed
Extreme rainfall value imputed
Categorical values standardised
```

### ERROR

Used when an error prevents part of the workflow from continuing normally.

Examples include:

```text
Input file cannot be loaded
Schema validation fails
Output file cannot be saved
Invalid CLI input
```

---

# Log File Behaviour

`pipeline.py` starts a new log file using write mode.

The later scripts use append mode so that their log messages are added to the same file.

Therefore, running the scripts in this order:

```text
pipeline.py
feature_engineering.py
visualizations.py
cli_app.py
```

creates a continuous logging trail covering the complete Part 2 workflow.

---

# Reproducibility

The project separates the workflow into individual Python scripts so that each stage can be run and reviewed independently.

```text
Raw CSV
   ↓
Data cleaning and validation
   ↓
cleaned_traffic.csv
   ↓
Feature engineering
   ↓
engineered_traffic.csv
   ↓
Visualisation and CLI analytics
```

Intermediate datasets are saved so that later stages can be reproduced without modifying the original raw dataset.

Git and GitHub are used for version control and to maintain an incremental history of project development.

---

# Part 2 Outputs

The main outputs produced by the project are:

```text
pipeline.py
feature_engineering.py
visualizations.py
cli_app.py

pipeline.log

cleaned_traffic.csv
engineered_traffic.csv

traffic_by_hour.png
weekday_vs_weekend.png
temperature_vs_traffic.png
```

These outputs provide a reproducible workflow for preparing, analysing and interacting with the Smart City traffic dataset.
