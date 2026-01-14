# Expected Danger Model

This project analyzes Premier League event data to calculate "Expected Danger" (ED) for passes. It includes a data processing pipeline, statistical analysis, and a Streamlit dashboard.

## How to use

Execute the entire pipeline (prepare data -> analysis -> dashboard(streamlit)) in one command:
```bash
python run_all.py
```

Alternatively, run steps individually:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Process the raw event data:
   ```bash
   python prepare_data.py
   ```
3. Run the analysis and generate stats:
   ```bash
   python expected_danger_model.py
   ```
4. Launch the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

## Data Storage & Parquet Files

The project uses the **Parquet** format for data storage. Parquet is a columnar storage format that provides several advantages over traditional CSV files like speed and compression.


Disclaimer: The dataset was obtained from Soccermatics Pro course and does not belong to me.

### Dataset Overview (located in `data_tables/`):
- `players.parquet`: Mapping of player IDs to names, roles, and nationalities
- `matches.parquet`: Metadata for all games, including match labels and dates
- `minutes.parquet`:  minutes played per player per match for normalized stat calculation
- `teams.parquet`: Reference data for the Premier League teams.
- `passes_dataset.parquet`: The processed dataset containing passes merged with subsequent shot outcomes
