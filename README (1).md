# Safe Semantic Planner using D* Lite

This project implements a **Safe Semantic Planner** using the **D* Lite** path planning algorithm. It computes safe paths while avoiding unsafe ("bad") states and supports efficient replanning when the environment changes. The project also compares the performance of **D* Lite** with **A*** search.

## Features

- Safe path planning
- Dynamic replanning using D* Lite
- A* implementation for comparison
- Six example test scenarios
- Performance evaluation on random graphs
- Unit tests

## Project Structure

```
safe_planner/
│── astar.py            # A* implementation
│── dstar_lite.py       # D* Lite implementation
│── graph.py            # Graph model
│── models.py           # Data models
│── scenarios.py        # Sample planning scenarios
│── main.py             # Demo program
│── experiments.py      # Performance experiments
│── test_planner.py     # Unit tests
│── requirements.txt
│── results.csv
│── results.png
```

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run the demo:

```bash
python main.py
```

Run the test cases:

```bash
python test_planner.py
```

Run the performance experiments:

```bash
python experiments.py
```

The experiment generates:

- `results.csv`
- `results.png`

## Algorithms Used

- D* Lite
- A*
- Euclidean heuristic

## Sample Results

The experiments compare:

- Number of expanded states
- Replanning time
- D* Lite vs A* performance

Example output:

- `results.csv`
- `results.png`

## Author

**MK Nandana**  
Roll No: **TCR24CS047**