## Data Version Control (DVC) Pipeline

This project uses DVC to maintain an auditable and reproducible data pipeline without bloating the Git repository history with massive text datasets.

### How to Reproduce the Data Environment

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd insurance-risk-analytics
   ```
2. **Install requirements:**

```bash
pip install -r requirements.txt
Pull the tracked dataset from local remote storage:
```

````bash
dvc pull```
````
