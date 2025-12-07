# Trading Bot API Backend

This is the new FastAPI-based backend for the Trading Bot. It serves as the core engine for the future React/Next.js frontend.

## Structure
*   **`backend/app`**: Contains the API logic (FastAPI).
    *   `api/endpoints`: Define the URL routes (e.g., `/system/health`, `/strategies/list`).
    *   `core`: Configuration.
*   **`backend/lib`**: Contains the core logic migrated from `v4` (Strategies, ML, Backtesting).

## Setup & Run

1.  **Install Dependencies**:
    ```bash
    pip install -r backend/requirements.txt
    ```

2.  **Run Server**:
    ```bash
    python backend/run.py
    ```

3.  **Access Documentation**:
    Open your browser to: `http://localhost:8000/docs`
    You will see the interactive Swagger UI with all available endpoints.

## Available Endpoints

### System
*   `GET /api/system/health`: Check if the API is online.

### Strategies
*   `GET /api/strategies/list`: List all strategy files available in the system.
*   `GET /api/strategies/ml-status`: Check if the Machine Learning Regime Detector is loaded and active.
