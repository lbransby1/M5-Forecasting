# M5 Forecasting Engine 

> **Live Production API:** [m5forecasting.info](https://m5forecasting.info)  
> **Infrastructure:** Docker + FastAPI + LightGBM + Railway + Streamlit

![M5 Engine Demo](https://raw.githubusercontent.com/lbransby1/M5-Forecasting/832b2211cd305b5034f34cde4fa2f5dd3bd75f35/images/m5-demo.gif)

##  The Challenge
Retailers face a multi-billion dollar "Inventory Gap." While baseline models predict the *mean*, real-world supply chain decisions require **Quantile Estimates** to calculate **Safety Stock** and mitigate the risks of intermittent demand. 

This project delivers a production-grade forecasting engine capable of localized inference across **10 distinct geographical regions**, handling **58M+ rows** of historical data with sub-second latency.

##  System Architecture
The system is architected as a decoupled microservice to ensure separation of concerns and computational efficiency:

1.  **Backend (FastAPI):** High-performance inference engine serving 9 distinct quantile boosters. It manages **recursive autoregressive loops** to maintain feature integrity over 28-day horizons.
2.  **ETL Pipeline (Polars):** Leverages Rust-backed Polars for server-side feature extraction, achieving a 98% reduction in memory overhead (46GB down to 700MB) compared to standard Pandas.
3.  **Frontend (Streamlit):** An interactive dashboard providing probabilistic "Fan Charts" and inventory metrics.
4.  **DevOps:** Containerized via multi-stage Docker builds and deployed with automated CI/CD via GitHub Actions and Railway.

##  Key Technical Milestones
* **Recursive Stability:** Engineered a custom loop that feeds predicted values back into rolling windows, utilizing a **weighted-quantile expectation** to prevent autoregressive decay and maintain forecast "energy".
* **Localized Inference:** Implemented store-specific filtering, allowing users to query unique item-store combinations (e.g., California vs. Wisconsin demand) to account for regional exogenous factors like SNAP benefit cycles.
* **Probabilistic Uncertainty:** Utilized **Quantile Regression** (0.005 to 0.995) to generate calibrated prediction intervals. This allows for "Safety Stock" calculation by targeting specific risk-tolerance levels (e.g., 75th or 95th percentile).
* **Human-Centric Mapping:** Integrated a mapping layer that translates granular SKU-level codes into human-readable product names (e.g., "Decorative Craft Stickers"), improving searchability for non-technical stakeholders.

##  Technical Stack
* **Modeling:** LightGBM (Gradient Boosted Trees)
* **Data Processing:** Polars, NumPy, Pandas
* **API Framework:** FastAPI, Uvicorn
* **Visualization:** Plotly, Streamlit
* **Environment:** Docker, Python 3.11

##  Local Development

### Prerequisites
* Docker and Docker Compose
* Python 3.11+

### Steps
```bash
# Clone the repository
git clone [https://github.com/lbransby1/M5-Forecasting.git](https://github.com/lbransby1/M5-Forecasting.git)

# Build and run with Docker Compose
docker-compose up --build
