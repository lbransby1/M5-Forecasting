# M5 Forecasting Engine 📈

> **Live Production API:** [m5forecasting.info](https://m5forecasting.info)  
> **Infrastructure:** Docker + FastAPI + LightGBM + Railway

![M5 Engine Demo](https://raw.githubusercontent.com/lbransby1/M5-Forecasting/832b2211cd305b5034f34cde4fa2f5dd3bd75f35/images/m5-demo.gif)

## The Challenge
Retailers face a multi-billion dollar "Inventory Gap": overstocking leads to waste, while understocking leads to lost revenue. Most baseline models predict the *mean* (the average), but real-world supply chain decisions require **Quantile Estimates** to calculate **Safety Stock**.

This project implements a high-performance forecasting engine designed to provide hierarchical quantile predictions for the M5 Uncertainty dataset.

## System Architecture
The system is built as a decoupled microservice to ensure scalability and separation of concerns:

1. **Inference Engine:** Optimized LightGBM model utilizing recursive lag features and rolling window statistics.
2. **API Layer:** FastAPI wrapper handles asynchronous requests and serves model weights via a RESTful endpoint.
3. **Containerization:** Multi-stage Docker build to minimize image size and ensure environment parity.
4. **Cloud Infrastructure:** Continuous Deployment (CD) via Railway, utilizing automated health checks and SSL termination.



## Technical Implementation
* **Model:** Gradient Boosted Trees (LightGBM) for efficient model training
* **Uncertainty:** Implemented **Quantile Regression** (0.005 to 0.995) to generate prediction intervals, allowing users to tune their risk-tolerance.
* **Optimization:** Leveraged **Polars** for feature engineering to achieve a 5x speedup over standard Pandas operations.

## Local Development
```bash
# Clone the repository
git clone [https://github.com/lbransby1/M5-Forecasting.git](https://github.com/lbransby1/M5-Forecasting.git)

# Build the Docker container
docker build -t m5-engine .

# Run the inference server
docker run -p 8000:8000 m5-engine
```

## Future Development
- Improve model calibration and accuracy through hyper-parameter tuning, tracking performance on Weights and Biases ( in progress )
- Implement Temporal Fusion Transformers to process future SNAP sales effectively ( in progress )
- Upscale system to use all data from all stores. Approx 30x size increase
- Compare several models to analyze in when each model is best fit 

