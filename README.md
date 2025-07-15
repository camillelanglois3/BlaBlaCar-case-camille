# BlaBlaCar Technical Case – Ride Booking Success Prediction

This repository contains the solution to the **technical case for the Junior Data Scientist position at BlaBlaCar**.  
The objective is to build a machine learning pipeline to predict the success of rides published on the platform.

---

## Project Structure

```
project_root/
│
├── data/
│   └── raw/
│       └── data_scientist_case.csv        # Input file required to run the project, you must add it
│
├── src/
│   ├── main.py                        # Main pipeline entry point
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── EDA.py
│   └── model.py
│
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── feature_engineering.ipynb
│   ├── EDA.ipynb
│   └── model.ipynb
├── results/                           # Saved models, plots, metrics...
├── blablacar_presentation.pdf
└── README.md
```

---

## Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/camillelanglois3/BlaBlaCar-case-camille.git
   cd BlaBlaCar-case-camille
   ```

2. **Install dependencies**  
   It’s recommended to use a virtual environment.  
   ```bash
   pip install -r requirements.txt
   ```

3. **Add the input data**  
   Place the provided file `data_scientist_case.csv` in the following directory:  
   ```
   data/raw/data_scientist_case.csv
   ```

---

## How to Run the Pipeline

From the `src/` folder, simply run:

```bash
cd src
python main.py
```

This will automatically:
- Load and preprocess the data  
- Perform feature engineering
- Train and evaluate the final model
- Generate confusion matrix and metrics
- Save all outputs in the `results/` folder

---

## Model Choice

The final model used is **XGBoost**, selected for its strong performance and interpretability using SHAP values.

---

## Context

This project was completed as part of the **BlaBlaCar technical case** for the **Junior Data Scientist** role.

---

## Contact

For any questions or feedback, feel free to reach out on GitHub or via email at camille.langlois.contact@gmail.com.

---

*Made with ❤️ for the BlaBlaCar Data Team*
