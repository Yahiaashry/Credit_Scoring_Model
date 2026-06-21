# Credit Scoring Model

##  Objective
The primary goal of this project is to predict an individual's creditworthiness based on their past financial data. This allows financial institutions to assess the risk of lending money to a client.

##  Approach
I implemented a **Random Forest Classifier**, which is highly effective for tabular financial data due to its ability to handle non-linear relationships and prevent overfitting.

### Key Implementation Details:
- **Dataset**: A synthetic financial dataset mimicking real-world data, including features like annual income, age, employment history, total debt, and payment history.
- **Feature Engineering**: I created a custom `debt_to_income` ratio, which is a critical indicator in the banking industry for assessing a borrower's ability to manage and repay debt.
- **Preprocessing**: Used `StandardScaler` to normalize features, ensuring that variables with large scales (like income) do not dominate the model.
- **Evaluation Metrics**: The model is assessed using:
  - **Accuracy**: Overall correctness.
  - **Precision & Recall**: To measure the trade-off between false positives and false negatives.
  - **F1-Score**: The harmonic mean of precision and recall.
  - **ROC-AUC**: To evaluate the model's ability to distinguish between classes.

## 📊 Performance Results
- **Accuracy**: ~88.25% (Exceeds the 85% minimum requirement)
- **Training Time**: $\approx 0.2$ seconds
- **Testing Time**: $\approx 0.05$ seconds (Well within the 1-minute limit)

## 📁 Files in this Folder
- `task1_credit_scoring.py`: The optimized production-ready Python script.
- `task1_credit_scoring.ipynb`: A detailed Jupyter Notebook explaining the process.
- `task1_plots.py`: Script to generate relationship visualizations.
- `task1_plots_colab.ipynb`: Colab-ready notebook for interactive graphs.
- `credit_scoring_plots.png`: Generated visualization of feature importance and correlations.

## 🛠️ How to Run
1. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
2. Execute the script: `python3 task1_credit_scoring.py`
3. Run the plotting script: `python3 task1_plots.py`
