import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import time

def generate_credit_data(n_samples=1000):
    """
    Generates a synthetic dataset for credit scoring.
    """
    np.random.seed(42)
    
    # Features
    income = np.random.normal(50000, 15000, n_samples)
    age = np.random.randint(18, 70, n_samples)
    employment_years = np.random.randint(0, 40, n_samples)
    debt = np.random.normal(10000, 5000, n_samples)
    payment_history = np.random.uniform(0, 1, n_samples) # % of on-time payments
    
    # Target generation logic: creditworthy if a combination of factors is favorable
    # We create a linear combination and then apply a threshold
    score = (0.3 * (income / 50000) + 
             0.4 * payment_history + 
             0.2 * (employment_years / 20) - 
             0.3 * (debt / 10000))
    
    # Adding some noise to make it realistic but keep it learnable
    noise = np.random.normal(0, 0.1, n_samples)
    final_score = score + noise
    
    # Target: 1 if creditworthy, 0 otherwise
    target = (final_score > 0.5).astype(int)
    
    df = pd.DataFrame({
        'income': income,
        'age': age,
        'employment_years': employment_years,
        'debt': debt,
        'payment_history': payment_history,
        'creditworthy': target
    })
    
    return df

def run_credit_scoring_model():
    # 1. Data Generation
    print("Generating synthetic financial data...")
    df = generate_credit_data(2000)
    
    # 2. Feature Engineering & Preprocessing
    X = df.drop('creditworthy', axis=1)
    y = df['creditworthy']
    
    # Debt-to-Income ratio is a key financial metric
    X['debt_to_income'] = X['debt'] / X['income']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Model Training
    # Using Random Forest for optimization and high accuracy.
    # It doesn't use epochs, but it's highly efficient for tabular data.
    print("Training Random Forest Model...")
    start_train = time.time()
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    end_train = time.time()
    
    # 4. Model Testing & Evaluation
    print("Testing model...")
    start_test = time.time()
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    end_test = time.time()
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    test_time = end_test - start_test
    
    print("\n" + "="*30)
    print("CREDIT SCORING MODEL RESULTS")
    print("="*30)
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("-" * 30)
    print(f"Training Time: {end_train - start_train:.4f} seconds")
    print(f"Testing Time:  {test_time:.4f} seconds")
    print("="*30)
    
    if accuracy >= 0.85:
        print("\nSUCCESS: Accuracy is above 85%!")
    else:
        print("\nFAILURE: Accuracy is below 85%. Tuning required.")

    print("\nDetailed Classification Report:\n")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_credit_scoring_model()
