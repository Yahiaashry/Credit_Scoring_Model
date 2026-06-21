import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def generate_credit_data(n_samples=2000):
    np.random.seed(42)
    income = np.random.normal(50000, 15000, n_samples)
    age = np.random.randint(18, 70, n_samples)
    employment_years = np.random.randint(0, 40, n_samples)
    debt = np.random.normal(10000, 5000, n_samples)
    payment_history = np.random.uniform(0, 1, n_samples)
    score = (0.3 * (income / 50000) + 0.4 * payment_history + 0.2 * (employment_years / 20) - 0.3 * (debt / 10000))
    target = ((score + np.random.normal(0, 0.1, n_samples)) > 0.5).astype(int)
    return pd.DataFrame({'income': income, 'age': age, 'employment_years': employment_years, 'debt': debt, 'payment_history': payment_history, 'creditworthy': target})

def run_credit_scoring_with_plots():
    df = generate_credit_data()
    X = df.drop('creditworthy', axis=1)
    y = df['creditworthy']
    X['debt_to_income'] = X['debt'] / X['income']
    
    # Model training for Feature Importance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # VISUALIZATIONS
    plt.figure(figsize=(18, 5))
    
    # Plot 1: Correlation Heatmap
    plt.subplot(1, 3, 1)
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    
    # Plot 2: Income vs Debt colored by Creditworthy
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df, x='income', y='debt', hue='creditworthy', alpha=0.6)
    plt.title("Income vs Debt Relationship")
    
    # Plot 3: Feature Importance
    plt.subplot(1, 3, 3)
    importances = model.feature_importances_
    sns.barplot(x=importances, y=X.columns)
    plt.title("Model Feature Importance")
    
    plt.tight_layout()
    plt.savefig(" Task 1 /credit_scoring_plots.png")
    print("Plots saved to Task 1 /credit_scoring_plots.png")

if __name__ == "__main__":
    run_credit_scoring_with_plots()
