import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --- Scenario: Predicting if someone buys a house based on Age and Salary ---

# Let's create some synthetic data.
# Feature 1: Age (range ~ 20 to 60)
# Feature 2: Salary (range ~ 30,000 to 150,000)
# Target: Bought house (1) or did not buy (0)

# Person A: 30 years old, $100,000 salary
# Person B: 31 years old, $50,000 salary
# Person C: 60 years old, $100,000 salary

print("="*60)
print("1. THE DISTANCE PROBLEM (KNN EXAMPLE)")
print("="*60)

X_train = np.array([
    [30, 100000],  # Person A
    [31, 50000],   # Person B
    [60, 100000]   # Person C
])
y_train = np.array([1, 0, 1])

# We have a NEW Person D. We want to know who they are most similar to.
# Person D is 30 years old with a $50,000 salary.
# Intuitively, are they closer to Person A (same age, huge salary diff) 
# or Person B (1 year diff, same salary)?
X_new = np.array([[30, 50000]])

print(f"Target Person D: Age 30, Salary $50,000")
print(f"Candidate A: Age 30, Salary $100,000")
print(f"Candidate B: Age 31, Salary $50,000")
print(f"Candidate C: Age 60, Salary $100,000\n")

# --- WITHOUT SCALING ---
print("--- WITHOUT SCALING ---")
# Let's manually calculate Euclidean distance
dist_A = np.sqrt( (30 - 30)**2 + (50000 - 100000)**2 )
dist_B = np.sqrt( (30 - 31)**2 + (50000 - 50000)**2 )

print(f"Distance to Person A: {dist_A:.2f}")
print(f"Distance to Person B: {dist_B:.2f}")
print("Conclusion: Without scaling, the algorithm thinks Person D is VERY different from Person A solely because 50000 is a big number. The Age difference is completely ignored!\n")

# --- WITH SCALING ---
print("--- WITH SCALING ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_new_scaled = scaler.transform(X_new)

print("Scaled values for A, B, C:")
print(np.round(X_train_scaled, 2))
print("Scaled values for D:")
print(np.round(X_new_scaled, 2), "\n")

dist_A_scaled = np.sqrt( np.sum((X_new_scaled[0] - X_train_scaled[0])**2) )
dist_B_scaled = np.sqrt( np.sum((X_new_scaled[0] - X_train_scaled[1])**2) )

print(f"Scaled Distance to Person A: {dist_A_scaled:.2f}")
print(f"Scaled Distance to Person B: {dist_B_scaled:.2f}")
print("Conclusion: With scaling, the algorithm pays attention to BOTH Age and Salary equally!\n")


print("="*60)
print("2. THE REGULARIZATION PROBLEM (RIDGE/LASSO EXAMPLE)")
print("="*60)
from sklearn.linear_model import Ridge

# Let's say we are predicting a house price.
# Actual formula: Price = 1000 * Age + 1 * Salary
X_train = np.array([
    [30, 100000], 
    [40, 120000],
    [50, 140000]
])
# Generate y roughly following the formula Price = 1000*Age + 1*Salary
y_train = 1000 * X_train[:, 0] + 1 * X_train[:, 1]

print("--- WITHOUT SCALING ---")
ridge_unscaled = Ridge(alpha=10000) 
ridge_unscaled.fit(X_train, y_train)

print(f"Learned Weight for Age: {ridge_unscaled.coef_[0]:.4f}")
print(f"Learned Weight for Salary: {ridge_unscaled.coef_[1]:.4f}")
print("Note: The model heavily penalized the Age weight because the feature range was small, forcing the model to rely too much on the unpenalized large Salary feature.\n")


print("--- WITH SCALING ---")
ridge_scaled = Ridge(alpha=10) # Smaller alpha now because values are smaller
ridge_scaled.fit(X_train_scaled, y_train)

print(f"Learned Weight for Age: {ridge_scaled.coef_[0]:.4f}")
print(f"Learned Weight for Salary: {ridge_scaled.coef_[1]:.4f}")
print("Note: Now both weights are treated fairly by the regularization penalty because their scales are matched!")
