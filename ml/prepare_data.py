import pandas as pd

employees = pd.read_csv(
    "../data/sprint_resource_allocation_1000_employees.csv"
)

history = pd.read_csv(
    "../data/historical_task_assignment_5000.csv"
)

print("Employee dataset:")
print(employees.head())

print("\nEmployee shape:")
print(employees.shape)

print("\nHistorical dataset:")
print(history.head())

print("\nHistorical shape:")
print(history.shape)

print("\nColumns:")
print(history.columns)