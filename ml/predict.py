import pandas as pd
import joblib


# ==========================================
# 1. LOAD MODEL
# ==========================================

model = joblib.load("model/developer_model.pkl")


# ==========================================
# 2. LOAD EMPLOYEE DATASET
# ==========================================

employees = pd.read_csv(
    "../data/sprint_resource_allocation_1000_employees.csv"
)


# ==========================================
# 3. NEW TASK DETAILS
# ==========================================

task_type = "Backend Development"

required_skills = [
    "Java",
    "Spring_Boot",
    "SQL"
]

story_points = 8

expected_hours = story_points * 5


# ==========================================
# 4. CALCULATE SKILL MATCH FOR EACH EMPLOYEE
# ==========================================

def calculate_skill_match(employee, skills):

    scores = []

    for skill in skills:

        column = skill + "_Proficiency"

        if column in employee:
            scores.append(employee[column])

    if len(scores) == 0:
        return 0

    # Convert average proficiency (1-5)
    # into percentage (0-100)

    return (sum(scores) / len(scores)) / 5 * 100


# ==========================================
# 5. CREATE CANDIDATE DATA
# ==========================================

candidates = []

for _, employee in employees.iterrows():

    skill_match = calculate_skill_match(
        employee,
        required_skills
    )

    candidates.append({

        "Employee_ID":
            employee["Employee_ID"],

        "Skill_Match_Percent":
            skill_match,

        "Experience_Years":
            employee["Experience_Years"],

        "Employee_Workload_Percent":
            employee["Workload_Percent"],

        "Employee_Availability_Percent":
            employee["Availability_Percent"],

        "Past_Performance_Score":
            employee["Past_Performance_Score"],

        "Story_Points":
            story_points,

        "Expected_Hours":
            expected_hours
    })


candidates_df = pd.DataFrame(candidates)


# ==========================================
# 6. FEATURES FOR XGBOOST
# ==========================================

features = [

    "Skill_Match_Percent",

    "Experience_Years",

    "Employee_Workload_Percent",

    "Employee_Availability_Percent",

    "Past_Performance_Score",

    "Story_Points",

    "Expected_Hours"
]


X = candidates_df[features]


# ==========================================
# 7. PREDICT FOR ALL EMPLOYEES
# ==========================================

probabilities = model.predict_proba(X)[:, 1]


candidates_df["Suitability_Score"] = (
    probabilities * 100
)


# ==========================================
# 8. SORT EMPLOYEES
# ==========================================

ranked = candidates_df.sort_values(
    by="Suitability_Score",
    ascending=False
)


# ==========================================
# 9. GET TOP 5
# ==========================================

top_5 = ranked.head(5).copy()


top_5.insert(
    0,
    "Rank",
    range(1, 6)
)


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

# ==========================================
# TOP 5 EMPLOYEES
# ==========================================

top_5 = ranked.head(5)

print("\n========================================")
print("       TOP 5 RECOMMENDED EMPLOYEES")
print("========================================")

for i, (_, employee) in enumerate(top_5.iterrows(), start=1):

    reasons = []

    if employee["Skill_Match_Percent"] >= 70:
        reasons.append("high skill match")

    if employee["Past_Performance_Score"] >= 4:
        reasons.append("strong past performance")

    if employee["Employee_Workload_Percent"] <= 50:
        reasons.append("low current workload")

    if employee["Employee_Availability_Percent"] >= 50:
        reasons.append("high availability")

    if employee["Experience_Years"] >= 3:
        reasons.append("relevant experience")

    reason = ", ".join(reasons)

    print(f"\n{i}. {employee['Employee_ID']}")
    print(f"   Reason: {reason}")