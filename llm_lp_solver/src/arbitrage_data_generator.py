import pandas as pd
import numpy as np
import json

# Sample data for Labor Arbitrage Optimization Problem
# This represents a company analyzing optimal role placement across global locations

# Define roles with their characteristics
roles_data = {
    'role_id': ['R001', 'R002', 'R003', 'R004', 'R005', 'R006', 'R007', 'R008', 'R009', 'R010'],
    'role_name': [
        'Software Engineer', 'Data Analyst', 'Customer Support', 'Product Manager',
        'DevOps Engineer', 'UI/UX Designer', 'QA Tester', 'Technical Writer',
        'Business Analyst', 'Project Manager'
    ],
    'skill_level_required': [4, 3, 2, 5, 4, 3, 2, 3, 4, 4],  # 1-5 scale
    'remote_work_compatibility': [5, 4, 3, 4, 5, 4, 2, 5, 3, 3],  # 1-5 scale
    'required_headcount': [15, 8, 12, 3, 6, 4, 10, 2, 5, 4],
    'communication_intensity': [3, 2, 4, 5, 3, 3, 2, 2, 4, 5],  # 1-5 scale (5 = high collaboration needs)
    'time_zone_sensitivity': [2, 1, 5, 4, 2, 2, 3, 1, 3, 4]  # 1-5 scale (5 = must work in business hours)
}

# Define geographical locations with their cost and capability profiles
locations_data = {
    'location_id': ['L001', 'L002', 'L003', 'L004', 'L005', 'L006', 'L007'],
    'location_name': [
        'San Francisco, USA', 'Austin, USA', 'Krakow, Poland', 'Bangalore, India',
        'Manila, Philippines', 'Buenos Aires, Argentina', 'Lisbon, Portugal'
    ],
    'cost_multiplier': [1.0, 0.75, 0.45, 0.35, 0.25, 0.40, 0.55],  # Relative to base cost
    'talent_availability': [5, 4, 4, 5, 3, 3, 4],  # 1-5 scale
    'infrastructure_quality': [5, 5, 4, 4, 3, 3, 4],  # 1-5 scale
    'time_zone_offset': [0, 0, 9, 10.5, 16, 5, 8],  # Hours from company HQ (SF)
    'language_compatibility': [5, 5, 4, 4, 4, 3, 4],  # 1-5 scale (English proficiency)
    'max_capacity': [50, 40, 30, 60, 35, 25, 20],  # Maximum employees per location
    'setup_cost': [0, 15000, 25000, 30000, 35000, 28000, 22000]  # One-time setup cost for new location
}

# Base salary data by role (annual, in USD)
base_salaries = {
    'R001': 140000,  # Software Engineer
    'R002': 95000,   # Data Analyst
    'R003': 55000,   # Customer Support
    'R004': 160000,  # Product Manager
    'R005': 130000,  # DevOps Engineer
    'R006': 85000,   # UI/UX Designer
    'R007': 70000,   # QA Tester
    'R008': 75000,   # Technical Writer
    'R009': 105000,  # Business Analyst
    'R010': 120000   # Project Manager
}

# Compatibility matrix (how well each role fits each location)
# Based on factors like skill availability, infrastructure needs, etc.
compatibility_matrix = {
    'R001': {'L001': 5, 'L002': 5, 'L003': 4, 'L004': 5, 'L005': 3, 'L006': 3, 'L007': 4},
    'R002': {'L001': 5, 'L002': 4, 'L003': 4, 'L004': 4, 'L005': 3, 'L006': 3, 'L007': 4},
    'R003': {'L001': 4, 'L002': 4, 'L003': 5, 'L004': 4, 'L005': 5, 'L006': 4, 'L007': 4},
    'R004': {'L001': 5, 'L002': 5, 'L003': 3, 'L004': 3, 'L005': 2, 'L006': 3, 'L007': 3},
    'R005': {'L001': 5, 'L002': 5, 'L003': 4, 'L004': 4, 'L005': 3, 'L006': 3, 'L007': 4},
    'R006': {'L001': 5, 'L002': 4, 'L003': 4, 'L004': 3, 'L005': 3, 'L006': 4, 'L007': 4},
    'R007': {'L001': 4, 'L002': 4, 'L003': 5, 'L004': 5, 'L005': 4, 'L006': 4, 'L007': 4},
    'R008': {'L001': 5, 'L002': 4, 'L003': 4, 'L004': 4, 'L005': 4, 'L006': 3, 'L007': 4},
    'R009': {'L001': 5, 'L002': 4, 'L003': 4, 'L004': 4, 'L005': 3, 'L006': 4, 'L007': 4},
    'R010': {'L001': 5, 'L002': 5, 'L003': 4, 'L004': 3, 'L005': 3, 'L006': 3, 'L007': 4}
}

# Create DataFrames
roles_df = pd.DataFrame(roles_data)
locations_df = pd.DataFrame(locations_data)

# Create cost matrix (annual cost per employee for each role-location combination)
cost_matrix = {}
for role_id in roles_df['role_id']:
    cost_matrix[role_id] = {}
    base_salary = base_salaries[role_id]
    for location_id in locations_df['location_id']:
        cost_multiplier = locations_df[locations_df['location_id'] == location_id]['cost_multiplier'].iloc[0]
        cost_matrix[role_id][location_id] = int(base_salary * cost_multiplier)

# Print sample data
print("=== LABOR ARBITRAGE OPTIMIZATION SAMPLE DATA ===\n")

print("ROLES:")
print(roles_df.to_string(index=False))
print("\n")

print("LOCATIONS:")
print(locations_df.to_string(index=False))
print("\n")

print("ANNUAL COST MATRIX (Role x Location):")
cost_df = pd.DataFrame(cost_matrix).T
print(cost_df.to_string())
print("\n")

print("COMPATIBILITY MATRIX (Role x Location, 1-5 scale):")
compatibility_df = pd.DataFrame(compatibility_matrix).T
print(compatibility_df.to_string())
print("\n")

# Save data to files for easy import
roles_df.to_csv('src/data/roles.csv', index=False)
locations_df.to_csv('src/data/locations.csv', index=False)
pd.DataFrame(cost_matrix).T.to_csv('src/data/cost_matrix.csv')
pd.DataFrame(compatibility_matrix).T.to_csv('src/data/compatibility_matrix.csv')

with open('src/data/labor_arbitrage_data.json', 'w') as f:
    json.dump({
        'roles': roles_data,
        'locations': locations_data,
        'cost_matrix': cost_matrix,
        'compatibility_matrix': compatibility_matrix,
        'base_salaries': base_salaries
    }, f, indent=2)

print("Data saved to CSV files and JSON file for easy import into your solver!")
