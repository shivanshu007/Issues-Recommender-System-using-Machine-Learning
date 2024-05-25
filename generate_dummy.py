import pandas as pd
import random

# List of sample car issues and corresponding solutions
issues = [
    ("Car won't start, battery is dead", "Replace or recharge the battery"),
    ("Engine overheating", "Check coolant levels and radiator"),
    ("Tire pressure warning", "Check and inflate tires to proper pressure"),
    ("Strange noise from brakes", "Inspect brake pads and rotors"),
    ("Car pulls to one side", "Check wheel alignment and tire condition"),
    ("Oil leak", "Inspect and repair or replace the oil gasket"),
    ("Headlights not working", "Check and replace the bulbs or fuse"),
    ("AC not cooling", "Check refrigerant levels and compressor"),
    ("Steering wheel vibration", "Inspect wheel balance and suspension"),
    ("Transmission slipping", "Check transmission fluid levels and condition"),
    ("Check engine light on", "Scan for error codes and address the issues"),
    ("Battery draining quickly", "Inspect the alternator and battery connections"),
    ("Brake pedal feels soft", "Check brake fluid levels and brake lines"),
    ("Unusual exhaust smoke", "Inspect the engine and exhaust system"),
    ("Car makes clicking noise when turning", "Inspect CV joints and axles"),
    ("Fuel efficiency has dropped", "Check the air filter and fuel injectors"),
    ("Windshield wipers not working", "Inspect the wiper motor and fuse"),
    ("Power windows not working", "Check the window motor and switch"),
    ("Heater not working", "Inspect the heater core and thermostat"),
    ("Car stalls while driving", "Check the fuel pump and ignition system")
]

# Generate 100 random entries
data = []
for i in range(100):
    issue, solution = random.choice(issues)
    data.append([issue, solution])

# Create a DataFrame
df = pd.DataFrame(data, columns=['issue_description', 'solution'])

# Save the DataFrame to a CSV file
df.to_csv('car_issues.csv', index=False)

print("Dummy dataset with 100 entries created and saved to 'car_issues.csv'.")