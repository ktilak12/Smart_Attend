import csv
import os

# Read existing CSV
csv_file = "StudentDetails/StudentDetails.csv"
rows = []

if os.path.exists(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

print("Current data:")
for row in rows:
    print(row)

# Check if we need to add serial 1234
serial_exists = False
for row in rows:
    if len(row) > 0 and row[0] == '1234':
        serial_exists = True
        break

if not serial_exists and len(rows) > 0:
    # Add the missing student
    name = input("Enter the name for Serial 1234: ")
    student_id = input("Enter the Student ID: ")
    
    new_row = ['1234', '', student_id, '', name]
    rows.append(new_row)
    
    # Write back
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"✅ Added student: Serial 1234, ID: {student_id}, Name: {name}")
else:
    print("Serial 1234 already exists or no data found")