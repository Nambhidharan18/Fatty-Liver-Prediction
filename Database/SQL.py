import sqlite3

# Connect to (or create) the database
conn = sqlite3.connect("reports.db")
cursor = conn.cursor()

# Create the table with your specified columns
cursor.execute("""
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT UNIQUE,
    name TEXT,
    age INTEGER,
    gender TEXT,
    disease TEXT,
    risk_level TEXT,
    report BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Database and table created successfully.")
