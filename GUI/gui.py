import sqlite3
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk

# Connect to SQLite database
def fetch_reports():
    conn = sqlite3.connect("C:/Users/ELCOT/Desktop/Final Year Project/Database/reports.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, patient_id, name, age, gender, disease, risk_level, created_at FROM reports")
    records = cursor.fetchall()
    conn.close()
    return records

def export_pdf():
    selected = tree.focus()
    if not selected:
        messagebox.showwarning("No selection", "Please select a report to export.")
        return

    item = tree.item(selected)
    patient_id = item['values'][1]

    conn = sqlite3.connect("C:/Users/ELCOT/Desktop/Final Year Project/Database/reports.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, report FROM reports WHERE patient_id = ?", (patient_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        name, pdf_data = result
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")],
            initialfile=f"{name}_report"
        )
        if filepath:
            with open(filepath, "wb") as f:
                f.write(pdf_data)
            messagebox.showinfo("Success", f"Report saved as: {filepath}")
    else:
        messagebox.showerror("Error", "Failed to retrieve report.")

# GUI Setup
root = tk.Tk()
root.title("Patient Report Viewer")
root.configure(bg="#1c1c1c")

# Set window icon
root.iconbitmap("C:/Users/ELCOT/Desktop/Final Year Project/Picture/icon.ico")

# Display logo at top
logo = Image.open("C:/Users/ELCOT/Desktop/Final Year Project/Picture/icon.png")
logo = logo.resize((800, 300))
photo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(root, image=photo)
logo_label.image = photo
logo_label.pack(pady=5)



# === Scrollable Treeview with Horizontal and Vertical Scrollbars ===
frame = tk.Frame(root)
frame.pack(expand=True, fill="both", padx=10, pady=10)

# Scrollbars
scroll_x = tk.Scrollbar(frame, orient="horizontal")
scroll_y = tk.Scrollbar(frame, orient="vertical")

# Treeview Widget
tree = ttk.Treeview(
    frame,
    columns=("ID", "Patient ID", "Name", "Age", "Gender", "Disease", "Risk Level", "Created At"),
    show="headings",
    xscrollcommand=scroll_x.set,
    yscrollcommand=scroll_y.set
)
style = ttk.Style()
style.theme_use("default")
style.configure("Treeview", background="#1c1c1c", foreground="white", fieldbackground="#1c1c1c")
style.configure("Treeview.Heading", background="#333333", foreground="white")


# Attach scrollbars to treeview
scroll_x.config(command=tree.xview)
scroll_y.config(command=tree.yview)
scroll_x.pack(side="bottom", fill="x")
scroll_y.pack(side="right", fill="y")
tree.pack(side="left", fill="both", expand=True)

# Define column headings
columns = ("ID", "Patient ID", "Name", "Age", "Gender", "Disease", "Risk Level", "Created At")
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=130, anchor="center")  # You can adjust width per column


# Load Data
records = fetch_reports()
for row in records:
    tree.insert("", "end", values=row)

# Export Button
btn_export = tk.Button(root, text="Export Selected PDF Report", command=export_pdf,  bg="#005a87", fg="white")
btn_export.pack(pady=10)

root.geometry("1000x400")
root.mainloop()

