import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog

# Assume SSCD is defined/imported elsewhere
from SSCD import SSCD

class SSCDGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SSCD Tool")

        # AI Label Selection
        self.label_frame = tk.LabelFrame(root, text="Select AI Label")
        self.label_frame.pack(padx=10, pady=10, fill="x")

        self.label_var = tk.StringVar(value='1')
        tk.Radiobutton(self.label_frame, text="Label 1", variable=self.label_var, value='1').pack(anchor='w')
        tk.Radiobutton(self.label_frame, text="Label 2", variable=self.label_var, value='2').pack(anchor='w')

        # Skimming option
        self.skim_var = tk.BooleanVar()
        tk.Checkbutton(root, text="Perform Skimming", variable=self.skim_var).pack(anchor='w', padx=10)

        self.skim_frame = tk.Frame(root)
        self.skim_frame.pack(padx=20, pady=5, fill="x")

        tk.Label(self.skim_frame, text="Search Terms (comma-separated):").pack(anchor='w')
        self.search_entry = tk.Entry(self.skim_frame)
        self.search_entry.pack(fill="x")

        tk.Label(self.skim_frame, text="Number of Rounds:").pack(anchor='w')
        self.rounds_entry = tk.Spinbox(self.skim_frame, from_=1, to=100)
        self.rounds_entry.pack(fill="x")

        tk.Label(self.skim_frame, text="Resolution (Width):").pack(anchor='w')
        # First variable (Entry)
        self.width_entry = tk.Entry(self.skim_frame)
        self.width_entry.pack(fill="x")
        self.width_entry.insert(0, self.root.winfo_screenwidth())

        # Second variable (Entry)
        tk.Label(self.skim_frame, text="Resolution (Height):").pack(anchor='w')
        self.height_entry = tk.Entry(self.skim_frame)
        self.height_entry.pack(fill="x")
        self.height_entry.insert(0, self.root.winfo_screenheight())

        # Segmentation option
        self.segment_var = tk.BooleanVar()
        tk.Checkbutton(root, text="Perform Detection and Clipping", variable=self.segment_var).pack(anchor='w', padx=10)

        # Submit button
        tk.Button(root, text="Run", command=self.run_sscd).pack(pady=20)

    def run_sscd(self):
        ai_label = self.label_var.get()
        sscd = SSCD(f"label{ai_label}")

        if self.skim_var.get():
            search_list = [term.strip() for term in self.search_entry.get().split(',')]
            folder_list = [term.replace(' ', '-') for term in search_list]
            sscd.skimmer(search_list, folder_list, int(self.rounds_entry.get()), int(self.width_entry.get()), int(self.height_entry.get()))

        if self.segment_var.get():
            sscd.load_dataset()
            sscd.image_segmentation()

        messagebox.showinfo("Done", "Operation completed successfully.")

# Usage
if __name__ == "__main__":
    root = tk.Tk()
    app = SSCDGuiApp(root)
    root.mainloop()
