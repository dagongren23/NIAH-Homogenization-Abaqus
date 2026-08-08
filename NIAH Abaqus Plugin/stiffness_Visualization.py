"""
NIAH-ABAQUS: 3. stiffness Visualization (Standalone GUI Version)
Author: Zhihui Liu
Date:   2026/04/07
Email:  dutme_lzh@163.com
"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# Local plotting and stiffness-file readers.
from Visualization_utility_function import visual, visual_plate_polar
from read_txt4property import read_stiffness_txt

def draw_CH(filepath, out_fig, stru_type, shell_thickness):
    """Load an effective stiffness matrix and create its visualization."""
    try:
        stiffness = read_stiffness_txt(filepath, stru_type)
        CH = stiffness['EH']

        # Validate the expected stiffness-matrix dimensions.
        if CH.shape == (6, 6):
            print("Matrix loaded successfully.")
            if stru_type == '3D':
                plt = visual(CH)
                plt.savefig(out_fig, dpi=600, bbox_inches='tight', pad_inches=0.05)
                plt.show()

            elif stru_type == 'shell':
                h = float(shell_thickness)
                plt, data = visual_plate_polar(CH, h, ntheta=361, title_prefix="NIAH ")
                plt.savefig(out_fig, dpi=600, bbox_inches='tight', pad_inches=0.02, transparent=True)
                plt.show()

            messagebox.showinfo("Success", f"Visualization successful!\nSaved to:\n{out_fig}")
        else:
            error_msg = f"Error: The loaded matrix is not 6x6. Its shape is {CH.shape}."
            print(error_msg)
            messagebox.showerror("Matrix Error", error_msg)

    except Exception as e:
        error_msg = f"Error loading the file or plotting: {e}"
        print(error_msg)
        messagebox.showerror("Execution Error", error_msg)


class NiahVisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NIAH Stiffness Visualization")
        self.root.geometry("450x280")
        self.root.resizable(False, False)

        # User-interface state.
        self.filepath_var = tk.StringVar()
        self.stru_type_var = tk.StringVar(value="shell")
        self.thickness_var = tk.StringVar(value="0.1")

        # Build the user interface.
        self._build_gui()

    def _build_gui(self):
        padding = {'padx': 10, 'pady': 10}

        # 1. File selection.
        frame_file = tk.LabelFrame(self.root, text="1. Select Data File")
        frame_file.pack(fill="x", **padding)

        tk.Entry(frame_file, textvariable=self.filepath_var, state='readonly', width=45).pack(side="left", padx=5, pady=5)
        tk.Button(frame_file, text="Browse...", command=self.browse_file).pack(side="right", padx=5, pady=5)

        # 2. Model parameters.
        frame_params = tk.LabelFrame(self.root, text="2. Parameters")
        frame_params.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_params, text="Structure Type:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Radiobutton(frame_params, text="3D", variable=self.stru_type_var, value="3D", command=self.toggle_thickness).grid(row=0, column=1, sticky="w")
        tk.Radiobutton(frame_params, text="Shell", variable=self.stru_type_var, value="shell", command=self.toggle_thickness).grid(row=0, column=2, sticky="w")

        tk.Label(frame_params, text="Shell Thickness (h):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.entry_thickness = tk.Entry(frame_params, textvariable=self.thickness_var, width=10)
        self.entry_thickness.grid(row=1, column=1, columnspan=2, sticky="w", padx=5)

        # 3. Visualization action.
        frame_run = tk.Frame(self.root)
        frame_run.pack(fill="x", **padding)

        tk.Button(frame_run, text="Run Visualization", font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white",
                  command=self.run_visualization, height=2).pack(fill="x")

    def toggle_thickness(self):
        """Disable the plate-thickness field for three-dimensional models."""
        if self.stru_type_var.get() == "3D":
            self.entry_thickness.config(state="disabled")
        else:
            self.entry_thickness.config(state="normal")

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Stiffness TXT File",
            filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))
        )
        if filename:
            self.filepath_var.set(filename)

    def run_visualization(self):
        filepath = self.filepath_var.get()
        stru_type = self.stru_type_var.get()
        thickness = self.thickness_var.get()

        if not filepath or not os.path.exists(filepath):
            messagebox.showwarning("Warning", "Please select a valid input TXT file.")
            return

        # Derive the figure directory from the selected result directory.
        file_dir, file_name = os.path.split(filepath)
        base_name, _ = os.path.splitext(file_name)

        # Use the sibling NIAH_CH_fig directory when the standard layout exists.
        parent_dir, current_folder = os.path.split(file_dir)
        if current_folder == 'NIAH_CH_txt':
            fig_path = os.path.join(parent_dir, 'NIAH_CH_fig')
        else:
            fig_path = os.path.join(file_dir, 'NIAH_CH_fig')

        os.makedirs(fig_path, exist_ok=True)
        out_fig = os.path.join(fig_path, f"{base_name}.svg")

        # Validate the plate-thickness input.
        try:
            thick_val = float(thickness)
        except ValueError:
            messagebox.showerror("Input Error", "Shell thickness must be a number.")
            return

        # Generate the requested plot.
        print(f"Reading from: {filepath}")
        print(f"Outputting to: {out_fig}")
        draw_CH(filepath, out_fig, stru_type, thick_val)


if __name__ == "__main__":
    root = tk.Tk()
    app = NiahVisApp(root)
    root.mainloop()
