import tkinter as tk
from tkinter import filedialog, messagebox
from Functions import calculate_stats

# Function to open files
def open_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

# Function to calculate stats
def calculate_lufs_gui():
    inputfile = input_file_entry.get()
    outputfile = output_file_entry.get()
    difffile = diff_file_entry.get()

    if not inputfile or not outputfile:
        messagebox.showerror("Error", "Please select both WAV files.")
        return

    try:
        results = calculate_stats(inputfile, outputfile, difffile)
        
        accuracy_score_text = f"Accuracy Score: {results['combined_score']:.3f}"
        
        result_text = [
            ("PEAQ:", f"{results['peaq']:.2f}"),
            ("LUFS:", f"{results['best_lufs']:.2f} [LUFS]"),
            ("SNR:", f"{results['snr']:.2f} [dB]"),
            ("THD:", f"{results['thd']:.6f} [%]"),
            ("Gain:", f"{results['best_scale']:.2f}"),
            ("MSE:", f"{results['mse']:.6f}")
        ]
       
        result_label_combined.config(text=accuracy_score_text)
        for i, (label_text, value_text) in enumerate(result_text):
            result_label_table[i][0].config(text=label_text)
            result_label_table[i][1].config(text=value_text)
        
        messagebox.showinfo("Calculation Complete", "Calculation was successful.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Main window creation
root = tk.Tk()
root.title("SoundMatchAnalyzer (SMA)")

# GUI design for Dark Mode
root.geometry("800x600")  # Larger window size
root.configure(bg="#2e2e2e")  # Dark background

# Frame for file selection
frame_files = tk.Frame(root, bg="#2e2e2e", pady=10)
frame_files.pack(fill="x")

input_file_label = tk.Label(frame_files, text="Amplifier File (WAV):", bg="#2e2e2e", fg="white", font=("Helvetica", 12, "bold"))
input_file_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
input_file_entry = tk.Entry(frame_files, bg="#cccccc", fg="black", insertbackground="white")  # Lighter entry field
input_file_entry.grid(row=0, column=1, padx=10, pady=10, sticky="we")
input_file_button = tk.Button(frame_files, text="Browse", command=lambda: open_file(input_file_entry), bg="#2196F3", fg="black")  # Blue color for the button
input_file_button.grid(row=0, column=2, padx=10, pady=10, sticky="e")

output_file_label = tk.Label(frame_files, text="Inference File (WAV):", bg="#2e2e2e", fg="white", font=("Helvetica", 12, "bold"))
output_file_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
output_file_entry = tk.Entry(frame_files, bg="#cccccc", fg="black", insertbackground="white")  # Lighter entry field
output_file_entry.grid(row=1, column=1, padx=10, pady=10, sticky="we")
output_file_button = tk.Button(frame_files, text="Browse", command=lambda: open_file(output_file_entry), bg="#2196F3", fg="black")  # Blue color for the button
output_file_button.grid(row=1, column=2, padx=10, pady=10, sticky="e")

diff_file_label = tk.Label(frame_files, text="Difference File (optional):", bg="#2e2e2e", fg="white", font=("Helvetica", 12, "bold"))
diff_file_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
diff_file_entry = tk.Entry(frame_files, bg="#cccccc", fg="black", insertbackground="white")  # Lighter entry field
diff_file_entry.grid(row=2, column=1, padx=10, pady=10, sticky="we")
diff_file_button = tk.Button(frame_files, text="Browse", command=lambda: open_file(diff_file_entry), bg="#2196F3", fg="black")  # Blue color for the button
diff_file_button.grid(row=2, column=2, padx=10, pady=10, sticky="e")

# Configure column weight to make entry fields take the remaining space
frame_files.columnconfigure(1, weight=1)

# Frame for calculation button and results
frame_buttons = tk.Frame(root, bg="#2e2e2e", pady=10)
frame_buttons.pack(fill="x")

calculate_button = tk.Button(frame_buttons, text="Calculate", command=calculate_lufs_gui, bg="#2196F3", fg="black", font=("Helvetica", 12, "bold"))
calculate_button.pack(pady=10)

result_label_combined = tk.Label(root, text="Accuracy Score will be displayed here...", bg="#2e2e2e", fg="white", font=("Helvetica", 18, "bold"), anchor="w", justify="left")
result_label_combined.pack(pady=10)

result_frame_table = tk.Frame(root, bg="#2e2e2e")
result_frame_table.pack()

result_label_table = []
for i in range(10):  # Adjusting the loop to 10 items
    frame = tk.Frame(result_frame_table, bg="#2e2e2e")
    frame.pack(fill="x", pady=2)
    label_text = tk.Label(frame, text="", bg="#2e2e2e", fg="white", font=("Helvetica", 12, "bold"), anchor="e", justify="right")
    label_text.pack(side="left", padx=(0, 5))
    label_value = tk.Label(frame, text="", bg="#2e2e2e", fg="white", font=("Helvetica", 12), anchor="w", justify="left")
    label_value.pack(side="left")
    result_label_table.append((label_text, label_value))

# Main window start
root.mainloop()
