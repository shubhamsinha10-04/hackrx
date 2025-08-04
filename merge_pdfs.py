import os
from PyPDF2 import PdfMerger

def merge_pdfs(input_folder: str, output_path: str):
    merger = PdfMerger()
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".pdf"):
            filepath = os.path.join(input_folder, filename)
            merger.append(filepath)
    merger.write(output_path)
    merger.close()
    print(f"Merged PDF saved to {output_path}")

if __name__ == "__main__":
    dataset_dir = "dataset"  # Change if your folder has different name/path
    output_file = "dataset/combined.pdf"
    merge_pdfs(dataset_dir, output_file)
