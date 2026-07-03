import fitz  # PyMuPDF
import os
import glob

def process_pdfs(input_folder):
    # Find all pdf files in the input folder
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))
    
    for pdf_path in pdf_files:
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Create an output folder inside the input folder
        output_folder = os.path.join(input_folder, f"{filename}_extracted")
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Processing {pdf_path}...")
        print(f"Saving data to {output_folder}")
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # 1. Render page to an image
            # High resolution zoom (2x)
            zoom_x = 2.0
            zoom_y = 2.0
            mat = fitz.Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=mat)
            image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
            pix.save(image_path)
            
            # 2. Extract text (optional, but good to have)
            text = page.get_text()
            text_path = os.path.join(output_folder, f"page_{page_num + 1}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            print(f"  Saved page {page_num + 1}")
            
        doc.close()
        print(f"Finished processing {filename}.")

if __name__ == "__main__":
    input_folder = os.path.join(os.path.dirname(__file__), "input")
    process_pdfs(input_folder)
