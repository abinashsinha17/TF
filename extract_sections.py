import fitz  # PyMuPDF
import os
import json
import shutil

def extract_page_data(pdf_path, output_dir, target_pages):
    print(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for page_num in target_pages:
        # fitz uses 0-based indexing
        page_index = page_num - 1
        try:
            page = doc.load_page(page_index)
        except Exception as e:
            print(f"Could not load page {page_num}: {e}")
            continue
            
        print(f"Processing Page {page_num}...")
        
        # 1. Render the full page image as a background reference
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_path = os.path.join(output_dir, f"page_{page_num}.png")
        pix.save(image_path)
        
        # 2. Extract detailed layout index (Text, Punctuation, embedded Images, and Bounding Boxes)
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])
        
        page_index_data = []
        
        for b_idx, block in enumerate(blocks):
            # Type 0 is Text
            if block["type"] == 0:
                block_text = ""
                # Keep exact punctuation and spacing
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "
                    block_text = block_text.rstrip() + "\n"
                    
                block_text = block_text.strip()
                if block_text:
                    page_index_data.append({
                        "id": f"block_{b_idx}",
                        "type": "text",
                        "bbox": block["bbox"],  # Coordinates [x0, y0, x1, y1]
                        "text": block_text
                    })
                    
            # Type 1 is Image
            elif block["type"] == 1:
                img_filename = f"page_{page_num}_img_{b_idx}.png"
                img_path_out = os.path.join(output_dir, img_filename)
                
                img_bytes = block.get("image")
                if img_bytes:
                    with open(img_path_out, "wb") as f:
                        f.write(img_bytes)
                        
                page_index_data.append({
                    "id": f"block_{b_idx}",
                    "type": "image",
                    "bbox": block["bbox"],  # Coordinates [x0, y0, x1, y1]
                    "src": img_filename
                })
                
        # 3. Save the exact coordinate mapping to JSON
        json_path = os.path.join(output_dir, f"page_{page_num}_index.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(page_index_data, f, indent=4, ensure_ascii=False)
            
        print(f"  -> Extracted text, images, and created coordinate index at {json_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "input", "D01710.pdf")
    output_dir = os.path.join(base_dir, "input", "section")
    
    if not os.path.exists(pdf_path):
        print(f"Error: Original PDF not found at {pdf_path}")
        return
        
    # The specific manual pages we are extracting sections from
    target_pages = [38, 39, 53, 54, 76, 77]
    
    extract_page_data(pdf_path, output_dir, target_pages)
    print("Process complete!")

if __name__ == "__main__":
    main()
