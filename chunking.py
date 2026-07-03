import os
import glob
import logging
import sys
import json

try:
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", # Clean format for reading JSON outputs
    handlers=[
        logging.FileHandler("chunking.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def production_hierarchical_chunking(layout_data, page_num=1, doc_id="manual_001"):
    """
    Implements Production-Grade Layout-Aware Hierarchical + Semantic Chunking.
    Processes a list of blocks (Text and Images) from the Layout Index JSON.
    Groups lines into Headings -> Paragraphs/Bullets/Captions/Images (Parent-Child).
    """
    logger.info("\n=== Running Production Layout-Aware Hierarchical Chunking ===\n")
    
    parents = []
    
    # Default parent if text starts without a heading
    current_parent = {
        "heading": "Document Start",
        "type": "parent",
        "children": []
    }
    parents.append(current_parent)
    
    current_child_type = "text"
    current_child_content = []

    def is_heading(line):
        # Headings are short, capitalized, and don't end in sentence punctuation
        if len(line) < 60 and not line.endswith(('.', ':', ';', ',', '!', '?')) and not line.startswith(('•', '-', '*')) and not line.lower().startswith("figure"):
            return True
        return False
        
    def is_bullet(line):
        return line.startswith(('•', '-', '*'))
        
    def is_figure_caption(line):
        return line.lower().startswith('figure')

    def save_current_child():
        if current_child_content:
            text = " ".join(current_child_content)
            # Create child metadata exactly as requested
            child = {
                "doc_id": doc_id,
                "page": page_num,
                "section": current_parent["heading"],
                "chunk_id": f"{page_num}_{len(parents)}_{len(current_parent['children'])+1}",
                "parent_chunk": f"{page_num}_{len(parents)}",
                "chunk_type": current_child_type,
                "text": text,
                "token_count": len(text.split()), # Simple proxy for tokens
            }
            current_parent["children"].append(child)
            current_child_content.clear()

    # Process each block from the layout index
    for block in layout_data:
        if block.get("type") == "image":
            # Save any text we were processing before hitting this image
            save_current_child()
            
            # Embed the image block explicitly as a chunk
            child = {
                "doc_id": doc_id,
                "page": page_num,
                "section": current_parent["heading"],
                "chunk_id": f"{page_num}_{len(parents)}_{len(current_parent['children'])+1}",
                "parent_chunk": f"{page_num}_{len(parents)}",
                "chunk_type": "image",
                "text": "[IMAGE EMBEDDING]",
                "src": block.get("src", ""),
                "bbox": block.get("bbox", []),
                "token_count": 0
            }
            current_parent["children"].append(child)
            
        elif block.get("type") == "text":
            lines = [line.strip() for line in block["text"].split('\n') if line.strip()]
            for line in lines:
                if is_heading(line):
                    # Save whatever was processing
                    save_current_child()
                    
                    # Start a new Parent Section
                    current_parent = {
                        "heading": line,
                        "type": "parent",
                        "children": []
                    }
                    parents.append(current_parent)
                    current_child_type = "text"
                    
                elif is_figure_caption(line):
                    save_current_child()
                    current_child_type = "figure_caption"
                    current_child_content.append(line)
                    save_current_child() # Captions are usually 1 line
                    
                elif is_bullet(line):
                    if current_child_type != "bullet":
                        save_current_child()
                        current_child_type = "bullet"
                    current_child_content.append(line)
                    
                else:
                    # Regular text or continuation of bullet
                    if current_child_type == "bullet" and not line.startswith(('•', '-', '*')):
                        current_child_content.append(line)
                    else:
                        if current_child_type != "text":
                            save_current_child()
                            current_child_type = "text"
                        current_child_content.append(line)
                        
                    # If line ends with a terminal punctuation, break the paragraph here.
                    if line.endswith(('.', '!', '?')) and current_child_type == "text":
                        save_current_child()
                        
    save_current_child()
    
    # Filter out empty parents
    return [p for p in parents if p["children"] or p["heading"] != "Document Start"]

def extract_layout_from_image(img_path):
    base_dir = os.path.dirname(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(base_dir, f"{base_name}_index.json")
    
    # 1. Try to read from the newly generated Layout Index JSON
    if os.path.exists(json_path):
        logger.info(f"Reading from Layout Index: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    # 2. Fallback to Tesseract OCR returning a single text block
    if not HAS_OCR:
        logger.error("pytesseract or PIL not installed. Cannot read PNG.")
        return []
    try:
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)
        return [{"type": "text", "text": text.strip()}]
    except Exception as e:
        logger.error(f"OCR failed for {img_path}: {e}")
        return []

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    section_dir = os.path.join(base_dir, "input", "section")
    
    if not os.path.exists(section_dir):
        logger.error(f"Directory not found: {section_dir}")
        return
        
    image_paths = sorted(glob.glob(os.path.join(section_dir, "*.png")))
    if not image_paths:
        logger.warning("No images found in section directory.")
        return
        
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Try to extract page number from "page_38" -> 38
        try:
            page_num = int(base_name.split("_")[1])
        except:
            page_num = 1
            
        logger.info(f"\n--- Extracting layout for {base_name} ---")
        layout_data = extract_layout_from_image(img_path)
        
        if layout_data:
            logger.info(f"\n--- Applying Chunking to {base_name} ---")
            
            # Run Production Chunking Strategy on the Layout Data
            hierarchy = production_hierarchical_chunking(layout_data, page_num=page_num)
                    
            # Log the beautifully structured output
            for parent in hierarchy:
                logger.info(f"\n[PARENT SECTION]: {parent['heading']}")
                logger.info("|")
                for child in parent["children"]:
                    logger.info(f"+-- [CHILD: {child['chunk_type'].upper()}] (Tokens: {child['token_count']})")
                    # Print text preview
                    text_preview = child['text'][:100] + "..." if len(child['text']) > 100 else child['text']
                    logger.info(f"|    {text_preview}")
                    logger.info(f"|    Metadata: {json.dumps({k:v for k,v in child.items() if k != 'text'})}")
                    logger.info("\\--")

if __name__ == "__main__":
    main()
