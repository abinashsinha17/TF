import os
import glob
import requests
import logging
import sys
import time
from dotenv import load_dotenv
import config

# Import from our beautifully structured chunking file
from chunking import production_hierarchical_chunking, extract_layout_from_image

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics
except ImportError:
    print("Please install reportlab: pip install reportlab")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("translation_chunks.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

NVIDIA_API_URL = os.getenv("INVOKE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
if not NVIDIA_API_URL.endswith("/chat/completions"):
    NVIDIA_API_URL = NVIDIA_API_URL.rstrip("/") + "/chat/completions"
GEMMA_API_KEY = os.getenv("GEMMA_MODEL_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_MODEL_API_KEY")

def translate_nvidia_api(text, model, target_lang, api_key):
    if not api_key:
        return f"Error: API Key missing for {model}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"Translate the following text into {target_lang}. Provide ONLY the translated text, without any additional explanations or intro.\n\nText:\n{text}"}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"API Translation failed: {e}"

def register_fonts():
    font_paths = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\msgothic.ttc"
    ]
    font_name = "Helvetica" # Default fallback
    for path in font_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont('UnicodeFont', path))
                font_name = 'UnicodeFont'
                logger.info(f"Registered unicode font from {path}")
                break
            except Exception as e:
                logger.warning(f"Could not register font {path}: {e}")
    return font_name

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    section_dir = os.path.join(base_dir, "input", "section")
    extracted_txt_dir = os.path.join(base_dir, "input", "D01710_extracted")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Only grab the root page images, ignore the extracted sub-images (which have _img_ in their name)
    all_pngs = sorted(glob.glob(os.path.join(section_dir, "page_*.png")))
    image_paths = [p for p in all_pngs if "_img_" not in os.path.basename(p)]
    
    if not image_paths:
        logger.warning("No base page images found.")
        return

    # PDF Setup
    font_name = register_fonts()
    pdf_path = os.path.join(output_dir, "Translated_Manual_Chunks.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    
    styles = getSampleStyleSheet()
    # Remove the blue/red colors, use black and native font sizing
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontName=font_name, fontSize=18, spaceAfter=20, alignment=1)
    
    # Main Heading: Black, large, bold
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontName=font_name, fontSize=16, spaceBefore=18, spaceAfter=10, textColor=colors.black)
    
    # Normal paragraph
    normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontName=font_name, fontSize=11, spaceAfter=10, leading=14)
    
    # Bullet paragraph (indented)
    bullet_style = ParagraphStyle('BulletStyle', parent=styles['Normal'], fontName=font_name, fontSize=11, spaceAfter=8, leftIndent=25, leading=14)
    
    # Caption style (bold)
    caption_style = ParagraphStyle('CaptionStyle', parent=styles['Normal'], fontName=font_name, fontSize=11, spaceBefore=6, spaceAfter=14, textColor=colors.black)
    
    # Header style (small, bold, top)
    header_style = ParagraphStyle('HeaderStyle', parent=styles['Normal'], fontName=font_name, fontSize=9, spaceAfter=20, textColor=colors.black)
    
    # Footer style (small, bold, bottom with lots of space before)
    footer_style = ParagraphStyle('FooterStyle', parent=styles['Normal'], fontName=font_name, fontSize=9, spaceBefore=40, textColor=colors.black)

    story.append(Paragraph("Multi-Lingual Chunked Manual", title_style))
    story.append(Spacer(1, 12))

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            page_num = int(base_name.split("_")[1])
        except:
            page_num = 1
            
        logger.info(f"\n======================================")
        logger.info(f"Processing Page: {base_name}")
        logger.info(f"======================================")
        
        # 1. (Removed printing the 'Source Page: page_X' debug header)

        # 2. Extract Layout Data via JSON Index (or fallback)
        layout_data = extract_layout_from_image(img_path)

        if not layout_data:
            logger.warning(f"No layout data found for {base_name}, skipping translations.")
            continue

        # 3. Get Parent-Child Chunks using the advanced chunking.py script
        hierarchy = production_hierarchical_chunking(layout_data, page_num=page_num)

        # 4. Translate Each Chunk & Match Original Styling
        for parent in hierarchy:
            # Reconstruct the Document Header
            if parent['heading'] and parent['heading'] != "Document Start":
                gemma_heading = translate_nvidia_api(parent['heading'], config.GEMMA_MODEL, "German", GEMMA_API_KEY)
                logger.info(f"[GERMAN TRANSLATION - HEADING]: {gemma_heading}")
                
                # Check if it's the header or footer
                if "Product Description" in parent['heading'] or "Produktbeschreibung" in gemma_heading or "Safety Devices" in parent['heading']:
                    story.append(Paragraph(f"<b>{gemma_heading}</b>", header_style))
                elif "4-8" in parent['heading'] or "Heratherm" in parent['heading'] or "Thermo" in parent['heading']:
                    story.append(Paragraph(f"<b>{gemma_heading}</b>", footer_style))
                else:
                    story.append(Paragraph(f"<b>{gemma_heading}</b>", heading_style))
            
            for child in parent["children"]:
                if child["chunk_type"] == "image":
                    logger.info(f"\n[IMAGE CHUNK DETECTED]: {child['src']}")
                    try:
                        # Append the specific extracted sub-image
                        img_path_child = os.path.join(section_dir, child["src"])
                        if os.path.exists(img_path_child):
                            # Make the image large and centered like the original
                            child_img = RLImage(img_path_child, width=450, height=220, kind='proportional')
                            story.append(child_img)
                            story.append(Spacer(1, 6))
                    except Exception as e:
                        logger.error(f"Failed to embed child image {child['src']}: {e}")
                    continue

                chunk_text = child["text"]
                
                logger.info(f"\n[ORIGINAL CHUNK]: {chunk_text}")
                
                # Translate
                gemma_german = translate_nvidia_api(chunk_text, config.GEMMA_MODEL, "German", GEMMA_API_KEY)
                logger.info(f"[GERMAN TRANSLATION]: {gemma_german}")
                
                # Apply exact original formatting style based on the semantic chunk type!
                if "4-8" in chunk_text or "Heratherm" in chunk_text or "Thermo" in chunk_text:
                    story.append(Paragraph(f"<b>{gemma_german}</b>", footer_style))
                elif "Product Description" in chunk_text or "Safety Devices" in chunk_text:
                    story.append(Paragraph(f"<b>{gemma_german}</b>", header_style))
                elif child["chunk_type"] == "bullet":
                    # Split if the LLM joined multiple bullets into a single line
                    bullets = [b.strip() for b in gemma_german.split('•') if b.strip()]
                    for b in bullets:
                        # Clean up any remaining asterisks or dashes
                        clean_text = b.lstrip('*- ')
                        story.append(Paragraph(f"• {clean_text}", bullet_style))
                elif child["chunk_type"] == "figure_caption":
                    story.append(Paragraph(f"<b>{gemma_german}</b>", caption_style))
                else:
                    story.append(Paragraph(f"{gemma_german}", normal_style))
                    
                # We removed the extra Spacer(1,8) here since spaceAfter is handled in the Styles directly!

    logger.info("\nBuilding Final PDF...")
    try:
        doc.build(story)
        logger.info(f"SUCCESS! Translated chunked PDF saved to {pdf_path}")
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")

if __name__ == "__main__":
    main()
