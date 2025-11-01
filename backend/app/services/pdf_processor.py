import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
from PIL import Image
from typing import List, Tuple
import logging
import io

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction and conversion to images"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        """
        Extract text directly from PDF using PyMuPDF
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Extracted text
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            
            doc.close()
            
            # If we got substantial text, return it
            if len(text.strip()) > 100:
                logger.info(f"Successfully extracted {len(text)} characters from PDF")
                return text
            
            logger.warning("PDF text extraction yielded minimal text, may need OCR")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    @staticmethod
    def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
        """
        Convert PDF pages to images for OCR processing
        
        Args:
            pdf_bytes: PDF file as bytes
            dpi: Resolution for conversion (higher = better quality)
            
        Returns:
            List of PIL Images
        """
        try:
            images = convert_from_bytes(pdf_bytes, dpi=dpi)
            logger.info(f"Converted PDF to {len(images)} images at {dpi} DPI")
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
    
    @staticmethod
    def is_scanned_pdf(pdf_bytes: bytes) -> bool:
        """
        Determine if PDF is scanned (image-based) or has extractable text
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            True if scanned (needs OCR), False if has text
        """
        text = PDFProcessor.extract_text_from_pdf(pdf_bytes)
        
        # Heuristic: if we get less than 100 characters, it's likely scanned
        is_scanned = len(text.strip()) < 100
        
        logger.info(f"PDF classification: {'Scanned' if is_scanned else 'Text-based'}")
        return is_scanned
    
    @staticmethod
    def get_pdf_preview(pdf_bytes: bytes, page_num: int = 0, max_width: int = 800) -> bytes:
        """
        Generate a preview image of a PDF page
        
        Args:
            pdf_bytes: PDF file as bytes
            page_num: Page number to preview
            max_width: Maximum width for preview
            
        Returns:
            JPEG image bytes
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc[page_num]
            
            # Calculate zoom to fit max_width
            zoom = max_width / page.rect.width
            mat = fitz.Matrix(zoom, zoom)
            
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("jpeg")
            
            doc.close()
            return img_bytes
            
        except Exception as e:
            logger.error(f"Error generating PDF preview: {e}")
            raise