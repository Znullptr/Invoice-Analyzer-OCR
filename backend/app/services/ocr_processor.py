import pytesseract
from PIL import Image
import numpy as np
from typing import List, Union
import logging
from .image_preprocessor import ImagePreprocessor
import google.generativeai as genai

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handles OCR text extraction from images with Gemini fallback"""
    
    def __init__(self, gemini_api_key: str, min_text_length: int = 3000):
        self.preprocessor = ImagePreprocessor()
        # Configure tesseract for French (Canadian) invoices
        self.config = '--oem 3 --psm 6 -l fra+eng'
        self.min_text_length = min_text_length
        
        # Initialize Gemini
        self.gemini_api_key = gemini_api_key
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Fallback to Gemini")
        else:
            self.gemini_model = None
    
    def _extract_with_gemini(self, image: Image.Image) -> str:
        """
        Extract text using Gemini Flash model
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted text from Gemini
        """
        if not self.gemini_model:
            raise ValueError("Gemini API key not configured")
        
        try:
            logger.info("Using Gemini Flash for text extraction")
            
            prompt = """Extract all text from this image. 
            This is an invoice or document that may contain French and English text.
            Please extract all visible text accurately, preserving the layout and structure as much as possible.
            Include all numbers, dates, amounts, and other details."""
            
            response = self.gemini_model.generate_content([prompt, image])
            text = response.text
            
            logger.info(f"Gemini extracted {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Gemini extraction error: {e}")
            raise
    
    def extract_text_from_image(self, image: Union[Image.Image, np.ndarray]) -> str:
        """
        Extract text from a single image using OCR with Gemini fallback
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Extracted text
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        try:
            # Try pytesseract first
            processed = self.preprocessor.preprocess_for_ocr(image)
            processed_pil = Image.fromarray(processed)
            processed_pil.save('debug_processed_pil.png')  # Use PIL's save method
            logger.info("Saved debug_processed_pil.png")
            
            text = pytesseract.image_to_string(processed_pil, config=self.config)
            
            logger.info(f"Pytesseract extracted {len(text)} characters")

            self._dump_text_to_file(text, method="tesseract_ocr")
            
            # Check if extraction was successful
            if len(text.strip()) < self.min_text_length:
                logger.warning(f"Pytesseract extracted only {len(text.strip())} characters (min: {self.min_text_length})")
                
                # Fallback to Gemini
                logger.info("Falling back to Gemini Flash")
                text = self._extract_with_gemini(pil_image)
                self._dump_text_to_file(text, method="gemini_ocr")
            return text
            
        except Exception as tesseract_e:
            logger.error(f"Pytesseract extraction error: {tesseract_e}")
            
            # Fallback to Gemini on error
            if self.gemini_model:
                logger.info("Falling back to Gemini Flash")
                try:
                    text = self._extract_with_gemini(pil_image)
                    self._dump_text_to_file(text, method="gemini_ocr")
                    return text
                except Exception as gemini_e:
                    logger.error(f"Gemini fallback also failed: {gemini_e}")
                    raise Exception(f"Both OCR methods failed. Pytesseract: {tesseract_e}, Gemini: {gemini_e}")
            else:
                raise
    
    def get_processed_image(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Get the preprocessed image without extracting text
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            processed = self.preprocessor.preprocess_for_ocr(image)
            return Image.fromarray(processed)
            
        except Exception as e:
            logger.error(f"Failed to get processed image: {e}")
            raise
    
    def _dump_text_to_file(self, text: str, method: str = "ocr") -> None:
        """
        Dump extracted text to a file for debugging
        
        Args:
            text: Extracted text
            method: Extraction method used (for filename)
        """
        try:
            import os
            from datetime import datetime
            
            # Create output directory if it doesn't exist
            output_dir = "ocr_output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/extracted_text_{method}_{timestamp}.txt"
            
            # Write text to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Extraction Method: {method}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Text Length: {len(text)} characters\n")
                f.write("=" * 80 + "\n\n")
                f.write(text)
            
            logger.info(f"Text dumped to: {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to dump text to file: {e}")
    
    def extract_text_from_images(self, images: List[Image.Image]) -> str:
        """
        Extract text from multiple images (multi-page documents)
        
        Args:
            images: List of PIL Images
            
        Returns:
            Combined extracted text
        """
        texts = []
        
        for idx, image in enumerate(images):
            logger.info(f"Processing page {idx + 1}/{len(images)}")
            text = self.extract_text_from_image(image)
            texts.append(text)
        
        combined_text = "\n\n--- PAGE BREAK ---\n\n".join(texts)
        logger.info(f"Total OCR text length: {len(combined_text)} characters")
        
        return combined_text
    
    def get_preprocessed_preview(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Get preprocessed version of image for preview
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed PIL Image
        """
        processed = self.preprocessor.preprocess_for_ocr(image)
        return self.preprocessor.enhance_for_display(processed)