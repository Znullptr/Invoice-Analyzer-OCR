import cv2
import numpy as np
from PIL import Image
from typing import Union
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Preprocesses images for optimal OCR performance"""

    def __init__(self):
        pass
    
    def preprocess_for_ocr(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Apply comprehensive preprocessing pipeline for OCR
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image as numpy array
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Resize if image is too small (improves OCR)
        height, width = gray.shape
        if height < 1000 or width < 1000:
            scale = max(1000 / height, 1000 / width)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)

        # Apply adaptive thresholding for better text extraction
        binary = cv2.adaptiveThreshold(
            contrast,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # Deskew the image
        #binary = self._deskew(binary)
        
        # Remove noise with morphological operations
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return processed
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image skew
        
        Args:
            image: Grayscale image
            
        Returns:
            Deskewed image
        """
        try:
            # Detect edges
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = np.degrees(theta) - 90
                    if abs(angle) < 45:  # Only consider reasonable angles
                        angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    logger.info(f"Detected skew angle: {median_angle:.2f} degrees")
                    
                    # Rotate image
                    if abs(median_angle) > 0.5:  # Only rotate if significant
                        (h, w) = image.shape
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        rotated = cv2.warpAffine(
                            image, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE
                        )
                        return rotated
            
            return image
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image
    
    
    def enhance_for_display(self, image: np.ndarray) -> Image.Image:
        """Convert processed image back to PIL for display"""
        return Image.fromarray(image)

