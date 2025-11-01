from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import io
import logging
from pydantic import BaseModel
from typing import AsyncGenerator, Literal, Optional, Dict
from dataclasses import dataclass
import json
import asyncio

from .services.pdf_processor import PDFProcessor
from .services.ocr_processor import OCRProcessor
from .services.llm_extractor import LLMExtractor, InvoiceField

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStatus(BaseModel):
    """Real-time processing status"""
    status: Literal["uploading", "preprocessing", "ocr", "extracting", "completed", "error"]
    message: str
    progress: int = 0
    data: Optional[InvoiceField] = None
    error: Optional[str] = None

app = FastAPI(title="Invoice Analyzer API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
api_key = ""
pdf_processor = PDFProcessor()
ocr_processor = OCRProcessor(gemini_api_key=api_key)
llm_extractor = LLMExtractor(gemini_api_key=api_key)


async def process_invoice_stream(file_bytes: bytes) -> AsyncGenerator[str, None]:
    """
    Stream processing status updates as Server-Sent Events
    """
    try:
        # Status: Uploading
        yield f"data: {json.dumps({'status': 'uploading', 'message': 'Fichier reçu', 'progress': 10})}\n\n"
        await asyncio.sleep(0.1)
        
        # Status: Preprocessing
        yield f"data: {json.dumps({'status': 'preprocessing', 'message': 'Analyse du PDF', 'progress': 20})}\n\n"
        await asyncio.sleep(0.1)
        
        # Check if PDF is scanned or has text
        is_scanned = pdf_processor.is_scanned_pdf(file_bytes)
        
        extracted_text = ""
        
        if is_scanned:
            # Status: OCR Processing
            yield f"data: {json.dumps({'status': 'ocr', 'message': 'PDF scanné détecté - OCR en cours', 'progress': 30})}\n\n"
            await asyncio.sleep(0.1)
            
            # Convert PDF to images
            images = pdf_processor.pdf_to_images(file_bytes)
            
            yield f"data: {json.dumps({'status': 'ocr', 'message': f'Traitement de {len(images)} page(s)', 'progress': 40})}\n\n"
            await asyncio.sleep(0.1)
            
            # Extract text with OCR
            extracted_text = ocr_processor.extract_text_from_images(images)
            method = "OCR"
            
        else:
            # Direct text extraction
            yield f"data: {json.dumps({'status': 'preprocessing', 'message': 'Extraction du texte', 'progress': 40})}\n\n"
            await asyncio.sleep(0.1)
            
            extracted_text = pdf_processor.extract_text_from_pdf(file_bytes)
            method = "Direct PDF"
        
        # Status: LLM Extraction
        yield f"data: {json.dumps({'status': 'extracting', 'message': 'Extraction des données avec IA', 'progress': 60})}\n\n"
        await asyncio.sleep(0.1)
        
        # Extract structured data using LLM
        invoice_data = llm_extractor.extract_invoice_data(extracted_text)
        invoice_data.processing_method = method
        
        yield f"data: {json.dumps({'status': 'extracting', 'message': 'Finalisation', 'progress': 90})}\n\n"
        await asyncio.sleep(0.1)
        
        # Status: Completed
        result = {
            'status': 'completed',
            'message': 'Extraction terminée',
            'progress': 100,
            'data': invoice_data.model_dump()
        }
        with open("result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        yield f"result: {json.dumps(result)}\n\n"
        
    except Exception as e:
        logger.error(f"Error processing invoice: {e}", exc_info=True)
        error_result = {
            'status': 'error',
            'message': 'Erreur lors du traitement',
            'progress': 0,
            'error': str(e)
        }
        yield f"result: {json.dumps(error_result)}\n\n"


@app.get("/")
async def root():
    return {"message": "Invoice Analyzer API", "version": "1.0.0"}


@app.post("/api/analyze")
async def analyze_invoice(file: UploadFile = File(...)):
    """
    Analyze an invoice PDF and return extracted data via Server-Sent Events
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read file
    file_bytes = await file.read()
    
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    
    # Return streaming response
    return StreamingResponse(
        process_invoice_stream(file_bytes),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/preview")
async def get_pdf_preview(file: UploadFile = File(...)):
    """
    Generate a preview image of the first page of the PDF
    """
    try:
        file_bytes = await file.read()
        
        # Generate preview
        preview_bytes = pdf_processor.get_pdf_preview(file_bytes)
        
        return StreamingResponse(
            io.BytesIO(preview_bytes),
            media_type="image/jpeg"
        )
        
    except Exception as e:
        logger.error(f"Error generating preview: {e}")
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)