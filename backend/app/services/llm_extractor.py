import google.generativeai as genai
import os
import json
import logging
from typing import Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class InvoiceField(BaseModel):
    """Individual extracted field with confidence"""
    value: Optional[str] = None
    confidence: Optional[float] = None
    found: bool = False
    
    @field_validator('value', mode='before')
    @classmethod
    def convert_to_string(cls, v):
        """Convert any value to string"""
        if v is None:
            return None
        return str(v)
    
    @field_validator('confidence', mode='before')
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is a float between 0 and 1"""
        if v is None:
            return None
        try:
            conf = float(v)
            return max(0.0, min(1.0, conf))  # Clamp between 0 and 1
        except (ValueError, TypeError):
            return None


class ExtractedInvoiceData(BaseModel):
    """Structured invoice data"""
    fournisseur: InvoiceField = Field(default_factory=InvoiceField)
    date_facture: InvoiceField = Field(default_factory=InvoiceField)
    numero_facture: InvoiceField = Field(default_factory=InvoiceField)
    sous_total: InvoiceField = Field(default_factory=InvoiceField)
    tps: InvoiceField = Field(default_factory=InvoiceField)
    tvq: InvoiceField = Field(default_factory=InvoiceField)
    total: InvoiceField = Field(default_factory=InvoiceField)
    devise: InvoiceField = Field(default_factory=InvoiceField)
    categorie_comptable: InvoiceField = Field(default_factory=InvoiceField)
    type_document: InvoiceField = Field(default_factory=InvoiceField)
    processing_method: Optional[str] = None


class LLMExtractor:
    """Uses Google Gemini to extract structured invoice data"""
    
    def __init__(self, gemini_api_key: str):
        api_key = gemini_api_key
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.extraction_prompt = """
Tu es un expert en extraction de données de factures canadiennes. Analyse le texte suivant et extrais les informations demandées.

TEXTE DE LA FACTURE:
{text}

INSTRUCTIONS:
Extrais les champs suivants de la facture. Si un champ n'est pas trouvé, réponds avec null.

1. **Nom du fournisseur**: Le nom complet de l'entreprise qui émet la facture
2. **Date de la facture**: Au format YYYY-MM-DD si possible
3. **Numéro de facture**: L'identifiant unique de la facture
4. **Sous-total**: Le montant avant taxes (en tant que STRING, par exemple "100.50")
5. **TPS**: La taxe TPS à 5% (en tant que STRING, par exemple "5.03")
6. **TVQ**: La taxe TVQ à 9.975% (en tant que STRING, par exemple "10.02")
7. **Total**: Le montant total incluant les taxes (en tant que STRING, par exemple "115.55")
8. **Devise**: La devise utilisée (généralement CAD)
9. **Catégorie comptable**: Détermine la catégorie appropriée:
   - 6010 pour Frais de réparation
   - 5000 pour Fournitures
   - 6300 pour Services professionnels
   - 6500 pour Équipement
   - 0000 si aucune catégorie évidente
10. **Type de document**: Identifie s'il s'agit d'une "facture", "reçu", ou "note de crédit"

IMPORTANT: Tous les montants doivent être des STRINGS, pas des nombres.

RÉPONDS UNIQUEMENT avec un objet JSON valide dans ce format exact:
{{
  "fournisseur": {{"value": "nom ou null", "confidence": 0.95, "found": true}},
  "date_facture": {{"value": "2025-01-15", "confidence": 0.90, "found": true}},
  "numero_facture": {{"value": "INV-12345", "confidence": 0.85, "found": true}},
  "sous_total": {{"value": "100.50", "confidence": 0.95, "found": true}},
  "tps": {{"value": "5.03", "confidence": 0.90, "found": true}},
  "tvq": {{"value": "9.975", "confidence": 0.90, "found": true}},
  "total": {{"value": "115.55", "confidence": 0.95, "found": true}},
  "devise": {{"value": "CAD", "confidence": 0.80, "found": true}},
  "categorie_comptable": {{"value": "5000", "confidence": 0.70, "found": true}},
  "type_document": {{"value": "facture", "confidence": 0.95, "found": true}}
}}

Le champ "confidence" indique ta confiance dans l'extraction (0.0 = aucune confiance, 1.0 = totalement certain).
Le champ "found" indique si l'information a été trouvée dans le document.
Si un champ n'est pas trouvé, utilise: {{"value": null, "confidence": 0.0, "found": false}}
"""
    
    def _safe_parse_field(self, field_data: Union[Dict, None]) -> InvoiceField:
        """
        Safely parse a field dictionary into InvoiceField
        
        Args:
            field_data: Dictionary with value, confidence, found
            
        Returns:
            InvoiceField object
        """
        if not field_data or not isinstance(field_data, dict):
            return InvoiceField()
        
        try:
            # Convert value to string if it exists
            value = field_data.get("value")
            if value is not None:
                value = str(value)
            
            # Ensure confidence is float
            confidence = field_data.get("confidence")
            if confidence is not None:
                confidence = float(confidence)
            
            # Ensure found is boolean
            found = bool(field_data.get("found", False))
            
            return InvoiceField(
                value=value,
                confidence=confidence,
                found=found
            )
        except Exception as e:
            logger.warning(f"Error parsing field: {e}, returning empty field")
            return InvoiceField()
    
    def extract_invoice_data(self, text: str) -> ExtractedInvoiceData:
        """
        Extract structured invoice data from text using LLM
        
        Args:
            text: OCR or extracted text from invoice
            
        Returns:
            ExtractedInvoiceData object
        """
        try:
            # Limit text length to avoid token limits
            text_sample = text[:8000] if len(text) > 8000 else text
            
            prompt = self.extraction_prompt.format(text=text_sample)
            
            logger.info("Sending prompt to Gemini...")
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            logger.info(f"Received response from Gemini: {len(response_text)} characters")
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON response
            extracted_data = json.loads(response_text)
            
            # Convert to Pydantic model using safe parsing
            invoice_data = ExtractedInvoiceData(
                fournisseur=self._safe_parse_field(extracted_data.get("fournisseur")),
                date_facture=self._safe_parse_field(extracted_data.get("date_facture")),
                numero_facture=self._safe_parse_field(extracted_data.get("numero_facture")),
                sous_total=self._safe_parse_field(extracted_data.get("sous_total")),
                tps=self._safe_parse_field(extracted_data.get("tps")),
                tvq=self._safe_parse_field(extracted_data.get("tvq")),
                total=self._safe_parse_field(extracted_data.get("total")),
                devise=self._safe_parse_field(extracted_data.get("devise")),
                categorie_comptable=self._safe_parse_field(extracted_data.get("categorie_comptable")),
                type_document=self._safe_parse_field(extracted_data.get("type_document")),
            )
            
            logger.info("Successfully extracted invoice data with LLM")
            return invoice_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.error(f"Response was: {response_text[:500]}")
            # Return empty data on parse error
            return ExtractedInvoiceData(raw_text=text)
            
        except Exception as e:
            logger.error(f"LLM extraction error: {e}", exc_info=True)
            # Return empty data on error
            return ExtractedInvoiceData(raw_text=text)