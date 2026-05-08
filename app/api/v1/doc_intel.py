from fastapi import APIRouter, HTTPException, Query
from app.schemas.request import DocIntelRequest
from app.schemas.response import DocIntelResponse
from app.core.downloader import download_single_image
from app.config import get_settings
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

SAMPLE_URL = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/DriverLicense.png"


def analyze_document(image_bytes: bytes, settings) -> dict:
    endpoint = settings.azure_doc_intel_endpoint
    key = settings.azure_doc_intel_key
    
    if not endpoint or not key:
        raise HTTPException(status_code=503, detail="Azure Document Intelligence not configured")
    
    logger.info(f"Azure Document Intelligence endpoint: {endpoint}")
    
    credential = AzureKeyCredential(key)
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=credential)
    
    logger.info("Starting document analysis with Azure SDK...")
    poller = client.begin_analyze_document("prebuilt-idDocument", AnalyzeDocumentRequest(bytes_source=image_bytes))
    result = poller.result()
    
    logger.info(f"Analysis completed. Documents found: {len(result.documents)}")
    
    return result


def extract_doc_fields(analysis_result) -> dict:
    documents = analysis_result.documents
    
    if not documents:
        return {}
    
    doc = documents[0]
    fields = doc.fields
    
    raw_doc_type = getattr(doc, "doc_type", "unknown")
    
    doc_type_mapping = {
        "idDocument.driverLicense": "driving_license",
        "idDocument.idCard": "id_card",
        "idDocument.passport": "passport",
    }
    
    doc_type = doc_type_mapping.get(raw_doc_type, raw_doc_type)
    
    result = {
        "doc_type": doc_type,
        "confidence": getattr(doc, "confidence", 0)
    }
    
    if fields.get("FirstName"):
        result["first_name"] = fields["FirstName"].value_string or ""
    
    if fields.get("LastName"):
        result["last_name"] = fields["LastName"].value_string or ""
    
    if fields.get("DateOfBirth"):
        dob = fields["DateOfBirth"]
        result["date_of_birth"] = str(dob.value_date) if dob.value_date else ""
    
    if fields.get("DocumentNumber"):
        result["document_number"] = fields["DocumentNumber"].value_string or ""
    
    if fields.get("CountryRegion"):
        country = fields["CountryRegion"]
        result["country"] = country.value_country_region or country.content or ""
    
    if fields.get("Region"):
        result["region"] = fields["Region"].value_string or ""
    
    if fields.get("DateOfExpiration"):
        result["date_of_expiration"] = str(fields["DateOfExpiration"].value_date) if fields["DateOfExpiration"].value_date else ""
    
    if fields.get("DateOfIssue"):
        result["date_of_issue"] = str(fields["DateOfIssue"].value_date) if fields["DateOfIssue"].value_date else ""
    
    if fields.get("Nationality"):
        result["nationality"] = fields["Nationality"].value_string or ""
    
    if fields.get("Sex"):
        result["sex"] = fields["Sex"].content or ""
    
    if fields.get("Address"):
        result["address"] = fields["Address"].value_address or fields["Address"].content or ""
    
    return result


@router.post("/doc-intel", response_model=DocIntelResponse)
async def analyze_document_intel(
    request: DocIntelRequest,
    use_sample: bool = Query(False, description="Use sample image for testing")
):
    settings = get_settings()
    
    if not settings.azure_doc_intel_endpoint or not settings.azure_doc_intel_key:
        raise HTTPException(status_code=503, detail="Azure Document Intelligence not configured")
    
    document_url = SAMPLE_URL if use_sample else request.document_url
    
    logger.info(f"Document URL: {document_url} (sample: {use_sample})")
    
    image_bytes = await download_single_image(
        document_url,
        timeout=settings.download_timeout,
        max_size=settings.max_image_size
    )
    
    if image_bytes is None:
        raise HTTPException(status_code=400, detail="Failed to download document image")
    
    analysis_result = analyze_document(image_bytes, settings)
    
    fields = extract_doc_fields(analysis_result)
    
    return DocIntelResponse(
        document_type=fields.get("doc_type", "unknown"),
        country=fields.get("country", ""),
        country_code=fields.get("country_code", ""),
        first_name=fields.get("first_name", ""),
        last_name=fields.get("last_name", ""),
        date_of_birth=fields.get("date_of_birth", ""),
        dob_formatted=fields.get("dob_formatted", ""),
        document_number=fields.get("document_number", ""),
        nationality=fields.get("nationality", ""),
        sex=fields.get("sex", ""),
        expiration_date=fields.get("date_of_expiration", ""),
        issue_date=fields.get("date_of_issue", ""),
        region=fields.get("region", ""),
        address=fields.get("address", ""),
        raw_confidence=fields.get("confidence", 0),
    )