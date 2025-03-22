import base64
import json
import time
from mistralai import Mistral

class MistralOCR:
    def __init__(self, api_key):
        self.client = Mistral(api_key=api_key)
        
    def process_url(self, urls, file_type="PDF"):
        """Process OCR from URLs"""
        results = []
        for url in urls:
            if file_type.upper() == "PDF":
                document = {"type": "document_url", "document_url": url.strip()}
            else:
                document = {"type": "image_url", "image_url": url.strip()}
            
            results.append(self._process_document(document))
        return results
    
    def process_file(self, file_paths, file_type="PDF"):
        """Process OCR from local files"""
        results = []
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
                
            if file_type.upper() == "PDF":
                encoded_file = base64.b64encode(file_bytes).decode("utf-8")
                document = {"type": "document_url", "document_url": f"data:application/pdf;base64,{encoded_file}"}
            else:
                # Determine mime type based on file extension
                mime_type = f"image/{file_path.split('.')[-1].lower()}"
                encoded_file = base64.b64encode(file_bytes).decode("utf-8")
                document = {"type": "image_url", "image_url": f"data:{mime_type};base64,{encoded_file}"}
            
            results.append(self._process_document(document))
        return results
    
    def _process_document(self, document):
        """Internal method to process a single document"""
        try:
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document=document,
                include_image_base64=True
            )
            time.sleep(1)  # Rate limit prevention
            
            pages = ocr_response.pages if hasattr(ocr_response, "pages") else (
                ocr_response if isinstance(ocr_response, list) else []
            )
            result_text = "\n\n".join(page.markdown for page in pages) or "No result found."
            return result_text
        except Exception as e:
            return f"Error extracting result: {e}"
    
    def save_results(self, results, output_dir="outputs"):
        """Save results in multiple formats"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, result in enumerate(results):
            # Save as JSON
            json_path = os.path.join(output_dir, f"Output_{idx+1}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({"ocr_result": result}, f, ensure_ascii=False, indent=2)
            
            # Save as plain text
            txt_path = os.path.join(output_dir, f"Output_{idx+1}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            # Save as markdown
            md_path = os.path.join(output_dir, f"Output_{idx+1}.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(result)

# Example usage
if __name__ == "__main__":
    api_key = ""
    ocr = MistralOCR(api_key)
    
    # # Process URLs
    # urls = ["https://example.com/document.pdf"]
    # results = ocr.process_url(urls, file_type="PDF")
    
    # Process local files
    file_paths = ["test.pdf"]
    results = ocr.process_file(file_paths, file_type="PDF")
    
    # Save results
    ocr.save_results(results)