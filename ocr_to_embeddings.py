import os
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Initialize SentenceTransformer with a high-quality model
model = SentenceTransformer('all-mpnet-base-v2')  # High-quality model for OCR text

# Connect to Qdrant running in Docker
client = QdrantClient(host="localhost", port=6333)

# Configure collection name and vector dimension
COLLECTION_NAME = "chapter7_sections"
VECTOR_SIZE = 768  # mpnet-base dimension

class RecordManager:
    """Simple record manager to track document hashes and metadata"""
    
    def __init__(self, storage_path="./record_manager"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.records = {}
        self._load_records()
    
    def _load_records(self):
        try:
            with open(f"{self.storage_path}/records.txt", "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        doc_hash, write_time, source_id = parts[0], parts[1], parts[2]
                        self.records[doc_hash] = {"write_time": write_time, "source_id": source_id}
        except FileNotFoundError:
            pass
    
    def _save_records(self):
        with open(f"{self.storage_path}/records.txt", "w") as f:
            for doc_hash, data in self.records.items():
                f.write(f"{doc_hash}\t{data['write_time']}\t{data['source_id']}\n")
    
    def compute_hash(self, content: str, metadata: Dict[str, Any]) -> str:
        """Compute hash based on content and metadata"""
        combined = content + str(metadata)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def add_record(self, content: str, metadata: Dict[str, Any], source_id: str) -> str:
        doc_hash = self.compute_hash(content, metadata)
        write_time = datetime.now().isoformat()
        self.records[doc_hash] = {"write_time": write_time, "source_id": source_id}
        self._save_records()
        return doc_hash
    
    def get_record(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        return self.records.get(doc_hash)
    
    def exists(self, content: str, metadata: Dict[str, Any]) -> bool:
        doc_hash = self.compute_hash(content, metadata)
        return doc_hash in self.records


def setup_qdrant_collection():
    """Set up Qdrant collection for Chapter 7 sections"""
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created new collection: {COLLECTION_NAME}")
    else:
        print(f"Collection {COLLECTION_NAME} already exists")


def extract_sections(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Extract sections with proper metadata from OCR text
    Focus on Chapter 7 content and properly identify section numbers
    """
    # Regular expressions to detect section headers
    section_patterns = [
        r'#+\s*([\d\.]+)\s+([^\n]+)',  # Markdown headers with numbers
        r'#+\s*([^\n]+)',              # General Markdown headers
        r'### ([\d\.]+)\s+([^\n]+)',   # Level 3 headers with numbers
        r'### ([^\n]+)',               # Level 3 headers
        r'## ([\d\.]+)\s+([^\n]+)',    # Level 2 headers with numbers
        r'## ([^\n]+)',                # Level 2 headers
        r'# ([\d\.]+)\s+([^\n]+)',     # Level 1 headers with numbers
        r'# ([^\n]+)',                 # Level 1 headers
    ]
    
    # Split text into lines for processing
    lines = text.split('\n')
    sections = []
    current_section = {"text": "", "metadata": {}}
    current_section_num = None
    current_header = None
    chapter = None
    unit = None
    
    for i, line in enumerate(lines):
        # Check if line contains a section header
        section_match = None
        header_level = None
        
        for pattern in section_patterns:
            match = re.match(pattern, line)
            if match:
                if len(match.groups()) == 2:  # Pattern with section number and title
                    section_num, section_title = match.groups()
                    if section_num.startswith("7."):  # Only process Chapter 7
                        section_match = (section_num, section_title)
                        header_level = pattern.count('#')
                elif len(match.groups()) == 1:  # Pattern with just title
                    section_title = match.group(1)
                    if section_title.upper() == "THERMODYNAMICS":
                        chapter = "7"
                        section_match = (None, section_title)
                        header_level = pattern.count('#')
                    elif section_title.startswith("Unit"):
                        unit = section_title
                        section_match = (None, section_title)
                        header_level = pattern.count('#')
                break
        
        # If we found a section header, store the previous section and start a new one
        if section_match:
            if current_section["text"].strip():
                sections.append((current_section["text"].strip(), current_section["metadata"]))
            
            section_num, section_title = section_match
            current_section = {"text": "", "metadata": {}}
            
            if section_num:  # If we have a section number
                current_section["metadata"]["section_id"] = section_num
                current_section["metadata"]["section_title"] = section_title
                current_section["metadata"]["chapter"] = "7"
                if unit:
                    current_section["metadata"]["unit"] = unit
                
                # Add the section header to the beginning of the text
                current_section["text"] += f"{section_num} {section_title}\n\n"
                current_header = line
                current_section_num = section_num
            else:
                if section_title.upper() == "THERMODYNAMICS":
                    current_section["metadata"]["chapter"] = "7"
                    current_section["metadata"]["chapter_title"] = section_title
                elif section_title.startswith("Unit"):
                    current_section["metadata"]["unit"] = section_title
                
                current_section["text"] += f"{section_title}\n\n"
                current_header = line
        
        # If this is not a section header, add it to the current section
        elif line and current_section is not None:
            # Filter out content not from Chapter 7
            if current_section_num and current_section_num.startswith("7."):
                current_section["text"] += line + "\n"
            elif "chapter" in current_section["metadata"] and current_section["metadata"]["chapter"] == "7":
                current_section["text"] += line + "\n"
    
    # Add the last section
    if current_section["text"].strip():
        sections.append((current_section["text"].strip(), current_section["metadata"]))
    
    # Filter sections to only include Chapter 7
    filtered_sections = []
    for text, metadata in sections:
        if "chapter" in metadata and metadata["chapter"] == "7":
            filtered_sections.append((text, metadata))
        elif "section_id" in metadata and metadata["section_id"].startswith("7."):
            filtered_sections.append((text, metadata))
    
    return filtered_sections


def chunk_text(sections: List[Tuple[str, Dict[str, Any]]], max_chunk_size: int = 500) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Chunk the sections into smaller pieces while preserving metadata
    """
    chunked_sections = []
    
    for text, metadata in sections:
        # If text is smaller than max chunk size, keep it as is
        if len(text.split()) <= max_chunk_size:
            chunked_sections.append((text, metadata))
            continue
        
        # Split text into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        chunk_num = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed the max chunk size, save current chunk
            if len(current_chunk.split()) + len(sentence.split()) > max_chunk_size and current_chunk:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_num"] = chunk_num
                chunked_sections.append((current_chunk.strip(), chunk_metadata))
                current_chunk = ""
                chunk_num += 1
            
            current_chunk += sentence + " "
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_num"] = chunk_num
            chunked_sections.append((current_chunk.strip(), chunk_metadata))
    
    return chunked_sections


def index_documents(chunked_sections: List[Tuple[str, Dict[str, Any]]], record_manager: RecordManager) -> None:
    """
    Generate embeddings and index documents in Qdrant with proper metadata
    """
    # Process documents in batches
    batch_size = 10
    for i in range(0, len(chunked_sections), batch_size):
        batch = chunked_sections[i:i+batch_size]
        texts = [text for text, _ in batch]
        metadatas = [metadata for _, metadata in batch]
        
        # Generate embeddings for this batch
        embeddings = model.encode(texts)
        
        # Prepare points for Qdrant
        points = []
        for j, (text, metadata) in enumerate(batch):
            # Check if document already exists in record manager
            if record_manager.exists(text, metadata):
                print(f"Document already indexed: {metadata.get('section_id', 'unknown')}")
                continue
            
            # Add to record manager
            doc_hash = record_manager.add_record(
                content=text,
                metadata=metadata,
                source_id=f"chapter7_{metadata.get('section_id', 'unknown')}"
            )
            
            # Create point for Qdrant
            points.append(models.PointStruct(
                id=doc_hash,
                vector=embeddings[j].tolist(),
                payload={
                    "text": text,
                    **metadata
                }
            ))
        
        # Upload to Qdrant if we have any points
        if points:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            print(f"Indexed {len(points)} documents")


def search_qdrant(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Search the Qdrant collection and return results with relevance scores
    """
    query_vector = model.encode(query).tolist()
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=k,
        with_payload=True,
        with_vectors=False
    )
    
    formatted_results = []
    for result in results:
        formatted_results.append({
            "text": result.payload.get("text", ""),
            "metadata": {k: v for k, v in result.payload.items() if k != "text"},
            "score": result.score
        })
    
    return formatted_results


def main():
    # Setup
    setup_qdrant_collection()
    record_manager = RecordManager()
    
    # Process the OCR output file
    ocr_file_path = "outputs/Output_1.txt"
    
    try:
        with open(ocr_file_path, "r", encoding="utf-8") as f:
            ocr_text = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {ocr_file_path}")
        return
    
    # Extract sections with metadata
    print("Extracting sections...")
    sections = extract_sections(ocr_text)
    print(f"Found {len(sections)} sections in Chapter 7")
    
    # Chunk the sections
    print("Chunking sections...")
    chunked_sections = chunk_text(sections)
    print(f"Created {len(chunked_sections)} chunks")
    
    # Index the documents
    print("Indexing documents...")
    index_documents(chunked_sections, record_manager)
    
    # Example search
    print("\nExample search:")
    query = "What is the zeroth law of thermodynamics?"
    results = search_qdrant(query)
    
    print(f"Search results for: '{query}'")
    for i, result in enumerate(results):
        print(f"Result {i+1} (Score: {result['score']:.4f}):")
        print(f"Section: {result['metadata'].get('section_id', 'N/A')}")
        print(f"Title: {result['metadata'].get('section_title', 'N/A')}")
        print(f"Text snippet: {result['text'][:150]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()