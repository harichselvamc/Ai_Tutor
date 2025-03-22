from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional, Tuple, Set
import uuid
import re
import random
from enum import Enum

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class VectorStorageProcessor:
    def __init__(self, collection_name: str = "thermodynamics_tutor", qdrant_url: Optional[str] = None, chunk_size: int = 1000):
        """Initialize Vector Storage processor for thermodynamics tutoring
        
        Args:
            collection_name (str): Name of the Qdrant collection
            qdrant_url (Optional[str]): URL for Qdrant server. If None, uses in-memory storage
            chunk_size (int): Maximum number of characters per chunk
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.concept_relationships = {}  # Store relationships between concepts
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(":memory:")
        self._create_collection()

    def _create_collection(self) -> Dict[str, str]:
        """Create a new collection if it doesn't exist"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )
            return {"status": "success", "message": "Collection created successfully"}
        except Exception as e:
            return {"status": "info", "message": f"Collection might already exist: {str(e)}"}

    def _extract_difficulty(self, text: str) -> DifficultyLevel:
        """Extract difficulty level from text content"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["complex", "advanced", "difficult", "challenging"]):
            return DifficultyLevel.HARD
        elif any(word in text_lower for word in ["intermediate", "moderate"]):
            return DifficultyLevel.MEDIUM
        return DifficultyLevel.EASY

    def _extract_related_concepts(self, text: str) -> Set[str]:
        """Extract related concepts from text"""
        related_concepts = set()
        # Look for concept references in the text
        concept_pattern = r'(?:related to|similar to|connected with|part of)\s+([A-Za-z\s]+)'
        matches = re.finditer(concept_pattern, text)
        for match in matches:
            related_concepts.add(match.group(1).strip())
        return related_concepts

    async def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk the text based on section headers, problems, and concepts"""
        chunks = []
        current_chunk = []
        current_section = ""
        current_type = ""
        current_length = 0
        current_concept = ""
        current_difficulty = DifficultyLevel.EASY
        
        lines = text.split('\n')
        
        for line in lines:
            # Enhanced regex patterns for better section and concept detection
            section_match = re.match(r'#{1,3}\s*(\d+\.\d+(?:\.\d+)?)\s*(.*)', line)
            problem_match = re.match(r'##\s*Problem[:\s]*(\d+\.\d+(?:\.\d+)?)', line)
            concept_match = re.match(r'###\s*([A-Za-z\s]+)', line)
            
            if current_length + len(line) > self.chunk_size and current_chunk:
                # Extract difficulty and related concepts
                chunk_text = '\n'.join(current_chunk)
                current_difficulty = self._extract_difficulty(chunk_text)
                related_concepts = self._extract_related_concepts(chunk_text)
                
                chunks.append({
                    'text': chunk_text,
                    'section': current_section,
                    'type': current_type,
                    'concept': current_concept,
                    'difficulty': current_difficulty.value,
                    'related_concepts': list(related_concepts)
                })
                current_chunk = []
                current_length = 0
            
            if section_match:
                current_section = section_match.group(1)
                current_type = "section"
                current_chunk.append(line)
                current_length += len(line)
                
            elif problem_match:
                current_section = problem_match.group(1)
                current_type = "problem"
                current_chunk.append(line)
                current_length += len(line)
                
            elif concept_match:
                current_concept = concept_match.group(1).strip()
                current_chunk.append(line)
                current_length += len(line)
                
            else:
                current_chunk.append(line)
                current_length += len(line)
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            current_difficulty = self._extract_difficulty(chunk_text)
            related_concepts = self._extract_related_concepts(chunk_text)
            
            chunks.append({
                'text': chunk_text,
                'section': current_section,
                'type': current_type,
                'concept': current_concept,
                'difficulty': current_difficulty.value,
                'related_concepts': list(related_concepts)
            })
        
        return chunks

    async def process_text(self, text: str) -> Dict[str, Any]:
        """Process and add text to the vector store with enhanced metadata"""
        try:
            chunks = await self._chunk_text(text)
            points = []
            ids = []
            
            for chunk in chunks:
                if chunk['text'].strip():
                    id_ = str(uuid.uuid4())
                    # Create comprehensive text for embedding
                    embedding_text = f"{chunk['section']} {chunk['type']} {chunk['concept']} {chunk['text']}"
                    embedding = self.model.encode(embedding_text)
                    
                    points.append(
                        models.PointStruct(
                            id=id_,
                            vector=embedding.tolist(),
                            payload={
                                "text": chunk['text'],
                                "section": chunk['section'],
                                "type": chunk['type'],
                                "concept": chunk['concept'],
                                "difficulty": chunk['difficulty'],
                                "related_concepts": chunk['related_concepts'],
                                "full_text": embedding_text
                            }
                        )
                    )
                    ids.append(id_)
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            return {
                "status": "success",
                "message": f"Successfully processed {len(ids)} chunks",
                "chunk_ids": ids
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing text: {str(e)}"
            }

    async def search_similar(self, query: str, limit: int = 5, section_filter: Optional[str] = None, 
                           type_filter: Optional[str] = None, concept_filter: Optional[str] = None,
                           difficulty_filter: Optional[DifficultyLevel] = None,
                           score_threshold: float = 0.5) -> Dict[str, Any]:
        """Enhanced search with concept filtering and difficulty levels"""
        try:
            if not query.strip():
                return {
                    "status": "error",
                    "message": "Empty query provided",
                    "results": []
                }

            # Enhance query with context
            enhanced_query = query
            if section_filter:
                enhanced_query = f"{section_filter} {query}"
            if type_filter:
                enhanced_query = f"{type_filter} {enhanced_query}"
            if concept_filter:
                enhanced_query = f"{concept_filter} {enhanced_query}"
            if difficulty_filter:
                enhanced_query = f"{difficulty_filter.value} {enhanced_query}"

            query_embedding = self.model.encode(enhanced_query)
            
            # Build search filter
            search_filter = None
            if section_filter or type_filter or concept_filter or difficulty_filter:
                must_conditions = []
                if section_filter:
                    must_conditions.append(models.FieldCondition(
                        key="section",
                        match=models.MatchValue(value=section_filter)
                    ))
                if type_filter:
                    must_conditions.append(models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value=type_filter)
                    ))
                if concept_filter:
                    must_conditions.append(models.FieldCondition(
                        key="concept",
                        match=models.MatchValue(value=concept_filter)
                    ))
                if difficulty_filter:
                    must_conditions.append(models.FieldCondition(
                        key="difficulty",
                        match=models.MatchValue(value=difficulty_filter.value)
                    ))
                search_filter = models.Filter(must=must_conditions)
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                query_filter=search_filter,
                score_threshold=score_threshold
            )
            
            results = []
            for result in search_results:
                results.append({
                    "section": result.payload["section"],
                    "type": result.payload["type"],
                    "concept": result.payload["concept"],
                    "text": result.payload["text"],
                    "difficulty": result.payload["difficulty"],
                    "related_concepts": result.payload["related_concepts"],
                    "score": float(result.score)
                })
                
            return {
                "status": "success",
                "query": query,
                "enhanced_query": enhanced_query,
                "results": results,
                "filters_applied": {
                    "section": section_filter,
                    "type": type_filter,
                    "concept": concept_filter,
                    "difficulty": difficulty_filter.value if difficulty_filter else None,
                    "score_threshold": score_threshold
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error performing search: {str(e)}",
                "results": []
            }

    async def generate_quiz(self, section: str, num_questions: int = 5, 
                          difficulty: Optional[DifficultyLevel] = None) -> Dict[str, Any]:
        """Generate quiz questions for a specific section with difficulty levels"""
        try:
            # Search for problems in the specified section
            search_result = await self.search_similar(
                query="problem",
                section_filter=section,
                type_filter="problem",
                difficulty_filter=difficulty,
                limit=10
            )
            
            if search_result["status"] != "success" or not search_result["results"]:
                return {
                    "status": "error",
                    "message": f"No problems found for section {section}",
                    "questions": []
                }
            
            # Randomly select problems for the quiz
            problems = search_result["results"]
            selected_problems = random.sample(problems, min(num_questions, len(problems)))
            
            questions = []
            for idx, problem in enumerate(selected_problems, 1):
                questions.append({
                    "question_number": idx,
                    "section": problem["section"],
                    "text": problem["text"],
                    "concept": problem["concept"],
                    "difficulty": problem["difficulty"],
                    "related_concepts": problem["related_concepts"]
                })
            
            return {
                "status": "success",
                "section": section,
                "num_questions": len(questions),
                "difficulty": difficulty.value if difficulty else "mixed",
                "questions": questions
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating quiz: {str(e)}",
                "questions": []
            }

    async def get_concept_relationships(self, concept: str) -> Dict[str, Any]:
        """Get related concepts and their relationships"""
        try:
            search_result = await self.search_similar(
                query=concept,
                concept_filter=concept,
                limit=5
            )
            
            if search_result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Concept {concept} not found",
                    "relationships": []
                }
            
            relationships = []
            for result in search_result["results"]:
                relationships.extend(result["related_concepts"])
            
            return {
                "status": "success",
                "concept": concept,
                "relationships": list(set(relationships))
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting concept relationships: {str(e)}",
                "relationships": []
            }

    async def delete_collection(self) -> Dict[str, str]:
        """Delete the current collection"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            return {
                "status": "success",
                "message": f"Collection {self.collection_name} deleted successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting collection: {str(e)}"
            }

if __name__ == "__main__":
    import asyncio
    
    async def test_vector_storage():
        # Initialize the processor
        processor = VectorStorageProcessor(chunk_size=2000)
        
        # Read the exercise text from file
        try:
            with open('outputs/Output_1.txt', 'r', encoding='utf-8') as file:
                exercise_text = file.read()
        except FileNotFoundError:
            print("Error: test.txt file not found!")
            return
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return
        
        # Process the exercise text
        print("Processing exercise text...")
        result = await processor.process_text(exercise_text)
        print(f"Processing result: {result}\n")
        
        # Test tutoring queries
        tutoring_queries = [
            "What is entropy?",
            "Explain internal energy",
            "What is a cyclic process?",
            "Define extensive properties"
        ]
        
        print("\nTesting tutoring queries:")
        for query in tutoring_queries:
            print(f"\nSearching for: '{query}'")
            search_result = await processor.search_similar(
                query=query,
                limit=3,
                score_threshold=0.3
            )
            
            if search_result["status"] == "success":
                print(f"\nFound {len(search_result['results'])} results for '{query}':")
                for idx, result in enumerate(search_result["results"], 1):
                    print(f"\nResult {idx}:")
                    print(f"Section: {result['section']}")
                    print(f"Concept: {result['concept']}")
                    print(f"Difficulty: {result['difficulty']}")
                    print(f"Related concepts: {', '.join(result['related_concepts'])}")
                    print(f"Score: {result['score']:.3f}")
                    print(f"Text: {result['text'][:200]}...")
            else:
                print(f"No results found for '{query}'")
        
        # Test quiz generation with different difficulties
        print("\nTesting quiz generation:")
        difficulties = [None, DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        for difficulty in difficulties:
            print(f"\nGenerating quiz for section 7.1 with difficulty: {difficulty.value if difficulty else 'mixed'}")
            quiz_result = await processor.generate_quiz(section="7.1", num_questions=3, difficulty=difficulty)
            if quiz_result["status"] == "success":
                print(f"\nGenerated quiz for section {quiz_result['section']}:")
                for question in quiz_result["questions"]:
                    print(f"\nQuestion {question['question_number']}:")
                    print(f"Section: {question['section']}")
                    print(f"Concept: {question['concept']}")
                    print(f"Difficulty: {question['difficulty']}")
                    print(f"Related concepts: {', '.join(question['related_concepts'])}")
                    print(f"Text: {question['text'][:200]}...")
            else:
                print(f"Error generating quiz: {quiz_result['message']}")
        
        # Test concept relationships
        print("\nTesting concept relationships:")
        concepts = ["entropy", "internal energy", "cyclic process"]
        for concept in concepts:
            print(f"\nFinding relationships for concept: {concept}")
            relationship_result = await processor.get_concept_relationships(concept)
            if relationship_result["status"] == "success":
                print(f"Related concepts: {', '.join(relationship_result['relationships'])}")
            else:
                print(f"Error: {relationship_result['message']}")
        
        # Clean up
        print("\nCleaning up...")
        await processor.delete_collection()
        print("Test completed!")
    
    # Run the test
    asyncio.run(test_vector_storage())
