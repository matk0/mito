#!/usr/bin/env python3

import json
import re
from typing import List, Dict, Set, Any, Tuple
from pathlib import Path
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
import logging
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Entity:
    """Represents an extracted entity with metadata."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    variants: List[str] = None
    
    def __post_init__(self):
        if self.variants is None:
            self.variants = []

class SlovakHealthEntityExtractor:
    """
    Specialized entity extractor for Slovak health and quantum biology content.
    Combines rule-based matching with pattern recognition for domain-specific terms.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the entity extractor.
        
        Args:
            model_name: SpaCy model name for Slovak language processing
        """
        self.model_name = model_name
        self.nlp = None
        self.matcher = None
        self.entity_patterns = {}
        self.entity_taxonomy = {}
        
        # Initialize components
        self._load_nlp_model()
        self._create_entity_taxonomy()
        self._create_patterns()
        self._setup_matcher()
        
        # Statistics
        self.extraction_stats = defaultdict(int)
    
    def _load_nlp_model(self):
        """Load and configure the Slovak spaCy model."""
        try:
            self.nlp = spacy.load(self.model_name)
            print(f"✅ Loaded spaCy model: {self.model_name}")
        except IOError:
            print(f"❌ Could not load {self.model_name}. Trying English model as fallback...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print(f"✅ Successfully loaded English model as fallback")
            except Exception as e:
                print(f"❌ Failed to load any model: {e}")
                raise
        
        # Configure for better entity recognition
        self.nlp.max_length = 2000000  # Handle long texts
    
    def _create_entity_taxonomy(self):
        """Create comprehensive taxonomy of Slovak health entities."""
        self.entity_taxonomy = {
            # BIOCHEMICAL COMPOUNDS
            "HORMONE": [
                "dopamín", "dopamine", "serotonín", "serotonin", "melatonín", "melatonin",
                "leptín", "leptin", "inzulín", "insulin", "kortizol", "cortisol",
                "adrenalín", "adrenaline", "noradrenalín", "noradrenaline",
                "testosterón", "testosterone", "estrogén", "estrogen", 
                "pregnenolón", "pregnenolone", "rastový hormón", "growth hormone",
                "tyroxín", "T3", "T4", "TSH", "štítna žľaza hormóny", "thyroid hormones"
            ],
            
            "FATTY_ACID": [
                "DHA", "kyselina dokosahexaenová", "docosahexaenoic acid",
                "cholesterol", "nasýtené mastné kyseliny", "saturated fatty acids",
                "nenasýtené mastné kyseliny", "unsaturated fatty acids",
                "PUFA", "polynenasýtené mastné kyseliny", "polyunsaturated fatty acids",
                "ketóny", "ketones", "VLDL", "LDL", "HDL", "triglyceridy", "triglycerides"
            ],
            
            "VITAMIN_MINERAL": [
                "vitamín D3", "vitamin D3", "vitamín C", "vitamin C", "vitamín A", "vitamin A",
                "vitamín B6", "vitamin B6", "vitamín B12", "vitamin B12", "folát", "folate",
                "molybdén", "molybdenum", "železo", "iron", "meď", "copper",
                "horčík", "magnesium", "jód", "iodine", "síra", "sulfur",
                "zinok", "zinc", "selén", "selenium", "vápnik", "calcium"
            ],
            
            # ANATOMICAL STRUCTURES
            "CELLULAR_COMPONENT": [
                "mitochondrie", "mitochondria", "chloroplast", "bunková membrána", "cell membrane",
                "jadro bunky", "cell nucleus", "cytoplazma", "cytoplasm",
                "respiračné komplexy", "respiratory complexes", "ATP-syntáza", "ATP synthase",
                "elektrónový transportný reťazec", "electron transport chain",
                "CRYSTA", "cristae", "vnútorná membrána", "inner membrane"
            ],
            
            "ORGAN_SYSTEM": [
                "centrálny nervový systém", "central nervous system", "CNS",
                "črevný trakt", "digestive tract", "žalúdok", "stomach",
                "pečeň", "liver", "nadoblička", "adrenal glands",
                "štítna žľaza", "thyroid gland", "hypotalamus", "hypothalamus",
                "epifýza", "pineal gland", "mozgomiešny mok", "cerebrospinal fluid"
            ],
            
            # BIOLOGICAL PROCESSES
            "METABOLIC_PROCESS": [
                "fotosyntéza", "photosynthesis", "oxidácia", "oxidation",
                "glykolýza", "glycolysis", "beta-oxidácia", "beta-oxidation",
                "Krebsov cyklus", "Krebs cycle", "chemiosmóza", "chemiosmosis",
                "apoptóza", "apoptosis", "autofágia", "autophagy",
                "glukoneogenéza", "gluconeogenesis"
            ],
            
            "GENETIC_PROCESS": [
                "epigenetika", "epigenetics", "metylácia DNA", "DNA methylation",
                "acetylácia histónov", "histone acetylation", "génová expresia", "gene expression",
                "endosymbióza", "endosymbiosis", "transkripcia", "transcription",
                "translácia", "translation"
            ],
            
            # PHYSICAL PHENOMENA
            "LIGHT_ELECTROMAGNETIC": [
                "UV svetlo", "UV light", "UVA", "UVB", "UVC",
                "infračervené svetlo", "infrared light", "NIR", "FIR",
                "modré svetlo", "blue light", "elektromagnetizmus", "electromagnetism",
                "fotón", "photon", "elektrón", "electron", "protón", "proton",
                "viditeľné svetlo", "visible light", "slnečné svetlo", "sunlight"
            ],
            
            "PHYSICS_CONCEPT": [
                "kvantová biológia", "quantum biology", "negatívna entropia", "negative entropy",
                "Gibbsova voľná energia", "Gibbs free energy", "elektrónvolt", "electron volt",
                "piezoelektrina", "piezoelectricity", "supravodivosť", "superconductivity",
                "koherenčné domény", "coherent domains", "kvantovanie", "quantization"
            ],
            
            # ENVIRONMENTAL FACTORS
            "ENVIRONMENTAL": [
                "cirkadiálny rytmus", "circadian rhythm", "deutérium", "deuterium",
                "štruktúrovaná voda", "structured water", "ozón", "ozone",
                "teplota", "temperature", "chlad", "cold", "teplo", "heat",
                "vlhkosť", "humidity", "tlak", "pressure"
            ],
            
            "LIFESTYLE": [
                "adaptácia na chlad", "cold adaptation", "uzemnenie", "grounding",
                "spánok", "sleep", "stres", "stress", "cvičenie", "exercise",
                "meditácia", "meditation", "dýchanie", "breathing",
                "post", "fasting", "ketóza", "ketosis"
            ],
            
            # PATHOLOGICAL CONDITIONS
            "DISEASE": [
                "cukrovka", "diabetes", "rakovina", "cancer", "depresia", "depression",
                "autizmus", "autism", "Parkinsonova choroba", "Parkinson's disease",
                "autoimunita", "autoimmunity", "inzulínová rezistencia", "insulin resistance",
                "hypotýrea", "hypothyroidism", "hyperbilirubinémia", "hyperbilirubinemia"
            ],
            
            "SYMPTOM_CONDITION": [
                "zápal", "inflammation", "oxidatívny stres", "oxidative stress",
                "mitochondriálna dysfunkcia", "mitochondrial dysfunction",
                "lepínová rezistencia", "leptin resistance", "neurodegenerácia", "neurodegeneration",
                "únava", "fatigue", "bolesť", "pain", "nespavosť", "insomnia"
            ],
            
            # SPECIALIZED TERMS
            "TECHNICAL_TERM": [
                "biohackovanie", "biohacking", "detoxikácia", "detoxification",
                "redox", "ROS", "reaktívne kyslíkové zlúčeniny", "reactive oxygen species",
                "ALAN", "umelé svetlo v noci", "artificial light at night",
                "molekulárny vodík", "molecular hydrogen", "ATP", "adenozín trifosfát"
            ],
            
            "MEASUREMENT_UNIT": [
                "elektrónvolt", "electron volt", "eV", "nanometer", "nm",
                "millivolt", "mV", "volt", "V", "watt", "W",
                "lux", "kelvin", "K", "celsius", "°C"
            ]
        }
    
    def _create_patterns(self):
        """Create spaCy matcher patterns for entity recognition."""
        self.entity_patterns = {}
        
        for entity_type, terms in self.entity_taxonomy.items():
            patterns = []
            
            for term in terms:
                # Handle multi-word terms
                words = term.split()
                if len(words) == 1:
                    # Single word pattern
                    patterns.append([{"LOWER": term.lower()}])
                else:
                    # Multi-word pattern
                    pattern = []
                    for word in words:
                        pattern.append({"LOWER": word.lower()})
                    patterns.append(pattern)
                
                # Add pattern for terms with parentheses (Slovak (English))
                if "(" in term and ")" in term:
                    base_term = term.split("(")[0].strip()
                    if len(base_term.split()) == 1:
                        patterns.append([{"LOWER": base_term.lower()}])
                    else:
                        pattern = []
                        for word in base_term.split():
                            pattern.append({"LOWER": word.lower()})
                        patterns.append(pattern)
            
            self.entity_patterns[entity_type] = patterns
    
    def _setup_matcher(self):
        """Set up the spaCy matcher with all patterns."""
        self.matcher = Matcher(self.nlp.vocab)
        
        for entity_type, patterns in self.entity_patterns.items():
            self.matcher.add(entity_type, patterns)
        
        print(f"✅ Set up matcher with {len(self.entity_patterns)} entity types")
        print(f"📊 Total patterns: {sum(len(patterns) for patterns in self.entity_patterns.values())}")
    
    def _extract_chemical_formulas(self, text: str) -> List[Entity]:
        """Extract chemical formulas using regex patterns."""
        entities = []
        
        # Pattern for chemical formulas
        chemical_pattern = r'\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)*\b'
        
        for match in re.finditer(chemical_pattern, text):
            formula = match.group()
            # Filter out common non-chemical words
            if len(formula) > 1 and not formula.lower() in ['the', 'and', 'or', 'to', 'in', 'on', 'at']:
                entities.append(Entity(
                    text=formula,
                    label="CHEMICAL_FORMULA",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        return entities
    
    def _extract_measurements(self, text: str) -> List[Entity]:
        """Extract measurements and units."""
        entities = []
        
        # Pattern for measurements (number + unit)
        measurement_patterns = [
            r'\b\d+(?:[.,]\d+)?\s*(?:nm|μm|mm|cm|m|km)\b',  # Length
            r'\b\d+(?:[.,]\d+)?\s*(?:mg|g|kg)\b',           # Weight
            r'\b\d+(?:[.,]\d+)?\s*(?:mV|V|kV)\b',           # Voltage
            r'\b\d+(?:[.,]\d+)?\s*(?:Hz|kHz|MHz|GHz)\b',    # Frequency
            r'\b\d+(?:[.,]\d+)?\s*(?:°C|K)\b',              # Temperature
            r'\b\d+(?:[.,]\d+)?\s*(?:lux|cd)\b',            # Light
        ]
        
        for pattern in measurement_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(),
                    label="MEASUREMENT",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        return entities
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract all entities from text using multiple methods.
        
        Args:
            text: Input text for entity extraction
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # 1. Use matcher for domain-specific terms
        matches = self.matcher(doc)
        matcher_entities = []
        
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            matcher_entities.append(Entity(
                text=span.text,
                label=label,
                start=span.start_char,
                end=span.end_char,
                confidence=1.0
            ))
        
        # Remove overlapping entities, keeping longer ones
        matcher_entities = self._remove_overlaps(matcher_entities)
        entities.extend(matcher_entities)
        
        # 2. Extract chemical formulas
        chemical_entities = self._extract_chemical_formulas(text)
        entities.extend(chemical_entities)
        
        # 3. Extract measurements
        measurement_entities = self._extract_measurements(text)
        entities.extend(measurement_entities)
        
        # 4. Use spaCy's built-in NER for additional entities
        for ent in doc.ents:
            # Only add if not already covered by our domain-specific extraction
            overlaps = any(
                e.start <= ent.start_char < e.end or 
                e.start < ent.end_char <= e.end
                for e in entities
            )
            
            if not overlaps:
                entities.append(Entity(
                    text=ent.text,
                    label=f"SPACY_{ent.label_}",
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.7
                ))
        
        # Update statistics
        for entity in entities:
            self.extraction_stats[entity.label] += 1
        
        return entities
    
    def _remove_overlaps(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping longer ones."""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x.start)
        
        filtered = []
        for entity in entities:
            # Check if this entity overlaps with any in filtered list
            overlaps = False
            for existing in filtered:
                if (entity.start < existing.end and entity.end > existing.start):
                    # There's an overlap
                    if len(entity.text) > len(existing.text):
                        # Remove the shorter one
                        filtered.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered
    
    def process_chunked_content(self, chunked_file: str) -> Dict[str, Any]:
        """
        Process chunked content and extract entities from all chunks.
        
        Args:
            chunked_file: Path to chunked content JSON file
            
        Returns:
            Dictionary with extracted entities and metadata
        """
        print(f"🔍 Processing chunked content: {chunked_file}")
        
        # Load chunked data
        with open(chunked_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get('chunks', [])
        total_chunks = len(chunks)
        
        print(f"📄 Processing {total_chunks} chunks...")
        
        all_entities = []
        chunk_entities = []
        
        for i, chunk in enumerate(chunks):
            # Extract entities from chunk text
            entities = self.extract_entities(chunk['text'])
            
            # Add context entities if available
            if chunk.get('preceding_context'):
                context_entities = self.extract_entities(chunk['preceding_context'])
                entities.extend(context_entities)
            
            if chunk.get('following_context'):
                context_entities = self.extract_entities(chunk['following_context'])
                entities.extend(context_entities)
            
            # Remove duplicates within chunk
            entities = self._remove_overlaps(entities)
            
            # Add chunk metadata to entities
            chunk_entity_data = {
                'chunk_id': chunk.get('global_chunk_id', i),
                'source_title': chunk.get('source_title', ''),
                'source_url': chunk.get('source_url', ''),
                'source_date': chunk.get('source_date', ''),
                'chunk_position': chunk.get('chunk_position', ''),
                'entities': [
                    {
                        'text': entity.text,
                        'label': entity.label,
                        'start': entity.start,
                        'end': entity.end,
                        'confidence': entity.confidence,
                        'variants': entity.variants
                    }
                    for entity in entities
                ]
            }
            
            chunk_entities.append(chunk_entity_data)
            all_entities.extend(entities)
            
            # Progress reporting
            if (i + 1) % 50 == 0 or (i + 1) == total_chunks:
                print(f"📊 Progress: {i + 1}/{total_chunks} chunks processed")
        
        # Generate statistics
        entity_stats = self._generate_entity_statistics(all_entities)
        
        result = {
            'chunk_entities': chunk_entities,
            'extraction_metadata': {
                'total_chunks_processed': total_chunks,
                'total_entities_extracted': len(all_entities),
                'entity_type_counts': dict(self.extraction_stats),
                'entity_statistics': entity_stats,
                'extraction_timestamp': __import__('datetime').datetime.now().isoformat()
            }
        }
        
        print(f"✅ Entity extraction complete!")
        print(f"📊 Total entities extracted: {len(all_entities)}")
        print(f"📈 Entity types found: {len(self.extraction_stats)}")
        
        return result
    
    def _generate_entity_statistics(self, entities: List[Entity]) -> Dict[str, Any]:
        """Generate comprehensive statistics about extracted entities."""
        stats = {
            'total_entities': len(entities),
            'unique_entities': len(set(entity.text.lower() for entity in entities)),
            'entity_types': len(set(entity.label for entity in entities)),
            'avg_confidence': sum(entity.confidence for entity in entities) / len(entities) if entities else 0,
            'top_entities': {},
            'top_entity_types': {}
        }
        
        # Count entity frequencies
        entity_counts = defaultdict(int)
        for entity in entities:
            entity_counts[entity.text.lower()] += 1
        
        # Top 20 most frequent entities
        stats['top_entities'] = dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        
        # Top entity types
        type_counts = defaultdict(int)
        for entity in entities:
            type_counts[entity.label] += 1
        
        stats['top_entity_types'] = dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))
        
        return stats
    
    def save_extracted_entities(self, entities_data: Dict[str, Any], output_file: str):
        """Save extracted entities to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Entities saved to: {output_file}")


def main():
    """Main function to run entity extraction."""
    print("Slovak Health Entity Extractor")
    print("=" * 40)
    
    # Initialize extractor
    extractor = SlovakHealthEntityExtractor()
    
    # Check if chunked data exists
    chunked_file = "./data/processed/chunked_data/chunked_content.json"
    if not Path(chunked_file).exists():
        print(f"❌ Chunked content file not found: {chunked_file}")
        print("Please run content_chunker.py first!")
        return
    
    # Process chunked content
    entities_data = extractor.process_chunked_content(chunked_file)
    
    # Save results
    output_file = "./data/processed/chunked_data/extracted_entities.json"
    extractor.save_extracted_entities(entities_data, output_file)
    
    # Print summary statistics
    stats = entities_data['extraction_metadata']
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"📄 Chunks processed: {stats['total_chunks_processed']}")
    print(f"🏷️  Total entities: {stats['total_entities_extracted']}")
    print(f"📈 Entity types: {len(stats['entity_type_counts'])}")
    
    print(f"\n🔝 Top 10 Entity Types:")
    for entity_type, count in list(stats['entity_type_counts'].items())[:10]:
        print(f"   {entity_type}: {count}")


if __name__ == "__main__":
    main()