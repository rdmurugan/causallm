"""
LLM Multi-Modal Causal Analysis Module

This module provides LLM-enhanced causal analysis capabilities across multiple data modalities:
- Text + Structured Data Analysis
- Image + Text Causal Reasoning  
- Time Series + Cross-Sectional Integration
- Multi-Modal Evidence Synthesis

Author: CausalLM Team
"""

import asyncio
import json
import logging
import base64
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from .llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    TEXT = "text"
    STRUCTURED = "structured"
    IMAGE = "image"
    TIME_SERIES = "time_series"
    CROSS_SECTIONAL = "cross_sectional"
    MIXED = "mixed"

class EvidenceStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class MultiModalEvidence:
    """Evidence from multiple data modalities"""
    modality: ModalityType
    content: Any
    causal_claim: str
    confidence: float
    supporting_details: List[str]
    contradicting_details: List[str]
    
class CausalEvidenceType(Enum):
    CORRELATION = "correlation"
    TEMPORAL_PRECEDENCE = "temporal_precedence"
    DOSE_RESPONSE = "dose_response"
    MECHANISM = "mechanism"
    CONFOUNDING_CONTROL = "confounding_control"
    EXPERIMENTAL = "experimental"

@dataclass
class CausalEvidence:
    """Structured causal evidence"""
    evidence_type: CausalEvidenceType
    strength: EvidenceStrength
    description: str
    source_modality: ModalityType
    reliability_score: float
    supporting_data: Dict[str, Any]

@dataclass
class MultiModalCausalAssessment:
    """Comprehensive multi-modal causal assessment"""
    treatment_variable: str
    outcome_variable: str
    causal_claim: str
    overall_evidence_strength: EvidenceStrength
    confidence_score: float
    evidence_pieces: List[CausalEvidence]
    modal_consensus: Dict[str, float]
    contradictions: List[str]
    limitations: List[str]
    recommendations: List[str]
    narrative_summary: str

class LLMMultiModalAnalysis:
    """LLM-enhanced multi-modal causal analysis system"""
    
    def __init__(self, llm_client: BaseLLMClient = None):
        if llm_client is None:
            from .llm_client import get_llm_client
            self.llm_client = get_llm_client()
        else:
            self.llm_client = llm_client
        self.analysis_history = []
        
        # Domain-specific prompts for different modalities
        self.modality_prompts = {
            ModalityType.TEXT: self._get_text_analysis_prompt(),
            ModalityType.STRUCTURED: self._get_structured_analysis_prompt(),
            ModalityType.IMAGE: self._get_image_analysis_prompt(),
            ModalityType.TIME_SERIES: self._get_timeseries_analysis_prompt(),
            ModalityType.CROSS_SECTIONAL: self._get_cross_sectional_prompt()
        }
        
    async def analyze_multimodal_causality(self,
                                         treatment_variable: str,
                                         outcome_variable: str,
                                         data_sources: Dict[str, Any],
                                         domain: str = "general",
                                         context: str = "") -> MultiModalCausalAssessment:
        """
        Perform comprehensive multi-modal causal analysis
        
        Args:
            treatment_variable: Treatment/intervention variable
            outcome_variable: Outcome variable of interest
            data_sources: Dictionary mapping modality types to data
            domain: Domain context (healthcare, business, etc.)
            context: Additional context for analysis
        """
        try:
            logger.info(f"Starting multi-modal causal analysis: {treatment_variable} -> {outcome_variable}")
            
            # Analyze each modality separately
            modal_evidence = []
            for modality_str, data in data_sources.items():
                try:
                    modality = ModalityType(modality_str)
                    evidence = await self._analyze_single_modality(
                        treatment_variable, outcome_variable, modality, data, domain, context
                    )
                    modal_evidence.extend(evidence)
                except ValueError:
                    logger.warning(f"Unknown modality type: {modality_str}")
                    continue
            
            # Synthesize evidence across modalities
            assessment = await self._synthesize_multimodal_evidence(
                treatment_variable, outcome_variable, modal_evidence, domain, context
            )
            
            self.analysis_history.append(assessment)
            return assessment
            
        except Exception as e:
            logger.error(f"Error in multi-modal causal analysis: {e}")
            # Return default assessment
            return MultiModalCausalAssessment(
                treatment_variable=treatment_variable,
                outcome_variable=outcome_variable,
                causal_claim=f"Unable to assess causality between {treatment_variable} and {outcome_variable}",
                overall_evidence_strength=EvidenceStrength.WEAK,
                confidence_score=0.1,
                evidence_pieces=[],
                modal_consensus={},
                contradictions=[f"Analysis failed: {str(e)}"],
                limitations=["Technical error prevented complete analysis"],
                recommendations=["Retry analysis with corrected data"],
                narrative_summary="Analysis could not be completed due to technical issues."
            )
    
    async def _analyze_single_modality(self,
                                     treatment: str,
                                     outcome: str,
                                     modality: ModalityType,
                                     data: Any,
                                     domain: str,
                                     context: str) -> List[CausalEvidence]:
        """Analyze causal evidence from a single modality"""
        
        prompt = self._build_modality_prompt(modality, treatment, outcome, data, domain, context)
        
        try:
            response = await self.llm_client.generate_response(
                prompt,
                max_tokens=1500,
                temperature=0.3
            )
            
            return self._parse_evidence_response(response, modality)
            
        except Exception as e:
            logger.error(f"Error analyzing {modality.value} modality: {e}")
            return []
    
    def _build_modality_prompt(self, modality: ModalityType, treatment: str, 
                             outcome: str, data: Any, domain: str, context: str) -> str:
        """Build analysis prompt for specific modality"""
        
        base_prompt = f"""
        You are a causal inference expert analyzing the relationship between '{treatment}' and '{outcome}' 
        in the {domain} domain using {modality.value} data.
        
        Context: {context}
        
        {self.modality_prompts.get(modality, "")}
        """
        
        # Add modality-specific data formatting
        if modality == ModalityType.TEXT:
            data_section = f"Text data to analyze:\n{str(data)[:2000]}"
        elif modality == ModalityType.STRUCTURED:
            data_section = f"Structured data summary:\n{self._format_structured_data(data)}"
        elif modality == ModalityType.IMAGE:
            data_section = f"Image data provided for analysis: {type(data).__name__}"
        elif modality == ModalityType.TIME_SERIES:
            data_section = f"Time series data characteristics:\n{self._format_timeseries_data(data)}"
        else:
            data_section = f"Data type: {type(data).__name__}"
        
        return base_prompt + "\n\n" + data_section + "\n\n" + self._get_evidence_format_instructions()
    
    def _format_structured_data(self, data: Any) -> str:
        """Format structured data for LLM analysis"""
        if isinstance(data, pd.DataFrame):
            return f"""
            DataFrame with {data.shape[0]} rows, {data.shape[1]} columns
            Columns: {list(data.columns)}
            Data types: {data.dtypes.to_dict()}
            Sample data:\n{data.head().to_string()}
            """
        elif isinstance(data, dict):
            return f"Dictionary with keys: {list(data.keys())}\nSample: {str(data)[:500]}"
        else:
            return f"Data type: {type(data).__name__}\nContent: {str(data)[:500]}"
    
    def _format_timeseries_data(self, data: Any) -> str:
        """Format time series data for LLM analysis"""
        if isinstance(data, pd.DataFrame):
            return f"""
            Time series with {len(data)} observations
            Columns: {list(data.columns)}
            Date range: {data.index.min() if hasattr(data.index, 'min') else 'Unknown'} to {data.index.max() if hasattr(data.index, 'max') else 'Unknown'}
            Frequency: {data.index.freq if hasattr(data.index, 'freq') else 'Unknown'}
            """
        else:
            return f"Time series data type: {type(data).__name__}"
    
    def _get_evidence_format_instructions(self) -> str:
        """Instructions for evidence formatting"""
        return """
        Please provide your analysis in the following JSON format:
        {
            "evidence_pieces": [
                {
                    "evidence_type": "correlation|temporal_precedence|dose_response|mechanism|confounding_control|experimental",
                    "strength": "weak|moderate|strong|very_strong",
                    "description": "Detailed description of the evidence",
                    "reliability_score": 0.0-1.0,
                    "supporting_data": {"key": "value"}
                }
            ],
            "overall_assessment": "Summary of findings",
            "limitations": ["List of limitations"],
            "confidence": 0.0-1.0
        }
        """
    
    def _parse_evidence_response(self, response: str, modality: ModalityType) -> List[CausalEvidence]:
        """Parse LLM response into structured evidence"""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                evidence_list = []
                for evidence_data in parsed.get('evidence_pieces', []):
                    try:
                        evidence = CausalEvidence(
                            evidence_type=CausalEvidenceType(evidence_data.get('evidence_type', 'correlation')),
                            strength=EvidenceStrength(evidence_data.get('strength', 'weak')),
                            description=evidence_data.get('description', ''),
                            source_modality=modality,
                            reliability_score=float(evidence_data.get('reliability_score', 0.5)),
                            supporting_data=evidence_data.get('supporting_data', {})
                        )
                        evidence_list.append(evidence)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing evidence piece: {e}")
                        continue
                
                return evidence_list
            else:
                # Fallback: create evidence from text analysis
                return self._create_fallback_evidence(response, modality)
                
        except json.JSONDecodeError:
            return self._create_fallback_evidence(response, modality)
    
    def _create_fallback_evidence(self, response: str, modality: ModalityType) -> List[CausalEvidence]:
        """Create evidence when JSON parsing fails"""
        confidence = 0.6 if "strong" in response.lower() else 0.4 if "moderate" in response.lower() else 0.3
        strength = EvidenceStrength.STRONG if "strong" in response.lower() else EvidenceStrength.MODERATE if "moderate" in response.lower() else EvidenceStrength.WEAK
        
        return [CausalEvidence(
            evidence_type=CausalEvidenceType.CORRELATION,
            strength=strength,
            description=response[:500],
            source_modality=modality,
            reliability_score=confidence,
            supporting_data={"raw_response": response}
        )]
    
    async def _synthesize_multimodal_evidence(self,
                                            treatment: str,
                                            outcome: str,
                                            evidence_pieces: List[CausalEvidence],
                                            domain: str,
                                            context: str) -> MultiModalCausalAssessment:
        """Synthesize evidence from multiple modalities into final assessment"""
        
        # Group evidence by modality
        modal_evidence = {}
        for evidence in evidence_pieces:
            modality = evidence.source_modality.value
            if modality not in modal_evidence:
                modal_evidence[modality] = []
            modal_evidence[modality].append(evidence)
        
        # Calculate consensus scores
        modal_consensus = {}
        for modality, evidences in modal_evidence.items():
            scores = [e.reliability_score for e in evidences]
            modal_consensus[modality] = np.mean(scores) if scores else 0.0
        
        # Build synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt(
            treatment, outcome, evidence_pieces, modal_consensus, domain, context
        )
        
        try:
            response = await self.llm_client.generate_response(
                synthesis_prompt,
                max_tokens=2000,
                temperature=0.2
            )
            
            return self._parse_synthesis_response(
                response, treatment, outcome, evidence_pieces, modal_consensus
            )
            
        except Exception as e:
            logger.error(f"Error in evidence synthesis: {e}")
            return self._create_default_assessment(
                treatment, outcome, evidence_pieces, modal_consensus
            )
    
    def _build_synthesis_prompt(self, treatment: str, outcome: str, 
                              evidence_pieces: List[CausalEvidence],
                              modal_consensus: Dict[str, float],
                              domain: str, context: str) -> str:
        """Build prompt for evidence synthesis"""
        
        evidence_summary = "\n".join([
            f"- {e.source_modality.value}: {e.evidence_type.value} ({e.strength.value}) - {e.description[:100]}"
            for e in evidence_pieces
        ])
        
        consensus_summary = "\n".join([
            f"- {modality}: {score:.2f} confidence"
            for modality, score in modal_consensus.items()
        ])
        
        return f"""
        You are synthesizing causal evidence from multiple data modalities to assess the causal relationship 
        between '{treatment}' and '{outcome}' in the {domain} domain.
        
        Context: {context}
        
        Evidence from different modalities:
        {evidence_summary}
        
        Modality consensus scores:
        {consensus_summary}
        
        Please provide a comprehensive synthesis addressing:
        1. Overall causal claim and evidence strength
        2. Areas of agreement/disagreement between modalities
        3. Key limitations and uncertainties
        4. Practical recommendations
        
        Respond in JSON format:
        {{
            "causal_claim": "Clear statement of causal relationship",
            "overall_strength": "weak|moderate|strong|very_strong",
            "confidence_score": 0.0-1.0,
            "contradictions": ["List of contradictions between modalities"],
            "limitations": ["Key limitations"],
            "recommendations": ["Practical recommendations"],
            "narrative_summary": "2-3 sentence summary of findings"
        }}
        """
    
    def _parse_synthesis_response(self, response: str, treatment: str, outcome: str,
                                evidence_pieces: List[CausalEvidence],
                                modal_consensus: Dict[str, float]) -> MultiModalCausalAssessment:
        """Parse synthesis response into assessment"""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                return MultiModalCausalAssessment(
                    treatment_variable=treatment,
                    outcome_variable=outcome,
                    causal_claim=parsed.get('causal_claim', f'Relationship between {treatment} and {outcome}'),
                    overall_evidence_strength=EvidenceStrength(parsed.get('overall_strength', 'weak')),
                    confidence_score=float(parsed.get('confidence_score', 0.5)),
                    evidence_pieces=evidence_pieces,
                    modal_consensus=modal_consensus,
                    contradictions=parsed.get('contradictions', []),
                    limitations=parsed.get('limitations', []),
                    recommendations=parsed.get('recommendations', []),
                    narrative_summary=parsed.get('narrative_summary', 'Analysis completed with mixed evidence.')
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        return self._create_default_assessment(treatment, outcome, evidence_pieces, modal_consensus)
    
    def _create_default_assessment(self, treatment: str, outcome: str,
                                 evidence_pieces: List[CausalEvidence],
                                 modal_consensus: Dict[str, float]) -> MultiModalCausalAssessment:
        """Create default assessment when parsing fails"""
        
        # Calculate overall confidence from evidence
        if evidence_pieces:
            avg_confidence = np.mean([e.reliability_score for e in evidence_pieces])
            strength_scores = {
                EvidenceStrength.WEAK: 1,
                EvidenceStrength.MODERATE: 2,
                EvidenceStrength.STRONG: 3,
                EvidenceStrength.VERY_STRONG: 4
            }
            avg_strength = np.mean([strength_scores[e.strength] for e in evidence_pieces])
            
            if avg_strength >= 3:
                overall_strength = EvidenceStrength.STRONG
            elif avg_strength >= 2:
                overall_strength = EvidenceStrength.MODERATE
            else:
                overall_strength = EvidenceStrength.WEAK
        else:
            avg_confidence = 0.3
            overall_strength = EvidenceStrength.WEAK
        
        return MultiModalCausalAssessment(
            treatment_variable=treatment,
            outcome_variable=outcome,
            causal_claim=f"Multi-modal analysis of {treatment} effect on {outcome}",
            overall_evidence_strength=overall_strength,
            confidence_score=avg_confidence,
            evidence_pieces=evidence_pieces,
            modal_consensus=modal_consensus,
            contradictions=["Unable to detect specific contradictions"],
            limitations=["Analysis based on automated assessment"],
            recommendations=["Consider additional validation studies"],
            narrative_summary=f"Multi-modal analysis found {overall_strength.value} evidence for causal relationship."
        )
    
    async def compare_modality_evidence(self, assessment: MultiModalCausalAssessment) -> Dict[str, Any]:
        """Compare evidence strength across different modalities"""
        
        modality_comparison = {}
        for evidence in assessment.evidence_pieces:
            modality = evidence.source_modality.value
            if modality not in modality_comparison:
                modality_comparison[modality] = {
                    'evidence_count': 0,
                    'avg_reliability': 0.0,
                    'evidence_types': [],
                    'strengths': []
                }
            
            modality_comparison[modality]['evidence_count'] += 1
            modality_comparison[modality]['evidence_types'].append(evidence.evidence_type.value)
            modality_comparison[modality]['strengths'].append(evidence.strength.value)
        
        # Calculate averages
        for modality_data in modality_comparison.values():
            reliabilities = [e.reliability_score for e in assessment.evidence_pieces 
                           if e.source_modality.value in modality_data]
            modality_data['avg_reliability'] = np.mean(reliabilities) if reliabilities else 0.0
        
        return modality_comparison
    
    def export_assessment(self, assessment: MultiModalCausalAssessment, 
                         export_path: str = None) -> str:
        """Export assessment to JSON file"""
        
        export_data = {
            'assessment': asdict(assessment),
            'timestamp': pd.Timestamp.now().isoformat(),
            'analysis_type': 'multimodal_causal_analysis'
        }
        
        if export_path:
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            return export_path
        else:
            return json.dumps(export_data, indent=2, default=str)
    
    def _get_text_analysis_prompt(self) -> str:
        """Prompt template for text data analysis"""
        return """
        Analyze the provided text data for causal evidence. Look for:
        1. Explicit causal claims and their support
        2. Temporal relationships and sequence of events
        3. Mechanism descriptions and pathways
        4. Discussion of confounding factors
        5. Experimental or quasi-experimental evidence
        """
    
    def _get_structured_analysis_prompt(self) -> str:
        """Prompt template for structured data analysis"""
        return """
        Analyze the structured dataset for causal evidence. Focus on:
        1. Statistical relationships and correlations
        2. Dose-response patterns
        3. Temporal precedence in longitudinal data
        4. Control for confounding variables
        5. Robustness of associations across subgroups
        """
    
    def _get_image_analysis_prompt(self) -> str:
        """Prompt template for image analysis"""
        return """
        Analyze visual evidence for causal relationships. Consider:
        1. Visual representations of causal processes
        2. Before/after comparisons
        3. Spatial or temporal patterns
        4. Process diagrams or flow charts
        5. Visual evidence of mechanisms
        """
    
    def _get_timeseries_analysis_prompt(self) -> str:
        """Prompt template for time series analysis"""
        return """
        Analyze time series data for causal evidence. Examine:
        1. Temporal precedence (cause before effect)
        2. Granger causality patterns
        3. Intervention timing and effects
        4. Lag structures and delayed effects
        5. Seasonal or cyclical patterns
        """
    
    def _get_cross_sectional_prompt(self) -> str:
        """Prompt template for cross-sectional analysis"""
        return """
        Analyze cross-sectional data for causal evidence. Focus on:
        1. Natural experiments and instrumental variables
        2. Regression discontinuity designs
        3. Matching and propensity score evidence
        4. Dose-response relationships
        5. Robustness across different specifications
        """


async def main():
    """Example usage of LLM Multi-Modal Causal Analysis"""
    
    # Initialize the analysis system
    analyzer = LLMMultiModalAnalysis()
    
    # Example multi-modal data
    data_sources = {
        "text": "Study found that exercise training programs led to significant improvements in cardiovascular health outcomes across multiple randomized controlled trials.",
        "structured": pd.DataFrame({
            'exercise_hours': [0, 2, 4, 6, 8],
            'cardiovascular_score': [60, 65, 72, 78, 85],
            'age': [45, 47, 52, 49, 51]
        }),
        "time_series": pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=12, freq='M'),
            'exercise_program_intensity': [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
            'health_outcomes': [60, 62, 65, 68, 72, 75, 78, 82, 85, 87, 90, 92]
        })
    }
    
    # Perform multi-modal analysis
    assessment = await analyzer.analyze_multimodal_causality(
        treatment_variable="exercise_program",
        outcome_variable="cardiovascular_health",
        data_sources=data_sources,
        domain="healthcare",
        context="Evaluating effectiveness of structured exercise interventions"
    )
    
    print("Multi-Modal Causal Assessment:")
    print(f"Claim: {assessment.causal_claim}")
    print(f"Evidence Strength: {assessment.overall_evidence_strength.value}")
    print(f"Confidence: {assessment.confidence_score:.2f}")
    print(f"Evidence Pieces: {len(assessment.evidence_pieces)}")
    print(f"Recommendations: {assessment.recommendations}")

if __name__ == "__main__":
    asyncio.run(main())