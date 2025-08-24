"""
Causal Argument Validator

This module validates the logical consistency of causal claims and arguments,
identifying potential fallacies, inconsistencies, and gaps in causal reasoning.
It helps ensure rigorous causal inference by checking arguments against
established principles and logical frameworks.
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import re
from collections import defaultdict

from causalllm.logging import get_logger


class ArgumentType(Enum):
    """Types of causal arguments."""
    DESCRIPTIVE_CLAIM = "descriptive_claim"        # X is associated with Y
    CAUSAL_CLAIM = "causal_claim"                 # X causes Y
    COUNTERFACTUAL_CLAIM = "counterfactual_claim" # If X had been different, Y would be different
    MECHANISTIC_CLAIM = "mechanistic_claim"       # X causes Y through mechanism Z
    PROBABILISTIC_CLAIM = "probabilistic_claim"   # X increases probability of Y
    COMPARATIVE_CLAIM = "comparative_claim"       # X has stronger effect than Z
    CONDITIONAL_CLAIM = "conditional_claim"       # X causes Y only if Z


class LogicalFallacy(Enum):
    """Types of logical fallacies in causal arguments."""
    CORRELATION_CAUSATION = "correlation_implies_causation"
    POST_HOC_ERGO_PROPTER_HOC = "post_hoc_ergo_propter_hoc"
    REVERSE_CAUSATION = "reverse_causation"
    CONFOUNDING_BIAS = "confounding_bias"
    SELECTION_BIAS = "selection_bias"
    CHERRY_PICKING = "cherry_picking"
    HASTY_GENERALIZATION = "hasty_generalization"
    ECOLOGICAL_FALLACY = "ecological_fallacy"
    SURVIVORS_BIAS = "survivors_bias"
    FALSE_DILEMMA = "false_dilemma"
    AFFIRMING_CONSEQUENT = "affirming_the_consequent"
    DENYING_ANTECEDENT = "denying_the_antecedent"
    SLIPPERY_SLOPE = "slippery_slope"
    CIRCULAR_REASONING = "circular_reasoning"


class ValidationCriterion(Enum):
    """Criteria for validating causal arguments."""
    TEMPORAL_PRECEDENCE = "temporal_precedence"
    COVARIATION = "covariation"
    ALTERNATIVE_EXPLANATIONS = "alternative_explanations"
    MECHANISM_PLAUSIBILITY = "mechanism_plausibility"
    DOSE_RESPONSE = "dose_response"
    CONSISTENCY = "consistency"
    REVERSIBILITY = "reversibility"
    BIOLOGICAL_PLAUSIBILITY = "biological_plausibility"
    STRENGTH_OF_ASSOCIATION = "strength_of_association"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"      # Argument is fundamentally flawed
    MAJOR = "major"           # Significant weaknesses that undermine argument
    MODERATE = "moderate"     # Notable issues that should be addressed
    MINOR = "minor"          # Minor concerns or suggestions for improvement
    INFO = "info"            # Informational notes or observations


@dataclass
class ValidationIssue:
    """A validation issue found in a causal argument."""
    
    issue_type: Union[LogicalFallacy, ValidationCriterion, str]
    severity: ValidationSeverity
    description: str
    location: str  # Where in the argument the issue occurs
    explanation: str
    suggested_fix: str
    evidence_needed: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)


@dataclass
class CausalArgument:
    """Structured representation of a causal argument."""
    
    main_claim: str
    argument_type: ArgumentType
    premises: List[str]
    evidence_cited: List[str]
    reasoning_chain: List[str]
    context: str = ""
    domain: str = "general"
    assumptions: List[str] = field(default_factory=list)
    potential_confounders: List[str] = field(default_factory=list)


@dataclass 
class ValidationResult:
    """Results from causal argument validation."""
    
    argument: CausalArgument
    overall_validity_score: float  # 0-1 scale
    logical_consistency_score: float
    evidence_adequacy_score: float
    
    validation_issues: List[ValidationIssue]
    critical_issues: List[ValidationIssue]
    major_issues: List[ValidationIssue]
    
    strengths_identified: List[str]
    weaknesses_identified: List[str]
    improvement_suggestions: List[str]
    
    alternative_explanations: List[str]
    missing_evidence: List[str]
    methodological_concerns: List[str]
    
    validation_summary: str
    confidence_in_validation: float


class CausalArgumentValidator:
    """LLM-enhanced causal argument validation system."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = get_logger("causalllm.causal_argument_validator")
        
        # Bradford Hill criteria for causal validation
        self.bradford_hill_criteria = [
            ValidationCriterion.STRENGTH_OF_ASSOCIATION,
            ValidationCriterion.CONSISTENCY,
            ValidationCriterion.TEMPORAL_PRECEDENCE,
            ValidationCriterion.DOSE_RESPONSE,
            ValidationCriterion.BIOLOGICAL_PLAUSIBILITY,
            ValidationCriterion.ALTERNATIVE_EXPLANATIONS,
            ValidationCriterion.REVERSIBILITY
        ]
        
        # Common fallacy patterns
        self.fallacy_patterns = {
            LogicalFallacy.CORRELATION_CAUSATION: [
                r"correlated|associated.*therefore.*caus",
                r"linked.*so.*caus",
                r"related.*means.*caus"
            ],
            LogicalFallacy.POST_HOC_ERGO_PROPTER_HOC: [
                r"after.*therefore.*because",
                r"following.*so.*caus",
                r"subsequent.*due to"
            ],
            LogicalFallacy.REVERSE_CAUSATION: [
                r"might actually be.*other way around",
                r"could be reverse"
            ]
        }
        
        # Domain-specific validation rules
        self.domain_rules = {
            "healthcare": {
                "required_evidence": ["randomized_trial", "observational_study", "mechanism"],
                "critical_assumptions": ["no_unmeasured_confounding", "stable_unit_treatment"],
                "common_fallacies": [LogicalFallacy.CONFOUNDING_BIAS, LogicalFallacy.SELECTION_BIAS]
            },
            "business": {
                "required_evidence": ["controlled_experiment", "natural_experiment", "time_series"],
                "critical_assumptions": ["market_stability", "customer_behavior_consistency"],
                "common_fallacies": [LogicalFallacy.CHERRY_PICKING, LogicalFallacy.SURVIVORS_BIAS]
            },
            "education": {
                "required_evidence": ["randomized_trial", "quasi_experiment", "longitudinal_study"],
                "critical_assumptions": ["no_spillover_effects", "implementation_fidelity"],
                "common_fallacies": [LogicalFallacy.SELECTION_BIAS, LogicalFallacy.HASTY_GENERALIZATION]
            }
        }
    
    async def validate_causal_argument(self, 
                                     argument: Union[CausalArgument, str],
                                     domain: str = "general",
                                     strict_mode: bool = False) -> ValidationResult:
        """
        Comprehensively validate a causal argument.
        
        Args:
            argument: Structured argument or raw text to validate
            domain: Domain context for validation
            strict_mode: Whether to apply strict validation criteria
            
        Returns:
            Comprehensive validation results
        """
        self.logger.info("Starting comprehensive causal argument validation")
        
        # Parse argument if provided as text
        if isinstance(argument, str):
            argument = await self._parse_argument_from_text(argument, domain)
        
        # Step 1: Structural validation
        structural_issues = await self._validate_argument_structure(argument)
        
        # Step 2: Logical consistency check
        logical_issues = await self._check_logical_consistency(argument)
        
        # Step 3: Evidence adequacy assessment
        evidence_issues = await self._assess_evidence_adequacy(argument, domain)
        
        # Step 4: Fallacy detection
        fallacy_issues = await self._detect_logical_fallacies(argument)
        
        # Step 5: Domain-specific validation
        domain_issues = await self._validate_domain_specific_rules(argument, domain)
        
        # Step 6: Alternative explanation analysis
        alternative_explanations = await self._identify_alternative_explanations(argument)
        
        # Step 7: Generate comprehensive assessment
        all_issues = structural_issues + logical_issues + evidence_issues + fallacy_issues + domain_issues
        
        # Categorize issues by severity
        critical_issues = [issue for issue in all_issues if issue.severity == ValidationSeverity.CRITICAL]
        major_issues = [issue for issue in all_issues if issue.severity == ValidationSeverity.MAJOR]
        
        # Calculate scores
        overall_score = self._calculate_overall_validity_score(all_issues, len(argument.premises))
        logical_score = self._calculate_logical_consistency_score(logical_issues + fallacy_issues)
        evidence_score = self._calculate_evidence_adequacy_score(evidence_issues)
        
        # Generate strengths and improvements
        strengths = await self._identify_argument_strengths(argument, all_issues)
        improvements = await self._generate_improvement_suggestions(all_issues, argument)
        
        # Create validation summary
        summary = await self._generate_validation_summary(argument, all_issues, overall_score)
        
        result = ValidationResult(
            argument=argument,
            overall_validity_score=overall_score,
            logical_consistency_score=logical_score,
            evidence_adequacy_score=evidence_score,
            validation_issues=all_issues,
            critical_issues=critical_issues,
            major_issues=major_issues,
            strengths_identified=strengths,
            weaknesses_identified=[issue.description for issue in major_issues + critical_issues],
            improvement_suggestions=improvements,
            alternative_explanations=alternative_explanations,
            missing_evidence=[issue.evidence_needed for issue in evidence_issues if issue.evidence_needed],
            methodological_concerns=[issue.description for issue in all_issues 
                                   if "methodological" in issue.description.lower()],
            validation_summary=summary,
            confidence_in_validation=0.85  # Base confidence, could be adjusted
        )
        
        self.logger.info(f"Argument validation completed. Overall score: {overall_score:.2f}")
        return result
    
    async def _parse_argument_from_text(self, text: str, domain: str) -> CausalArgument:
        """Parse a causal argument from natural language text."""
        
        parse_prompt = f"""
        Parse this text into a structured causal argument. Identify:
        
        TEXT: {text}
        DOMAIN: {domain}
        
        Extract:
        1. Main causal claim (the primary assertion)
        2. Type of argument (descriptive, causal, counterfactual, etc.)
        3. Supporting premises (evidence or reasons given)
        4. Evidence cited (studies, data, examples mentioned)
        5. Reasoning chain (logical steps in the argument)
        6. Underlying assumptions (implicit beliefs required)
        
        Respond in JSON format:
        {{
            "main_claim": "the primary causal claim",
            "argument_type": "causal_claim|counterfactual_claim|etc",
            "premises": ["premise 1", "premise 2"],
            "evidence_cited": ["evidence 1", "evidence 2"],
            "reasoning_chain": ["step 1", "step 2"],
            "assumptions": ["assumption 1", "assumption 2"],
            "context": "relevant context or setting"
        }}
        """
        
        try:
            response = await self.llm_client.generate_response(parse_prompt)
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                
                return CausalArgument(
                    main_claim=parsed_data.get("main_claim", ""),
                    argument_type=ArgumentType(parsed_data.get("argument_type", "causal_claim")),
                    premises=parsed_data.get("premises", []),
                    evidence_cited=parsed_data.get("evidence_cited", []),
                    reasoning_chain=parsed_data.get("reasoning_chain", []),
                    context=parsed_data.get("context", ""),
                    domain=domain,
                    assumptions=parsed_data.get("assumptions", [])
                )
                
        except Exception as e:
            self.logger.error(f"Argument parsing failed: {e}")
        
        # Fallback parsing
        return CausalArgument(
            main_claim=text[:200] + "..." if len(text) > 200 else text,
            argument_type=ArgumentType.CAUSAL_CLAIM,
            premises=[text],
            evidence_cited=[],
            reasoning_chain=[],
            context="",
            domain=domain
        )
    
    async def _validate_argument_structure(self, argument: CausalArgument) -> List[ValidationIssue]:
        """Validate the basic structure of the argument."""
        
        issues = []
        
        # Check if main claim is clear
        if not argument.main_claim or len(argument.main_claim.strip()) < 10:
            issues.append(ValidationIssue(
                issue_type="unclear_main_claim",
                severity=ValidationSeverity.CRITICAL,
                description="Main causal claim is unclear or missing",
                location="main_claim",
                explanation="A clear causal claim is essential for evaluation",
                suggested_fix="Restate the main claim clearly and specifically",
                evidence_needed=["clear_causal_statement"]
            ))
        
        # Check for supporting premises
        if not argument.premises or len(argument.premises) == 0:
            issues.append(ValidationIssue(
                issue_type="no_supporting_premises",
                severity=ValidationSeverity.MAJOR,
                description="No supporting premises provided",
                location="premises",
                explanation="Causal claims require supporting evidence or reasoning",
                suggested_fix="Provide premises that support the causal claim",
                evidence_needed=["supporting_evidence"]
            ))
        
        # Check reasoning chain
        if not argument.reasoning_chain or len(argument.reasoning_chain) == 0:
            issues.append(ValidationIssue(
                issue_type="missing_reasoning_chain",
                severity=ValidationSeverity.MODERATE,
                description="No explicit reasoning chain provided",
                location="reasoning_chain",
                explanation="Clear reasoning helps evaluate argument validity",
                suggested_fix="Explicitly state the logical steps connecting premises to conclusion",
                evidence_needed=["logical_reasoning"]
            ))
        
        return issues
    
    async def _check_logical_consistency(self, argument: CausalArgument) -> List[ValidationIssue]:
        """Check for logical consistency in the argument."""
        
        consistency_prompt = f"""
        Check this causal argument for logical consistency and coherence:
        
        MAIN CLAIM: {argument.main_claim}
        PREMISES: {argument.premises}
        REASONING: {argument.reasoning_chain}
        ASSUMPTIONS: {argument.assumptions}
        
        Identify any:
        1. Contradictions between premises
        2. Gaps in logical reasoning
        3. Inconsistent assumptions
        4. Circular reasoning
        5. Non-sequiturs (conclusions that don't follow)
        
        For each issue found, specify:
        - Type of logical problem
        - Where it occurs
        - Why it's problematic
        - How to fix it
        
        Respond as JSON array of issues:
        [
            {{
                "issue_type": "contradiction|logical_gap|circular_reasoning|etc",
                "severity": "critical|major|moderate|minor",
                "description": "brief description",
                "location": "where the issue occurs",
                "explanation": "why this is problematic",
                "suggested_fix": "how to address it"
            }}
        ]
        """
        
        try:
            response = await self.llm_client.generate_response(consistency_prompt)
            
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                issues_data = json.loads(json_match.group())
                
                issues = []
                for issue_data in issues_data:
                    issue = ValidationIssue(
                        issue_type=issue_data.get("issue_type", "logical_inconsistency"),
                        severity=ValidationSeverity(issue_data.get("severity", "moderate")),
                        description=issue_data.get("description", ""),
                        location=issue_data.get("location", ""),
                        explanation=issue_data.get("explanation", ""),
                        suggested_fix=issue_data.get("suggested_fix", "")
                    )
                    issues.append(issue)
                
                return issues
                
        except Exception as e:
            self.logger.error(f"Logical consistency check failed: {e}")
        
        return []
    
    async def _assess_evidence_adequacy(self, argument: CausalArgument, domain: str) -> List[ValidationIssue]:
        """Assess whether the evidence is adequate for the causal claim."""
        
        issues = []
        
        # Check for evidence
        if not argument.evidence_cited:
            issues.append(ValidationIssue(
                issue_type="no_evidence_cited",
                severity=ValidationSeverity.MAJOR,
                description="No empirical evidence cited",
                location="evidence_cited",
                explanation="Causal claims require empirical support",
                suggested_fix="Provide relevant studies, data, or experiments",
                evidence_needed=["empirical_studies", "data_analysis", "experiments"]
            ))
        
        # Domain-specific evidence requirements
        domain_rules = self.domain_rules.get(domain, {})
        required_evidence = domain_rules.get("required_evidence", [])
        
        for req_evidence in required_evidence:
            if not any(req_evidence in str(evidence).lower() for evidence in argument.evidence_cited):
                issues.append(ValidationIssue(
                    issue_type=f"missing_{req_evidence}",
                    severity=ValidationSeverity.MODERATE,
                    description=f"Missing {req_evidence.replace('_', ' ')} evidence",
                    location="evidence_cited",
                    explanation=f"{req_evidence.replace('_', ' ').title()} is typically required in {domain}",
                    suggested_fix=f"Include {req_evidence.replace('_', ' ')} evidence",
                    evidence_needed=[req_evidence]
                ))
        
        # Check evidence quality
        if len(argument.evidence_cited) < 2:
            issues.append(ValidationIssue(
                issue_type="insufficient_evidence",
                severity=ValidationSeverity.MODERATE,
                description="Limited evidence sources",
                location="evidence_cited",
                explanation="Single sources may be biased or insufficient",
                suggested_fix="Provide multiple independent sources of evidence",
                evidence_needed=["multiple_sources"]
            ))
        
        return issues
    
    async def _detect_logical_fallacies(self, argument: CausalArgument) -> List[ValidationIssue]:
        """Detect logical fallacies in the causal argument."""
        
        fallacy_prompt = f"""
        Analyze this causal argument for logical fallacies commonly found in causal reasoning:
        
        ARGUMENT: {argument.main_claim}
        PREMISES: {argument.premises}
        EVIDENCE: {argument.evidence_cited}
        CONTEXT: {argument.context}
        
        Check for these specific fallacies:
        1. Correlation implies causation
        2. Post hoc ergo propter hoc (after this, therefore because of this)
        3. Reverse causation
        4. Confounding bias (third variable explanation)
        5. Selection bias
        6. Cherry picking evidence
        7. Hasty generalization
        8. Ecological fallacy
        9. Survivor's bias
        10. False dilemma
        11. Circular reasoning
        12. Slippery slope
        
        For each fallacy found:
        - Identify the specific fallacy
        - Explain how it appears in the argument
        - Assess severity (critical/major/moderate/minor)
        - Suggest how to address it
        
        Respond as JSON array of fallacies found.
        """
        
        try:
            response = await self.llm_client.generate_response(fallacy_prompt)
            
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                fallacies_data = json.loads(json_match.group())
                
                issues = []
                for fallacy_data in fallacies_data:
                    try:
                        fallacy_type = fallacy_data.get("fallacy_type", "unknown_fallacy")
                        # Try to map to enum
                        if fallacy_type in [f.value for f in LogicalFallacy]:
                            issue_type = LogicalFallacy(fallacy_type)
                        else:
                            issue_type = fallacy_type
                        
                        issue = ValidationIssue(
                            issue_type=issue_type,
                            severity=ValidationSeverity(fallacy_data.get("severity", "moderate")),
                            description=fallacy_data.get("description", ""),
                            location=fallacy_data.get("location", "argument"),
                            explanation=fallacy_data.get("explanation", ""),
                            suggested_fix=fallacy_data.get("suggested_fix", "")
                        )
                        issues.append(issue)
                    except (ValueError, KeyError):
                        continue
                
                return issues
                
        except Exception as e:
            self.logger.error(f"Fallacy detection failed: {e}")
        
        # Pattern-based fallback detection
        return self._pattern_based_fallacy_detection(argument)
    
    def _pattern_based_fallacy_detection(self, argument: CausalArgument) -> List[ValidationIssue]:
        """Fallback pattern-based fallacy detection."""
        
        issues = []
        full_text = f"{argument.main_claim} {' '.join(argument.premises)}"
        
        for fallacy, patterns in self.fallacy_patterns.items():
            for pattern in patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    issues.append(ValidationIssue(
                        issue_type=fallacy,
                        severity=ValidationSeverity.MODERATE,
                        description=f"Potential {fallacy.value.replace('_', ' ')} detected",
                        location="argument_text",
                        explanation=f"Pattern matching suggests {fallacy.value.replace('_', ' ')} fallacy",
                        suggested_fix=f"Review argument for {fallacy.value.replace('_', ' ')} issues"
                    ))
                    break
        
        return issues
    
    async def _validate_domain_specific_rules(self, argument: CausalArgument, domain: str) -> List[ValidationIssue]:
        """Validate against domain-specific rules and standards."""
        
        issues = []
        domain_rules = self.domain_rules.get(domain, {})
        
        # Check critical assumptions
        critical_assumptions = domain_rules.get("critical_assumptions", [])
        for assumption in critical_assumptions:
            if not any(assumption.replace("_", " ") in str(arg_assumption).lower() 
                      for arg_assumption in argument.assumptions):
                issues.append(ValidationIssue(
                    issue_type=f"missing_assumption_{assumption}",
                    severity=ValidationSeverity.MAJOR,
                    description=f"Missing critical assumption: {assumption.replace('_', ' ')}",
                    location="assumptions",
                    explanation=f"This assumption is critical for valid {domain} causal claims",
                    suggested_fix=f"Explicitly state and justify the {assumption.replace('_', ' ')} assumption",
                    evidence_needed=[f"justification_for_{assumption}"]
                ))
        
        # Check for common domain fallacies
        common_fallacies = domain_rules.get("common_fallacies", [])
        # This would be enhanced with domain-specific detection logic
        
        return issues
    
    async def _identify_alternative_explanations(self, argument: CausalArgument) -> List[str]:
        """Identify potential alternative explanations for the claimed causal relationship."""
        
        alternatives_prompt = f"""
        Given this causal argument, identify plausible alternative explanations:
        
        CLAIM: {argument.main_claim}
        CONTEXT: {argument.context}
        DOMAIN: {argument.domain}
        
        Consider:
        1. Reverse causation
        2. Third variable/confounding explanations
        3. Selection effects
        4. Measurement artifacts
        5. Coincidence/spurious correlation
        6. Different mechanisms than proposed
        
        List 3-5 most plausible alternative explanations.
        """
        
        try:
            response = await self.llm_client.generate_response(alternatives_prompt)
            
            # Extract alternatives (assuming they're listed)
            alternatives = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or 
                           line.startswith(tuple('123456789'))):
                    # Clean up formatting
                    alt = re.sub(r'^[-•\d\.\)]\s*', '', line)
                    if alt:
                        alternatives.append(alt)
            
            return alternatives[:5]  # Limit to top 5
            
        except Exception as e:
            self.logger.error(f"Alternative explanation identification failed: {e}")
            return ["Consider reverse causation", "Check for confounding variables", "Examine selection bias"]
    
    def _calculate_overall_validity_score(self, issues: List[ValidationIssue], num_premises: int) -> float:
        """Calculate overall validity score based on issues found."""
        
        if not issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.CRITICAL: -0.4,
            ValidationSeverity.MAJOR: -0.2,
            ValidationSeverity.MODERATE: -0.1,
            ValidationSeverity.MINOR: -0.05,
            ValidationSeverity.INFO: 0.0
        }
        
        penalty = sum(severity_weights[issue.severity] for issue in issues)
        
        # Adjust for argument complexity (more premises = more opportunity for issues)
        complexity_bonus = min(num_premises * 0.05, 0.2)
        
        score = max(0.0, min(1.0, 1.0 + penalty + complexity_bonus))
        return score
    
    def _calculate_logical_consistency_score(self, logical_issues: List[ValidationIssue]) -> float:
        """Calculate logical consistency score."""
        
        if not logical_issues:
            return 1.0
        
        critical_logical = sum(1 for issue in logical_issues if issue.severity == ValidationSeverity.CRITICAL)
        major_logical = sum(1 for issue in logical_issues if issue.severity == ValidationSeverity.MAJOR)
        
        # Heavy penalty for critical logical issues
        penalty = critical_logical * 0.5 + major_logical * 0.2
        
        return max(0.0, 1.0 - penalty)
    
    def _calculate_evidence_adequacy_score(self, evidence_issues: List[ValidationIssue]) -> float:
        """Calculate evidence adequacy score."""
        
        if not evidence_issues:
            return 1.0
        
        no_evidence_issues = sum(1 for issue in evidence_issues 
                               if "no_evidence" in str(issue.issue_type))
        insufficient_issues = sum(1 for issue in evidence_issues 
                                if "insufficient" in str(issue.issue_type))
        
        penalty = no_evidence_issues * 0.4 + insufficient_issues * 0.2
        
        return max(0.0, 1.0 - penalty)
    
    async def _identify_argument_strengths(self, argument: CausalArgument, issues: List[ValidationIssue]) -> List[str]:
        """Identify strengths in the argument."""
        
        strengths = []
        
        # Structural strengths
        if argument.main_claim and len(argument.main_claim) > 20:
            strengths.append("Clear and specific main claim")
        
        if len(argument.premises) >= 3:
            strengths.append("Multiple supporting premises provided")
        
        if argument.evidence_cited and len(argument.evidence_cited) >= 2:
            strengths.append("Multiple sources of evidence cited")
        
        if argument.reasoning_chain and len(argument.reasoning_chain) >= 2:
            strengths.append("Explicit reasoning chain provided")
        
        if argument.assumptions:
            strengths.append("Underlying assumptions made explicit")
        
        # Check for absence of critical issues
        critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
        if not critical_issues:
            strengths.append("No critical logical flaws detected")
        
        return strengths
    
    async def _generate_improvement_suggestions(self, issues: List[ValidationIssue], argument: CausalArgument) -> List[str]:
        """Generate specific improvement suggestions."""
        
        suggestions = []
        
        # Collect suggestions from issues
        for issue in issues:
            if issue.suggested_fix and issue.suggested_fix not in suggestions:
                suggestions.append(issue.suggested_fix)
        
        # General suggestions based on argument type
        if argument.argument_type == ArgumentType.CAUSAL_CLAIM:
            suggestions.append("Consider alternative explanations for the observed relationship")
            suggestions.append("Address potential confounding variables")
        
        # Limit suggestions
        return suggestions[:8]
    
    async def _generate_validation_summary(self, argument: CausalArgument, issues: List[ValidationIssue], score: float) -> str:
        """Generate a comprehensive validation summary."""
        
        critical_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)
        major_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.MAJOR)
        
        summary_prompt = f"""
        Create a concise validation summary for this causal argument:
        
        ARGUMENT: {argument.main_claim}
        OVERALL SCORE: {score:.2f}/1.0
        CRITICAL ISSUES: {critical_count}
        MAJOR ISSUES: {major_count}
        TOTAL ISSUES: {len(issues)}
        
        Issues found: {[issue.description for issue in issues[:5]]}
        
        Provide a 2-3 sentence summary of the argument's validity and main concerns.
        """
        
        try:
            response = await self.llm_client.generate_response(summary_prompt)
            return response.strip()
        
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            
            if score >= 0.8:
                return f"Strong argument with score {score:.2f}. Minor issues identified but overall reasoning is sound."
            elif score >= 0.6:
                return f"Moderate argument with score {score:.2f}. Some significant issues need addressing."
            else:
                return f"Weak argument with score {score:.2f}. Major logical or evidential problems identified."
    
    async def compare_arguments(self, arguments: List[CausalArgument], domain: str = "general") -> Dict[str, Any]:
        """Compare multiple causal arguments and rank them by validity."""
        
        self.logger.info(f"Comparing {len(arguments)} causal arguments")
        
        # Validate each argument
        validation_results = []
        for arg in arguments:
            result = await self.validate_causal_argument(arg, domain)
            validation_results.append(result)
        
        # Rank by overall validity score
        ranked_results = sorted(validation_results, key=lambda r: r.overall_validity_score, reverse=True)
        
        # Generate comparison summary
        comparison_prompt = f"""
        Compare these {len(arguments)} causal arguments based on their validation results:
        
        Arguments and scores:
        """
        
        for i, result in enumerate(ranked_results):
            comparison_prompt += f"""
        Argument {i+1}: "{result.argument.main_claim[:100]}..."
        - Score: {result.overall_validity_score:.2f}
        - Critical Issues: {len(result.critical_issues)}
        - Major Issues: {len(result.major_issues)}
        """
        
        comparison_prompt += """
        
        Provide:
        1. Ranking from strongest to weakest
        2. Key differentiating factors
        3. Which argument is most convincing and why
        4. Common weaknesses across arguments
        """
        
        try:
            comparison_response = await self.llm_client.generate_response(comparison_prompt)
            
            return {
                "ranked_results": ranked_results,
                "comparison_analysis": comparison_response,
                "best_argument": ranked_results[0] if ranked_results else None,
                "summary_statistics": {
                    "avg_score": sum(r.overall_validity_score for r in validation_results) / len(validation_results),
                    "score_range": (min(r.overall_validity_score for r in validation_results),
                                  max(r.overall_validity_score for r in validation_results)),
                    "total_critical_issues": sum(len(r.critical_issues) for r in validation_results)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Argument comparison failed: {e}")
            return {"ranked_results": ranked_results, "comparison_analysis": "Comparison failed"}


# Convenience functions
def create_argument_validator(llm_client) -> CausalArgumentValidator:
    """Create a causal argument validator."""
    return CausalArgumentValidator(llm_client)


async def validate_causal_claim(claim: str, 
                              llm_client,
                              domain: str = "general") -> ValidationResult:
    """Quick function to validate a causal claim from text."""
    
    validator = create_argument_validator(llm_client)
    return await validator.validate_causal_argument(claim, domain)


async def check_argument_fallacies(argument_text: str, llm_client) -> List[str]:
    """Quick function to check for logical fallacies in an argument."""
    
    validator = create_argument_validator(llm_client)
    result = await validator.validate_causal_argument(argument_text)
    
    fallacies = []
    for issue in result.validation_issues:
        if isinstance(issue.issue_type, LogicalFallacy):
            fallacies.append(f"{issue.issue_type.value}: {issue.description}")
    
    return fallacies