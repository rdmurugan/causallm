
from typing import List, Tuple
import re
from causalllm.llm_client import BaseLLMClient
from causalllm.logging import get_logger, get_structured_logger

class SCMExplainer:
    def __init__(self, llm_client: BaseLLMClient, model: str = "gpt-4") -> None:
        self.logger = get_logger("causalllm.scm_explainer")
        self.struct_logger = get_structured_logger("scm_explainer")
        
        self.logger.info("Initializing SCMExplainer")
        
        self.llm_client = llm_client
        self.model = model
        
        self.struct_logger.log_interaction(
            "explainer_initialization",
            {
                "model": model,
                "llm_client_type": type(llm_client).__name__
            }
        )
        
        self.logger.info(f"SCMExplainer initialized with model: {model}")

    def extract_variables_and_edges(self, scenario_description: str) -> List[Tuple[str, str]]:
        self.logger.info("Extracting variables and edges from scenario")
        self.logger.debug(f"Scenario length: {len(scenario_description)}")
        
        try:
            prompt = f"""
You're a causal inference modeler.

Read the following scenario and extract causal relationships in the form of edges (A -> B).

Respond only with a list of pairs like:
(A, B)
(B, C)

Scenario:
{scenario_description.strip()}
        """.strip()

            self.logger.debug(f"Generated prompt with length: {len(prompt)}")
            
            response = self.llm_client.chat(prompt, model=self.model, temperature=0.3)
            
            self.logger.debug(f"LLM response length: {len(response)}")
            
            edges = self._parse_edges(response)
            
            self.struct_logger.log_interaction(
                "extract_variables_and_edges",
                {
                    "scenario_length": len(scenario_description),
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    "extracted_edges": edges,
                    "edges_count": len(edges),
                    "model": self.model,
                    "temperature": 0.3
                }
            )
            
            self.logger.info(f"Successfully extracted {len(edges)} edges from scenario")
            return edges
            
        except Exception as e:
            self.logger.error(f"Error extracting variables and edges: {e}")
            self.struct_logger.log_error(e, {
                "scenario_length": len(scenario_description),
                "model": self.model
            })
            raise

    def _parse_edges(self, raw_text: str) -> List[Tuple[str, str]]:
        self.logger.debug("Parsing edges from LLM response")
        
        try:
            pattern = r"\(([^,]+),\s*([^)]+)\)"
            edges = re.findall(pattern, raw_text)
            
            # Clean up the edges (strip whitespace)
            cleaned_edges = [(a.strip(), b.strip()) for a, b in edges]
            
            self.logger.debug(f"Parsed {len(cleaned_edges)} edges from response")
            self.struct_logger.log_interaction(
                "parse_edges",
                {
                    "raw_text_length": len(raw_text),
                    "pattern": pattern,
                    "raw_edges": edges,
                    "cleaned_edges": cleaned_edges,
                    "edges_count": len(cleaned_edges)
                }
            )
            
            return cleaned_edges
            
        except Exception as e:
            self.logger.error(f"Error parsing edges: {e}")
            self.struct_logger.log_error(e, {"raw_text_length": len(raw_text)})
            return []
