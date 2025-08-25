from typing import Any, Dict, List, Optional, Tuple
from causalllm.dag_parser import DAGParser
from causalllm.do_operator import DoOperatorSimulator
from causalllm.prompt_templates import PromptTemplates
from causalllm.counterfactual_engine import CounterfactualEngine
from causalllm.scm_explainer import SCMExplainer
from causalllm.llm_client import BaseLLMClient, get_llm_client
from causalllm.logging import get_logger, get_structured_logger
from causalllm.llm_prompting import CausalPromptEngine
from causalllm.llm_agents import MultiAgentCausalAnalyzer
from causalllm.causal_rag import DynamicCausalRAG, create_rag_system
from causalllm.llm_statistical_interpreter import LLMStatisticalInterpreter
from causalllm.llm_confounder_reasoning import LLMConfounderReasoning
from causalllm.llm_multimodal_analysis import LLMMultiModalAnalysis
from causalllm.llm_causal_discovery import LLMCausalDiscoveryAgent
from causalllm.interactive_causal_qa import InteractiveCausalQA
from causalllm.domain_causal_templates import DomainTemplateEngine
from causalllm.automatic_confounder_detection import AutomaticConfounderDetector
from causalllm.visual_causal_graphs import VisualCausalGraphGenerator


class CausalLLMCore:
    """
    Core class that coordinates various modules to perform causal reasoning
    using DAG parsing, do-calculus simulation, counterfactuals, and SCM explanations.
    """

    def __init__(
        self,
        context: str,
        variables: Dict[str, str],
        dag_edges: List[Tuple[str, str]],
        llm_client: Optional[BaseLLMClient] = None,
    ) -> None:
        """
        Initializes the core engine for causal LLM tasks.

        Args:
            context (str): The textual context of the data or system.
            variables (Dict[str, str]): Mapping of variable names to their descriptions.
            dag_edges (List[Tuple[str, str]]): Directed edges representing the DAG structure.
            llm_client (Optional[BaseLLMClient]): A language model client implementing the chat interface.
                                                 If not provided, it is inferred from environment.
        """
        self.logger = get_logger("causalllm.core")
        self.struct_logger = get_structured_logger("core")
        
        self.logger.info("Initializing CausalLLMCore")
        self.logger.debug(f"Context length: {len(context)}, Variables: {list(variables.keys())}, DAG edges: {len(dag_edges)}")
        
        self.context = context
        self.variables = variables
        self.dag = DAGParser(dag_edges)
        self.do_operator = DoOperatorSimulator(context, variables)
        self.templates = PromptTemplates()
        
        # Default to configured LLM if none provided
        self.llm_client = llm_client or get_llm_client()
        self.logger.info(f"Using LLM client: {type(self.llm_client).__name__}")
        
        # Check if we're using MCP client for special handling
        self.is_mcp_client = type(self.llm_client).__name__ == "MCPClient"
        if self.is_mcp_client:
            self.logger.info("MCP client detected - enabling MCP-specific features")
        
        self.counterfactual = CounterfactualEngine(self.llm_client)
        self.scm = SCMExplainer(self.llm_client)
        
        # Initialize Tier 1 LLM enhancements
        self.intelligent_prompting = CausalPromptEngine()  # Just use default initialization
        self.multi_agent_analyzer = MultiAgentCausalAnalyzer(self.llm_client)
        self.rag_system = create_rag_system(llm_client=self.llm_client)
        
        self.logger.info("Tier 1 LLM enhancements initialized: intelligent prompting, multi-agent analysis, RAG system")
        
        # Initialize Tier 2 LLM enhancements
        self.statistical_interpreter = LLMStatisticalInterpreter(self.llm_client)
        self.confounder_reasoning = LLMConfounderReasoning(self.llm_client)
        self.multimodal_analysis = LLMMultiModalAnalysis(self.llm_client)
        
        self.logger.info("Tier 2 LLM enhancements initialized: statistical interpretation, confounder reasoning, multi-modal analysis")
        
        # Initialize Tier 3 Advanced LLM Features
        self.causal_discovery_agent = LLMCausalDiscoveryAgent(self.llm_client)
        self.interactive_qa = InteractiveCausalQA(self.llm_client)
        self.template_engine = DomainTemplateEngine(self.llm_client)
        self.confounder_detector = AutomaticConfounderDetector(self.llm_client)
        self.graph_visualizer = VisualCausalGraphGenerator()
        
        self.logger.info("Tier 3 Advanced LLM features initialized: causal discovery, interactive Q&A, domain templates, auto-confounder detection, graph visualization")
        
        self.struct_logger.log_interaction(
            "initialization",
            {
                "variables_count": len(variables),
                "dag_edges_count": len(dag_edges),
                "context_length": len(context),
                "llm_client_type": type(self.llm_client).__name__
            }
        )
        
        self.logger.info("CausalLLMCore initialization completed")

    def simulate_do(self, intervention: Dict[str, str], question: Optional[str] = None) -> str:
        """
        Generates a prompt simulating an intervention using do-calculus.

        Args:
            intervention (Dict[str, str]): Variable intervention values.
            question (Optional[str]): Optional question to include in the prompt.

        Returns:
            str: Generated prompt simulating the do-intervention.
        """
        self.logger.info(f"Simulating do-operation with intervention: {intervention}")
        self.logger.debug(f"Question: {question}")
        
        try:
            result = self.do_operator.generate_do_prompt(intervention, question)
            
            self.struct_logger.log_interaction(
                "do_simulation",
                {
                    "intervention": intervention,
                    "question": question,
                    "result_length": len(result)
                }
            )
            
            self.logger.info("Do-operation simulation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in do-operation simulation: {e}")
            self.struct_logger.log_error(e, {"intervention": intervention, "question": question})
            raise

    def simulate_counterfactual(
        self, 
        factual: str, 
        intervention: str, 
        instruction: Optional[str] = None
    ) -> str:
        """
        Generates a counterfactual explanation based on a factual scenario and intervention.

        Args:
            factual (str): The factual statement.
            intervention (str): The hypothetical intervention.
            instruction (Optional[str]): Additional instruction for the model.

        Returns:
            str: Counterfactual explanation or reasoning prompt.
        """
        self.logger.info("Starting counterfactual simulation")
        self.logger.debug(f"Factual: {factual[:100]}..., Intervention: {intervention[:100]}...")
        
        try:
            result = self.counterfactual.simulate_counterfactual(
                self.context, factual, intervention, instruction
            )
            
            self.struct_logger.log_interaction(
                "counterfactual_simulation",
                {
                    "factual_length": len(factual),
                    "intervention_length": len(intervention),
                    "instruction": instruction,
                    "result_length": len(result)
                }
            )
            
            self.logger.info("Counterfactual simulation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in counterfactual simulation: {e}")
            self.struct_logger.log_error(e, {
                "factual_length": len(factual),
                "intervention_length": len(intervention),
                "instruction": instruction
            })
            raise

    def generate_reasoning_prompt(self, task: str = "") -> str:
        """
        Generates a reasoning prompt from the DAG structure and task.

        Args:
            task (str): Optional reasoning task name.

        Returns:
            str: Prompt derived from the DAG.
        """
        self.logger.info(f"Generating reasoning prompt for task: {task}")
        
        try:
            result = self.dag.to_prompt(task)
            
            self.struct_logger.log_interaction(
                "reasoning_prompt_generation",
                {
                    "task": task,
                    "result_length": len(result)
                }
            )
            
            self.logger.info("Reasoning prompt generated successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning prompt: {e}")
            self.struct_logger.log_error(e, {"task": task})
            raise

    def get_mcp_tools(self) -> List[str]:
        """
        Get list of available MCP tools if using MCP client.
        
        Returns:
            List[str]: List of available MCP tool names.
        """
        if not self.is_mcp_client:
            self.logger.warning("get_mcp_tools called on non-MCP client")
            return []
            
        self.logger.info("Retrieving available MCP tools")
        
        try:
            # Access the underlying MCP client
            if hasattr(self.llm_client, 'mcp_client'):
                import asyncio
                tools = asyncio.run(self.llm_client.mcp_client.list_tools())
                tool_names = [tool['name'] for tool in tools]
                
                self.logger.info(f"Found {len(tool_names)} MCP tools: {tool_names}")
                return tool_names
            else:
                self.logger.warning("MCP client does not have expected mcp_client attribute")
                return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving MCP tools: {e}")
            self.struct_logger.log_error(e, {"operation": "get_mcp_tools"})
            return []

    def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Direct call to MCP tool if using MCP client.
        
        Args:
            tool_name (str): Name of the MCP tool to call.
            arguments (Dict[str, Any]): Arguments to pass to the tool.
            
        Returns:
            Dict[str, Any]: Tool execution result.
        """
        if not self.is_mcp_client:
            raise ValueError("call_mcp_tool can only be used with MCP client")
            
        self.logger.info(f"Calling MCP tool: {tool_name}")
        self.logger.debug(f"Arguments: {arguments}")
        
        try:
            # Access the underlying MCP client
            if hasattr(self.llm_client, 'mcp_client'):
                import asyncio
                result = asyncio.run(self.llm_client.mcp_client.call_tool(tool_name, arguments))
                
                self.struct_logger.log_interaction(
                    "mcp_tool_call",
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result_type": type(result).__name__
                    }
                )
                
                self.logger.info(f"MCP tool {tool_name} completed successfully")
                return result
            else:
                raise RuntimeError("MCP client does not have expected mcp_client attribute")
                
        except Exception as e:
            self.logger.error(f"Error calling MCP tool {tool_name}: {e}")
            self.struct_logger.log_error(e, {
                "tool_name": tool_name,
                "arguments": arguments,
                "operation": "call_mcp_tool"
            })
            raise

    def create_causal_mcp_core(self) -> Dict[str, Any]:
        """
        Create a comprehensive causal core representation for MCP.
        Useful for initializing MCP servers with this core's configuration.
        
        Returns:
            Dict[str, Any]: Core configuration suitable for MCP tools.
        """
        self.logger.info("Creating causal MCP core representation")
        
        try:
            config = {
                "context": self.context,
                "variables": self.variables,
                "dag_edges": list(self.dag.graph.edges()),
                "capabilities": {
                    "simulate_do": True,
                    "simulate_counterfactual": True,
                    "generate_reasoning_prompt": True,
                    "extract_causal_edges": True,
                    "mcp_integration": self.is_mcp_client
                }
            }
            
            self.struct_logger.log_interaction(
                "mcp_core_creation",
                {
                    "variables_count": len(self.variables),
                    "dag_edges_count": self.dag.graph.number_of_edges(),
                    "context_length": len(self.context),
                    "is_mcp_client": self.is_mcp_client
                }
            )
            
            self.logger.info("Causal MCP core representation created")
            return config
            
        except Exception as e:
            self.logger.error(f"Error creating causal MCP core: {e}")
            self.struct_logger.log_error(e, {"operation": "create_causal_mcp_core"})
            raise

    async def enhanced_counterfactual_analysis(self, factual: str, intervention: str, 
                                             domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform enhanced counterfactual analysis using Tier 1 capabilities.
        Combines RAG-enhanced context, intelligent prompting, and multi-agent analysis.
        
        Args:
            factual (str): The factual scenario.
            intervention (str): The counterfactual intervention.
            domain (Optional[str]): Domain for specialized knowledge retrieval.
            
        Returns:
            Dict[str, Any]: Comprehensive analysis results.
        """
        self.logger.info("Starting enhanced counterfactual analysis")
        
        try:
            # Step 1: RAG-enhanced query processing
            query = f"Counterfactual analysis: If {factual}, what would happen with {intervention}?"
            rag_response = await self.rag_system.enhance_query(
                query=query,
                context=self.context,
                domain=domain,
                causal_concepts=["counterfactuals", "treatment_effects", "confounding"]
            )
            
            # Step 2: Intelligent prompt generation with RAG context
            enhanced_prompt_text = self.intelligent_prompting.generate_chain_of_thought_prompt(
                task_type="counterfactual",
                domain=domain or "general",
                context=rag_response.enhanced_context,
                factual=factual,
                intervention=intervention
            )
            
            # Create mock enhanced prompt result for compatibility
            class MockEnhancedPrompt:
                def __init__(self, prompt):
                    self.enhanced_prompt = prompt
                    self.reasoning_strategy = "chain_of_thought"
                    self.examples_used = []
                    self.quality_score = 0.8
            
            enhanced_prompt = MockEnhancedPrompt(enhanced_prompt_text)
            
            # Step 3: Multi-agent collaborative analysis
            collaborative_result = await self.multi_agent_analyzer.analyze_counterfactual(
                context=rag_response.enhanced_context,
                factual=factual,
                intervention=intervention
            )
            
            # Step 4: Synthesize comprehensive result
            result = {
                "query": query,
                "factual_scenario": factual,
                "intervention": intervention,
                "domain": domain,
                "rag_analysis": {
                    "enhanced_context": rag_response.enhanced_context,
                    "retrieved_documents": len(rag_response.retrieved_documents),
                    "confidence_score": rag_response.confidence_score,
                    "knowledge_gaps": rag_response.knowledge_gaps,
                    "recommendations": rag_response.recommendations
                },
                "intelligent_prompting": {
                    "enhanced_prompt": enhanced_prompt.enhanced_prompt,
                    "reasoning_strategy": enhanced_prompt.reasoning_strategy,
                    "examples_used": len(enhanced_prompt.examples_used),
                    "quality_score": enhanced_prompt.quality_score
                },
                "multi_agent_analysis": {
                    "domain_expert_analysis": collaborative_result.domain_expert_analysis,
                    "statistical_analysis": collaborative_result.statistical_analysis,
                    "skeptic_analysis": collaborative_result.skeptic_analysis,
                    "synthesized_conclusion": collaborative_result.synthesized_conclusion,
                    "confidence_score": collaborative_result.confidence_score,
                    "key_assumptions": collaborative_result.key_assumptions,
                    "recommendations": collaborative_result.recommendations
                },
                "overall_confidence": (
                    rag_response.confidence_score * 0.3 +
                    enhanced_prompt.quality_score * 0.3 +
                    collaborative_result.confidence_score * 0.4
                )
            }
            
            self.struct_logger.log_interaction(
                "enhanced_counterfactual_analysis",
                {
                    "factual_length": len(factual),
                    "intervention_length": len(intervention),
                    "domain": domain,
                    "documents_retrieved": len(rag_response.retrieved_documents),
                    "agents_involved": len(collaborative_result.domain_expert_analysis),
                    "overall_confidence": result["overall_confidence"]
                }
            )
            
            self.logger.info("Enhanced counterfactual analysis completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced counterfactual analysis: {e}")
            self.struct_logger.log_error(e, {
                "factual": factual[:100],
                "intervention": intervention[:100],
                "domain": domain
            })
            raise

    async def enhanced_treatment_effect_analysis(self, treatment: str, outcome: str,
                                               domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform enhanced treatment effect analysis using Tier 1 capabilities.
        
        Args:
            treatment (str): Treatment variable description.
            outcome (str): Outcome variable description.
            domain (Optional[str]): Domain for specialized analysis.
            
        Returns:
            Dict[str, Any]: Comprehensive treatment effect analysis.
        """
        self.logger.info("Starting enhanced treatment effect analysis")
        
        try:
            # RAG-enhanced query for treatment effect analysis
            query = f"Treatment effect analysis: Impact of {treatment} on {outcome}"
            rag_response = await self.rag_system.enhance_query(
                query=query,
                context=self.context,
                domain=domain,
                causal_concepts=["treatment_effects", "confounding", "randomized_trials"]
            )
            
            # Multi-agent analysis for treatment effects
            collaborative_result = await self.multi_agent_analyzer.analyze_treatment_effect(
                context=rag_response.enhanced_context,
                treatment=treatment,
                outcome=outcome
            )
            
            result = {
                "treatment": treatment,
                "outcome": outcome,
                "domain": domain,
                "rag_enhancement": {
                    "confidence_score": rag_response.confidence_score,
                    "knowledge_gaps": rag_response.knowledge_gaps,
                    "recommendations": rag_response.recommendations
                },
                "collaborative_analysis": collaborative_result,
                "overall_assessment": {
                    "confidence": collaborative_result.confidence_score,
                    "key_findings": collaborative_result.synthesized_conclusion,
                    "methodological_recommendations": collaborative_result.recommendations
                }
            }
            
            self.logger.info("Enhanced treatment effect analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced treatment effect analysis: {e}")
            raise

    @classmethod
    def from_mcp_config(cls, mcp_config: Dict[str, Any], llm_client: Optional[BaseLLMClient] = None) -> "CausalLLMCore":
        """
        Create CausalLLMCore instance from MCP configuration.
        
        Args:
            mcp_config (Dict[str, Any]): MCP configuration dictionary.
            llm_client (Optional[BaseLLMClient]): Optional LLM client to use.
            
        Returns:
            CausalLLMCore: Initialized core instance.
        """
        logger = get_logger("causalllm.core.from_mcp")
        logger.info("Creating CausalLLMCore from MCP configuration")
        
        try:
            # Extract required fields from MCP config
            context = mcp_config.get("context", "")
            variables = mcp_config.get("variables", {})
            dag_edges = [tuple(edge) for edge in mcp_config.get("dag_edges", [])]
            
            if not context:
                raise ValueError("MCP configuration missing required 'context' field")
            if not variables:
                raise ValueError("MCP configuration missing required 'variables' field")
            if not dag_edges:
                raise ValueError("MCP configuration missing required 'dag_edges' field")
            
            # Create core instance
            core = cls(
                context=context,
                variables=variables,
                dag_edges=dag_edges,
                llm_client=llm_client
            )
            
            logger.info("CausalLLMCore created successfully from MCP configuration")
            return core
            
        except Exception as e:
            logger.error(f"Error creating CausalLLMCore from MCP config: {e}")
            raise RuntimeError(f"Failed to create core from MCP config: {e}") from e

    # ============================================================================
    # Tier 3 Advanced LLM Methods
    # ============================================================================

    async def discover_causal_structure_from_data(self, 
                                                data,
                                                variable_descriptions: Dict[str, str],
                                                domain: str = "general",
                                                context: str = "",
                                                target_variable: Optional[str] = None):
        """
        Automatically discover causal structure from data using Tier 3 capabilities.
        
        Args:
            data: Dataset for causal discovery
            variable_descriptions: Descriptions of variables  
            domain: Domain context
            context: Additional context
            target_variable: Optional target variable to focus on
            
        Returns:
            Discovered causal structure with confidence scores
        """
        self.logger.info("Starting automated causal structure discovery")
        
        try:
            from causalllm.llm_causal_discovery import DiscoveryMethod
            
            structure = await self.causal_discovery_agent.discover_causal_structure(
                data=data,
                variable_descriptions=variable_descriptions,
                domain=domain,
                context=context,
                method=DiscoveryMethod.HYBRID_LLM_STATISTICAL,
                target_variable=target_variable
            )
            
            self.logger.info(f"Causal discovery completed. Found {len(structure.edges)} causal relationships.")
            return structure
            
        except Exception as e:
            self.logger.error(f"Error in causal structure discovery: {e}")
            raise

    async def ask_causal_question(self, 
                                question: str,
                                data=None,
                                variable_descriptions: Optional[Dict[str, str]] = None,
                                domain: str = "general") -> Dict[str, Any]:
        """
        Answer a causal question using the interactive Q&A system.
        
        Args:
            question: Natural language causal question
            data: Optional dataset
            variable_descriptions: Optional variable descriptions
            domain: Domain context
            
        Returns:
            Structured answer with evidence and confidence
        """
        self.logger.info(f"Processing causal question: {question[:100]}...")
        
        try:
            # Start Q&A session
            context = self.interactive_qa.start_conversation(
                domain=domain,
                data=data,
                variable_descriptions=variable_descriptions or {}
            )
            
            # Get answer
            answer = await self.interactive_qa.ask_causal_question(question, context)
            
            # Convert to dictionary for easier access
            result = {
                "question": question,
                "main_answer": answer.main_answer,
                "confidence_level": answer.confidence_level.value,
                "supporting_evidence": answer.supporting_evidence,
                "limitations": answer.limitations,
                "alternative_explanations": answer.alternative_explanations,
                "follow_up_questions": answer.follow_up_questions
            }
            
            self.logger.info("Causal question answered successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error answering causal question: {e}")
            raise

    async def get_causal_template_for_analysis(self, 
                                             data_description: str,
                                             research_question: str,
                                             domain: Optional[str] = None):
        """
        Get recommended causal analysis template for the given context.
        
        Args:
            data_description: Description of available data
            research_question: Research question or analysis goal
            domain: Optional domain specification
            
        Returns:
            Recommended causal analysis template
        """
        self.logger.info("Getting causal analysis template recommendation")
        
        try:
            templates = await self.template_engine.suggest_template(
                data_description=data_description,
                research_question=research_question,
                domain=domain
            )
            
            if templates:
                recommended = templates[0]  # Top recommendation
                
                result = {
                    "template_name": recommended.template_name,
                    "description": recommended.description,
                    "domain": recommended.domain.value,
                    "variables": [{"name": v.name, "role": v.role, "type": v.data_type} 
                                for v in recommended.variables],
                    "causal_edges": [{"source": e.source, "target": e.target, 
                                    "type": e.relationship_type} 
                                   for e in recommended.causal_edges],
                    "common_confounders": recommended.common_confounders,
                    "key_assumptions": recommended.key_assumptions,
                    "interpretation_guidelines": recommended.interpretation_guidelines,
                    "example_questions": recommended.example_questions
                }
                
                self.logger.info(f"Recommended template: {recommended.template_name}")
                return result
            else:
                self.logger.warning("No suitable template found")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting causal template: {e}")
            raise

    async def detect_confounders_automatically(self,
                                             data,
                                             treatment_variable: str,
                                             outcome_variable: str,
                                             variable_descriptions: Optional[Dict[str, str]] = None,
                                             domain: str = "general",
                                             context: str = ""):
        """
        Automatically detect confounders using multiple methods.
        
        Args:
            data: Dataset for confounder detection
            treatment_variable: Treatment/exposure variable
            outcome_variable: Outcome variable
            variable_descriptions: Optional variable descriptions
            domain: Domain context
            context: Additional context
            
        Returns:
            Comprehensive confounder detection summary
        """
        self.logger.info(f"Starting automatic confounder detection for {treatment_variable} -> {outcome_variable}")
        
        try:
            summary = await self.confounder_detector.detect_confounders(
                data=data,
                treatment_variable=treatment_variable,
                outcome_variable=outcome_variable,
                variable_descriptions=variable_descriptions,
                domain=domain,
                context=context
            )
            
            # Convert to dictionary format
            result = {
                "treatment_variable": summary.treatment_variable,
                "outcome_variable": summary.outcome_variable,
                "detected_confounders": [
                    {
                        "variable": c.variable_name,
                        "confounding_score": c.confounding_score,
                        "confidence": c.confidence,
                        "reasoning": c.reasoning,
                        "recommended_adjustment": [adj.value for adj in c.recommended_adjustment]
                    }
                    for c in summary.detected_confounders
                ],
                "overall_confounding_risk": summary.overall_confounding_risk,
                "recommended_adjustment_set": summary.recommended_adjustment_set,
                "alternative_adjustment_sets": summary.alternative_adjustment_sets,
                "validation_suggestions": summary.validation_suggestions
            }
            
            self.logger.info(f"Confounder detection completed. Found {len(summary.detected_confounders)} confounders.")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in automatic confounder detection: {e}")
            raise

    def create_causal_graph_visualization(self,
                                        discovered_edges: List[Tuple[str, str]],
                                        treatment_vars: List[str],
                                        outcome_vars: List[str],
                                        confounder_vars: Optional[List[str]] = None,
                                        title: str = "Causal Graph",
                                        save_path: Optional[str] = None,
                                        interactive: bool = False):
        """
        Create visual causal graph from discovered relationships.
        
        Args:
            discovered_edges: List of causal relationships (source, target)
            treatment_vars: Treatment variables
            outcome_vars: Outcome variables  
            confounder_vars: Optional confounder variables
            title: Graph title
            save_path: Optional path to save visualization
            interactive: Whether to create interactive visualization
            
        Returns:
            Matplotlib or Plotly figure
        """
        self.logger.info("Creating causal graph visualization")
        
        try:
            import networkx as nx
            from causalllm.visual_causal_graphs import NodeType, VisualizationType, GraphLayout
            
            # Create graph
            graph = nx.DiGraph()
            graph.add_edges_from(discovered_edges)
            
            # Assign node types
            node_types = {}
            for node in graph.nodes():
                if node in treatment_vars:
                    node_types[node] = NodeType.TREATMENT.value
                elif node in outcome_vars:
                    node_types[node] = NodeType.OUTCOME.value
                elif confounder_vars and node in confounder_vars:
                    node_types[node] = NodeType.CONFOUNDER.value
                else:
                    node_types[node] = NodeType.COVARIATE.value
            
            # Create visualization
            output_format = (VisualizationType.INTERACTIVE_PLOTLY if interactive 
                           else VisualizationType.STATIC_MATPLOTLIB)
            
            fig = self.graph_visualizer.create_causal_graph_visualization(
                graph=graph,
                node_types=node_types,
                layout=GraphLayout.HIERARCHICAL,
                theme="default",
                title=title,
                output_format=output_format,
                save_path=save_path
            )
            
            self.logger.info("Causal graph visualization created successfully")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating causal graph visualization: {e}")
            raise

    async def comprehensive_causal_analysis(self,
                                          data,
                                          treatment_variable: str,
                                          outcome_variable: str,
                                          variable_descriptions: Optional[Dict[str, str]] = None,
                                          domain: str = "general",
                                          context: str = "",
                                          create_visualization: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive causal analysis using all Tier 3 capabilities.
        
        Args:
            data: Dataset for analysis
            treatment_variable: Treatment variable
            outcome_variable: Outcome variable
            variable_descriptions: Variable descriptions
            domain: Domain context
            context: Additional context
            create_visualization: Whether to create graph visualization
            
        Returns:
            Comprehensive analysis results
        """
        self.logger.info("Starting comprehensive causal analysis")
        
        try:
            results = {}
            
            # 1. Discover causal structure
            structure = await self.discover_causal_structure_from_data(
                data, variable_descriptions or {}, domain, context, outcome_variable
            )
            results["causal_structure"] = {
                "edges": [(e.source, e.target) for e in structure.edges],
                "confidence": structure.overall_confidence,
                "assumptions": structure.assumptions,
                "limitations": structure.limitations
            }
            
            # 2. Detect confounders
            confounder_summary = await self.detect_confounders_automatically(
                data, treatment_variable, outcome_variable, variable_descriptions, domain, context
            )
            results["confounder_analysis"] = confounder_summary
            
            # 3. Get recommended template
            data_desc = f"Dataset with {len(data)} observations and variables: {list(data.columns)}"
            research_q = f"Analyze the causal effect of {treatment_variable} on {outcome_variable}"
            
            template = await self.get_causal_template_for_analysis(data_desc, research_q, domain)
            results["recommended_template"] = template
            
            # 4. Create visualization if requested
            if create_visualization:
                edges = [(e.source, e.target) for e in structure.edges]
                confounders = [c["variable"] for c in confounder_summary["detected_confounders"]]
                
                fig = self.create_causal_graph_visualization(
                    discovered_edges=edges,
                    treatment_vars=[treatment_variable],
                    outcome_vars=[outcome_variable],
                    confounder_vars=confounders,
                    title=f"Causal Analysis: {treatment_variable} → {outcome_variable}"
                )
                results["visualization"] = fig
            
            # 5. Generate summary insights
            insights = await self._generate_analysis_insights(results, domain)
            results["key_insights"] = insights
            
            self.logger.info("Comprehensive causal analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive causal analysis: {e}")
            raise

    async def _generate_analysis_insights(self, results: Dict[str, Any], domain: str) -> List[str]:
        """Generate key insights from comprehensive analysis."""
        
        insights = []
        
        # Structure insights
        if results.get("causal_structure"):
            structure = results["causal_structure"]
            insights.append(f"Discovered {len(structure['edges'])} causal relationships with {structure['confidence']:.2f} overall confidence")
        
        # Confounder insights
        if results.get("confounder_analysis"):
            conf_analysis = results["confounder_analysis"]
            n_confounders = len(conf_analysis["detected_confounders"])
            insights.append(f"Identified {n_confounders} potential confounders with {conf_analysis['overall_confounding_risk'].lower()} bias risk")
            
            if conf_analysis["recommended_adjustment_set"]:
                adj_vars = ", ".join(conf_analysis["recommended_adjustment_set"])
                insights.append(f"Recommended adjustment variables: {adj_vars}")
        
        # Template insights
        if results.get("recommended_template"):
            template = results["recommended_template"]
            insights.append(f"Analysis follows {template['template_name']} pattern for {domain} domain")
        
        return insights
