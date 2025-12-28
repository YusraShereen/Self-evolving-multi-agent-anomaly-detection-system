

import torch
from PIL import Image
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path

from mcp_server import MCPServer


# XML tags for structured output
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


class ReasoningExtractionTool:
    """MCP Tool to extract reasoning from VLM response"""
    
    def __init__(self):
        from mcp_server import MCPToolSchema, MCPTool, MCPToolType
        
        schema = MCPToolSchema(
            name="extract_reasoning",
            description="Parse REASONING and SOLUTION from VLM response",
            input_schema={
                "type": "object",
                "properties": {
                    "response": {"type": "string"}
                },
                "required": ["response"]
            },
            category=MCPToolType.ANALYSIS
        )
        self.schema = schema
        self.usage_count = 0
        self.last_execution_time = None
    
    async def execute(self, response: str):
        """Extract structured output from VLM response"""
        from mcp_server import MCPToolResult
        
        try:
            reasoning_start = response.find(REASONING_START)
            reasoning_end = response.find(REASONING_END)
            solution_start = response.find(SOLUTION_START)
            solution_end = response.find(SOLUTION_END)
            
            reasoning = ""
            solution = 0.5
            
            # Extract reasoning
            if reasoning_start != -1 and reasoning_end != -1:
                reasoning = response[reasoning_start + len(REASONING_START):reasoning_end].strip()
            
            # Extract solution
            if solution_start != -1 and solution_end != -1:
                solution_text = response[solution_start + len(SOLUTION_START):solution_end].strip()
                try:
                    solution = float(solution_text)
                    solution = max(0.0, min(1.0, solution))  # Clip to [0, 1]
                except:
                    # Fallback: keyword detection
                    if any(word in solution_text.lower() for word in ["abnormal", "patholog", "anomal"]):
                        solution = 1.0
                    elif any(word in solution_text.lower() for word in ["normal", "healthy"]):
                        solution = 0.0
            
            self.usage_count += 1
            import time
            self.last_execution_time = time.time()
            
            return MCPToolResult(
                success=True,
                data={
                    "reasoning": reasoning,
                    "solution": solution,
                    "confidence": 0.9 if reasoning else 0.5,
                    "has_structured_output": bool(reasoning and solution_text)
                }
            )
        except Exception as e:
            return MCPToolResult(success=False, data=None, error=str(e))


class VLMExplainerAgent:
    """Qwen3-VL agent for detailed analysis of flagged samples"""
    
    def __init__(self, agent_id: int, mcp_server: MCPServer, 
                 model, tokenizer, device: str = 'cuda'):
        """
        Args:
            agent_id: Unique agent identifier
            mcp_server: MCP server instance
            model: Qwen3-VL model
            tokenizer: Qwen3 tokenizer
            device: 'cuda' or 'cpu'
        """
        self.id = agent_id
        self.mcp_server = mcp_server
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.tokenizer = tokenizer
        
        # Training history
        self.reward_history = []
        self.analysis_count = 0
        
        print(f"✓ VLM Explainer Agent {agent_id} initialized on {self.device}")
    
    def create_analysis_prompt(self, reconstruction_error: float, 
                              threshold: float) -> str:
        """
        Create prompt for VLM analysis
        
        Args:
            reconstruction_error: Anomaly score from Phase 1
            threshold: Decision threshold
            
        Returns:
            prompt: Text prompt for VLM
        """
        prompt = (
            f"You are an expert radiologist analyzing brain MRI scans. "
            f"This scan has been flagged by an AI system with an anomaly score of {reconstruction_error:.4f} "
            f"(threshold: {threshold:.4f}).\n\n"
            f"Carefully analyze this brain MRI and provide:\n\n"
            f"1. Your detailed clinical reasoning between {REASONING_START} and {REASONING_END}\n"
            f"   Consider:\n"
            f"   - Tissue contrast and signal intensity\n"
            f"   - Brain symmetry and anatomical structure\n"
            f"   - Presence of mass effect or midline shift\n"
            f"   - White matter and gray matter appearance\n"
            f"   - Ventricle size and shape\n"
            f"   - Any visible lesions or abnormalities\n\n"
            f"2. Your final abnormality score between {SOLUTION_START} and {SOLUTION_END}\n"
            f"   - 0.0 means completely normal\n"
            f"   - 1.0 means definitely abnormal\n"
            f"   - Provide a single number between 0.0 and 1.0\n\n"
            f"Be thorough and precise in your analysis."
        )
        return prompt
    
    @torch.inference_mode()
    async def analyze_sample(self, image_path: str, reconstruction_error: float,
                            threshold: float) -> str:
        """
        Analyze a single sample with VLM
        
        Args:
            image_path: Path to MRI image
            reconstruction_error: Anomaly score from Phase 1
            threshold: Decision threshold
            
        Returns:
            response: VLM generated response
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Create prompt
            text_prompt = self.create_analysis_prompt(reconstruction_error, threshold)
            
            # Prepare input in chat format
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
            
            # Apply chat template
            prompt_text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                image,
                prompt_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # Decode
            response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            
            # Extract only the generated part (remove prompt)
            if prompt_text in response:
                response = response[len(prompt_text):].strip()
            
            self.analysis_count += 1
            return response
            
        except Exception as e:
            print(f"  ⚠ VLM analysis error: {e}")
            return f"{REASONING_START}Error during analysis{REASONING_END}{SOLUTION_START}0.5{SOLUTION_END}"
    
    def compute_reward(self, prediction: float, ground_truth: float) -> float:
        """
        Compute reward for GRPO training
        
        Args:
            prediction: Predicted abnormality score [0, 1]
            ground_truth: True label (0 or 1)
            
        Returns:
            reward: Reward value
        """
        # Simple accuracy-based reward
        error = abs(prediction - ground_truth)
        reward = 1.0 - error
        
        self.reward_history.append(reward)
        return reward
    
    def get_agent_info(self) -> Dict:
        """Get agent statistics"""
        return {
            'agent_id': self.id,
            'analysis_count': self.analysis_count,
            'mean_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'total_rewards': len(self.reward_history)
        }


class Phase2ExplainerSystem:
    """Phase 2: VLM-based explanation system"""
    
    def __init__(self, mcp_server: MCPServer, model=None, tokenizer=None):
        """
        Args:
            mcp_server: MCP server instance
            model: Qwen3-VL model (optional, loaded later)
            tokenizer: Qwen3 tokenizer (optional, loaded later)
        """
        self.mcp_server = mcp_server
        self.agents = []
        self.model = model
        self.tokenizer = tokenizer
        
        # Register reasoning extraction tool
        reasoning_tool = ReasoningExtractionTool()
        self.mcp_server.register_tool(reasoning_tool)
        
        print(f"\n{'='*70}")
        print(f"Phase 2 Explainer System")
        print(f"  • VLM: Qwen3-VL")
        print(f"  • Training: GRPO")
        print(f"  • Task: Explanation generation")
        print(f"{'='*70}\n")
    
    def load_vlm_model(self, model, tokenizer):
        """Load VLM model and create agents"""
        self.model = model
        self.tokenizer = tokenizer
        
        # Create VLM agent
        agent = VLMExplainerAgent(0, self.mcp_server, model, tokenizer)
        self.agents.append(agent)
        
        print(f"✓ VLM model loaded with {len(self.agents)} agent(s)")
    
    async def analyze_flagged_samples(self, flagged_indices: np.ndarray,
                                     test_dataset, phase1_scores: np.ndarray,
                                     threshold: float) -> Dict:
        """
        Analyze samples flagged by Phase 1
        
        Args:
            flagged_indices: Indices of flagged samples
            test_dataset: Test dataset
            phase1_scores: Anomaly scores from Phase 1
            threshold: Decision threshold
            
        Returns:
            results: Dictionary of analysis results
        """
        if len(self.agents) == 0:
            raise ValueError("No VLM agents loaded. Call load_vlm_model() first.")
        
        print(f"\n🔬 Phase 2: Analyzing {len(flagged_indices)} flagged samples...")
        
        agent = self.agents[0]
        results = {
            'explanations': [],
            'refined_scores': phase1_scores.copy(),
            'rewards': []
        }
        
        for idx in flagged_indices:
            image_path = test_dataset.samples[idx]
            reconstruction_error = phase1_scores[idx]
            ground_truth = test_dataset.labels[idx]
            
            # Get VLM analysis
            response = await agent.analyze_sample(
                image_path, reconstruction_error, threshold
            )
            
            # Extract reasoning and solution
            reasoning_result = await self.mcp_server.execute_tool(
                "extract_reasoning",
                response=response
            )
            
            if reasoning_result.success:
                vlm_score = reasoning_result.data['solution']
                reasoning = reasoning_result.data['reasoning']
                
                # Combine Phase 1 and Phase 2 scores
                # 60% reconstruction error + 40% VLM score
                refined_score = 0.6 * reconstruction_error + 0.4 * vlm_score
                results['refined_scores'][idx] = refined_score
                
                # Compute reward
                reward = agent.compute_reward(vlm_score, float(ground_truth))
                results['rewards'].append(reward)
                
                # Store explanation
                results['explanations'].append({
                    'index': int(idx),
                    'image_path': image_path,
                    'phase1_score': float(reconstruction_error),
                    'vlm_score': float(vlm_score),
                    'refined_score': float(refined_score),
                    'reasoning': reasoning,
                    'ground_truth': int(ground_truth),
                    'reward': float(reward)
                })
        
        mean_reward = np.mean(results['rewards']) if results['rewards'] else 0.0
        print(f"  ✓ Analysis complete")
        print(f"  Mean reward: {mean_reward:.4f}")
        
        return results
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            'num_agents': len(self.agents),
            'agents_info': [agent.get_agent_info() for agent in self.agents],
            'model_loaded': self.model is not None
        }

