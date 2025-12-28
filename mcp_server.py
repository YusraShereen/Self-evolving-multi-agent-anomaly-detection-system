

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class MCPToolType(Enum):
    """Tool categories for MCP server"""
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    COMMUNICATION = "communication"
    VALIDATION = "validation"


@dataclass
class MCPToolSchema:
    """Schema definition for MCP tools"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    category: MCPToolType


@dataclass
class MCPToolResult:
    """Standardized tool execution result"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPTool:
    """Base class for MCP tools"""
    
    def __init__(self, schema: MCPToolSchema):
        self.schema = schema
        self.usage_count = 0
        self.last_execution_time = None

    async def execute(self, **kwargs) -> MCPToolResult:
        """Execute tool - to be implemented by subclasses"""
        raise NotImplementedError

    def record_execution(self, success: bool):
        """Track tool usage statistics"""
        self.usage_count += 1
        self.last_execution_time = time.time()


class ThresholdOptimizationTool(MCPTool):
    """3-sigma threshold optimization tool"""
    
    def __init__(self):
        schema = MCPToolSchema(
            name="optimize_threshold",
            description="Compute 3-sigma threshold for anomaly detection",
            input_schema={
                "type": "object",
                "properties": {
                    "scores": {"type": "array", "description": "Anomaly scores"},
                    "method": {"type": "string", "enum": ["3sigma", "percentile"]}
                },
                "required": ["scores"]
            },
            category=MCPToolType.OPTIMIZATION
        )
        super().__init__(schema)

    async def execute(self, scores: np.ndarray, method: str = "3sigma") -> MCPToolResult:
        """Calculate threshold using statistical methods"""
        try:
            mean = np.mean(scores)
            std = np.std(scores)

            if method == "3sigma":
                threshold = mean + 3 * std
                lower_bound = mean - 3 * std
            else:  # percentile
                threshold = np.percentile(scores, 95)
                lower_bound = np.percentile(scores, 5)

            self.record_execution(True)
            return MCPToolResult(
                success=True,
                data={
                    "threshold": float(threshold),
                    "lower_bound": float(lower_bound),
                    "mean": float(mean),
                    "std": float(std),
                    "method": method
                },
                metadata={"sample_size": len(scores)}
            )
        except Exception as e:
            self.record_execution(False)
            return MCPToolResult(success=False, data=None, error=str(e))


class EnsembleConsensusTool(MCPTool):
    """Multi-agent consensus computation"""
    
    def __init__(self):
        schema = MCPToolSchema(
            name="ensemble_consensus",
            description="Compute consensus from multiple agent predictions",
            input_schema={
                "type": "object",
                "properties": {
                    "predictions": {"type": "array"},
                    "method": {"type": "string", "enum": ["mean", "median", "weighted"]}
                },
                "required": ["predictions"]
            },
            category=MCPToolType.COMMUNICATION
        )
        super().__init__(schema)

    async def execute(self, predictions: List[np.ndarray], 
                     method: str = "mean", weights: Optional[List[float]] = None) -> MCPToolResult:
        """Aggregate predictions from multiple agents"""
        try:
            pred_array = np.array(predictions)

            if method == "mean":
                consensus = pred_array.mean(axis=0)
            elif method == "median":
                consensus = np.median(pred_array, axis=0)
            elif method == "weighted" and weights is not None:
                consensus = np.average(pred_array, axis=0, weights=weights)
            else:
                consensus = pred_array.mean(axis=0)

            # Agreement score: inverse of standard deviation
            agreement = 1.0 - pred_array.std(axis=0).mean()

            self.record_execution(True)
            return MCPToolResult(
                success=True,
                data={
                    "consensus": consensus,
                    "agreement_score": float(agreement),
                    "num_agents": len(predictions)
                }
            )
        except Exception as e:
            self.record_execution(False)
            return MCPToolResult(success=False, data=None, error=str(e))


class DriftDetectionTool(MCPTool):
    """Detect distribution drift using 3-sigma rule"""
    
    def __init__(self):
        schema = MCPToolSchema(
            name="detect_drift",
            description="Detect if test distribution drifts from training",
            input_schema={
                "type": "object",
                "properties": {
                    "train_mean": {"type": "number"},
                    "train_std": {"type": "number"},
                    "test_scores": {"type": "array"}
                },
                "required": ["train_mean", "train_std", "test_scores"]
            },
            category=MCPToolType.VALIDATION
        )
        super().__init__(schema)

    async def execute(self, train_mean: float, train_std: float, 
                     test_scores: np.ndarray) -> MCPToolResult:
        """Check if test scores fall within 3-sigma range"""
        try:
            lower_bound = train_mean - 3 * train_std
            upper_bound = train_mean + 3 * train_std
            
            test_mean = np.mean(test_scores)
            
            # Check if test mean is within bounds
            within_bounds = lower_bound <= test_mean <= upper_bound
            
            # Calculate deviation
            if test_mean > upper_bound:
                deviation = (test_mean - upper_bound) / train_std
            elif test_mean < lower_bound:
                deviation = (lower_bound - test_mean) / train_std
            else:
                deviation = 0.0

            # Calculate percentage of samples out of bounds
            out_of_bounds = ((test_scores < lower_bound) | (test_scores > upper_bound)).mean()

            self.record_execution(True)
            return MCPToolResult(
                success=True,
                data={
                    "within_bounds": bool(within_bounds),
                    "requires_retraining": not within_bounds or out_of_bounds > 0.1,
                    "test_mean": float(test_mean),
                    "deviation_sigma": float(deviation),
                    "out_of_bounds_percentage": float(out_of_bounds),
                    "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                }
            )
        except Exception as e:
            self.record_execution(False)
            return MCPToolResult(success=False, data=None, error=str(e))


class MCPServer:
    """MCP Server - Central tool registry and orchestrator"""
    
    def __init__(self, name: str = "BrainMRI_MCP_Server"):
        self.name = name
        self.tools: Dict[str, MCPTool] = {}
        self.execution_log: List[Dict] = []
        print(f"✓ MCP Server '{name}' initialized")

    def register_tool(self, tool: MCPTool):
        """Register a tool with the server"""
        self.tools[tool.schema.name] = tool
        print(f"  ✓ Registered: {tool.schema.name} ({tool.schema.category.value})")

    async def execute_tool(self, tool_name: str, **kwargs) -> MCPToolResult:
        """Execute a registered tool"""
        if tool_name not in self.tools:
            return MCPToolResult(
                success=False, 
                data=None, 
                error=f"Tool '{tool_name}' not found"
            )
        
        result = await self.tools[tool_name].execute(**kwargs)
        
        # Log execution
        self.execution_log.append({
            "tool": tool_name,
            "success": result.success,
            "timestamp": self.tools[tool_name].last_execution_time
        })
        
        return result

    def get_tool_stats(self) -> Dict:
        """Get usage statistics for all tools"""
        return {
            name: {
                "usage_count": tool.usage_count,
                "category": tool.schema.category.value
            }
            for name, tool in self.tools.items()
        }

    def get_execution_log(self) -> List[Dict]:
        """Get full execution log"""
        return self.execution_log

    def reset_statistics(self):
        """Reset all tool statistics"""
        for tool in self.tools.values():
            tool.usage_count = 0
        self.execution_log.clear()
        print("✓ Statistics reset")


def create_mcp_server() -> MCPServer:
    """Factory function to create and configure MCP server"""
    server = MCPServer()
    
    # Register all tools
    server.register_tool(ThresholdOptimizationTool())
    server.register_tool(EnsembleConsensusTool())
    server.register_tool(DriftDetectionTool())
    
    return server


if __name__ == "__main__":
    # Test MCP server
    import asyncio
    
    async def test_mcp_server():
        server = create_mcp_server()
        
        # Test threshold optimization
        test_scores = np.random.randn(100) * 0.1 + 0.5
        result = await server.execute_tool(
            "optimize_threshold",
            scores=test_scores,
            method="3sigma"
        )
        print(f"\nThreshold result: {result.data}")
        
        # Test consensus
        predictions = [
            np.random.randn(50) * 0.1 + 0.5,
            np.random.randn(50) * 0.1 + 0.52,
            np.random.randn(50) * 0.1 + 0.48
        ]
        result = await server.execute_tool(
            "ensemble_consensus",
            predictions=predictions,
            method="mean"
        )
        print(f"\nConsensus agreement: {result.data['agreement_score']:.4f}")
        
        # Print statistics
        print("\n" + "="*50)
        print("MCP Server Statistics:")
        print("="*50)
        stats = server.get_tool_stats()
        for tool, data in stats.items():
            print(f"{tool}: {data['usage_count']} calls")
    
    asyncio.run(test_mcp_server())