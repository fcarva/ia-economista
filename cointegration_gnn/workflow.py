"""
LangGraph Workflow Orchestration
================================

This module defines the agentic workflow using LangGraph, orchestrating
the Analyst, Risk Manager, and Executor agents.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    TRADING WORKFLOW                         │
    │                                                             │
    │   [Market Data]                                            │
    │        │                                                    │
    │        ▼                                                    │
    │   ┌─────────┐    ┌─────────────┐    ┌──────────┐          │
    │   │ ANALYST │───▶│ RISK_MANAGER│───▶│ EXECUTOR │          │
    │   └─────────┘    └─────────────┘    └──────────┘          │
    │        │              │                   │                 │
    │        ▼              ▼                   ▼                 │
    │   [Signals]    [Filtered]           [Orders]               │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

LangGraph Implementation:
    Each agent is a "node" in the graph. State flows between nodes
    carrying market data, signals, and decisions. The graph handles:
    - State management
    - Error recovery
    - Checkpointing (for debugging)
"""

from typing import TypedDict, Annotated, Literal
import operator
from dataclasses import dataclass, field

import pandas as pd
from langgraph.graph import StateGraph, END

from .agents.analyst import AnalystAgent, SignalOutput
from .agents.risk_manager import RiskManagerAgent, RiskAdjustedSignals
from .agents.executor import ExecutorAgent, ExecutionReport
from .models.gnn_model import CointegrationGNN


class WorkflowState(TypedDict):
    """State passed between workflow nodes.
    
    Each node reads from and writes to specific fields.
    """
    # Input data (set at start)
    prices: pd.DataFrame
    features: pd.DataFrame
    current_date: pd.Timestamp
    current_positions: dict[str, float]
    current_equity: float
    current_prices: dict[str, float]
    portfolio_returns: pd.Series
    
    # Analyst output
    analyst_output: SignalOutput | None
    
    # Risk Manager output
    risk_output: RiskAdjustedSignals | None
    
    # Executor output
    execution_report: ExecutionReport | None
    
    # Workflow metadata
    error: str | None
    completed: bool


def create_analyst_node(analyst: AnalystAgent):
    """Create Analyst node function.
    
    This node:
    1. Reads market data from state
    2. Generates signals via GNN
    3. Writes signals to state
    """
    def analyst_node(state: WorkflowState) -> WorkflowState:
        try:
            output = analyst.generate_signals(
                prices=state["prices"],
                features=state["features"],
                current_date=state["current_date"],
            )
            return {"analyst_output": output, "error": None}
        except Exception as e:
            return {"analyst_output": None, "error": f"Analyst error: {str(e)}"}
    
    return analyst_node


def create_risk_manager_node(risk_mgr: RiskManagerAgent):
    """Create Risk Manager node function.
    
    This node:
    1. Reads analyst signals from state
    2. Applies risk filters
    3. Writes adjusted signals to state
    """
    def risk_manager_node(state: WorkflowState) -> WorkflowState:
        if state.get("error"):
            return state
        
        analyst_output = state.get("analyst_output")
        if analyst_output is None:
            return {"error": "No analyst output available"}
        
        try:
            output = risk_mgr.filter_signals(
                analyst_output=analyst_output,
                portfolio_returns=state["portfolio_returns"],
                current_equity=state["current_equity"],
            )
            return {"risk_output": output, "error": None}
        except Exception as e:
            return {"risk_output": None, "error": f"Risk Manager error: {str(e)}"}
    
    return risk_manager_node


def create_executor_node(executor: ExecutorAgent):
    """Create Executor node function.
    
    This node:
    1. Reads approved signals from state
    2. Generates executable orders
    3. Writes execution report to state
    """
    def executor_node(state: WorkflowState) -> WorkflowState:
        if state.get("error"):
            return state
        
        risk_output = state.get("risk_output")
        if risk_output is None:
            return {"error": "No risk output available"}
        
        try:
            report = executor.generate_orders(
                signals=risk_output,
                current_prices=state["current_prices"],
                current_positions=state["current_positions"],
            )
            return {"execution_report": report, "completed": True, "error": None}
        except Exception as e:
            return {"execution_report": None, "error": f"Executor error: {str(e)}"}
    
    return executor_node


def should_continue(state: WorkflowState) -> Literal["risk_manager", "end"]:
    """Conditional edge: continue or abort based on errors."""
    if state.get("error"):
        return "end"
    return "risk_manager"


def should_execute(state: WorkflowState) -> Literal["executor", "end"]:
    """Conditional edge: execute or abort based on risk status."""
    if state.get("error"):
        return "end"
    
    risk_output = state.get("risk_output")
    if risk_output and risk_output.risk_status == "critical":
        # In critical mode, we still execute (with reduced positions)
        pass
    
    return "executor"


class TradingWorkflow:
    """LangGraph-based trading workflow.
    
    Orchestrates the three agents in sequence:
        Analyst → Risk Manager → Executor
    
    Example:
        >>> workflow = TradingWorkflow(
        ...     tickers=['XLK', 'XLF'],
        ...     model=trained_gnn,
        ... )
        >>> result = workflow.run(
        ...     prices=price_data,
        ...     features=feature_data,
        ...     current_date=pd.Timestamp('2023-06-30'),
        ...     current_positions={'XLK': 100, 'XLF': 200},
        ...     portfolio_returns=recent_returns,
        ...     current_equity=1_000_000,
        ... )
        >>> print(result['execution_report'])
    """
    
    def __init__(
        self,
        tickers: list[str],
        model: CointegrationGNN,
        initial_capital: float = 1_000_000.0,
    ) -> None:
        """Initialize workflow with agents.
        
        Args:
            tickers: Asset universe.
            model: Trained GNN model.
            initial_capital: Portfolio value for position sizing.
        """
        self.tickers = tickers
        
        # Initialize agents
        self.analyst = AnalystAgent(tickers=tickers, model=model)
        self.risk_manager = RiskManagerAgent(tickers=tickers)
        self.executor = ExecutorAgent(tickers=tickers, portfolio_value=initial_capital)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph."""
        # Create the graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("analyst", create_analyst_node(self.analyst))
        workflow.add_node("risk_manager", create_risk_manager_node(self.risk_manager))
        workflow.add_node("executor", create_executor_node(self.executor))
        
        # Add edges
        workflow.set_entry_point("analyst")
        
        workflow.add_conditional_edges(
            "analyst",
            should_continue,
            {
                "risk_manager": "risk_manager",
                "end": END,
            }
        )
        
        workflow.add_conditional_edges(
            "risk_manager",
            should_execute,
            {
                "executor": "executor",
                "end": END,
            }
        )
        
        workflow.add_edge("executor", END)
        
        # Compile
        return workflow.compile()
    
    def run(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        current_date: pd.Timestamp,
        current_positions: dict[str, float],
        portfolio_returns: pd.Series,
        current_equity: float,
    ) -> WorkflowState:
        """Execute the workflow.
        
        Args:
            prices: Historical price data.
            features: Node features for GNN.
            current_date: Date to generate signals for.
            current_positions: Current shares held.
            portfolio_returns: Recent portfolio returns (for VaR).
            current_equity: Current portfolio value.
            
        Returns:
            Final workflow state with all outputs.
        """
        # Get current prices
        if current_date in prices.index:
            current_prices = prices.loc[current_date].to_dict()
        else:
            # Use most recent available
            current_prices = prices.iloc[-1].to_dict()
        
        # Initial state
        initial_state: WorkflowState = {
            "prices": prices,
            "features": features,
            "current_date": current_date,
            "current_positions": current_positions,
            "current_equity": current_equity,
            "current_prices": current_prices,
            "portfolio_returns": portfolio_returns,
            "analyst_output": None,
            "risk_output": None,
            "execution_report": None,
            "error": None,
            "completed": False,
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return result
    
    def run_backtest(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        initial_capital: float = 1_000_000.0,
    ) -> list[WorkflowState]:
        """Run workflow for each date in backtest period.
        
        Args:
            prices: Full price data.
            features: Full feature data.
            start_date: Backtest start.
            end_date: Backtest end.
            initial_capital: Starting capital.
            
        Returns:
            List of workflow states for each trading day.
        """
        results: list[WorkflowState] = []
        
        positions: dict[str, float] = {t: 0.0 for t in self.tickers}
        equity = initial_capital
        returns: list[float] = []
        
        trading_dates = prices[
            (prices.index >= start_date) & (prices.index <= end_date)
        ].index
        
        for i, date in enumerate(trading_dates):
            if i < 60:  # Need history for returns calculation
                continue
            
            # Prepare returns series
            portfolio_returns = pd.Series(returns[-60:]) if len(returns) >= 60 else pd.Series([0.0])
            
            # Run workflow
            state = self.run(
                prices=prices[prices.index <= date],
                features=features[features.index <= date],
                current_date=date,
                current_positions=positions.copy(),
                portfolio_returns=portfolio_returns,
                current_equity=equity,
            )
            
            results.append(state)
            
            # Update positions based on execution (simplified)
            if state.get("execution_report"):
                for order in state["execution_report"].orders:
                    if order.direction == "buy":
                        positions[order.ticker] = positions.get(order.ticker, 0) + order.target_shares
                    else:
                        positions[order.ticker] = positions.get(order.ticker, 0) - order.target_shares
            
            # Calculate day's return (simplified)
            if i > 0:
                day_return = 0.0  # Placeholder - real implementation tracks PnL
                returns.append(day_return)
        
        return results
