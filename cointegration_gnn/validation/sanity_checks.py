"""
Sanity Checks and Validation Module
====================================

Este módulo implementa os testes de validação críticos para garantir
robustez acadêmica e comercial da estratégia.

PEER REVIEW CHECKLIST (MIT/DeepMind Style):
===========================================

1. SPREAD STATIONARITY TEST (Crítico)
   - O spread negociado deve ser I(0) (estacionário)
   - Se não for, a estratégia é economicamente inválida
   - Usamos ADF + KPSS com confirmação dupla

2. RANDOMIZATION TEST (Detector de Look-ahead Bias)
   - Embaralhamos os retornos, quebrando estrutura causal
   - Re-treinamos o modelo nos dados aleatórios
   - Esperado: Sharpe ≈ 0. Se positivo → BUG GRAVE

3. COST SENSITIVITY ANALYSIS (Viabilidade Comercial)
   - Sweep de custos de 0bps a 50bps
   - Se Sharpe < 0.5 com 10bps → estratégia inviável
   - Plota curva de degradação

4. REGIME ROBUSTNESS TEST
   - Treina em Bull Market, testa em Bear Market
   - Já implementado via splits regime-aware

References:
    - De Prado (2018), "Advances in Financial Machine Learning", Ch. 11-12
    - Bailey et al. (2014), "Probability of Backtest Overfitting"
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt


@dataclass
class StationarityResult:
    """Resultado do teste de estacionariedade em um spread."""
    
    spread_name: str
    adf_statistic: float
    adf_pvalue: float
    kpss_statistic: float
    kpss_pvalue: float
    is_stationary: bool
    half_life: float  # Em dias
    verdict: str


@dataclass  
class RandomizationResult:
    """Resultado do teste de randomização."""
    
    original_sharpe: float
    shuffled_sharpes: list[float]
    mean_shuffled_sharpe: float
    std_shuffled_sharpe: float
    z_score: float  # (original - mean_shuffled) / std_shuffled
    p_value: float  # Probabilidade de obter Sharpe por sorte
    has_look_ahead_bias: bool
    verdict: str


@dataclass
class CostSensitivityResult:
    """Resultado da análise de sensibilidade a custos."""
    
    cost_levels_bps: list[float]
    sharpe_ratios: list[float]
    breakeven_cost_bps: float  # Custo onde Sharpe ≈ 0
    sharpe_at_10bps: float
    is_commercially_viable: bool
    verdict: str


class SpreadStationarityValidator:
    """Valida que os spreads negociados são estacionários.
    
    Teoria Econômica:
        Se o spread não for I(0), então:
        - Não há mean-reversion (fundamento do pairs trading)
        - O modelo está apostando em tendências, não em equilíbro
        - A estratégia é economicamente inválida
    
    Implementação:
        Usamos confirmação dupla ADF + KPSS:
        - ADF: H0 = tem unit root (queremos REJEITAR para I(0))
        - KPSS: H0 = é estacionário (queremos NÃO REJEITAR para I(0))
    
    Example:
        >>> validator = SpreadStationarityValidator()
        >>> spread = compute_portfolio_spread(prices, weights)
        >>> result = validator.test(spread, "GNN_Portfolio")
        >>> if not result.is_stationary:
        ...     raise ValueError(f"ALERTA: Spread não é I(0)! {result.verdict}")
    """
    
    def __init__(
        self,
        significance: float = 0.05,
        max_half_life_days: int = 60,  # Máximo aceitável para trading
    ) -> None:
        """Inicializa o validador.
        
        Args:
            significance: Nível de significância (5% padrão)
            max_half_life_days: Meia-vida máxima aceitável do spread
        """
        self.significance = significance
        self.max_half_life = max_half_life_days
    
    def test(
        self,
        spread: pd.Series,
        spread_name: str = "Spread",
    ) -> StationarityResult:
        """Testa estacionariedade do spread.
        
        Args:
            spread: Série temporal do spread/resíduo
            spread_name: Nome para identificação
            
        Returns:
            StationarityResult com diagnóstico completo
        """
        spread = spread.dropna()
        
        if len(spread) < 60:
            return StationarityResult(
                spread_name=spread_name,
                adf_statistic=0.0,
                adf_pvalue=1.0,
                kpss_statistic=0.0,
                kpss_pvalue=1.0,
                is_stationary=False,
                half_life=float("inf"),
                verdict="ERRO: Dados insuficientes (< 60 observações)",
            )
        
        # 1. Teste ADF (H0: unit root)
        adf_result = adfuller(spread, autolag="AIC")
        adf_stat = adf_result[0]
        adf_pval = adf_result[1]
        
        # 2. Teste KPSS (H0: estacionário)
        kpss_result = kpss(spread, regression="c", nlags="auto")
        kpss_stat = kpss_result[0]
        kpss_pval = kpss_result[1]
        
        # 3. Calcular meia-vida (half-life) via OU process
        half_life = self._compute_half_life(spread)
        
        # 4. Determinar estacionariedade
        adf_rejects_unit_root = adf_pval < self.significance
        kpss_fails_to_reject_stationary = kpss_pval > self.significance
        half_life_acceptable = half_life < self.max_half_life
        
        is_stationary = (
            adf_rejects_unit_root and 
            kpss_fails_to_reject_stationary and
            half_life_acceptable
        )
        
        # 5. Gerar veredito
        if is_stationary:
            verdict = f"✅ ESTACIONÁRIO: ADF p={adf_pval:.4f}, KPSS p={kpss_pval:.4f}, τ½={half_life:.1f}d"
        else:
            issues = []
            if not adf_rejects_unit_root:
                issues.append(f"ADF não rejeita unit root (p={adf_pval:.4f})")
            if not kpss_fails_to_reject_stationary:
                issues.append(f"KPSS rejeita estacionariedade (p={kpss_pval:.4f})")
            if not half_life_acceptable:
                issues.append(f"Meia-vida muito longa ({half_life:.0f} dias)")
            verdict = f"❌ NÃO ESTACIONÁRIO: {'; '.join(issues)}"
        
        return StationarityResult(
            spread_name=spread_name,
            adf_statistic=adf_stat,
            adf_pvalue=adf_pval,
            kpss_statistic=kpss_stat,
            kpss_pvalue=kpss_pval,
            is_stationary=is_stationary,
            half_life=half_life,
            verdict=verdict,
        )
    
    def _compute_half_life(self, spread: pd.Series) -> float:
        """Calcula meia-vida via processo Ornstein-Uhlenbeck.
        
        Modelo: dS = θ(μ - S)dt + σdW
        
        Half-life = ln(2) / θ
        
        Onde θ é estimado via regressão:
            ΔS_t = α + β S_{t-1} + ε_t
            θ = -β
        """
        spread_lag = spread.shift(1)
        spread_diff = spread.diff()
        
        # Remover NaNs
        valid_idx = spread_lag.notna() & spread_diff.notna()
        y = spread_diff[valid_idx].values
        x = spread_lag[valid_idx].values
        
        if len(y) < 10:
            return float("inf")
        
        # Regressão OLS: ΔS = α + β S_{t-1}
        # Adicionar constante
        X = np.column_stack([np.ones(len(x)), x])
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            theta = -beta[1]  # Mean-reversion speed
            
            if theta <= 0:
                return float("inf")  # Não mean-reverting
            
            half_life = np.log(2) / theta
            return max(min(half_life, 1000), 0)  # Clamp to reasonable range
            
        except Exception:
            return float("inf")


class RandomizationTest:
    """Teste de randomização para detectar look-ahead bias.
    
    Metodologia:
        1. Calcula Sharpe da estratégia original
        2. Embaralha (shuffle) os retornos N vezes
        3. Re-executa backtest em cada shuffle
        4. Compara Sharpe original vs distribuição shuffled
        
    Interpretação:
        - Z-score > 3: Estratégia tem sinal real
        - Z-score < 2: Pode ser ruído
        - Sharpe_shuffled ≈ Sharpe_original: LOOK-AHEAD BIAS!
    
    Example:
        >>> test = RandomizationTest(n_shuffles=100)
        >>> result = test.run(returns, signal_generator, backtest_func)
        >>> if result.has_look_ahead_bias:
        ...     raise ValueError("CRÍTICO: Look-ahead bias detectado!")
    """
    
    def __init__(
        self,
        n_shuffles: int = 100,
        seed: int = 42,
    ) -> None:
        """Inicializa o teste.
        
        Args:
            n_shuffles: Número de permutações aleatórias
            seed: Seed para reprodutibilidade
        """
        self.n_shuffles = n_shuffles
        self.rng = np.random.default_rng(seed)
    
    def run(
        self,
        prices: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame, pd.Timestamp], dict[str, float]],
        backtest_func: Callable[[pd.DataFrame, Callable], float],
    ) -> RandomizationResult:
        """Executa o teste de randomização.
        
        Args:
            prices: DataFrame de preços originais
            signal_generator: Função que gera sinais
            backtest_func: Função de backtest que retorna Sharpe
            
        Returns:
            RandomizationResult com diagnóstico
        """
        # 1. Sharpe original
        original_sharpe = backtest_func(prices, signal_generator)
        
        # 2. Sharpes com dados shuffled
        shuffled_sharpes = []
        
        for i in range(self.n_shuffles):
            # Shuffle os retornos (mantém estrutura de cross-section)
            shuffled_prices = self._shuffle_returns(prices)
            
            # Re-backtest
            try:
                shuffled_sharpe = backtest_func(shuffled_prices, signal_generator)
                shuffled_sharpes.append(shuffled_sharpe)
            except Exception:
                shuffled_sharpes.append(0.0)
        
        # 3. Estatísticas
        mean_shuffled = np.mean(shuffled_sharpes)
        std_shuffled = np.std(shuffled_sharpes) + 1e-8
        
        z_score = (original_sharpe - mean_shuffled) / std_shuffled
        
        # P-value: probabilidade de obter Sharpe >= original por sorte
        p_value = np.mean([s >= original_sharpe for s in shuffled_sharpes])
        
        # 4. Detectar bias
        # Se mean_shuffled > 0.3, há vazamento de informação
        has_bias = mean_shuffled > 0.3 or p_value > 0.1
        
        # 5. Veredito
        if has_bias:
            verdict = f"🚨 LOOK-AHEAD BIAS PROVÁVEL: Sharpe shuffled={mean_shuffled:.2f}, p-value={p_value:.3f}"
        elif z_score < 2:
            verdict = f"⚠️ SINAL FRACO: Z-score={z_score:.2f} (< 2), pode ser ruído"
        else:
            verdict = f"✅ SINAL VÁLIDO: Z-score={z_score:.2f}, p-value={p_value:.4f}"
        
        return RandomizationResult(
            original_sharpe=original_sharpe,
            shuffled_sharpes=shuffled_sharpes,
            mean_shuffled_sharpe=mean_shuffled,
            std_shuffled_sharpe=std_shuffled,
            z_score=z_score,
            p_value=p_value,
            has_look_ahead_bias=has_bias,
            verdict=verdict,
        )
    
    def _shuffle_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Embaralha retornos mantendo estrutura cross-section.
        
        Isso quebra a estrutura temporal (causalidade) mas mantém
        as correlações contemporâneas entre ativos.
        """
        returns = prices.pct_change().dropna()
        
        # Shuffle linhas (mantém correlações cross-section)
        shuffled_idx = self.rng.permutation(len(returns))
        shuffled_returns = returns.iloc[shuffled_idx].reset_index(drop=True)
        
        # Reconstruir preços
        shuffled_prices = (1 + shuffled_returns).cumprod() * prices.iloc[0]
        shuffled_prices.index = prices.index[:len(shuffled_prices)]
        
        return shuffled_prices


class CostSensitivityAnalyzer:
    """Análise de sensibilidade a custos de transação.
    
    Teoria:
        Estratégias de alta frequência/turnover são extremamente
        sensíveis a custos. Uma estratégia válida deve manter
        Sharpe > 0.5 com custos realistas (5-10 bps para ETFs).
    
    Metodologia:
        1. Sweep de custos: 0, 2, 5, 10, 15, 20, 30, 50 bps
        2. Para cada nível, re-executa backtest
        3. Gera curva de degradação
        4. Calcula breakeven cost (onde Sharpe = 0)
    
    Example:
        >>> analyzer = CostSensitivityAnalyzer()
        >>> result = analyzer.analyze(prices, strategy, cost_range=[0, 5, 10, 20])
        >>> analyzer.plot(result)
        >>> if not result.is_commercially_viable:
        ...     print("Estratégia morre com custos realistas")
    """
    
    def __init__(
        self,
        min_viable_sharpe: float = 0.5,
        realistic_cost_bps: float = 10.0,
    ) -> None:
        """Inicializa o analisador.
        
        Args:
            min_viable_sharpe: Sharpe mínimo aceitável
            realistic_cost_bps: Custo realista para avaliar viabilidade
        """
        self.min_viable_sharpe = min_viable_sharpe
        self.realistic_cost = realistic_cost_bps
    
    def analyze(
        self,
        prices: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame, float], float],
        cost_levels_bps: list[float] | None = None,
    ) -> CostSensitivityResult:
        """Executa análise de sensibilidade.
        
        Args:
            prices: DataFrame de preços
            strategy_func: Função(prices, cost_bps) -> Sharpe
            cost_levels_bps: Níveis de custo a testar
            
        Returns:
            CostSensitivityResult com diagnóstico
        """
        if cost_levels_bps is None:
            cost_levels_bps = [0, 2, 5, 10, 15, 20, 30, 50]
        
        sharpes = []
        
        for cost in cost_levels_bps:
            try:
                sharpe = strategy_func(prices, cost)
                sharpes.append(sharpe)
            except Exception:
                sharpes.append(0.0)
        
        # Encontrar breakeven
        breakeven = self._find_breakeven(cost_levels_bps, sharpes)
        
        # Sharpe no custo realista
        sharpe_at_realistic = self._interpolate(
            cost_levels_bps, sharpes, self.realistic_cost
        )
        
        # Avaliação
        is_viable = sharpe_at_realistic >= self.min_viable_sharpe
        
        if is_viable:
            verdict = f"✅ VIÁVEL COMERCIALMENTE: Sharpe={sharpe_at_realistic:.2f} @ {self.realistic_cost}bps"
        else:
            verdict = f"❌ INVIÁVEL: Sharpe={sharpe_at_realistic:.2f} @ {self.realistic_cost}bps (mín: {self.min_viable_sharpe})"
        
        return CostSensitivityResult(
            cost_levels_bps=cost_levels_bps,
            sharpe_ratios=sharpes,
            breakeven_cost_bps=breakeven,
            sharpe_at_10bps=sharpe_at_realistic,
            is_commercially_viable=is_viable,
            verdict=verdict,
        )
    
    def _find_breakeven(
        self,
        costs: list[float],
        sharpes: list[float],
    ) -> float:
        """Encontra custo onde Sharpe = 0 via interpolação."""
        for i in range(len(sharpes) - 1):
            if sharpes[i] > 0 and sharpes[i + 1] <= 0:
                # Interpolação linear
                slope = (sharpes[i + 1] - sharpes[i]) / (costs[i + 1] - costs[i])
                if abs(slope) < 1e-8:
                    return costs[i + 1]
                return costs[i] - sharpes[i] / slope
        
        return costs[-1] if sharpes[-1] <= 0 else float("inf")
    
    def _interpolate(
        self,
        x: list[float],
        y: list[float],
        x_target: float,
    ) -> float:
        """Interpolação linear simples."""
        for i in range(len(x) - 1):
            if x[i] <= x_target <= x[i + 1]:
                ratio = (x_target - x[i]) / (x[i + 1] - x[i])
                return y[i] + ratio * (y[i + 1] - y[i])
        return y[0] if x_target < x[0] else y[-1]
    
    def plot(
        self,
        result: CostSensitivityResult,
        save_path: str | None = None,
    ) -> None:
        """Gera gráfico de sensibilidade.
        
        Args:
            result: Resultado da análise
            save_path: Caminho para salvar (opcional)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(
            result.cost_levels_bps,
            result.sharpe_ratios,
            "b-o",
            linewidth=2,
            markersize=8,
        )
        
        # Linhas de referência
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="Breakeven")
        ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.7, label="Min Viável (0.5)")
        ax.axvline(x=10, color="orange", linestyle=":", alpha=0.7, label="Custo Realista (10bps)")
        
        # Preencher área viável
        ax.fill_between(
            result.cost_levels_bps,
            result.sharpe_ratios,
            0.5,
            where=[s >= 0.5 for s in result.sharpe_ratios],
            alpha=0.2,
            color="green",
        )
        
        ax.set_xlabel("Custo de Transação (bps)", fontsize=12)
        ax.set_ylabel("Sharpe Ratio", fontsize=12)
        ax.set_title("Análise de Sensibilidade a Custos de Transação", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # Anotação do breakeven
        ax.annotate(
            f"Breakeven: {result.breakeven_cost_bps:.1f}bps",
            xy=(result.breakeven_cost_bps, 0),
            xytext=(result.breakeven_cost_bps + 5, 0.3),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10,
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        plt.show()


@dataclass
class ValidationSuiteResult:
    """Resultado consolidado de todos os testes de validação."""
    
    stationarity: StationarityResult
    randomization: RandomizationResult | None
    cost_sensitivity: CostSensitivityResult
    overall_pass: bool
    executive_summary: str


class ValidationSuite:
    """Suite completa de validação para estratégias de trading.
    
    Executa todos os testes de sanidade em sequência:
    1. Estacionariedade dos spreads
    2. Randomization test (se habilitado)
    3. Sensibilidade a custos
    
    Example:
        >>> suite = ValidationSuite()
        >>> result = suite.run_all(
        ...     spread=portfolio_spread,
        ...     prices=price_data,
        ...     strategy_func=backtest_with_cost,
        ... )
        >>> print(result.executive_summary)
        >>> if not result.overall_pass:
        ...     raise ValueError("Estratégia falhou na validação!")
    """
    
    def __init__(
        self,
        run_randomization: bool = False,  # Computacionalmente caro
        n_shuffles: int = 100,
    ) -> None:
        """Inicializa a suite.
        
        Args:
            run_randomization: Se deve executar teste de randomização
            n_shuffles: Número de shuffles (se habilitado)
        """
        self.stationarity_validator = SpreadStationarityValidator()
        self.randomization_test = RandomizationTest(n_shuffles=n_shuffles) if run_randomization else None
        self.cost_analyzer = CostSensitivityAnalyzer()
        self.run_randomization = run_randomization
    
    def run_all(
        self,
        spread: pd.Series,
        prices: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame, float], float],
        signal_generator: Callable | None = None,
        backtest_func: Callable | None = None,
    ) -> ValidationSuiteResult:
        """Executa suite completa de validação.
        
        Args:
            spread: Spread do portfólio para teste de estacionariedade
            prices: Dados de preços para backtests
            strategy_func: Função(prices, cost_bps) -> Sharpe
            signal_generator: Para randomization test (opcional)
            backtest_func: Para randomization test (opcional)
            
        Returns:
            ValidationSuiteResult com diagnóstico completo
        """
        print("=" * 60)
        print("VALIDATION SUITE - PEER REVIEW CHECKLIST")
        print("=" * 60)
        
        # 1. Teste de Estacionariedade
        print("\n[1/3] Teste de Estacionariedade (ADF + KPSS)...")
        stationarity_result = self.stationarity_validator.test(spread, "Portfolio_Spread")
        print(f"      {stationarity_result.verdict}")
        
        # 2. Randomization Test (se habilitado)
        randomization_result = None
        if self.run_randomization and signal_generator and backtest_func:
            print("\n[2/3] Randomization Test (Detector de Look-ahead Bias)...")
            randomization_result = self.randomization_test.run(
                prices, signal_generator, backtest_func
            )
            print(f"      {randomization_result.verdict}")
        else:
            print("\n[2/3] Randomization Test: PULADO (não habilitado)")
        
        # 3. Cost Sensitivity
        print("\n[3/3] Análise de Sensibilidade a Custos...")
        cost_result = self.cost_analyzer.analyze(prices, strategy_func)
        print(f"      {cost_result.verdict}")
        
        # Consolidação
        tests_passed = [
            stationarity_result.is_stationary,
            cost_result.is_commercially_viable,
        ]
        
        if randomization_result:
            tests_passed.append(not randomization_result.has_look_ahead_bias)
        
        overall_pass = all(tests_passed)
        
        # Executive Summary
        summary_lines = [
            "",
            "=" * 60,
            "EXECUTIVE SUMMARY",
            "=" * 60,
            f"Estacionariedade: {'✅ PASS' if stationarity_result.is_stationary else '❌ FAIL'}",
            f"  - Half-life: {stationarity_result.half_life:.1f} dias",
        ]
        
        if randomization_result:
            summary_lines.append(
                f"Look-ahead Bias: {'❌ DETECTADO' if randomization_result.has_look_ahead_bias else '✅ LIMPO'}"
            )
            summary_lines.append(f"  - Z-score: {randomization_result.z_score:.2f}")
        
        summary_lines.extend([
            f"Viabilidade Comercial: {'✅ PASS' if cost_result.is_commercially_viable else '❌ FAIL'}",
            f"  - Sharpe @ 10bps: {cost_result.sharpe_at_10bps:.2f}",
            f"  - Breakeven: {cost_result.breakeven_cost_bps:.1f}bps",
            "",
            f"RESULTADO FINAL: {'✅ APROVADO' if overall_pass else '❌ REPROVADO'}",
            "=" * 60,
        ])
        
        executive_summary = "\n".join(summary_lines)
        print(executive_summary)
        
        return ValidationSuiteResult(
            stationarity=stationarity_result,
            randomization=randomization_result,
            cost_sensitivity=cost_result,
            overall_pass=overall_pass,
            executive_summary=executive_summary,
        )
