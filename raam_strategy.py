"""
Ranked Asset Allocation Model (RAAM) Strategy
Inspired by:
- Generalized Momentum by Wouter J. Keller & Hugo van Putten
- Ranked Asset Allocation Model by Gioele Giordano (Dow Award 2018)

This implementation preserves the original strategy logic while improving
modularity, readability, and diagnostic capabilities.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
from datetime import datetime, timedelta

# Set styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class DataDownloader:
    """Handles asset data downloading and preprocessing."""
    
    def __init__(self, start_date: str = "2010-01-01", end_date: Optional[str] = None):
        """
        Initialize the data downloader.
        
        Args:
            start_date: Start date for data download (YYYY-MM-DD)
            end_date: End date for data download (YYYY-MM-DD), defaults to today
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    
    def download_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Download adjusted close prices for given tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with adjusted close prices
        """
        print(f"Downloading data for {len(tickers)} assets from {self.start_date} to {self.end_date}")
        
        try:
            data = yf.download(tickers, start=self.start_date, end=self.end_date, 
                             progress=False)['Adj Close']
            
            if isinstance(data, pd.Series):
                data = data.to_frame(tickers[0])
            
            # Handle missing data
            missing_pct = data.isnull().sum() / len(data) * 100
            if missing_pct.max() > 5:
                print(f"Warning: Some assets have >5% missing data:")
                print(missing_pct[missing_pct > 5])
            
            data = data.fillna(method='ffill').dropna()
            print(f"Data shape: {data.shape}")
            print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            
            return data
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise

class TechnicalIndicators:
    """Calculates technical indicators used in the ranking system."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI calculation period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, period: int = 21) -> pd.Series:
        """
        Calculate rolling volatility (annualized).
        
        Args:
            returns: Return series
            period: Rolling window period
            
        Returns:
            Volatility series
        """
        return returns.rolling(window=period).std() * np.sqrt(252)
    
    @staticmethod
    def calculate_average_correlation(returns: pd.DataFrame, asset: str, period: int = 63) -> pd.Series:
        """
        Calculate average correlation of an asset with all other assets.
        
        Args:
            returns: DataFrame of all asset returns
            asset: Target asset name
            period: Rolling correlation window
            
        Returns:
            Average correlation series
        """
        if asset not in returns.columns:
            return pd.Series(index=returns.index, dtype=float)
        
        correlations = []
        for other_asset in returns.columns:
            if other_asset != asset:
                corr = returns[asset].rolling(window=period).corr(returns[other_asset])
                correlations.append(corr)
        
        if correlations:
            avg_corr = pd.concat(correlations, axis=1).mean(axis=1)
        else:
            avg_corr = pd.Series(0, index=returns.index)
        
        return avg_corr

class RankingEngine:
    """Implements the asset ranking methodology."""
    
    def __init__(self, rsi_weight: float = 0.33, vol_weight: float = 0.33, corr_weight: float = 0.34):
        """
        Initialize ranking engine with factor weights.
        
        Args:
            rsi_weight: Weight for RSI factor
            vol_weight: Weight for volatility factor  
            corr_weight: Weight for correlation factor
        """
        self.rsi_weight = rsi_weight
        self.vol_weight = vol_weight
        self.corr_weight = corr_weight
        
        # Validate weights sum to 1
        total_weight = rsi_weight + vol_weight + corr_weight
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Factor weights sum to {total_weight:.3f}, not 1.0")
    
    def calculate_all_indicators(self, prices: pd.DataFrame, 
                               rsi_period: int = 14,
                               vol_period: int = 21, 
                               corr_period: int = 63) -> Dict[str, pd.DataFrame]:
        """
        Calculate all technical indicators for ranking.
        
        Args:
            prices: Price data DataFrame
            rsi_period: RSI calculation period
            vol_period: Volatility calculation period
            corr_period: Correlation calculation period
            
        Returns:
            Dictionary containing all indicator DataFrames
        """
        print("Calculating technical indicators...")
        
        returns = prices.pct_change()
        
        indicators = {}
        
        # Calculate RSI for all assets
        rsi_data = {}
        for asset in prices.columns:
            rsi_data[asset] = TechnicalIndicators.calculate_rsi(prices[asset], rsi_period)
        indicators['rsi'] = pd.DataFrame(rsi_data)
        
        # Calculate volatility for all assets
        vol_data = {}
        for asset in prices.columns:
            vol_data[asset] = TechnicalIndicators.calculate_volatility(returns[asset], vol_period)
        indicators['volatility'] = pd.DataFrame(vol_data)
        
        # Calculate average correlation for all assets
        corr_data = {}
        for asset in prices.columns:
            corr_data[asset] = TechnicalIndicators.calculate_average_correlation(returns, asset, corr_period)
        indicators['correlation'] = pd.DataFrame(corr_data)
        
        print(f"Indicators calculated for {len(prices.columns)} assets")
        return indicators
    
    def calculate_composite_scores(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate composite ranking scores.
        
        Higher RSI = Higher score (momentum)
        Lower volatility = Higher score (stability)  
        Lower correlation = Higher score (diversification)
        
        Args:
            indicators: Dictionary of indicator DataFrames
            
        Returns:
            DataFrame of composite scores
        """
        rsi_scores = indicators['rsi'] / 100  # Normalize to 0-1
        vol_scores = 1 - (indicators['volatility'] / indicators['volatility'].max())  # Invert volatility
        corr_scores = 1 - indicators['correlation']  # Invert correlation
        
        # Handle NaN values
        rsi_scores = rsi_scores.fillna(0)
        vol_scores = vol_scores.fillna(0) 
        corr_scores = corr_scores.fillna(0)
        
        # Calculate weighted composite score
        composite_scores = (self.rsi_weight * rsi_scores + 
                          self.vol_weight * vol_scores + 
                          self.corr_weight * corr_scores)
        
        return composite_scores
    
    def rank_assets(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Rank assets based on composite scores (1 = best).
        
        Args:
            scores: Composite scores DataFrame
            
        Returns:
            DataFrame of asset rankings
        """
        # Rank assets (1 = highest score = best rank)
        rankings = scores.rank(axis=1, method='min', ascending=False)
        return rankings

class PortfolioConstructor:
    """Handles portfolio weight assignment and construction."""
    
    def __init__(self, top_n: int = 5):
        """
        Initialize portfolio constructor.
        
        Args:
            top_n: Number of top-ranked assets to select
        """
        self.top_n = top_n
    
    def assign_equal_weights(self, rankings: pd.DataFrame) -> pd.DataFrame:
        """
        Assign equal weights to top N ranked assets.
        
        Args:
            rankings: Asset rankings DataFrame
            
        Returns:
            DataFrame of portfolio weights
        """
        weights = pd.DataFrame(0.0, index=rankings.index, columns=rankings.columns)
        
        for date in rankings.index:
            # Get rankings for this date
            date_ranks = rankings.loc[date]
            
            # Remove NaN rankings
            valid_ranks = date_ranks.dropna()
            
            if len(valid_ranks) == 0:
                continue
            
            # Select top N assets
            top_assets = valid_ranks.nsmallest(self.top_n).index
            
            # Assign equal weights
            weight_per_asset = 1.0 / len(top_assets)
            weights.loc[date, top_assets] = weight_per_asset
            
            # Log selection for diagnostics
            if date == rankings.index[-1]:  # Log final selection
                print(f"\nFinal Portfolio Selection ({date.strftime('%Y-%m-%d')}):")
                for i, asset in enumerate(top_assets, 1):
                    rank = valid_ranks[asset]
                    print(f"  {i}. {asset}: Rank {rank:.1f}, Weight {weight_per_asset:.1%}")
        
        return weights

class BacktestEngine:
    """Handles portfolio backtesting and performance calculation."""
    
    def __init__(self, rebalance_freq: str = 'M'):
        """
        Initialize backtest engine.
        
        Args:
            rebalance_freq: Rebalancing frequency ('M' for monthly, 'Q' for quarterly)
        """
        self.rebalance_freq = rebalance_freq
    
    def run_backtest(self, prices: pd.DataFrame, weights: pd.DataFrame) -> Dict:
        """
        Run portfolio backtest.
        
        Args:
            prices: Asset price data
            weights: Portfolio weights
            
        Returns:
            Dictionary containing backtest results
        """
        print(f"\nRunning backtest with {self.rebalance_freq} rebalancing...")
        
        returns = prices.pct_change()
        
        # Align weights with returns
        if self.rebalance_freq == 'M':
            rebal_weights = weights.resample('M').last().fillna(method='ffill')
        elif self.rebalance_freq == 'Q':
            rebal_weights = weights.resample('Q').last().fillna(method='ffill')
        else:
            rebal_weights = weights
        
        # Forward fill weights between rebalancing dates
        aligned_weights = rebal_weights.reindex(returns.index).fillna(method='ffill')
        
        # Calculate portfolio returns
        portfolio_returns = (returns * aligned_weights.shift(1)).sum(axis=1)
        portfolio_returns = portfolio_returns.dropna()
        
        # Calculate cumulative returns
        portfolio_cumret = (1 + portfolio_returns).cumprod()
        
        # Calculate performance metrics
        total_return = portfolio_cumret.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Calculate drawdown
        rolling_max = portfolio_cumret.expanding().max()
        drawdown = (portfolio_cumret - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        results = {
            'returns': portfolio_returns,
            'cumulative_returns': portfolio_cumret,
            'weights': aligned_weights,
            'drawdown': drawdown,
            'metrics': {
                'Total Return': f"{total_return:.1%}",
                'Annual Return': f"{annual_return:.1%}",
                'Annual Volatility': f"{annual_vol:.1%}",
                'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                'Max Drawdown': f"{max_drawdown:.1%}"
            }
        }
        
        return results

class Visualizer:
    """Handles all visualization and output generation."""
    
    @staticmethod
    def plot_equity_curve(results: Dict, benchmark_prices: Optional[pd.Series] = None):
        """Plot portfolio equity curve vs benchmark."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Equity curve
        portfolio_cumret = results['cumulative_returns']
        ax1.plot(portfolio_cumret.index, portfolio_cumret.values, 
                label='RAAM Strategy', linewidth=2, color='darkblue')
        
        if benchmark_prices is not None:
            bench_returns = benchmark_prices.pct_change().dropna()
            bench_cumret = (1 + bench_returns).cumprod()
            # Align with portfolio dates
            bench_cumret = bench_cumret.reindex(portfolio_cumret.index).fillna(method='ffill')
            ax1.plot(bench_cumret.index, bench_cumret.values, 
                    label='Benchmark', linewidth=2, color='gray', alpha=0.7)
        
        ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        drawdown = results['drawdown']
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        alpha=0.7, color='red', label='Drawdown')
        ax2.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_rolling_metrics(results: Dict, window: int = 252):
        """Plot rolling performance metrics."""
        returns = results['returns']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Rolling Sharpe ratio
        rolling_sharpe = (returns.rolling(window).mean() / 
                         returns.rolling(window).std() * np.sqrt(252))
        ax1.plot(rolling_sharpe.index, rolling_sharpe.values, 
                linewidth=2, color='green')
        ax1.set_title(f'Rolling {window}-Day Sharpe Ratio', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        ax2.plot(rolling_vol.index, rolling_vol.values, 
                linewidth=2, color='orange')
        ax2.set_title(f'Rolling {window}-Day Volatility', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volatility')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_weight_heatmap(weights: pd.DataFrame, top_n: int = 10):
        """Plot portfolio weights heatmap."""
        # Get most frequently selected assets
        weight_sums = weights.sum().nlargest(top_n)
        top_assets = weight_sums.index
        
        # Resample to monthly for cleaner visualization
        monthly_weights = weights[top_assets].resample('M').last()
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(monthly_weights.T, cmap='YlOrRd', cbar_kws={'label': 'Weight'})
        plt.title('Portfolio Weights Heatmap (Monthly)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Assets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def print_performance_summary(results: Dict):
        """Print comprehensive performance summary."""
        print("\n" + "="*60)
        print("RAAM STRATEGY PERFORMANCE SUMMARY")
        print("="*60)
        
        metrics = results['metrics']
        for metric, value in metrics.items():
            print(f"{metric:<20}: {value}")
        
        print("\n" + "-"*40)
        print("ADDITIONAL STATISTICS")
        print("-"*40)
        
        returns = results['returns']
        print(f"{'Number of Trades':<20}: {len(returns)}")
        print(f"{'Positive Days':<20}: {(returns > 0).sum()} ({(returns > 0).mean():.1%})")
        print(f"{'Best Day':<20}: {returns.max():.2%}")
        print(f"{'Worst Day':<20}: {returns.min():.2%}")

class RAAMStrategy:
    """Main strategy class that orchestrates all components."""
    
    def __init__(self, 
                 tickers: List[str],
                 start_date: str = "2010-01-01",
                 end_date: Optional[str] = None,
                 top_n: int = 5,
                 rebalance_freq: str = 'M'):
        """
        Initialize RAAM strategy.
        
        Args:
            tickers: List of asset tickers
            start_date: Strategy start date
            end_date: Strategy end date
            top_n: Number of assets to select
            rebalance_freq: Rebalancing frequency
        """
        self.tickers = tickers
        self.top_n = top_n
        
        # Initialize components
        self.data_downloader = DataDownloader(start_date, end_date)
        self.ranking_engine = RankingEngine()
        self.portfolio_constructor = PortfolioConstructor(top_n)
        self.backtest_engine = BacktestEngine(rebalance_freq)
        self.visualizer = Visualizer()
        
        # Storage for results
        self.prices = None
        self.indicators = None
        self.rankings = None
        self.weights = None
        self.results = None
    
    def run_strategy(self, benchmark_ticker: Optional[str] = None):
        """
        Execute the complete RAAM strategy.
        
        Args:
            benchmark_ticker: Benchmark ticker for comparison
        """
        print("Starting RAAM Strategy Execution")
        print("="*50)
        
        # 1. Download data
        self.prices = self.data_downloader.download_data(self.tickers)
        
        # 2. Calculate indicators
        self.indicators = self.ranking_engine.calculate_all_indicators(self.prices)
        
        # 3. Calculate scores and rankings
        scores = self.ranking_engine.calculate_composite_scores(self.indicators)
        self.rankings = self.ranking_engine.rank_assets(scores)
        
        # 4. Assign weights
        self.weights = self.portfolio_constructor.assign_equal_weights(self.rankings)
        
        # 5. Run backtest
        self.results = self.backtest_engine.run_backtest(self.prices, self.weights)
        
        # 6. Generate outputs
        self._generate_outputs(benchmark_ticker)
    
    def _generate_outputs(self, benchmark_ticker: Optional[str] = None):
        """Generate all outputs and visualizations."""
        
        # Print performance summary
        self.visualizer.print_performance_summary(self.results)
        
        # Download benchmark if provided
        benchmark_prices = None
        if benchmark_ticker:
            try:
                benchmark_data = self.data_downloader.download_data([benchmark_ticker])
                benchmark_prices = benchmark_data[benchmark_ticker]
            except:
                print(f"Warning: Could not download benchmark {benchmark_ticker}")
        
        # Generate visualizations
        self.visualizer.plot_equity_curve(self.results, benchmark_prices)
        self.visualizer.plot_rolling_metrics(self.results)
        self.visualizer.plot_weight_heatmap(self.weights)
        
        # Diagnostic outputs
        self._print_diagnostics()
    
    def _print_diagnostics(self):
        """Print diagnostic information for strategy validation."""
        print("\n" + "="*60)
        print("STRATEGY DIAGNOSTICS")
        print("="*60)
        
        # Check for ranking ties
        final_rankings = self.rankings.iloc[-1].dropna()
        tied_ranks = final_rankings[final_rankings.duplicated()].index
        if len(tied_ranks) > 0:
            print(f"Warning: Tied rankings detected for assets: {list(tied_ranks)}")
        
        # Weight distribution check
        final_weights = self.weights.iloc[-1]
        selected_assets = final_weights[final_weights > 0]
        print(f"\nFinal Portfolio Allocation:")
        print(f"Selected Assets: {len(selected_assets)}")
        print(f"Weight Distribution: {selected_assets.describe()}")
        
        # Indicator validation
        print(f"\nIndicator Summary (Final Values):")
        for indicator_name, indicator_df in self.indicators.items():
            final_values = indicator_df.iloc[-1].dropna()
            print(f"{indicator_name.upper()}:")
            print(f"  Mean: {final_values.mean():.3f}")
            print(f"  Range: {final_values.min():.3f} - {final_values.max():.3f}")

# Example usage and main execution
if __name__ == "__main__":
    # Define asset universe (example portfolio)
    TICKERS = [
        'SPY',  # S&P 500
        'QQQ',  # NASDAQ
        'IWM',  # Russell 2000
        'EFA',  # Developed Markets
        'EEM',  # Emerging Markets
        'TLT',  # Long-term Treasuries
        'IEF',  # Intermediate Treasuries
        'GLD',  # Gold
        'VNQ',  # Real Estate
        'DBC'   # Commodities
    ]
    
    # Initialize and run strategy
    strategy = RAAMStrategy(
        tickers=TICKERS,
        start_date="2010-01-01",
        top_n=5,
        rebalance_freq='M'
    )
    
    # Execute strategy with SPY as benchmark
    strategy.run_strategy(benchmark_ticker='SPY')
    
    print("\n" + "="*60)
    print("FUTURE ENHANCEMENT SUGGESTIONS")
    print("="*60)
    print("1. Add transaction costs and slippage modeling")
    print("2. Implement dynamic lookback periods based on market regime")
    print("3. Add risk budgeting constraints")
    print("4. Include momentum and mean reversion factors")
    print("5. Add regime detection for factor weight adjustment")
    print("6. Implement portfolio optimization beyond equal weighting")
    print("7. Add sector/geographic constraints")
    print("8. Include ESG scoring as additional factor") 