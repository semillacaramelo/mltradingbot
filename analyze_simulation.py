"""
Script to analyze trading simulation results and generate a comprehensive report
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_simulation_results(file_path='simulation_results.csv'):
    """Analyze trading simulation results and generate visualizations"""
    try:
        # Read simulation results
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic statistics
        total_trades = len(df)
        profitable_trades = len(df[df['profit'] > 0])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = df['profit'].sum()
        avg_profit_per_trade = df['profit'].mean()
        
        # Calculate prediction accuracy
        df['prediction_error'] = abs(df['predicted_change'] - df['actual_change'])
        df['prediction_direction_correct'] = (
            ((df['type'] == 'CALL') & (df['actual_change'] > 0)) |
            ((df['type'] == 'PUT') & (df['actual_change'] < 0))
        )
        direction_accuracy = df['prediction_direction_correct'].mean() * 100
        avg_prediction_error = df['prediction_error'].mean()
        
        # Calculate drawdown
        df['cumulative_profit'] = df['profit'].cumsum()
        df['peak'] = df['cumulative_profit'].cummax()
        df['drawdown'] = df['peak'] - df['cumulative_profit']
        max_drawdown = df['drawdown'].max()
        
        # Generate report
        print("\n=== Trading Simulation Analysis Report ===")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nBasic Statistics:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Profit: {total_profit:.4f}")
        print(f"Average Profit per Trade: {avg_profit_per_trade:.4f}")
        print(f"Maximum Drawdown: {max_drawdown:.4f}")
        
        print("\nPrediction Performance:")
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        print(f"Average Prediction Error: {avg_prediction_error:.2f}%")
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Cumulative Profit
        plt.subplot(2, 2, 1)
        plt.plot(df['timestamp'], df['cumulative_profit'])
        plt.title('Cumulative Profit Over Time')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Plot 2: Prediction Error Distribution
        plt.subplot(2, 2, 2)
        plt.hist(df['prediction_error'], bins=20)
        plt.title('Prediction Error Distribution')
        plt.grid(True)
        
        # Plot 3: Actual vs Predicted Changes
        plt.subplot(2, 2, 3)
        plt.scatter(df['predicted_change'], df['actual_change'])
        plt.xlabel('Predicted Change (%)')
        plt.ylabel('Actual Change (%)')
        plt.title('Predicted vs Actual Price Changes')
        plt.grid(True)
        
        # Plot 4: Drawdown Over Time
        plt.subplot(2, 2, 4)
        plt.plot(df['timestamp'], df['drawdown'])
        plt.title('Drawdown Over Time')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('simulation_analysis.png')
        
        # Generate recommendations
        print("\nRecommendations:")
        
        if avg_prediction_error > 1.0:  # More than 1% average error
            print("- High prediction error detected. Consider:")
            print("  * Reducing prediction horizon")
            print("  * Adding more relevant features")
            print("  * Adjusting model architecture")
            
        if direction_accuracy < 55:  # Less than 55% direction accuracy
            print("- Poor directional accuracy. Consider:")
            print("  * Reviewing feature importance")
            print("  * Implementing ensemble methods")
            print("  * Adding market regime detection")
            
        if max_drawdown > abs(total_profit):  # Drawdown larger than total profit
            print("- High drawdown relative to profits. Consider:")
            print("  * Implementing stricter risk management")
            print("  * Reducing position sizes")
            print("  * Adding stop-loss mechanisms")
            
        print("\nAnalysis complete. Visualizations saved to 'simulation_analysis.png'")
        
    except Exception as e:
        print(f"Error analyzing simulation results: {str(e)}")

if __name__ == "__main__":
    analyze_simulation_results()
