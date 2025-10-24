"""
Results Display Module

This module provides functions for formatting and displaying optimization results,
including comparison tables, currency formatting, and conditional highlighting.
"""

import pandas as pd
import numpy as np


def create_comparison_table(item_names, base_budgets, opt_budgets):
    """
    Build results DataFrame comparing base and optimized budgets.
    
    Args:
        item_names: List or array of item names (products and channels)
        base_budgets: Array of base budget values
        opt_budgets: Array of optimized budget values
    
    Returns:
        pd.DataFrame with columns: Item, Base Budget, Optimized Budget, Change ($), Change (%)
    
    Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9
    """
    # Calculate changes
    change_dollars = opt_budgets - base_budgets
    change_percent = (change_dollars / base_budgets) * 100
    
    # Create DataFrame
    comparison_df = pd.DataFrame({
        'Item': item_names,
        'Base Budget': base_budgets,
        'Optimized Budget': opt_budgets,
        'Change ($)': change_dollars,
        'Change (%)': change_percent
    })
    
    return comparison_df


def format_currency(value):
    """
    Format a numeric value as currency with two decimal places.
    
    Args:
        value: Numeric value to format
    
    Returns:
        String formatted as "$X,XXX.XX"
    
    Requirements: 11.2, 11.3, 11.7
    """
    if pd.isna(value):
        return "$0.00"
    
    # Handle negative values
    if value < 0:
        return f"-${abs(value):,.2f}"
    
    return f"${value:,.2f}"


def highlight_large_changes(row, threshold=0.10):
    """
    Return list of CSS styles for DataFrame styling to highlight large budget changes.
    
    Highlights rows where absolute percentage change exceeds the threshold.
    
    Args:
        row: DataFrame row containing 'Change (%)' column
        threshold: Percentage threshold (default 0.10 for 10%)
    
    Returns:
        List of CSS style strings for each column in the row
    
    Requirements: 11.8
    """
    # Get the change percentage value
    change_pct = row.get('Change (%)', 0)
    
    # Check if absolute change exceeds threshold
    if abs(change_pct) > threshold * 100:  # threshold is in decimal, change_pct is in percentage
        # Return yellow background for all columns in this row
        return ['background-color: #fff3cd'] * len(row)
    else:
        # Return empty styles (no highlighting)
        return [''] * len(row)
