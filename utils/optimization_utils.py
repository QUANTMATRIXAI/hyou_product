"""
Optimization Utilities for Marketing Budget Optimizer

This module consolidates all optimization-related calculations including:
- Catalog budget distribution
- Impression calculations
- Volume prediction
- Revenue calculation
- Budget optimization
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize


# ============================================================================
# CATALOG DISTRIBUTION
# ============================================================================

def distribute_catalog_budget(budgets, item_names, catalog_idx, other_products_idx):
    """No distribution needed - catalog budget is already in 'Other Products'."""
    # Catalog has been removed and its budget moved to Other Products before optimization
    # So we just return budgets as-is
    return budgets.copy()


# ============================================================================
# IMPRESSION CALCULATIONS
# ============================================================================

def calculate_impressions(budgets, cpm_values):
    """Convert budget allocations to impression counts using CPM values."""
    safe_cpm_values = cpm_values.copy()
    
    invalid_cpm_mask = safe_cpm_values <= 0
    if np.any(invalid_cpm_mask):
        safe_cpm_values[invalid_cpm_mask] = 1.0
    
    nan_cpm_mask = ~np.isfinite(safe_cpm_values)
    if np.any(nan_cpm_mask):
        safe_cpm_values[nan_cpm_mask] = 1.0
    
    impressions = np.divide(budgets, safe_cpm_values) * 1000
    impressions = np.where(np.isfinite(impressions), impressions, 0.0)
    impressions = np.maximum(impressions, 0.0)
    
    return impressions


def create_impression_dict(impressions, beta_column_names):
    """Create a dictionary mapping beta column names to impression values."""
    impression_dict = {}
    
    for beta_col, impression_value in zip(beta_column_names, impressions):
        if beta_col is not None:
            impression_dict[beta_col] = float(impression_value)
    
    return impression_dict


# ============================================================================
# VOLUME PREDICTION
# ============================================================================

def predict_volume(beta_row, impression_dict):
    """Calculate predicted volume for a single product using beta coefficients and impressions."""
    # Start with the intercept (B0) - handle variations like "B0 (Original)"
    volume = 0.0
    for col_name in beta_row.index:
        if col_name.startswith('B0'):
            volume = beta_row.get(col_name, 0.0)
            break
    
    if pd.isna(volume) or not np.isfinite(volume):
        volume = 0.0
    
    # Iterate through all Beta_ columns
    for col_name in beta_row.index:
        if not col_name.startswith('Beta_'):
            continue
        
        beta_value = beta_row[col_name]
        
        if pd.isna(beta_value) or not np.isfinite(beta_value):
            continue
        
        if col_name in impression_dict:
            impression_value = impression_dict[col_name]
            
            if not np.isfinite(impression_value):
                continue
            
            contribution = beta_value * impression_value
            
            if np.isfinite(contribution):
                volume += contribution
    
    if not np.isfinite(volume):
        volume = 0.0
    
    return max(0.0, volume)


def predict_all_volumes(beta_df, impression_dict, modeling_means=None):
    """Predict volumes for all products that have models in the beta DataFrame."""
    if beta_df.empty:
        return pd.Series([], dtype=float, name='Predicted Volume')
    
    # Create a dictionary mapping lowercase product names to (original_name, beta_row)
    product_map = {}
    
    for idx, row in beta_df.iterrows():
        try:
            product_name = row.get('Product title', f'Product_{idx}')
            
            if pd.isna(product_name) or product_name == '':
                product_name = f'Product_{idx}'
            
            product_map[product_name.lower()] = (product_name, row)
        except Exception:
            continue
    
    # Predict volumes for all products in the impression dict
    volumes = {}
    
    for beta_col in impression_dict.keys():
        if beta_col and '_meta_impression' in beta_col:
            product_part = beta_col.replace('Beta_', '').replace('_meta_impression', '')
            product_name_lower = product_part  # Keep spaces as-is
            
            if product_name_lower in product_map:
                original_name, beta_row = product_map[product_name_lower]
                
                if original_name not in volumes:
                    # Create product-specific impression dict with modeling means
                    product_impression_dict = impression_dict.copy()
                    
                    # Add product-specific modeling means
                    if modeling_means is not None and product_name_lower in modeling_means:
                        product_impression_dict.update(modeling_means[product_name_lower])
                    
                    volume = predict_volume(beta_row, product_impression_dict)
                    volumes[original_name] = volume
    
    # Also predict for "Other Products" if it exists
    if 'other products' in product_map:
        original_name, beta_row = product_map['other products']
        if original_name not in volumes:
            # Create product-specific impression dict with modeling means
            product_impression_dict = impression_dict.copy()
            
            # Add product-specific modeling means
            if modeling_means is not None and 'other products' in modeling_means:
                product_impression_dict.update(modeling_means['other products'])
            
            volume = predict_volume(beta_row, product_impression_dict)
            volumes[original_name] = volume
    
    return pd.Series(volumes, name='Predicted Volume')


# ============================================================================
# REVENUE CALCULATION
# ============================================================================

def calculate_revenue(volumes, prices):
    """Calculate total revenue from predicted volumes and product prices."""
    if isinstance(volumes, pd.Series):
        volumes = volumes.values
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    if len(volumes) == 0 or len(prices) == 0:
        return 0.0
    
    min_length = min(len(volumes), len(prices))
    volumes = volumes[:min_length]
    prices = prices[:min_length]
    
    volumes = np.where(np.isfinite(volumes), volumes, 0.0)
    prices = np.where(np.isfinite(prices), prices, 0.0)
    
    volumes = np.maximum(volumes, 0.0)
    prices = np.maximum(prices, 0.0)
    
    product_revenues = volumes * prices
    product_revenues = np.where(np.isfinite(product_revenues), product_revenues, 0.0)
    
    total_revenue = np.sum(product_revenues)
    
    if not np.isfinite(total_revenue):
        total_revenue = 0.0
    
    return max(0.0, float(total_revenue))


# ============================================================================
# OPTIMIZATION
# ============================================================================

def calculate_revenue_for_display(budgets, beta_df, cpm_values, price_dict, item_names,
                                   beta_column_names, google_trends_value=50.0, modeling_means=None):
    """
    Calculate revenue and volume for display purposes (not used in optimization).
    
    This function runs the same prediction pipeline as the objective function but
    multiplies volumes by prices to calculate revenue. It's used to show revenue
    metrics alongside volume metrics in the results display.
    
    Args:
        budgets: Array of budget allocations
        beta_df: DataFrame containing beta coefficients
        cpm_values: Array of CPM values
        price_dict: Dictionary mapping product names to prices
        item_names: Array of product/channel names
        beta_column_names: Array of beta column names
        google_trends_value: Google Trends seasonality value (default 50.0)
        modeling_means: Optional dict of product-specific mean values
    
    Returns:
        tuple: (revenue, total_volume) - Revenue in dollars and total volume in units
    """
    try:
        # Calculate impressions
        impressions = calculate_impressions(budgets, cpm_values)
        
        # Create impression dict
        impression_dict = create_impression_dict(impressions, beta_column_names)
        impression_dict['Beta_google_trends'] = google_trends_value
        
        # Predict volumes
        volumes = predict_all_volumes(beta_df, impression_dict, modeling_means)
        
        # Calculate total volume
        total_volume = volumes.sum()
        
        # Calculate revenue using prices
        price_dict_lower = {k.lower(): v for k, v in price_dict.items()}
        product_prices = np.array([
            price_dict_lower.get(product_name.lower(), 0.0) 
            for product_name in volumes.index
        ])
        
        revenue = calculate_revenue(volumes, product_prices)
        
        return revenue, total_volume
        
    except Exception:
        return 0.0, 0.0


def create_objective_function(beta_df, cpm_values, item_names,
                               beta_column_names, google_trends_value=50.0, modeling_means=None):
    """
    Create an objective function for volume-based optimization.
    
    This function creates and returns an objective function that maximizes total predicted
    volume across all products. The objective function does NOT use product prices - it
    optimizes purely based on predicted sales quantity.
    
    Args:
        beta_df: DataFrame containing beta coefficients for volume prediction
        cpm_values: Array of CPM values for impression calculation
        item_names: Array of product/channel names
        beta_column_names: Array of beta column names corresponding to items
        google_trends_value: Google Trends seasonality value (default 50.0)
        modeling_means: Optional dict of product-specific mean values for variables
    
    Returns:
        objective: Function that takes budget array and returns negative total volume
                  (negative because optimizer minimizes)
    """
    
    def objective(budgets):
        """Calculate negative total volume for given budget allocation."""
        try:
            if np.any(budgets < 0):
                return 1e10
            
            # No distribution needed - Other Products is already in the budget file
            distributed_budgets = budgets.copy()
            
            impressions = calculate_impressions(distributed_budgets, cpm_values)
            
            if np.any(~np.isfinite(impressions)):
                return 1e10
            
            impression_dict = create_impression_dict(impressions, beta_column_names)
            
            # Add Google Trends value to impression dict
            impression_dict['Beta_google_trends'] = google_trends_value
            
            # Note: modeling_means are product-specific, so they need to be added per product
            # This will be handled in predict_all_volumes
            
            volumes = predict_all_volumes(beta_df, impression_dict, modeling_means)
            
            if volumes.isna().any() or np.any(~np.isfinite(volumes.values)):
                return 1e10
            
            # Sum volumes directly (no price multiplication)
            total_volume = volumes.sum()
            
            if not np.isfinite(total_volume) or total_volume < 0:
                return 1e10
            
            return -total_volume  # Negative because optimizer minimizes
            
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return 1e10
        except Exception:
            return 1e10
    
    return objective


def create_bounds(base_budgets, lower_pct=0.75, upper_pct=1.25):
    """Generate budget constraint bounds for optimization."""
    bounds = []
    
    for base_budget in base_budgets:
        lower_bound = base_budget * lower_pct
        upper_bound = base_budget * upper_pct
        bounds.append((lower_bound, upper_bound))
    
    return bounds


def optimize_budgets(objective_fn, base_budgets, bounds, constraints=None):
    """Run SLSQP optimization to find optimal budget allocation."""
    try:
        if np.any(~np.isfinite(base_budgets)):
            raise ValueError("Base budgets contain invalid values (NaN or Inf)")
        
        if np.any(base_budgets < 0):
            raise ValueError("Base budgets contain negative values")
        
        result = minimize(
            fun=objective_fn,
            x0=base_budgets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 1000,
                'ftol': 1e-6
            }
        )
        
        # Note: result.fun is negative volume (because we minimize -volume)
        optimized_volume = -result.fun
        
        if not result.success:
            base_volume = -objective_fn(base_budgets)
            if optimized_volume > base_volume and np.isfinite(optimized_volume):
                optimization_result = {
                    'success': False,
                    'optimized_budgets': result.x,
                    'optimized_volume': optimized_volume,
                    'message': f"Partial convergence: {result.message}",
                    'iterations': result.nit if hasattr(result, 'nit') else 0,
                    'function_evals': result.nfev if hasattr(result, 'nfev') else 0
                }
            else:
                optimization_result = {
                    'success': False,
                    'optimized_budgets': base_budgets,
                    'optimized_volume': base_volume,
                    'message': f"Optimization failed to improve results: {result.message}",
                    'iterations': result.nit if hasattr(result, 'nit') else 0,
                    'function_evals': result.nfev if hasattr(result, 'nfev') else 0
                }
        else:
            optimization_result = {
                'success': result.success,
                'optimized_budgets': result.x,
                'optimized_volume': optimized_volume,
                'message': result.message,
                'iterations': result.nit if hasattr(result, 'nit') else 0,
                'function_evals': result.nfev if hasattr(result, 'nfev') else 0
            }
        
        return optimization_result
        
    except (ValueError, ZeroDivisionError, FloatingPointError) as e:
        return {
            'success': False,
            'optimized_budgets': base_budgets,
            'optimized_volume': 0.0,
            'message': f"Numerical error during optimization: {str(e)}",
            'iterations': 0,
            'function_evals': 0
        }
    except Exception as e:
        return {
            'success': False,
            'optimized_budgets': base_budgets,
            'optimized_volume': 0.0,
            'message': f"Optimization failed: {str(e)}",
            'iterations': 0,
            'function_evals': 0
        }
