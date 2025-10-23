"""
Data Utilities for Marketing Budget Optimizer

This module consolidates all data loading, validation, preparation, and mapping functions.
"""

import pandas as pd
import numpy as np


# ============================================================================
# FILE LOADING
# ============================================================================

def load_file(uploaded_file):
    """Load a CSV or Excel file into a pandas DataFrame."""
    try:
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if df.empty:
            raise ValueError(f"File {file_name} is empty")
        
        return df
    except Exception as e:
        raise ValueError(f"Error reading file {uploaded_file.name}: {str(e)}")


# ============================================================================
# FILE VALIDATION
# ============================================================================

def validate_budget_file(df):
    """Validate budget allocation file structure."""
    try:
        if df.empty:
            return False, "Budget file is empty"
        if len(df.columns) < 2:
            return False, "Budget file must have at least one product/channel column and one week column"
        if df[df.columns[0]].isna().all():
            return False, "Product/channel name column is empty"
        return True, ""
    except Exception as e:
        return False, f"Error validating budget file: {str(e)}"


def validate_cpm_file(df):
    """Validate CPM file structure."""
    try:
        if df.empty:
            return False, "CPM file is empty"
        cpm_col = next((col for col in df.columns if 'cpm' in col.lower()), None)
        if cpm_col is None:
            return False, "CPM file must have a 'CPM' column"
        if len(df.columns) < 2:
            return False, "CPM file must have a product/channel name column and a CPM column"
        return True, ""
    except Exception as e:
        return False, f"Error validating CPM file: {str(e)}"


def validate_beta_file(df):
    """Validate beta coefficients file structure."""
    try:
        if df.empty:
            return False, "Beta file is empty"
        
        # Check for Product title column
        product_col = next((col for col in df.columns if 'product' in col.lower() and 'title' in col.lower()), None)
        if product_col is None:
            return False, "Beta file must have a 'Product title' column"
        
        # Check for B0 column (handle variations like "B0 (Original)")
        b0_col = next((col for col in df.columns if col.startswith('B0')), None)
        if b0_col is None:
            return False, "Beta file must have a 'B0' column (intercept)"
        
        # Check for at least one Beta_ column
        beta_columns = [col for col in df.columns if col.startswith('Beta_')]
        if len(beta_columns) == 0:
            return False, "Beta file must have at least one column starting with 'Beta_'"
        
        return True, ""
    except Exception as e:
        return False, f"Error validating beta file: {str(e)}"


def validate_attribution_file(df):
    """Validate catalog attribution file structure."""
    try:
        if df.empty:
            return False, "Attribution file is empty"
        attr_col = next((col for col in df.columns if 'attribution' in col.lower() and 'ratio' in col.lower()), None)
        if attr_col is None:
            return False, "Attribution file must have an 'Attribution_Ratio' column"
        if len(df.columns) < 2:
            return False, "Attribution file must have a product name column and an Attribution_Ratio column"
        ratio_sum = df[attr_col].sum()
        if not np.isclose(ratio_sum, 1.0, atol=0.01):
            return False, f"Attribution ratios must sum to approximately 1.0 (current sum: {ratio_sum:.4f})"
        return True, ""
    except Exception as e:
        return False, f"Error validating attribution file: {str(e)}"


def validate_price_file(df):
    """Validate product price file structure."""
    try:
        if df.empty:
            return False, "Price file is empty"
        price_col = next((col for col in df.columns if 'price' in col.lower()), None)
        if price_col is None:
            return False, "Price file must have a 'Price' column"
        if len(df.columns) < 2:
            return False, "Price file must have a product name column and a Price column"
        return True, ""
    except Exception as e:
        return False, f"Error validating price file: {str(e)}"


# ============================================================================
# DATA PREPARATION
# ============================================================================

def extract_week_columns(budget_df):
    """Extract available week column names from budget file."""
    if budget_df.empty or len(budget_df.columns) < 2:
        return []
    return budget_df.columns[1:].tolist()


def extract_base_budgets(budget_df, week_col):
    """Extract budget values for a selected week."""
    if week_col not in budget_df.columns:
        raise ValueError(f"Week column '{week_col}' not found in budget file")
    
    name_col = budget_df.columns[0]
    budget_values = budget_df[week_col].values
    budget_values = np.where(np.isfinite(budget_values), budget_values, 0.0)
    budget_values = np.maximum(budget_values, 0.0)
    
    return pd.Series(budget_values, index=budget_df[name_col].values, name='base_budget')


def normalize_product_names(df, name_col=None):
    """Normalize product/channel names for case-insensitive matching."""
    df_copy = df.copy()
    if df_copy.empty:
        return df_copy
    
    if name_col is None:
        name_col = df_copy.columns[0]
    
    df_copy[name_col] = (
        df_copy[name_col]
        .fillna('')
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r'\s+', ' ', regex=True)
    )
    
    return df_copy


def merge_data(budget_df, cpm_df, attr_df, price_df, week_col):
    """Merge all data sources into a master DataFrame."""
    base_budgets = extract_base_budgets(budget_df, week_col)
    
    budget_name_col = budget_df.columns[0]
    master_df = pd.DataFrame({
        'item_name': budget_df[budget_name_col].values,
        'base_budget': base_budgets.values
    })
    
    master_normalized = normalize_product_names(master_df, 'item_name')
    master_normalized['item_name_normalized'] = master_normalized['item_name']
    master_normalized['item_name'] = master_df['item_name']
    
    # Merge CPM data
    cpm_name_col = cpm_df.columns[0]
    cpm_col = next((col for col in cpm_df.columns if 'cpm' in col.lower()), None)
    
    if cpm_col is None:
        master_normalized['cpm'] = 1.0
    else:
        cpm_normalized = normalize_product_names(cpm_df, cpm_name_col)
        cpm_normalized = cpm_normalized.rename(columns={cpm_name_col: 'item_name_normalized', cpm_col: 'cpm'})
        master_normalized = master_normalized.merge(cpm_normalized[['item_name_normalized', 'cpm']], on='item_name_normalized', how='left')
        master_normalized['cpm'] = master_normalized['cpm'].fillna(1.0)
    
    # Attribution data is now optional (not needed anymore)
    # Catalog budget goes directly to "Other Products"
    master_normalized['attribution_ratio'] = 0.0
    
    # Merge price data
    price_name_col = price_df.columns[0]
    price_col = next((col for col in price_df.columns if 'price' in col.lower()), None)
    
    if price_col is None:
        master_normalized['price'] = 0.0
    else:
        price_normalized = normalize_product_names(price_df, price_name_col)
        price_normalized = price_normalized.rename(columns={price_name_col: 'item_name_normalized', price_col: 'price'})
        master_normalized = master_normalized.merge(price_normalized[['item_name_normalized', 'price']], on='item_name_normalized', how='left')
        master_normalized['price'] = master_normalized['price'].fillna(0.0)
    
    master_df = master_normalized.drop(columns=['item_name_normalized'])
    
    # Ensure all numeric columns have valid values
    numeric_cols = ['base_budget', 'cpm', 'attribution_ratio', 'price']
    for col in numeric_cols:
        master_df[col] = master_df[col].fillna(1.0 if col == 'cpm' else 0.0)
        master_df[col] = np.where(np.isfinite(master_df[col]), master_df[col], 1.0 if col == 'cpm' else 0.0)
    
    return master_df


# ============================================================================
# BETA MAPPING
# ============================================================================

def detect_beta_columns(beta_df):
    """Identify impression and other beta columns from the beta DataFrame."""
    beta_columns = [col for col in beta_df.columns if col.startswith('Beta_')]
    impression_betas = [col for col in beta_columns if 'impression' in col.lower()]
    other_betas = [col for col in beta_columns if 'impression' not in col.lower()]
    
    return {
        'impression_betas': impression_betas,
        'other_betas': other_betas
    }


def product_to_beta_column(product_name):
    """Convert a product name to its corresponding beta column format."""
    normalized = product_name.lower()
    return f"Beta_{normalized}_meta_impression"


def get_channel_beta_mapping():
    """Return the hardcoded mapping of channel names to beta column names."""
    return {
        'google campaigns': 'Beta_Google_Impression',
        "traffic(all sku's)": 'Beta_Daily_Impressions_OUTCOME_ENGAGEMENT',
        "catalog campaign(all sku's)": None
    }
