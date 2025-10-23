"""
Beta Mapping Module

This module provides functions to map products and channels to beta coefficient columns
in the beta file, enabling the volume prediction pipeline to correctly identify which
beta coefficients to use for each product and channel.
"""

import pandas as pd


def detect_beta_columns(beta_df):
    """
    Identify impression and other beta columns from the beta DataFrame.
    
    Separates beta columns into two categories:
    - impression_betas: Beta columns containing "impression" in the name
    - other_betas: Beta columns not containing "impression"
    
    Args:
        beta_df (pd.DataFrame): DataFrame containing beta coefficients with columns
                                starting with "Beta_"
    
    Returns:
        dict: Dictionary with keys 'impression_betas' and 'other_betas', each containing
              a list of column names
    
    Requirements: 5.1, 5.2
    """
    # Find all columns starting with "Beta_"
    beta_columns = [col for col in beta_df.columns if col.startswith('Beta_')]
    
    # Separate into impression and other betas
    impression_betas = [col for col in beta_columns if 'impression' in col.lower()]
    other_betas = [col for col in beta_columns if 'impression' not in col.lower()]
    
    return {
        'impression_betas': impression_betas,
        'other_betas': other_betas
    }

def product_to_beta_column(product_name):
    """
    Convert a product name to its corresponding beta column format.
    
    Transformation rules:
    1. Convert to lowercase
    2. Replace spaces with underscores  # <-- ADD THIS
    3. Add "Beta_" prefix
    4. Add "_meta_impression" suffix
    """
    # Convert to lowercase and replace spaces with underscores
    # NEW (correct):
    normalized = product_name.lower()  # Keep spaces!
    # Creates: Beta_contemporary open diamond band_meta_impression

        
    # Add prefix and suffix
    beta_column = f"Beta_{normalized}_meta_impression"
    
    return beta_column



def get_channel_beta_mapping():
    """
    Return the hardcoded mapping of channel names to beta column names.
    
    Channel mappings:
    - "Google Campaigns" -> "Beta_Google_Impression"
    - "Traffic(All Sku's)" -> "Beta_Daily_Impressions_OUTCOME_ENGAGEMENT"
    - "Catalog Campaign(All sku's)" -> None (distributed to products)
    
    Returns:
        dict: Dictionary mapping channel names (lowercase) to beta column names (or None)
    
    Requirements: 5.4, 5.5
    """
    return {
        'google campaigns': 'Beta_Google_Impression',
        "traffic(all sku's)": 'Beta_Daily_Impressions_OUTCOME_ENGAGEMENT',
        "catalog campaign(all sku's)": None  # Distributed to products via attribution
    }
