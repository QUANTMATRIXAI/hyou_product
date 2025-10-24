"""
Marketing Budget Optimizer - Clean Version
Consolidated app with 2 main tabs and minimal expanders
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.data_utils import (
    load_file, validate_budget_file, validate_cpm_file, validate_beta_file,
    validate_attribution_file, validate_price_file, extract_week_columns, merge_data,
    detect_beta_columns, product_to_beta_column, get_channel_beta_mapping
)
from utils.optimization_utils import (
    distribute_catalog_budget, calculate_impressions, create_impression_dict,
    predict_all_volumes, calculate_revenue, create_objective_function,
    create_bounds, optimize_budgets, calculate_revenue_for_display
)
from utils.results_display import create_comparison_table, format_currency


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_validate_files(budget_file, cpm_file, beta_file, attribution_file, price_file, google_trends_file=None):
    """Load and validate all uploaded files silently."""
    try:
        # Helper to load file from path or uploaded file
        def load_file_smart(file_input):
            if isinstance(file_input, str):
                # It's a file path
                if file_input.endswith('.csv'):
                    return pd.read_csv(file_input)
                else:
                    return pd.read_excel(file_input, engine='openpyxl')
            else:
                # It's an uploaded file
                return load_file(file_input)
        
        budget_df = load_file_smart(budget_file)
        is_valid, error_msg = validate_budget_file(budget_df)
        if not is_valid:
            st.error(f"❌ Budget File Error: {error_msg}")
            return None
        
        cpm_df = load_file_smart(cpm_file)
        is_valid, error_msg = validate_cpm_file(cpm_df)
        if not is_valid:
            st.error(f"❌ CPM File Error: {error_msg}")
            return None
        
        beta_df = load_file_smart(beta_file)
        is_valid, error_msg = validate_beta_file(beta_df)
        if not is_valid:
            st.error(f"❌ Beta File Error: {error_msg}")
            return None
        
        # Attribution file is now optional
        if attribution_file is not None:
            attribution_df = load_file_smart(attribution_file)
            is_valid, error_msg = validate_attribution_file(attribution_df)
            if not is_valid:
                st.error(f"❌ Attribution File Error: {error_msg}")
                return None
        else:
            attribution_df = None
        
        price_df = load_file_smart(price_file)
        is_valid, error_msg = validate_price_file(price_df)
        if not is_valid:
            st.error(f"❌ Price File Error: {error_msg}")
            return None
        
        # Load Google Trends (optional)
        google_trends_df = None
        if google_trends_file is not None:
            google_trends_df = load_file_smart(google_trends_file)
        
        return {
            'budget': budget_df,
            'cpm': cpm_df,
            'beta': beta_df,
            'attribution': attribution_df,
            'price': price_df,
            'google_trends': google_trends_df
        }
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        return None


def prepare_master_dataframe(budget_df, cpm_df, attr_df, price_df, week_col):
    """Merge all data sources."""
    try:
        # Attribution is now optional (None)
        master_df = merge_data(budget_df, cpm_df, attr_df, price_df, week_col)
        return master_df
    except Exception as e:
        st.error(f"❌ Error preparing data: {str(e)}")
        return None


def get_google_trends_value(google_trends_df, selected_week):
    """Extract Google Trends value for the selected week, using cyclical seasonality pattern."""
    if google_trends_df is None:
        return 50.0, "No Google Trends data available", None, None
    
    try:
        from datetime import datetime
        import re
        
        # Parse budget week format (e.g., "6th-12th" means October 6-12, 2025)
        date_match = re.search(r'(\d+)(?:st|nd|rd|th)', selected_week)
        if not date_match:
            return 50.0, "Could not parse week format", None, None
        
        # Extract the start day from the range
        start_day = int(date_match.group(1))
        
        # Budget weeks are in October 2025
        year = 2025
        month = 10  # October
        
        # Create the budget date
        budget_date = datetime(year, month, start_day)
        
        # Convert Google Trends Week column to datetime
        google_trends_df_copy = google_trends_df.copy()
        google_trends_df_copy['date'] = pd.to_datetime(google_trends_df_copy['Week'], format='%d-%m-%Y')
        google_trends_df_copy['week_num'] = google_trends_df_copy['date'].dt.isocalendar().week
        
        # Find the Google Trends week where the date is <= budget_date and closest to it
        # This ensures we use the week that STARTS on or before the budget start date
        valid_weeks = google_trends_df_copy[google_trends_df_copy['date'] <= budget_date]
        
        if len(valid_weeks) > 0:
            # Get the closest date that is <= budget_date
            closest_idx = (budget_date - valid_weeks['date']).abs().idxmin()
            matching_week = google_trends_df_copy.loc[[closest_idx]]
        else:
            # If no weeks before budget date, use cyclical mapping
            budget_week_num = budget_date.isocalendar()[1]
            cyclical_week = ((budget_week_num - 1) % 52) + 1
            matching_week = google_trends_df_copy[google_trends_df_copy['week_num'] == cyclical_week]
        
        budget_week_num = budget_date.isocalendar()[1]
        
        if len(matching_week) > 0:
            # If multiple matches, take the first one
            matching_week = matching_week.iloc[[0]]
            
            # Use SUM of all trend columns (not average)
            trend_cols = [col for col in google_trends_df_copy.columns if col not in ['Week', 'date', 'week_num']]
            sum_value = matching_week[trend_cols].sum(axis=1).values[0]
            matched_date = matching_week['Week'].values[0]
            matched_week_num = matching_week['week_num'].values[0]
            
            # Convert numpy datetime64 to Python datetime for subtraction
            matched_datetime = pd.Timestamp(matching_week['date'].values[0]).to_pydatetime()
            days_diff = (budget_date - matched_datetime).days
            
            message = f"Budget date Oct {start_day}, 2025 (Week {budget_week_num}) → Matched with {matched_date} (Week {matched_week_num})"
            
            return sum_value, message, google_trends_df_copy, budget_week_num
        
        return 50.0, "No matching week found, using default value", google_trends_df_copy, budget_week_num
    except Exception as e:
        # Still return the dataframe even on error
        try:
            google_trends_df_copy = google_trends_df.copy()
            google_trends_df_copy['date'] = pd.to_datetime(google_trends_df_copy['Week'], format='%d-%m-%Y')
            google_trends_df_copy['week_num'] = google_trends_df_copy['date'].dt.isocalendar().week
            return 50.0, f"Error extracting Google Trends: {str(e)}", google_trends_df_copy, None
        except:
            return 50.0, f"Error extracting Google Trends: {str(e)}", None, None


def prepare_contribution_impression_dict(master_df, google_trends_value):
    """
    Prepare impression dictionary for contribution calculation.
    Uses same logic as optimizer to ensure consistency.
    
    This function calculates actual impressions from base budgets using the formula:
    Impressions = Budget / CPM × 1000
    
    This matches exactly what the optimizer uses, ensuring the contribution chart
    reflects the actual budget allocation rather than historical means.
    
    Args:
        master_df: Master dataframe with base_budget, cpm, item_name columns
        google_trends_value: Google Trends value for selected week
    
    Returns:
        impression_dict: Dictionary mapping beta columns to impression values
    """
    from utils.optimization_utils import calculate_impressions, create_impression_dict
    from utils.data_utils import get_channel_beta_mapping, product_to_beta_column
    
    # Extract arrays
    base_budgets = master_df['base_budget'].values
    cpm_values = master_df['cpm'].values
    item_names = master_df['item_name'].values
    
    # Create beta column names (same logic as optimizer)
    beta_column_names = []
    channel_mapping = get_channel_beta_mapping()
    
    for name in item_names:
        name_lower = name.lower()
        if name_lower in channel_mapping:
            beta_column_names.append(channel_mapping[name_lower])
        else:
            beta_column_names.append(product_to_beta_column(name))
    
    # Calculate impressions from base budgets
    impressions = calculate_impressions(base_budgets, cpm_values)
    
    # Create impression dictionary
    impression_dict = create_impression_dict(impressions, beta_column_names)
    
    # Add fixed variables
    impression_dict['Beta_google_trends'] = google_trends_value
    
    return impression_dict


def run_optimization(master_df, beta_df, constraint_pct, google_trends_value=50.0, modeling_data_df=None):
    """Run the complete optimization pipeline."""
    try:
        # Budget file now has "Other Products" directly, no catalog campaign
        item_names = master_df['item_name'].values
        base_budgets = master_df['base_budget'].values
        cpm_values = master_df['cpm'].fillna(1.0).values
        
        # Create price_dict for display purposes (not used in optimization)
        price_dict = {}
        for idx, row in master_df.iterrows():
            if pd.notna(row['price']) and row['price'] > 0:
                price_dict[row['item_name']] = row['price']
        
        # Calculate modeling means for each product
        modeling_means = {}
        if modeling_data_df is not None:
            try:
                # Group by product and calculate means
                product_col = None
                for col in modeling_data_df.columns:
                    if 'product' in col.lower() and 'title' in col.lower():
                        product_col = col
                        break
                
                if product_col is not None:
                    # Get numeric columns (excluding impressions which are handled separately)
                    numeric_cols = modeling_data_df.select_dtypes(include=[np.number]).columns.tolist()
                    exclude_cols = ['date', 'week', 'amount', 'gross', 'net', 'discount', 'sold', 'margin']
                    numeric_cols = [col for col in numeric_cols if not any(ex in col.lower() for ex in exclude_cols)]
                    numeric_cols = [col for col in numeric_cols if 'impression' not in col.lower()]
                    
                    # Calculate means by product
                    means_by_product = modeling_data_df.groupby(product_col)[numeric_cols].mean()
                    
                    # Create a dictionary with Beta_ prefix for each variable
                    for product_name in means_by_product.index:
                        product_means = {}
                        for col in numeric_cols:
                            beta_col_name = f'Beta_{col.replace(" ", "_").lower()}'
                            product_means[beta_col_name] = means_by_product.loc[product_name, col]
                        
                        # Store by product name (lowercase for matching)
                        modeling_means[product_name.lower()] = product_means
                    
                    # For product_variant_price, use the price from price_dict
                    for product_name, price in price_dict.items():
                        if product_name.lower() in modeling_means:
                            modeling_means[product_name.lower()]['Beta_product_variant_price'] = price
            except Exception as e:
                st.warning(f"⚠️ Could not calculate modeling means: {str(e)}")
                modeling_means = {}
        
        beta_column_names = []
        channel_mapping = get_channel_beta_mapping()
        
        for name in item_names:
            name_lower = name.lower()
            if name_lower in channel_mapping:
                beta_column_names.append(channel_mapping[name_lower])
            else:
                beta_column_names.append(product_to_beta_column(name))
        
        # Create volume-based objective function (no price_dict parameter)
        objective_fn = create_objective_function(
            beta_df=beta_df,
            cpm_values=cpm_values,
            item_names=item_names,
            beta_column_names=beta_column_names,
            google_trends_value=google_trends_value,
            modeling_means=modeling_means
        )
        
        # Calculate base volume from objective function
        base_volume = -objective_fn(base_budgets)
        
        # Calculate base revenue for display purposes
        base_revenue, base_volume_verify = calculate_revenue_for_display(
            base_budgets, beta_df, cpm_values, price_dict, item_names,
            beta_column_names, google_trends_value, modeling_means
        )
        
        lower_pct = 1.0 - (constraint_pct / 100.0)
        upper_pct = 1.0 + (constraint_pct / 100.0)
        bounds = create_bounds(base_budgets, lower_pct=lower_pct, upper_pct=upper_pct)
        
        # Add constraint to keep total budget constant
        total_budget = base_budgets.sum()
        constraints = {'type': 'eq', 'fun': lambda x: x.sum() - total_budget}
        
        result = optimize_budgets(objective_fn, base_budgets, bounds, constraints)
        
        # Get optimized volume from result (already positive)
        optimized_volume = result['optimized_volume']
        
        # Calculate optimized revenue for display
        optimized_revenue, optimized_volume_verify = calculate_revenue_for_display(
            result['optimized_budgets'], beta_df, cpm_values, price_dict, item_names,
            beta_column_names, google_trends_value, modeling_means
        )
        
        # Update result dictionary with both volume and revenue metrics
        result['base_volume'] = base_volume
        result['optimized_volume'] = optimized_volume
        result['base_revenue'] = base_revenue
        result['optimized_revenue'] = optimized_revenue
        result['item_names'] = item_names
        result['base_budgets'] = base_budgets
        
        return result
        
    except Exception as e:
        st.error(f"❌ Optimization error: {str(e)}")
        return None


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Media Budget Optimizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💰 Media Budget Optimizer")
    st.markdown("---")
    
    # ========== AUTO-LOAD FILES IF AVAILABLE ==========
    import os
    from pathlib import Path
    
    # Define expected file names
    file_mappings = {
        'budget': ['budget_allocation.csv', 'budget_allocation.xlsx', 'budget.csv', 'budget.xlsx'],
        'cpm': ['cpm.csv', 'cpm.xlsx'],
        'beta': ['betas.csv', '2025-10-20T14-17_export.csv'],
        'attribution': ['catalog_attribution_ratios.csv', 'attribution.csv'],
        'price': ['product_prices.csv', 'prices.csv'],
        'google_trends': ['Seasonality_google_trend_extended.csv', 'Seasonaility_google_trend.csv', 'seasonality_google_trend.csv', 'google_trends.csv'],
        'modeling_data': ['Data_for_model.xlsx', 'data_for_model.xlsx', 'modeling_data.xlsx']
    }
    
    auto_loaded_files = {}
    
    # Try to auto-load files
    for file_type, possible_names in file_mappings.items():
        for filename in possible_names:
            if os.path.exists(filename):
                try:
                    auto_loaded_files[file_type] = filename
                    break
                except:
                    continue
    
    # ========== SIDEBAR: FILE UPLOADS ==========
    with st.sidebar:
        st.header("📁 Data Upload")
        
        if auto_loaded_files:
            st.success(f"✅ Auto-loaded {len(auto_loaded_files)} file(s)")
        
        # Budget file
        if 'budget' in auto_loaded_files:
            st.info(f"📄 Using: {auto_loaded_files['budget']}")
            budget_file = auto_loaded_files['budget']
        else:
            budget_file = st.file_uploader("1️⃣ Budget Allocation", type=["csv", "xlsx"], key="budget")
        
        # CPM file
        if 'cpm' in auto_loaded_files:
            st.info(f"📄 Using: {auto_loaded_files['cpm']}")
            cpm_file = auto_loaded_files['cpm']
        else:
            cpm_file = st.file_uploader("2️⃣ CPM Data", type=["csv", "xlsx"], key="cpm")
        
        # Beta file
        if 'beta' in auto_loaded_files:
            st.info(f"📄 Using: {auto_loaded_files['beta']}")
            beta_file = auto_loaded_files['beta']
        else:
            beta_file = st.file_uploader("3️⃣ Beta Coefficients", type=["csv"], key="beta")
        
        # Attribution file (OPTIONAL - no longer needed)
        attribution_file = None
        st.info("ℹ️ Catalog Attribution file is no longer required (Catalog budget goes to 'Other Products')")
        
        # Price file
        if 'price' in auto_loaded_files:
            st.info(f"📄 Using: {auto_loaded_files['price']}")
            price_file = auto_loaded_files['price']
        else:
            price_file = st.file_uploader("5️⃣ Product Prices", type=["csv"], key="price")
        
        # Google Trends file (optional)
        if 'google_trends' in auto_loaded_files:
            st.info(f"📄 Using: {auto_loaded_files['google_trends']}")
            google_trends_file = auto_loaded_files['google_trends']
        else:
            google_trends_file = st.file_uploader("6️⃣ Google Trends (Optional)", type=["csv"], key="google_trends")
        
        # Modeling Data file (optional)
        if 'modeling_data' in auto_loaded_files:
            st.info(f"📄 Using: {auto_loaded_files['modeling_data']}")
            modeling_data_file = auto_loaded_files['modeling_data']
        else:
            modeling_data_file = st.file_uploader("7️⃣ Modeling Data (Optional)", type=["csv", "xlsx"], key="modeling_data")
        
        st.markdown("---")
        
        files_uploaded = sum([
            budget_file is not None,
            cpm_file is not None,
            beta_file is not None,
            price_file is not None
        ])
        
        st.metric("Files Uploaded", f"{files_uploaded}/4")
        
        st.markdown("---")
        if st.button("🔄 Clear Cache & Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Check if required files uploaded
    if not all([budget_file, cpm_file, beta_file, price_file]):
        st.info("👈 Please upload all 4 required files in the sidebar to continue")
        return
    
    # Load files
    with st.spinner("Loading data files..."):
        data_files = load_and_validate_files(budget_file, cpm_file, beta_file, None, price_file, google_trends_file)
        
        # Load modeling data if provided
        if modeling_data_file is not None:
            try:
                if isinstance(modeling_data_file, str):
                    # It's a file path
                    modeling_data_df = pd.read_excel(modeling_data_file, engine='openpyxl')
                else:
                    # It's an uploaded file
                    modeling_data_df = pd.read_csv(modeling_data_file) if modeling_data_file.name.endswith('.csv') else pd.read_excel(modeling_data_file, engine='openpyxl')
                data_files['modeling_data'] = modeling_data_df
            except Exception as e:
                st.warning(f"⚠️ Could not load modeling data: {str(e)}")
                data_files['modeling_data'] = None
        else:
            data_files['modeling_data'] = None
    
    if data_files is None:
        return
    
    st.success("✅ All data files loaded successfully")
    
    # Initialize session state for optimization results
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    if 'selected_week' not in st.session_state:
        st.session_state.selected_week = None
    if 'constraint_pct' not in st.session_state:
        st.session_state.constraint_pct = 25
    
    # ========== MAIN TABS ==========
    tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📈 Results", "📊 Contribution Analysis"])
    
    # ========== TAB 1: CONFIGURATION ==========
    with tab1:
        st.info("🎯 **Optimization Objective:** This optimizer maximizes total predicted volume across all products, not revenue.")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            week_columns = extract_week_columns(data_files['budget'])
            if not week_columns:
                st.error("❌ No week columns found")
                return
            selected_week = st.selectbox("📅 Select Week:", week_columns, key="week_select")
            st.session_state.selected_week = selected_week
        
        with col2:
            constraint_pct = st.slider("📊 Budget Change Limit (±%):", 5, 50, 25, 5, key="constraint_slider")
            st.session_state.constraint_pct = constraint_pct
        
        with col3:
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
            run_btn = st.button("🚀 Run Optimization", type="primary", use_container_width=True, key="run_opt_btn")
        
        # Budget Editor
        st.markdown("### 💰 Edit Base Budgets")
        master_df_temp = prepare_master_dataframe(
            data_files['budget'], data_files['cpm'],
            data_files['attribution'], data_files['price'], selected_week
        )
        
        if master_df_temp is not None:
            # Create editable dataframe
            if 'edited_budgets' not in st.session_state:
                st.session_state.edited_budgets = master_df_temp[['item_name', 'base_budget']].copy()
            
            edited_df = st.data_editor(
                st.session_state.edited_budgets,
                column_config={
                    "item_name": st.column_config.TextColumn("Product/Channel", disabled=True),
                    "base_budget": st.column_config.NumberColumn("Base Budget ($)", min_value=0, format="$%.2f")
                },
                hide_index=True,
                use_container_width=True,
                key="budget_editor"
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("💾 Save Changes", use_container_width=True):
                    st.session_state.edited_budgets = edited_df.copy()
                    st.success("✅ Budget changes saved!")
            with col2:
                if st.button("🔄 Reset to File", use_container_width=True):
                    st.session_state.edited_budgets = master_df_temp[['item_name', 'base_budget']].copy()
                    st.success("✅ Reset to original values!")
                    st.rerun()
            with col3:
                total_budget = edited_df['base_budget'].sum()
                st.metric("Total Budget", f"${total_budget:,.2f}")
        
        # Expanders for advanced details
        with st.expander("🔍 Advanced: View Data Details"):
            master_df = prepare_master_dataframe(
                data_files['budget'], data_files['cpm'],
                data_files['attribution'], data_files['price'], selected_week
            )
            
            if master_df is not None:
                # Update with edited budgets if available
                if 'edited_budgets' in st.session_state:
                    for idx, row in st.session_state.edited_budgets.iterrows():
                        item_name = row['item_name']
                        new_budget = row['base_budget']
                        master_df.loc[master_df['item_name'] == item_name, 'base_budget'] = new_budget
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Items", len(master_df))
                with col2:
                    st.metric("Products with Prices", len(master_df[master_df['price'] > 0]))
                with col3:
                    st.metric("Total Base Budget", f"${master_df['base_budget'].sum():,.0f}")
                with col4:
                    st.metric("Beta Models", len(data_files['beta']))
                
                st.markdown("**Current Budgets (including any edits):**")
                st.dataframe(master_df, use_container_width=True, height=300)
        
        # Google Trends Seasonality Expander
        if data_files.get('google_trends') is not None:
            with st.expander("📈 Google Trends Seasonality Pattern"):
                google_trends_value, message, trends_df, week_num = get_google_trends_value(data_files['google_trends'], selected_week)
                
                # Calculate cyclical week
                if week_num is not None:
                    cyclical_week = ((week_num - 1) % 52) + 1
                else:
                    cyclical_week = 1
                
                # Show calculation explanation
                st.markdown("### 🎯 How the Google Trends Value is Calculated")
                
                if week_num is not None and week_num > 52:
                    st.warning(f"**Budget Week {week_num}** exceeds 52 weeks. Using cyclical mapping: Week {week_num} → Week {cyclical_week}")
                elif week_num is not None:
                    st.info(f"**Budget Week {week_num}** maps directly to Week {cyclical_week}")
                else:
                    st.info(f"**Using matched week data**")
                
                # Always show the basic info
                st.markdown(f"**Google Trends Value:** {google_trends_value:.2f}")
                st.markdown(f"**Matching Info:** {message}")
                
                if trends_df is not None:
                    # Calculate average trend value per week
                    trend_cols = [col for col in trends_df.columns if col not in ['Week', 'date', 'date_diff', 'week_num']]
                    trends_df['avg_trend'] = trends_df[trend_cols].mean(axis=1)
                    
                    # Get the selected row (find the one closest to budget date that is <= budget date)
                    from datetime import datetime
                    import re
                    date_match = re.search(r'(\d+)(?:st|nd|rd|th)', selected_week)
                    if date_match:
                        start_day = int(date_match.group(1))
                        budget_date = datetime(2025, 10, start_day)
                        
                        valid_rows = trends_df[trends_df['date'] <= budget_date]
                        if len(valid_rows) > 0:
                            closest_idx = (budget_date - valid_rows['date']).dt.days.idxmin()
                            selected_row = trends_df.loc[[closest_idx]]
                            selected_idx = closest_idx
                        else:
                            selected_row = None
                            selected_idx = None
                    else:
                        selected_row = None
                        selected_idx = None
                    
                    st.markdown(f"**Calculation Steps:**")
                    st.markdown(f"1. **Budget Week:** {selected_week} (October 2025)")
                    st.markdown(f"2. **Google Trends Value:** {google_trends_value:.2f}")
                    st.markdown(f"3. **Message:** {message}")
                    st.markdown(f"4. **Product Categories:** {len(trend_cols)} categories")
                    
                    if selected_row is not None and len(selected_row) > 0:
                        matched_date_str = selected_row['Week'].values[0]
                        matched_week_num = selected_row['week_num'].values[0]
                        st.markdown(f"5. **Matched Google Trends Date:** {matched_date_str} (Week {matched_week_num})")
                    
                    # Show individual category values
                    if selected_row is not None and len(selected_row) > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Individual Category Values:**")
                            for i, col in enumerate(trend_cols[:6]):  # Show first 6
                                val = selected_row[col].values[0]
                                st.text(f"  • {col.split(':')[0]}: {val}")
                        
                        with col2:
                            if len(trend_cols) > 6:
                                st.markdown("&nbsp;")
                                for col in trend_cols[6:]:  # Show remaining
                                    val = selected_row[col].values[0]
                                    st.text(f"  • {col.split(':')[0]}: {val}")
                    
                    st.markdown(f"**Sum Calculation:** Sum of all {len(trend_cols)} categories = **{google_trends_value:.2f}**")
                    st.success(f"✅ **Final Google Trends Value Used: {google_trends_value:.2f}**")
                    
                    st.markdown("---")
                    
                    # Show full trend data table
                    st.markdown("### 📊 Full Google Trends Data (Week 1 to Latest)")
                    
                    # Prepare display dataframe with all columns
                    display_df = trends_df[['week_num', 'Week', 'date', 'avg_trend'] + trend_cols].copy()
                    display_df = display_df.sort_values('date').reset_index(drop=True)
                    
                    st.dataframe(display_df, use_container_width=True, height=400)
                    
                    if selected_
