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
    create_bounds, optimize_budgets
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


def run_optimization(master_df, beta_df, constraint_pct, google_trends_value=50.0, modeling_data_df=None):
    """Run the complete optimization pipeline."""
    try:
        # Budget file now has "Other Products" directly, no catalog campaign
        item_names = master_df['item_name'].values
        base_budgets = master_df['base_budget'].values
        cpm_values = master_df['cpm'].fillna(1.0).values
        
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
        
        objective_fn = create_objective_function(
            beta_df=beta_df,
            cpm_values=cpm_values,
            price_dict=price_dict,
            item_names=item_names,
            beta_column_names=beta_column_names,
            google_trends_value=google_trends_value,
            modeling_means=modeling_means
        )
        
        base_revenue = -objective_fn(base_budgets)
        
        lower_pct = 1.0 - (constraint_pct / 100.0)
        upper_pct = 1.0 + (constraint_pct / 100.0)
        bounds = create_bounds(base_budgets, lower_pct=lower_pct, upper_pct=upper_pct)
        
        # Add constraint to keep total budget constant
        total_budget = base_budgets.sum()
        constraints = {'type': 'eq', 'fun': lambda x: x.sum() - total_budget}
        
        result = optimize_budgets(objective_fn, base_budgets, bounds, constraints)
        
        result['base_revenue'] = base_revenue
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
                    
                    if selected_idx is not None:
                        st.info(f"⭐ **Selected Week:** Week {cyclical_week} is highlighted in the chart above")
                    
                    st.markdown("---")
                    
                    # Create visualization
                    st.markdown("### 📈 Seasonality Trend Visualization")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trends_df['date'],
                        y=trends_df['avg_trend'],
                        mode='lines+markers',
                        name='Average Trend',
                        line=dict(color='blue', width=2),
                        marker=dict(size=6),
                        hovertemplate='Date: %{x|%b %d, %Y}<br>Trend: %{y:.2f}<extra></extra>'
                    ))
                    
                    # Highlight selected week - recalculate to ensure we have it
                    try:
                        from datetime import datetime
                        import re
                        date_match = re.search(r'(\d+)(?:st|nd|rd|th)', selected_week)
                        if date_match:
                            start_day = int(date_match.group(1))
                            budget_date = datetime(2025, 10, start_day)
                            
                            valid_rows = trends_df[trends_df['date'] <= budget_date]
                            if len(valid_rows) > 0:
                                closest_idx = (budget_date - valid_rows['date']).dt.days.idxmin()
                                highlight_row = trends_df.loc[[closest_idx]]
                                
                                if len(highlight_row) > 0:
                                    matched_date_str = highlight_row['Week'].values[0]
                                    matched_date_val = highlight_row['date'].values[0]
                                    matched_trend_val = highlight_row['avg_trend'].values[0]
                                    
                                    st.write(f"DEBUG: Adding red star at date={matched_date_val}, value={matched_trend_val}")
                                    
                                    fig.add_trace(go.Scatter(
                                        x=[matched_date_val],
                                        y=[matched_trend_val],
                                        mode='markers',
                                        name=f'Selected: {selected_week}',
                                        marker=dict(size=20, color='red', symbol='star'),
                                        hovertemplate=f'Budget Week: {selected_week}<br>Matched Date: {matched_date_str}<br>Value: {google_trends_value:.2f}<extra></extra>'
                                    ))
                                else:
                                    st.warning("DEBUG: highlight_row is empty")
                            else:
                                st.warning("DEBUG: No valid rows found")
                        else:
                            st.warning("DEBUG: Could not parse date from selected_week")
                    except Exception as e:
                        st.error(f"DEBUG: Error adding marker: {str(e)}")
                    
                    fig.update_layout(
                        title=f"Google Trends Seasonality Pattern (Highlighted: {selected_week})",
                        xaxis_title="Date",
                        yaxis_title="Average Trend Value",
                        height=400,
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Modeling Data Variables Expander
        if data_files.get('modeling_data') is not None:
            with st.expander("📊 Modeling Data - Variable Means by Product"):
                modeling_df = data_files['modeling_data']
                
                st.markdown("### Average Values of Model Variables by Product")
                st.info("These are the mean values of variables used in the MMM model for each product")
                
                # Check if Product column exists
                product_col = None
                for col in modeling_df.columns:
                    if 'product' in col.lower():
                        product_col = col
                        break
                
                if product_col is not None:
                    # Get numeric columns
                    numeric_cols = modeling_df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    # Exclude index columns
                    numeric_cols = [col for col in numeric_cols if not col.lower().startswith('unnamed')]
                    
                    if len(numeric_cols) > 0:
                        # Separate columns into two groups:
                        # 1. Global means (impressions, google_trends, Category Discount) - use "Other Products" mean
                        # 2. Product-specific means (everything else) - use product-specific mean
                        
                        global_mean_cols = []
                        product_specific_cols = []
                        
                        for col in numeric_cols:
                            col_lower = col.lower()
                            if 'impression' in col_lower or 'google_trends' in col_lower or 'category discount' in col_lower:
                                global_mean_cols.append(col)
                            else:
                                product_specific_cols.append(col)
                        
                        # Get "Other Products" data for global means
                        other_products_data = modeling_df[modeling_df[product_col].str.lower() == 'other products']
                        
                        if len(other_products_data) > 0:
                            # Calculate global means from "Other Products"
                            # All use MEAN (impressions, google_trends, Category Discount)
                            global_means = {}
                            for col in global_mean_cols:
                                global_means[col] = other_products_data[col].mean()
                            
                            # Calculate product-specific means
                            product_means_df = modeling_df.groupby(product_col)[product_specific_cols].mean().reset_index()
                            
                            # Add global means to each product row
                            for col in global_mean_cols:
                                product_means_df[col] = global_means[col]
                            
                            # Reorder columns to match original order
                            final_cols = [product_col] + [col for col in numeric_cols if col in product_means_df.columns]
                            means_df = product_means_df[final_cols]
                            
                            st.markdown(f"**Products found:** {len(means_df)}")
                            st.markdown(f"**Variables:** {len(numeric_cols)}")
                            st.info(f"ℹ️ **Note:** Impression variables, Google Trends, and Category Discount use MEAN from 'Other Products' (global) | Other variables use product-specific means")
                            
                            # Show the means table
                            st.dataframe(means_df, use_container_width=True, height=400)
                            
                            # Show summary statistics
                            st.markdown("### Summary Statistics")
                            summary_df = means_df[numeric_cols].describe()
                            st.dataframe(summary_df, use_container_width=True)
                        else:
                            st.error("'Other Products' not found in modeling data. Cannot calculate global means.")
                    else:
                        st.warning("No numeric columns found in modeling data")
                else:
                    st.warning("No product column found in modeling data. Please ensure the file has a 'Product' column.")
        
        # Equation Breakdown Expander
        with st.expander("🔢 Volume Prediction Equations by Product"):
            st.markdown("### Complete MMM Equations for Each Product")
            st.info("Shows the full equation with beta coefficients and their values (🔄 Variable = changes with budget, 🔒 Fixed = constant)")
            
            # Get beta file and modeling data
            beta_df = data_files['beta']
            modeling_df = data_files.get('modeling_data')
            
            # Get Google Trends value for the selected week
            google_trends_value, _, _, _ = get_google_trends_value(data_files.get('google_trends'), selected_week)
            
            # Prepare master dataframe to get prices
            master_df_temp = prepare_master_dataframe(
                data_files['budget'], data_files['cpm'],
                data_files['attribution'], data_files['price'], selected_week
            )
            
            # For each product in beta file
            for idx, row in beta_df.iterrows():
                product_name = row.get('Product title', f'Product_{idx}')
                
                if pd.isna(product_name) or product_name == '':
                    continue
                
                st.markdown(f"#### **{product_name}**")
                
                # Get B0 (intercept)
                b0_value = None
                for col in row.index:
                    if col.startswith('B0'):
                        b0_value = row[col]
                        break
                
                # Start equation
                equation_parts = []
                if b0_value is not None and pd.notna(b0_value):
                    equation_parts.append(f"**{b0_value:.4f}** (Intercept)")
                
                # Get all Beta columns
                variable_terms = []
                fixed_terms = []
                
                for col in row.index:
                    if not col.startswith('Beta_'):
                        continue
                    
                    beta_value = row[col]
                    if pd.isna(beta_value) or beta_value == 0:
                        continue
                    
                    # Determine if variable or fixed
                    if 'impression' in col.lower():
                        # Variable - changes with budget
                        var_name = col.replace('Beta_', '').replace('_', ' ').title()
                        variable_terms.append(f"🔄 **{beta_value:.6f}** × {var_name}")
                    elif 'google_trends' in col.lower():
                        # Fixed - Google Trends value
                        fixed_terms.append(f"🔒 **{beta_value:.6f}** × Google Trends (≈ {google_trends_value:.2f})")
                    else:
                        # Fixed - from modeling data means or product prices
                        var_name = col.replace('Beta_', '').replace('_', ' ').title()
                        original_col_name = col.replace('Beta_', '')
                        
                        # Try to get mean value
                        mean_value = "N/A"
                        
                        # Special case: product_variant_price - get from price_dict
                        if 'product_variant_price' in col.lower():
                            # Get price from master_df
                            try:
                                price_row = master_df_temp[master_df_temp['item_name'].str.lower() == product_name.lower()]
                                if len(price_row) > 0 and 'price' in price_row.columns:
                                    price_val = price_row['price'].values[0]
                                    if pd.notna(price_val):
                                        mean_value = f"{price_val:.2f}"
                            except:
                                pass
                        
                        # Try to get from modeling data
                        elif modeling_df is not None:
                            try:
                                product_col = None
                                for c in modeling_df.columns:
                                    if 'product' in c.lower() and 'title' in c.lower():
                                        product_col = c
                                        break
                                
                                if product_col is not None:
                                    product_data = modeling_df[modeling_df[product_col].str.lower() == product_name.lower()]
                                    
                                    if len(product_data) > 0:
                                        # Try exact match first
                                        if original_col_name in modeling_df.columns:
                                            mean_value = f"{product_data[original_col_name].mean():.4f}"
                                        else:
                                            # Try case-insensitive match with space/underscore variations
                                            for c in modeling_df.columns:
                                                c_normalized = c.lower().replace(' ', '_')
                                                col_normalized = original_col_name.lower().replace(' ', '_')
                                                if c_normalized == col_normalized:
                                                    mean_value = f"{product_data[c].mean():.4f}"
                                                    break
                            except:
                                pass
                        
                        fixed_terms.append(f"🔒 **{beta_value:.6f}** × {var_name} (≈ {mean_value})")
                
                # Display equation
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown("**Equation:**")
                with col2:
                    full_equation = " + ".join(equation_parts + variable_terms + fixed_terms)
                    st.markdown(f"Volume = {full_equation}")
                
                # Show breakdown
                if variable_terms:
                    st.markdown("**🔄 Variable Terms (optimized):**")
                    for term in variable_terms:
                        st.markdown(f"  - {term}")
                
                if fixed_terms:
                    st.markdown("**🔒 Fixed Terms (constant):**")
                    for term in fixed_terms:
                        st.markdown(f"  - {term}")
                
                st.markdown("---")
        
        # Beta Coefficients and Mean Values Table
        with st.expander("📊 Beta Coefficients & Mean Values Table"):
            st.markdown("### Complete Beta and Mean Values for All Products")
            st.info("Rows = Products with models | Columns = Variables | Values = Beta coefficient / Mean value")
            
            # Get beta file and modeling data
            beta_df = data_files['beta']
            modeling_df = data_files.get('modeling_data')
            
            # Prepare master dataframe to get prices
            master_df_temp = prepare_master_dataframe(
                data_files['budget'], data_files['cpm'],
                data_files['attribution'], data_files['price'], selected_week
            )
            
            if modeling_df is not None:
                # Get product column from modeling data
                product_col = None
                for col in modeling_df.columns:
                    if 'product' in col.lower() and 'title' in col.lower():
                        product_col = col
                        break
                
                if product_col is not None:
                    # Build the table
                    table_data = []
                    
                    # Get Google Trends value for the selected week
                    google_trends_value_temp, _, _, _ = get_google_trends_value(data_files.get('google_trends'), selected_week)
                    
                    # Get "Other Products" data for global means (impressions, google_trends, Category Discount)
                    other_products_data = modeling_df[modeling_df[product_col].str.lower() == 'other products']
                    
                    # For each product in beta file
                    for idx, beta_row in beta_df.iterrows():
                        product_name = beta_row.get('Product title', '')
                        if pd.isna(product_name) or product_name == '':
                            continue
                        
                        row_data = {'Product': product_name}
                        
                        # Get product data from modeling file (for product-specific means)
                        product_data = modeling_df[modeling_df[product_col].str.lower() == product_name.lower()]
                        
                        # Get product price
                        product_price = 0
                        if master_df_temp is not None:
                            price_row = master_df_temp[master_df_temp['item_name'].str.lower() == product_name.lower()]
                            if len(price_row) > 0 and 'price' in price_row.columns:
                                product_price = price_row['price'].values[0]
                        
                        # Get B0 (Intercept)
                        for col in beta_row.index:
                            if col.startswith('B0'):
                                b0_value = beta_row[col]
                                if pd.notna(b0_value):
                                    row_data['B0 (Intercept)'] = f"β={b0_value:.4f}"
                                break
                        
                        # Process all Beta columns
                        for col in beta_row.index:
                            if not col.startswith('Beta_'):
                                continue
                            
                            beta_value = beta_row[col]
                            if pd.isna(beta_value):
                                continue
                            
                            # Get the variable name
                            var_name = col.replace('Beta_', '')
                            display_name = var_name.replace('_', ' ').title()
                            
                            # Get mean value
                            mean_value = None
                            
                            # Special case: product_variant_price (use product-specific price)
                            if 'product_variant_price' in var_name.lower() or 'product variant price' in var_name.lower():
                                mean_value = product_price
                            
                            # Special case: google_trends (use selected week value)
                            elif 'google_trends' in var_name.lower() or 'google trends' in var_name.lower():
                                mean_value = google_trends_value_temp
                            
                            # For impressions and Category Discount: use "Other Products" mean (global)
                            elif 'impression' in var_name.lower() or 'category discount' in var_name.lower():
                                if len(other_products_data) > 0:
                                    # Try multiple matching strategies
                                    found = False
                                    
                                    # Strategy 1: Exact match
                                    if var_name in modeling_df.columns:
                                        mean_value = other_products_data[var_name].mean()
                                        found = True
                                    
                                    # Strategy 2: Case-insensitive with underscore/space
                                    if not found:
                                        for data_col in modeling_df.columns:
                                            col_normalized = data_col.lower().replace(' ', '_').replace('-', '_')
                                            var_normalized = var_name.lower().replace(' ', '_').replace('-', '_')
                                            if col_normalized == var_normalized:
                                                mean_value = other_products_data[data_col].mean()
                                                found = True
                                                break
                            
                            # For other variables: use product-specific mean
                            elif len(product_data) > 0:
                                # Try multiple matching strategies
                                found = False
                                
                                # Strategy 1: Exact match
                                if var_name in modeling_df.columns:
                                    mean_value = product_data[var_name].mean()
                                    found = True
                                
                                # Strategy 2: Case-insensitive with underscore/space
                                if not found:
                                    for data_col in modeling_df.columns:
                                        col_normalized = data_col.lower().replace(' ', '_').replace('-', '_')
                                        var_normalized = var_name.lower().replace(' ', '_').replace('-', '_')
                                        if col_normalized == var_normalized:
                                            mean_value = product_data[data_col].mean()
                                            found = True
                                            break
                            
                            # Format the cell value
                            if mean_value is not None and pd.notna(mean_value):
                                row_data[display_name] = f"β={beta_value:.6f} | μ={mean_value:.4f}"
                            else:
                                row_data[display_name] = f"β={beta_value:.6f} | μ=N/A"
                        
                        table_data.append(row_data)
                    
                    # Create DataFrame
                    if table_data:
                        table_df = pd.DataFrame(table_data)
                        
                        # Fill NaN with empty string
                        table_df = table_df.fillna('')
                        
                        st.markdown(f"**{len(table_df)} Products × {len(table_df.columns)-1} Variables**")
                        st.markdown("**Legend:** β = Beta coefficient | μ = Mean value")
                        st.info("ℹ️ **Mean Value Sources:** Impressions & Category Discount use 'Other Products' mean (global) | Product Price uses product-specific value | Google Trends uses selected week value | Other variables use product-specific means")
                        
                        # Display the table
                        st.dataframe(
                            table_df.set_index('Product'),
                            use_container_width=True,
                            height=600
                        )
                        
                        # Add download button
                        csv = table_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Table as CSV",
                            data=csv,
                            file_name="beta_mean_values_table.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No data to display")
                else:
                    st.error("Could not find product column in modeling data")
            else:
                st.warning("⚠️ Please upload modeling data file to see this table")
        
        if run_btn:
            # Run optimization and store in session state
            with st.spinner("Running optimization..."):
                master_df_temp = prepare_master_dataframe(
                    data_files['budget'], data_files['cpm'],
                    data_files['attribution'], data_files['price'], selected_week
                )
                
                if master_df_temp is not None:
                    # Use edited budgets if available
                    if 'edited_budgets' in st.session_state:
                        # Update master_df with edited budgets
                        for idx, row in st.session_state.edited_budgets.iterrows():
                            item_name = row['item_name']
                            new_budget = row['base_budget']
                            master_df_temp.loc[master_df_temp['item_name'] == item_name, 'base_budget'] = new_budget
                    
                    google_trends_value, message, _, _ = get_google_trends_value(data_files.get('google_trends'), selected_week)
                    result_temp = run_optimization(master_df_temp, data_files['beta'], constraint_pct, google_trends_value, data_files.get('modeling_data'))
                    
                    if result_temp is not None:
                        st.session_state.optimization_result = result_temp
                        st.success("✅ Optimization Complete! 👉 Go to the **Results** tab to view your optimized budget allocation")
                        # st.balloons()
                    else:
                        st.error("❌ Optimization failed. Please check your data and try again.")
                else:
                    st.error("❌ Data preparation failed. Please check your files.")
        
        if st.session_state.optimization_result is None:
            st.info("👆 Click 'Run Optimization' to see results in the Results tab")
    
    # ========== TAB 2: RESULTS ==========
    with tab2:
        if st.session_state.optimization_result is None:
            st.info("👈 Please run optimization in the Configuration tab first")
            return
        
        # Get results from session state
        result = st.session_state.optimization_result
        
        # # Display results
        # base_revenue = result['base_revenue']
        # optimized_revenue = result['optimized_revenue']
        # revenue_increase = optimized_revenue - base_revenue
        # revenue_increase_pct = (revenue_increase / base_revenue * 100) if base_revenue > 0 else 0
        
        # # # # Commented out for cleaner interface
        # # # if result['success']:
        # # #     st.success(f"✅ Optimization completed in {result['iterations']} iterations")
        # # # else:
        # # #     st.warning(f"⚠️ {result['message']}")
        
        # # # # Debug: Check budget constraint
        # # # total_base = result['base_budgets'].sum()
        # # # total_opt = result['optimized_budgets'].sum()
        # # # budget_diff = abs(total_opt - total_base)
        
        # # # if budget_diff > 1.0:
        # # #     st.warning(f"⚠️ Total budget changed by ${budget_diff:,.2f} (Base: ${total_base:,.2f}, Optimized: ${total_opt:,.2f})")
        # # # else:
        # # #     st.info(f"✅ Total budget maintained: ${total_base:,.2f}")
        
        # # # # Check how many items hit bounds
        # # # hitting_lower = np.sum(np.abs(result['optimized_budgets'] - result['base_budgets'] * (1 - constraint_pct/100)) < 1.0)
        # # # hitting_upper = np.sum(np.abs(result['optimized_budgets'] - result['base_budgets'] * (1 + constraint_pct/100)) < 1.0)
        
        # # # if hitting_lower > 0 or hitting_upper > 0:
        # # #     st.info(f"📊 {hitting_upper} items increased to max (+{constraint_pct}%), {hitting_lower} items decreased to max (-{constraint_pct}%)")
        
        # # # Metrics
        # # col1, col2, col3 = st.columns(3)
        # # with col1:
        # #     st.metric("Base Revenue", format_currency(base_revenue))
        # # with col2:
        # #     st.metric("Optimized Revenue", format_currency(optimized_revenue), delta=format_currency(revenue_increase))
        # # with col3:
        # #     st.metric("Revenue Increase", f"{revenue_increase_pct:.2f}%")
        
        # # st.markdown("---")
        
        # Results tabs
        result_tab1, result_tab2, result_tab3 = st.tabs(["💰 Budget Changes", "📊 Visualizations", "📋 Detailed Table"])
        
        comparison_df = create_comparison_table(result['item_names'], result['base_budgets'], result['optimized_budgets'])
        
        with result_tab1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Base Budget", format_currency(result['base_budgets'].sum()))
            with col2:
                st.metric("Total Optimized Budget", format_currency(result['optimized_budgets'].sum()))
            with col3:
                st.metric("Avg Change", f"{comparison_df['Change (%)'].abs().mean():.2f}%")
            with col4:
                st.metric("Max Increase", f"{comparison_df['Change (%)'].max():.2f}%")
            
            st.dataframe(comparison_df, use_container_width=True, height=500)
        
        with result_tab2:
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Base', x=comparison_df['Item'], y=comparison_df['Base Budget'], marker_color='lightblue'))
            fig.add_trace(go.Bar(name='Optimized', x=comparison_df['Item'], y=comparison_df['Optimized Budget'], marker_color='darkblue'))
            fig.update_layout(title="Budget Allocation Comparison", barmode='group', height=500, xaxis={'tickangle': -45})
            st.plotly_chart(fig, use_container_width=True)
        
        with result_tab3:
            st.dataframe(
                comparison_df.style.format({
                    'Base Budget': lambda x: f"${x:,.2f}",
                    'Optimized Budget': lambda x: f"${x:,.2f}",
                    'Change ($)': lambda x: f"${x:,.2f}",
                    'Change (%)': lambda x: f"{x:.2f}%"
                }).background_gradient(subset=['Change (%)'], cmap='RdYlGn', vmin=-25, vmax=25),
                use_container_width=True,
                height=600
            )
    
    # ========== TAB 3: CONTRIBUTION ANALYSIS ==========
    with tab3:
        # Check if required data is available
        if data_files.get('modeling_data') is None:
            st.warning("⚠️ Please upload modeling data file to see contribution analysis")
            return
        
        # Get required data
        beta_df = data_files['beta']
        modeling_df = data_files['modeling_data']
        
        # Prepare master dataframe for prices
        master_df_contrib = prepare_master_dataframe(
            data_files['budget'], data_files['cpm'],
            data_files['attribution'], data_files['price'], selected_week
        )
        
        # Apply edited budgets if available (same as optimizer)
        if 'edited_budgets' in st.session_state:
            for idx, row in st.session_state.edited_budgets.iterrows():
                item_name = row['item_name']
                new_budget = row['base_budget']
                master_df_contrib.loc[master_df_contrib['item_name'] == item_name, 'base_budget'] = new_budget
        
        # Get Google Trends value
        google_trends_value_contrib, _, _, _ = get_google_trends_value(data_files.get('google_trends'), selected_week)
        
        # Get "Other Products" data for global means
        product_col = None
        for col in modeling_df.columns:
            if 'product' in col.lower() and 'title' in col.lower():
                product_col = col
                break
        
        if product_col is None:
            st.error("Could not find product column in modeling data")
            return
        
        other_products_data = modeling_df[modeling_df[product_col].str.lower() == 'other products']
        
        # Create two sub-tabs
        contrib_tab1, contrib_tab2 = st.tabs(["📊 Portfolio Analysis", "📋 Product Analysis"])
        
        # ========== PORTFOLIO LEVEL CONTRIBUTION ==========
        with contrib_tab1:
            try:
                # Dictionary to store contributions by variable (REVENUE-based)
                variable_contributions = {}
                
                # For each product in beta file
                for idx, beta_row in beta_df.iterrows():
                    product_name = beta_row.get('Product title', '')
                    if pd.isna(product_name) or product_name == '':
                        continue
                    
                    # Get product-specific data
                    product_data = modeling_df[modeling_df[product_col].str.lower() == product_name.lower()]
                    
                    # Get product price (CRITICAL for revenue calculation)
                    product_price = 0
                    if master_df_contrib is not None:
                        price_row = master_df_contrib[master_df_contrib['item_name'].str.lower() == product_name.lower()]
                        if len(price_row) > 0 and 'price' in price_row.columns:
                            product_price = price_row['price'].values[0]
                    
                    # Skip products with no price (can't calculate revenue)
                    if product_price == 0 or pd.isna(product_price):
                        continue
                    
                    # Process all Beta columns (same logic as Beta table)
                    for col in beta_row.index:
                        if not col.startswith('Beta_'):
                            continue
                        
                        beta_value = beta_row[col]
                        if pd.isna(beta_value):
                            continue
                        
                        # Get the variable name
                        var_name = col.replace('Beta_', '')
                        display_name = var_name.replace('_', ' ').title()
                        
                        # Get mean value (SAME LOGIC AS BETA TABLE)
                        mean_value = None
                        
                        # Special case: product_variant_price (use product-specific price)
                        if 'product_variant_price' in var_name.lower() or 'product variant price' in var_name.lower():
                            mean_value = product_price
                        
                        # Special case: google_trends (use selected week value)
                        elif 'google_trends' in var_name.lower() or 'google trends' in var_name.lower():
                            mean_value = google_trends_value_contrib
                        
                        # For impressions and Category Discount: use "Other Products" mean (global)
                        elif 'impression' in var_name.lower() or 'category discount' in var_name.lower():
                            if len(other_products_data) > 0:
                                # Try matching
                                found = False
                                if var_name in modeling_df.columns:
                                    mean_value = other_products_data[var_name].mean()
                                    found = True
                                
                                if not found:
                                    for data_col in modeling_df.columns:
                                        col_normalized = data_col.lower().replace(' ', '_').replace('-', '_')
                                        var_normalized = var_name.lower().replace(' ', '_').replace('-', '_')
                                        if col_normalized == var_normalized:
                                            mean_value = other_products_data[data_col].mean()
                                            found = True
                                            break
                        
                        # For other variables: use product-specific mean
                        elif len(product_data) > 0:
                            found = False
                            if var_name in modeling_df.columns:
                                mean_value = product_data[var_name].mean()
                                found = True
                            
                            if not found:
                                for data_col in modeling_df.columns:
                                    col_normalized = data_col.lower().replace(' ', '_').replace('-', '_')
                                    var_normalized = var_name.lower().replace(' ', '_').replace('-', '_')
                                    if col_normalized == var_normalized:
                                        mean_value = product_data[data_col].mean()
                                        found = True
                                        break
                        
                        # Calculate Beta × Mean × Price = REVENUE CONTRIBUTION
                        if mean_value is not None and pd.notna(mean_value):
                            volume_contribution = beta_value * mean_value
                            revenue_contribution = volume_contribution * product_price  # Convert to revenue!
                            
                            if display_name not in variable_contributions:
                                variable_contributions[display_name] = 0
                            variable_contributions[display_name] += revenue_contribution
                
                # Check if we have any contributions
                if len(variable_contributions) == 0:
                    st.warning("⚠️ No revenue contributions calculated. This might be because:")
                    st.markdown("- Products don't have prices in the price file")
                    st.markdown("- No products match between budget file and beta file")
                    st.markdown("- All products were filtered out")
                    return
                
                # Separate impression and non-impression variables
                impression_vars = {k: v for k, v in variable_contributions.items() if 'impression' in k.lower()}
                non_impression_vars = {k: v for k, v in variable_contributions.items() if 'impression' not in k.lower()}
                
                # Calculate total using ONLY impression variables as denominator
                total_impression_contribution = sum(impression_vars.values())
                
                # Create dataframe with all variables
                contrib_data = []
                for var_name, contribution in variable_contributions.items():
                    # Calculate percentage using impression total as denominator
                    contrib_pct = (contribution / total_impression_contribution * 100) if total_impression_contribution != 0 else 0
                    is_impression = 'impression' in var_name.lower()
                    contrib_data.append({
                        'Variable': var_name,
                        'Revenue Contribution': contribution,
                        'Contribution %': contrib_pct,
                        'Type': 'Impression' if is_impression else 'Other'
                    })
                
                contrib_df = pd.DataFrame(contrib_data)
                contrib_df = contrib_df.sort_values('Contribution %', ascending=False)
                
                # Get all variable names
                all_variables = contrib_df['Variable'].tolist()
                impression_variables = contrib_df[contrib_df['Type'] == 'Impression']['Variable'].tolist()
                non_impression_variables = contrib_df[contrib_df['Type'] == 'Other']['Variable'].tolist()
                
                # Multiselect to REMOVE variables (default: remove non-impression variables)
                st.markdown("### 🎯 Remove Variables (Optional)")
                vars_to_remove = st.multiselect(
                    "Select variables to exclude from analysis:",
                    options=all_variables,
                    default=non_impression_variables,
                    help="By default, only impression variables are shown. You can add back other variables by deselecting them."
                )
                
                # Filter dataframe - show all EXCEPT removed ones
                filtered_df = contrib_df[~contrib_df['Variable'].isin(vars_to_remove)].copy()
                
                if len(filtered_df) == 0:
                    st.warning("You've removed all variables. Please keep at least one variable.")
                else:
                    # Recalculate percentages based on FILTERED variables (excluding removed ones)
                    total_filtered_contribution = filtered_df['Revenue Contribution'].sum()
                    filtered_df['Contribution %'] = (filtered_df['Revenue Contribution'] / total_filtered_contribution * 100) if total_filtered_contribution != 0 else 0
                    
                    # Show total revenue metric
                    # st.metric("💰 Total Revenue Contribution (Shown Variables)", f"${total_filtered_contribution:,.2f}")
                    # st.caption(f"This is the sum of revenue contributions from {len(filtered_df)} variables shown below (excluding {len(vars_to_remove)} removed variables)")
                    
                    # Create display names for better presentation
                    def get_display_name(var_name):
                        if 'Google Impression' in var_name or 'Google_Impression' in var_name:
                            return 'Google Ads Impressions'
                        elif 'Daily Impressions Outcome Engagement' in var_name or 'Daily_Impressions_OUTCOME_ENGAGEMENT' in var_name:
                            return 'Traffic Ads Impressions'
                        elif var_name == 'Impressions':
                            return 'Other Products (Meta Ads)'
                        else:
                            # Replace "_meta_impression" with " (Meta Ads)"
                            display = var_name.replace(' Meta Impression', ' (Meta Ads)')
                            display = display.replace('_Meta_Impression', ' (Meta Ads)')
                            return display
                    
                    filtered_df['Display_Name'] = filtered_df['Variable'].apply(get_display_name)
                    
                    # Create horizontal bar chart with contribution AND budget allocation
                    st.markdown("### 📊 Portfolio Contribution Analysis")
                    
                    fig = go.Figure()
                    
                    # Add contribution % trace
                    fig.add_trace(go.Bar(
                        y=filtered_df['Display_Name'],
                        x=filtered_df['Contribution %'],
                        name='Revenue Contribution %',
                        text=filtered_df['Contribution %'].apply(lambda x: f"{x:.2f}%"),
                        textposition='auto',
                        orientation='h',
                        marker_color='#2ecc71'
                    ))
                    
                    # Add budget allocation traces if optimization has been run
                    if st.session_state.get('optimization_result') is not None:
                        result = st.session_state.optimization_result
                        item_names = result['item_names']
                        base_budgets = result['base_budgets']
                        opt_budgets = result['optimized_budgets']
                        
                        # Calculate percentages
                        total_base = base_budgets.sum()
                        total_opt = opt_budgets.sum()
                        base_pct = (base_budgets / total_base * 100)
                        opt_pct = (opt_budgets / total_opt * 100)
                        
                        # Create budget dataframe
                        budget_df = pd.DataFrame({
                            'Item': item_names,
                            'Base %': base_pct,
                            'Optimized %': opt_pct
                        })
                        
                        # Apply display name mapping (must match contribution display names exactly)
                        def get_display_name_budget(item_name):
                            item_lower = item_name.lower()
                            if 'google' in item_lower and 'campaign' in item_lower:
                                return 'Google Ads Impressions'
                            elif 'traffic' in item_lower:
                                return 'Traffic Ads Impressions'
                            elif 'catalog' in item_lower or item_lower == 'other products':
                                return 'Other Products (Meta Ads)'
                            else:
                                # For products: convert to title case and add " (Meta Ads)"
                                # This matches the contribution display name format
                                return item_name.title() + ' (Meta Ads)'
                        
                        budget_df['Display_Name'] = budget_df['Item'].apply(get_display_name_budget)
                        
                        # Filter budget_df to only include items that are in filtered_df
                        budget_df_filtered = budget_df[budget_df['Display_Name'].isin(filtered_df['Display_Name'])].copy()
                        
                        # Add base budget trace (hidden by default)
                        fig.add_trace(go.Bar(
                            y=budget_df_filtered['Display_Name'],
                            x=budget_df_filtered['Base %'],
                            name='Base Budget %',
                            orientation='h',
                            marker_color='#3498db',
                            text=budget_df_filtered['Base %'].apply(lambda x: f"{x:.1f}%"),
                            textposition='auto',
                            visible='legendonly'
                        ))
                        
                        # Add optimized budget trace (hidden by default)
                        fig.add_trace(go.Bar(
                            y=budget_df_filtered['Display_Name'],
                            x=budget_df_filtered['Optimized %'],
                            name='Optimized Budget %',
                            orientation='h',
                            marker_color='#e74c3c',
                            text=budget_df_filtered['Optimized %'].apply(lambda x: f"{x:.1f}%"),
                            textposition='auto',
                            visible='legendonly'
                        ))
                    
                    # Add Impression Share % trace if modeling data available (hidden by default)
                    if modeling_df is not None:
                        try:
                            # Get "Other Products" rows
                            other_products_rows = modeling_df[modeling_df[product_col].str.lower() == 'other products']
                            
                            if len(other_products_rows) > 0:
                                # Get all Beta columns with "impression"
                                beta_impression_cols = [col for col in beta_df.columns if col.startswith('Beta_') and 'impression' in col.lower()]
                                
                                # Calculate sums for each impression variable
                                impression_sums = {}
                                for beta_col in beta_impression_cols:
                                    var_name = beta_col.replace('Beta_', '')
                                    
                                    # Try to find matching column
                                    found = False
                                    if var_name in modeling_df.columns:
                                        sum_val = other_products_rows[var_name].sum()
                                        impression_sums[beta_col] = sum_val
                                        found = True
                                    
                                    if not found:
                                        for data_col in modeling_df.columns:
                                            col_normalized = data_col.lower().replace(' ', '_').replace('-', '_')
                                            var_normalized = var_name.lower().replace(' ', '_').replace('-', '_')
                                            if col_normalized == var_normalized:
                                                sum_val = other_products_rows[data_col].sum()
                                                impression_sums[beta_col] = sum_val
                                                break
                                
                                # Calculate total
                                total_impressions = sum(impression_sums.values())
                                
                                # Create impression share dataframe
                                impression_chart_data = []
                                for beta_col, sum_val in impression_sums.items():
                                    share_pct = (sum_val / total_impressions * 100) if total_impressions > 0 else 0
                                    
                                    # Create display name (same logic as contribution)
                                    var_name = beta_col.replace('Beta_', '')
                                    display_name = var_name.replace('_', ' ').title()
                                    
                                    if 'Google Impression' in display_name or 'Google_Impression' in display_name:
                                        display_name = 'Google Ads Impressions'
                                    elif 'Daily Impressions Outcome Engagement' in display_name:
                                        display_name = 'Traffic Ads Impressions'
                                    elif display_name == 'Impressions':
                                        display_name = 'Other Products (Meta Ads)'
                                    else:
                                        display_name = display_name.replace(' Meta Impression', ' (Meta Ads)')
                                    
                                    impression_chart_data.append({
                                        'Display_Name': display_name,
                                        'Share %': share_pct
                                    })
                                
                                impression_chart_df = pd.DataFrame(impression_chart_data)
                                
                                # Filter to only show variables in the chart
                                impression_chart_filtered = impression_chart_df[
                                    impression_chart_df['Display_Name'].isin(filtered_df['Display_Name'])
                                ].copy()
                                
                                if len(impression_chart_filtered) > 0:
                                    # Add impression share trace
                                    fig.add_trace(go.Bar(
                                        y=impression_chart_filtered['Display_Name'],
                                        x=impression_chart_filtered['Share %'],
                                        name='Impression Share %',
                                        orientation='h',
                                        marker_color='#9b59b6',
                                        text=impression_chart_filtered['Share %'].apply(lambda x: f"{x:.1f}%"),
                                        textposition='auto',
                                        visible='legendonly'
                                    ))
                        except Exception:
                            pass  # Silently skip if impression share calculation fails
                    
                    fig.update_layout(
                        title="Portfolio Contribution: Revenue, Budget & Impression Share",
                        xaxis_title="Percentage",
                        yaxis_title="",
                        height=len(filtered_df) * 80,  # 80px per variable - full height
                        barmode='group',
                        yaxis={'autorange': 'reversed'},
                        font=dict(size=15),
                        margin=dict(l=300, r=80, t=100, b=70),
                        legend=dict(
                            orientation="h", 
                            yanchor="bottom", 
                            y=1.02, 
                            xanchor="right", 
                            x=1,
                            font=dict(size=14)
                        ),
                        bargap=0.1,
                        bargroupgap=0.05
                    )
                    
                    # Display chart in Streamlit container with height limit
                    with st.container(height=800):
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                    
                    # Show data table
                    st.markdown("### 📋 Portfolio Contribution Details")
                    st.info("ℹ️ **Note:** Revenue Contribution = (Beta × Mean × Price) summed across all products. Percentages = (Variable Revenue / Total Revenue of All Shown Variables) × 100%")
                    st.dataframe(
                        filtered_df[['Display_Name', 'Revenue Contribution', 'Contribution %', 'Type']].rename(columns={'Display_Name': 'Variable'}).style.format({
                            'Revenue Contribution': lambda x: f"${x:,.2f}",
                            'Contribution %': lambda x: f"{x:.2f}%"
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Expander: Impression Share Analysis
                    if modeling_df is not None:
                        with st.expander("📊 Impression Share % (from Data_for_model.xlsx)"):
                            st.markdown("### 🎯 Impression Share Calculation")
                            st.info("Shows the % share of impressions for each variable based on 'Other Products' data from modeling file")
                            
                            try:
                                # Filter for "Other Products" rows
                                other_products_rows = modeling_df[modeling_df[product_col].str.lower() == 'other products']
                                
                                if len(other_products_rows) == 0:
                                    st.warning("No 'Other Products' data found in modeling file")
                                else:
                                    st.success(f"✅ Found {len(other_products_rows)} rows for 'Other Products'")
                                    
                                    # Get all Beta columns from beta file that have "impression" in name
                                    beta_impression_cols = [col for col in beta_df.columns if col.startswith('Beta_') and 'impression' in col.lower()]
                                    
                                    st.markdown(f"**Beta impression columns found:** {len(beta_impression_cols)}")
                                    
                                    # For each beta impression column, find corresponding column in modeling data
                                    impression_means = {}
                                    for beta_col in beta_impression_cols:
                                        # Remove "Beta_" prefix to get the variable name
                                        var_name = beta_col.replace('Beta_', '')
                                        
                                        # Try to find matching column in modeling data
                                        found = False
                                        
                                        # Try exact match first
                                        if var_name in modeling_df.columns:
                                            sum_val = other_products_rows[var_name].sum()
                                            impression_means[beta_col] = sum_val
                                            found = True
                                        
                                        # Try normalized match (case-insensitive, underscore/space variations)
                                        if not found:
                                            for data_col in modeling_df.columns:
                                                col_normalized = data_col.lower().replace(' ', '_').replace('-', '_')
                                                var_normalized = var_name.lower().replace(' ', '_').replace('-', '_')
                                                if col_normalized == var_normalized:
                                                    sum_val = other_products_rows[data_col].sum()
                                                    impression_means[beta_col] = sum_val
                                                    found = True
                                                    break
                                    
                                    # Calculate total of all impression sums
                                    total_all_impressions = sum(impression_means.values())
                                    
                                    # Create dataframe with all impression variables
                                    impression_share_data = []
                                    for beta_col, sum_val in impression_means.items():
                                        share_pct = (sum_val / total_all_impressions * 100) if total_all_impressions > 0 else 0
                                        
                                        # Create display name
                                        var_name = beta_col.replace('Beta_', '')
                                        display_name = var_name.replace('_', ' ').title()
                                        
                                        # Apply same display name mapping as contribution chart
                                        if 'Google Impression' in display_name or 'Google_Impression' in display_name:
                                            display_name = 'Google Ads Impressions'
                                        elif 'Daily Impressions Outcome Engagement' in display_name or 'Daily_Impressions_OUTCOME_ENGAGEMENT' in display_name:
                                            display_name = 'Traffic Ads Impressions'
                                        elif display_name == 'Impressions':
                                            display_name = 'Other Products (Meta Ads)'
                                        else:
                                            display_name = display_name.replace(' Meta Impression', ' (Meta Ads)')
                                            display_name = display_name.replace('_Meta_Impression', ' (Meta Ads)')
                                        
                                        impression_share_data.append({
                                            'Beta Column': beta_col,
                                            'Variable': display_name,
                                            'Total Impressions': sum_val,
                                            'Share %': share_pct
                                        })
                                    
                                    impression_share_df = pd.DataFrame(impression_share_data)
                                    impression_share_df = impression_share_df.sort_values('Share %', ascending=False)
                                    
                                    st.markdown("---")
                                    st.markdown("### 📈 Impression Share % (All Variables with Beta)")
                                    st.success(f"✅ Showing {len(impression_share_df)} impression variables that have Beta coefficients")
                                    
                                    st.metric("Total Impressions (Sum)", f"{total_all_impressions:,.0f}")
                                    
                                    # Display table
                                    st.dataframe(
                                        impression_share_df[['Variable', 'Total Impressions', 'Share %']].style.format({
                                            'Total Impressions': lambda x: f"{x:,.0f}",
                                            'Share %': lambda x: f"{x:.2f}%"
                                        }).background_gradient(subset=['Share %'], cmap='Blues'),
                                        use_container_width=True,
                                        height=400
                                    )
                                    
                                    st.markdown("---")
                                    st.markdown("### 🧮 Calculation Details")
                                    st.code(f"""
Step 1: Find all Beta columns with "impression" in name
        → Found {len(beta_impression_cols)} Beta impression columns

Step 2: Filter Data_for_model.xlsx for "Other Products"
        → Found {len(other_products_rows)} rows

Step 3: Calculate SUM of impressions for each variable
        → Matched {len(impression_means)} variables with modeling data

Step 4: Calculate % share
        Formula: (Variable Sum / Total Sum) × 100%
        
        Total Impressions (Sum) = {total_all_impressions:,.0f}
        
Step 5: Display ALL impression variables with Beta
        → {len(impression_share_df)} variables shown
        → Percentages sum to 100%
                                    """, language="python")
                                    
                                    st.markdown("**Interpretation:**")
                                    st.markdown("- Shows impression distribution based on SUM of values from 'Other Products'")
                                    st.markdown("- Only includes variables that have Beta coefficients")
                                    st.markdown("- Percentages sum to 100% across all shown variables")
                                    st.markdown("- Compare with Revenue Contribution % to see efficiency")
                                
                            except Exception as e:
                                st.error(f"Error calculating impression share: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                    
                    # Expander: Product Prices & Calculation Details
                    with st.expander("💰 Product Prices & Revenue Calculation Details"):
                        st.markdown("### 📊 Product Prices Used in Revenue Calculation")
                        st.info("These are the prices from the **product_prices.csv** file, used by both the optimizer and contribution analysis.")
                        
                        # Create product price table
                        product_price_data = []
                        for idx, beta_row in beta_df.iterrows():
                            product_name = beta_row.get('Product title', '')
                            if pd.isna(product_name) or product_name == '':
                                continue
                            
                            # Get product price
                            product_price = 0
                            if master_df_contrib is not None:
                                price_row = master_df_contrib[master_df_contrib['item_name'].str.lower() == product_name.lower()]
                                if len(price_row) > 0 and 'price' in price_row.columns:
                                    product_price = price_row['price'].values[0]
                            
                            product_price_data.append({
                                'Product': product_name,
                                'Price': product_price,
                                'Status': '✅ Used' if product_price > 0 else '❌ No Price (Excluded)'
                            })
                        
                        price_df = pd.DataFrame(product_price_data)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Products with Prices", len(price_df[price_df['Price'] > 0]))
                        with col2:
                            st.metric("Products without Prices", len(price_df[price_df['Price'] == 0]))
                        
                        st.dataframe(
                            price_df.style.format({
                                'Price': lambda x: f"${x:,.2f}" if x > 0 else "$0.00"
                            }),
                            use_container_width=True,
                            height=400
                        )
                        
                        st.markdown("---")
                        
                        # Show calculation equation for each variable
                        st.markdown("### 🧮 Revenue Contribution Calculation by Variable")
                        st.markdown("**Formula:** For each variable, sum across all products:")
                        st.code("Revenue Contribution = Σ (Beta_product,variable × Mean_variable × Price_product)", language="python")
                        
                        st.markdown("**Example Breakdown for Each Variable:**")
                        
                        # Pick a sample variable to show detailed calculation
                        if len(filtered_df) > 0:
                            sample_var = filtered_df.iloc[0]['Variable']
                            sample_display_name = filtered_df.iloc[0]['Display_Name']
                            sample_contribution = filtered_df.iloc[0]['Revenue Contribution']
                            
                            st.markdown(f"#### Example: **{sample_display_name}**")
                            st.markdown(f"**Total Revenue Contribution: ${sample_contribution:,.2f}**")
                            
                            # Show breakdown by product
                            st.markdown("**Calculation Breakdown by Product:**")
                            
                            breakdown_data = []
                            for idx, beta_row in beta_df.iterrows():
                                product_name = beta_row.get('Product title', '')
                                if pd.isna(product_name) or product_name == '':
                                    continue
                                
                                # Get product price
                                product_price = 0
                                if master_df_contrib is not None:
                                    price_row = master_df_contrib[master_df_contrib['item_name'].str.lower() == product_name.lower()]
                                    if len(price_row) > 0 and 'price' in price_row.columns:
                                        product_price = price_row['price'].values[0]
                                
                                if product_price == 0:
                                    continue
                                
                                # Get product data
                                product_data = modeling_df[modeling_df[product_col].str.lower() == product_name.lower()]
                                
                                # Find the beta column for this variable
                                var_name_normalized = sample_var.lower().replace(' ', '_')
                                beta_col_name = None
                                beta_value = None
                                mean_value = None
                                
                                for col in beta_row.index:
                                    if not col.startswith('Beta_'):
                                        continue
                                    
                                    col_normalized = col.replace('Beta_', '').lower().replace(' ', '_')
                                    if col_normalized == var_name_normalized:
                                        beta_col_name = col
                                        beta_value = beta_row[col]
                                        
                                        # Get mean value (same logic as before)
                                        var_name = col.replace('Beta_', '')
                                        
                                        if 'product_variant_price' in var_name.lower():
                                            mean_value = product_price
                                        elif 'google_trends' in var_name.lower():
                                            mean_value = google_trends_value_contrib
                                        elif 'impression' in var_name.lower() or 'category discount' in var_name.lower():
                                            if len(other_products_data) > 0:
                                                if var_name in modeling_df.columns:
                                                    mean_value = other_products_data[var_name].mean()
                                                else:
                                                    for data_col in modeling_df.columns:
                                                        col_norm = data_col.lower().replace(' ', '_').replace('-', '_')
                                                        var_norm = var_name.lower().replace(' ', '_').replace('-', '_')
                                                        if col_norm == var_norm:
                                                            mean_value = other_products_data[data_col].mean()
                                                            break
                                        elif len(product_data) > 0:
                                            if var_name in modeling_df.columns:
                                                mean_value = product_data[var_name].mean()
                                            else:
                                                for data_col in modeling_df.columns:
                                                    col_norm = data_col.lower().replace(' ', '_').replace('-', '_')
                                                    var_norm = var_name.lower().replace(' ', '_').replace('-', '_')
                                                    if col_norm == var_norm:
                                                        mean_value = product_data[data_col].mean()
                                                        break
                                        break
                                
                                if beta_value is not None and mean_value is not None and pd.notna(beta_value) and pd.notna(mean_value):
                                    volume_contrib = beta_value * mean_value
                                    revenue_contrib = volume_contrib * product_price
                                    
                                    breakdown_data.append({
                                        'Product': product_name,
                                        'Beta': beta_value,
                                        'Mean': mean_value,
                                        'Price': product_price,
                                        'Volume Contribution': volume_contrib,
                                        'Revenue Contribution': revenue_contrib,
                                        'Equation': f"{beta_value:.6f} × {mean_value:.2f} × ${product_price:.2f}"
                                    })
                            
                            if breakdown_data:
                                breakdown_df = pd.DataFrame(breakdown_data)
                                
                                st.dataframe(
                                    breakdown_df.style.format({
                                        'Beta': lambda x: f"{x:.6f}",
                                        'Mean': lambda x: f"{x:.2f}",
                                        'Price': lambda x: f"${x:.2f}",
                                        'Volume Contribution': lambda x: f"{x:.4f}",
                                        'Revenue Contribution': lambda x: f"${x:.2f}"
                                    }),
                                    use_container_width=True,
                                    height=400
                                )
                                
                                total_revenue = breakdown_df['Revenue Contribution'].sum()
                                st.success(f"✅ **Total Revenue Contribution for {sample_display_name}: ${total_revenue:,.2f}**")
                                
                                st.markdown("**Interpretation:**")
                                st.markdown(f"- Each row shows how {sample_display_name} contributes to revenue for that specific product")
                                st.markdown(f"- The sum of all rows gives the total revenue contribution: **${total_revenue:,.2f}**")
                                st.markdown(f"- Products with higher prices have proportionally higher revenue contributions")
                            else:
                                st.warning("No breakdown data available for this variable")
                
            except Exception as e:
                st.error(f"Error calculating contributions: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # ========== BY PRODUCT CONTRIBUTION ==========
        with contrib_tab2:
            try:
                # Create contribution grid
                contribution_grid = []
                
                # For each product
                for idx, beta_row in beta_df.iterrows():
                    product_name = beta_row.get('Product title', '')
                    if pd.isna(product_name) or product_name == '':
                        continue
                    
                    # Get product data
                    product_data = modeling_df[modeling_df[product_col].str.lower() == product_name.lower()]
                    
                    # Get product price
                    product_price = 0
                    if master_df_contrib is not None:
                        price_row = master_df_contrib[master_df_contrib['item_name'].str.lower() == product_name.lower()]
                        if len(price_row) > 0 and 'price' in price_row.columns:
                            product_price = price_row['price'].values[0]
                    
                    # Calculate Beta × Mean for each variable
                    variable_contributions = {}
                    
                    for col in beta_row.index:
                        if not col.startswith('Beta_'):
                            continue
                        
                        beta_value = beta_row[col]
                        if pd.isna(beta_value):
                            continue
                        
                        var_name = col.replace('Beta_', '')
                        display_name = var_name.replace('_', ' ').title()
                        
                        # Get mean value (SAME LOGIC)
                        mean_value = None
                        
                        if 'product_variant_price' in var_name.lower() or 'product variant price' in var_name.lower():
                            mean_value = product_price
                        elif 'google_trends' in var_name.lower() or 'google trends' in var_name.lower():
                            mean_value = google_trends_value_contrib
                        elif 'impression' in var_name.lower() or 'category discount' in var_name.lower():
                            if len(other_products_data) > 0:
                                found = False
                                if var_name in modeling_df.columns:
                                    mean_value = other_products_data[var_name].mean()
                                    found = True
                                if not found:
                                    for data_col in modeling_df.columns:
                                        col_normalized = data_col.lower().replace(' ', '_').replace('-', '_')
                                        var_normalized = var_name.lower().replace(' ', '_').replace('-', '_')
                                        if col_normalized == var_normalized:
                                            mean_value = other_products_data[data_col].mean()
                                            break
                        elif len(product_data) > 0:
                            found = False
                            if var_name in modeling_df.columns:
                                mean_value = product_data[var_name].mean()
                                found = True
                            if not found:
                                for data_col in modeling_df.columns:
                                    col_normalized = data_col.lower().replace(' ', '_').replace('-', '_')
                                    var_normalized = var_name.lower().replace(' ', '_').replace('-', '_')
                                    if col_normalized == var_normalized:
                                        mean_value = product_data[data_col].mean()
                                        break
                        
                        if mean_value is not None and pd.notna(mean_value):
                            beta_x_mean = beta_value * mean_value
                            variable_contributions[display_name] = beta_x_mean
                    
                    # Calculate total and percentages
                    total_beta_x = sum(variable_contributions.values())
                    
                    row_data = {'Product': product_name}
                    for var_name, beta_x in variable_contributions.items():
                        contribution_pct = (beta_x / total_beta_x * 100) if total_beta_x != 0 else 0
                        row_data[var_name] = contribution_pct
                    
                    contribution_grid.append(row_data)
                
                # Create DataFrame
                if contribution_grid:
                    grid_df = pd.DataFrame(contribution_grid)
                    grid_df = grid_df.fillna(0)
                    
                    # Get all variable columns (exclude 'Product')
                    all_var_columns = [col for col in grid_df.columns if col != 'Product']
                    impression_columns = [col for col in all_var_columns if 'impression' in col.lower()]
                    non_impression_columns = [col for col in all_var_columns if 'impression' not in col.lower()]
                    
                    # Multiselect to REMOVE variables (default: remove non-impression variables)
                    st.markdown("### 🎯 Remove Variables (Optional)")
                    cols_to_remove = st.multiselect(
                        "Select variables to exclude from grid:",
                        options=all_var_columns,
                        default=non_impression_columns,
                        help="By default, only impression variables are shown. You can add back other variables by deselecting them.",
                        key="by_product_remove"
                    )
                    
                    # Filter columns
                    columns_to_show = ['Product'] + [col for col in all_var_columns if col not in cols_to_remove]
                    filtered_grid_df = grid_df[columns_to_show].copy()
                    
                    if len(columns_to_show) == 1:  # Only 'Product' column left
                        st.warning("You've removed all variables. Please keep at least one variable.")
                    else:
                        # Rename columns for better display
                        def get_display_name_grid(col_name):
                            if col_name == 'Product':
                                return col_name
                            elif 'Google Impression' in col_name or 'Google_Impression' in col_name:
                                return 'Google Ads Impressions'
                            elif 'Daily Impressions Outcome Engagement' in col_name or 'Daily_Impressions_OUTCOME_ENGAGEMENT' in col_name:
                                return 'Traffic Ads Impressions'
                            elif col_name == 'Impressions':
                                return 'Other Products (Meta Ads)'
                            else:
                                # Replace "_meta_impression" or " Meta Impression" with " (Meta Ads)"
                                display = col_name.replace(' Meta Impression', ' (Meta Ads)')
                                display = display.replace('_Meta_Impression', ' (Meta Ads)')
                                return display
                        
                        filtered_grid_df.columns = [get_display_name_grid(col) for col in filtered_grid_df.columns]
                        
                        st.markdown(f"**{len(filtered_grid_df)} Products × {len(columns_to_show)-1} Variables**")
                        st.info("ℹ️ **Note:** Shows relative volume contribution % within each product (sums to 100% per row). Portfolio-level revenue contribution is shown in the 'Portfolio Level' tab.")
                        
                        # Display as heatmap with better colors
                        st.dataframe(
                            filtered_grid_df.set_index('Product').style.background_gradient(cmap='Blues', axis=None).format("{:.2f}%"),
                            use_container_width=True,
                            height=600
                        )
                else:
                    st.warning("No contribution data calculated")
                    
            except Exception as e:
                st.error(f"Error calculating contributions: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
