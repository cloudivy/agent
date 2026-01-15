import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

st.set_page_config(page_title="Pipeline Digging vs Leak Analyzer", layout="wide", page_icon="🛢️")

st.title("🛢️ Pipeline Digging vs Leak Events Analyzer")
st.markdown("Upload your manual digging and LDS datasets to visualize correlations by chainage.")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    digging_file = st.file_uploader("Upload Manual Digging Data (CSV/Excel)", type=['csv', 'xlsx', 'xls'], key="digging")
with col2:
    leaks_file = st.file_uploader("Upload LDS Leak Data (CSV/Excel)", type=['csv', 'xlsx', 'xls'], key="leaks")

# Fixed tolerance
tolerance = 1.0
st.info(f"🔧 Fixed Chainage Tolerance: {tolerance} km")

if digging_file is not None and leaks_file is not None:
    try:
        df_manual_digging = pd.read_csv(digging_file) if digging_file.name.endswith('.csv') else pd.read_excel(digging_file)
        df_lds_IV = pd.read_csv(leaks_file) if leaks_file.name.endswith('.csv') else pd.read_excel(leaks_file)
        
        st.success("✅ Data loaded successfully!")
        st.dataframe(df_manual_digging.head(), use_container_width=True)
        st.dataframe(df_lds_IV.head(), use_container_width=True)
        
        # Extract unique chainages
        unique_chainages_dig = sorted(df_manual_digging['Original_chainage'].unique())
        unique_chainages_leak = sorted(df_lds_IV['chainage'].unique())
        unique_chainages = sorted(set(unique_chainages_dig).union(unique_chainages_leak))
        
        st.subheader("Select Chainage(s) to Analyze")
        col_a, col_b = st.columns(2)
        with col_a:
            selected_chainages = st.multiselect("Chainages", unique_chainages, default=unique_chainages[:3])
        with col_b:
            max_plots = st.slider("Max Plots per Run", 1, 10, 5)
        
        if selected_chainages:
            st.subheader("Visualizations (Tolerance: 1.0 km)")
            for i, target_chainage_val in enumerate(selected_chainages[:max_plots]):
                df_digging_filtered = df_manual_digging[abs(df_manual_digging['Original_chainage'] - target_chainage_val) <= tolerance]
                df_leaks_filtered = df_lds_IV[abs(df_lds_IV['chainage'] - target_chainage_val) <= tolerance]
                
                if not df_digging_filtered.empty or not df_leaks_filtered.empty:
                    plt.figure(figsize=(18, 10))
                    
                    if not df_digging_filtered.empty:
                        sns.scatterplot(data=df_digging_filtered, x='DateTime', y='Original_chainage', 
                                      color='blue', label='Digging Events', marker='o', s=50)
                    
                    if not df_leaks_filtered.empty:
                        sns.scatterplot(data=df_leaks_filtered, x='DateTime', y='chainage', 
                                      color='red', label='Leak Events', marker='X', s=80)
                    
                    plt.title(f'Digging vs. Leak Events at Chainage {target_chainage_val:.1f} (Tolerance: {tolerance:.1f} km)')
                    plt.xlabel('Date and Time')
                    plt.ylabel('Chainage')
                    plt.grid(True)
                    plt.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.subplots_adjust(right=0.75)
                    st.pyplot(plt)
                    plt.close()  # Prevent memory buildup
                    st.caption(f"Chainage {target_chainage_val:.1f}: {len(df_digging_filtered)} digging, {len(df_leaks_filtered)} leaks")
                else:
                    st.warning(f"No events at chainage {target_chainage_val:.1f} ± {tolerance} km.")
            
            # Export
            st.subheader("Export Results")
            all_results = []
            for target_chainage_val in selected_chainages:
                df_digging_filtered = df_manual_digging[abs(df_manual_digging['Original_chainage'] - target_chainage_val) <= tolerance]
                df_leaks_filtered = df_lds_IV[abs(df_lds_IV['chainage'] - target_chainage_val) <= tolerance]
                df_digging_filtered['target_chainage'] = target_chainage_val
                df_leaks_filtered['target_chainage'] = target_chainage_val
                all_results.extend([df_digging_filtered, df_leaks_filtered])
            
            if all_results:
                combined_df = pd.concat(all_results, ignore_index=True)
                csv_buffer = io.StringIO()
                combined_df.to_csv(csv_buffer, index=False)
                st.download_button("📥 Download Results CSV", csv_buffer.getvalue(), "chainage_analysis.csv")
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}. Check columns: 'DateTime', 'Original_chainage' / 'chainage'.")
else:
    st.info("👆 Upload both files to begin. Fixed tolerance: 1.0 km.")

# requirements.txt remains the same
