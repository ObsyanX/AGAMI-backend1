# KMRL AI Induction Planning - Streamlit Frontend
# Requirements: streamlit, pandas, requests, plotly
# Install: pip install streamlit pandas requests plotly
# Run: streamlit run frontend.py

import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configuration
st.set_page_config(
    page_title="KMRL AI Induction Planning",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
try:
    API_BASE_URL = st.secrets.get("API_URL", "http://localhost:5000")
except:
    # API_BASE_URL = "http://localhost:5000"
    API_BASE_URL = "https://kmrl-train-backend.onrender.com"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_all_trains():
    """Fetch all trains from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/trains")
        if response.status_code == 200:
            return response.json().get("trains", [])
        return []
    except Exception as e:
        st.error(f"Failed to fetch trains: {e}")
        return []

def upload_csv(file):
    """Upload CSV file to backend"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_BASE_URL}/trains/bulk", files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def run_optimization(params):
    """Run optimization via API"""
    try:
        response = requests.post(f"{API_BASE_URL}/optimize", json=params)
        if response.status_code == 200:
            return response.json()
        return {"error": response.json().get("error", "Optimization failed")}
    except Exception as e:
        return {"error": str(e)}

def get_train_explanation(train_id):
    """Get explanation for specific train"""
    try:
        response = requests.get(f"{API_BASE_URL}/explain/{train_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_kpis():
    """Get fleet KPIs"""
    try:
        response = requests.get(f"{API_BASE_URL}/kpis")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def whatif_analysis(train_id, changes):
    """Run what-if analysis"""
    try:
        response = requests.post(f"{API_BASE_URL}/whatif", 
                                json={"train_id": train_id, "changes": changes})
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def delete_train(train_id):
    """Delete a train"""
    try:
        response = requests.delete(f"{API_BASE_URL}/trains/{train_id}")
        return response.status_code == 200
    except:
        return False

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">🚆 KMRL AI-Driven Train Induction Planning</div>', 
                unsafe_allow_html=True)
    st.markdown("**Smart India Hackathon 2025** | Kochi Metro Rail Limited")
    
    # Check backend connection
    if not check_backend_health():
        st.error("⚠️ Backend server is not running! Please start the backend at " + API_BASE_URL)
        st.info("Run: `python backend.py` in a separate terminal")
        st.stop()
    
    st.success("✅ Connected to backend server")
    
    # Sidebar Navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.radio("Select Page", [
        "📊 Dashboard",
        "📤 Upload Trains",
        "⚙️ Run Optimization",
        "🔍 Explain Train",
        "🔮 What-If Analysis",
        "📈 Fleet KPIs",
        "🗂️ Manage Trains"
    ])
    
    # Page Routing
    if page == "📊 Dashboard":
        dashboard_page()
    elif page == "📤 Upload Trains":
        upload_page()
    elif page == "⚙️ Run Optimization":
        optimization_page()
    elif page == "🔍 Explain Train":
        explain_page()
    elif page == "🔮 What-If Analysis":
        whatif_page()
    elif page == "📈 Fleet KPIs":
        kpis_page()
    elif page == "🗂️ Manage Trains":
        manage_trains_page()

# ==================== PAGES ====================

def dashboard_page():
    """Main dashboard with overview"""
    st.header("📊 Fleet Dashboard")
    
    trains = get_all_trains()
    
    if not trains:
        st.warning("⚠️ No trains in the system. Please upload train data.")
        st.info("👈 Go to 'Upload Trains' to add train data via CSV")
        return
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trains", len(trains))
    
    with col2:
        valid_fitness = sum(1 for t in trains 
                          if t.get("fitness_rolling") == "Valid" 
                          and t.get("fitness_signal") == "Valid" 
                          and t.get("fitness_telecom") == "Valid")
        st.metric("Fitness Valid", valid_fitness)
    
    with col3:
        open_jobs = sum(1 for t in trains if t.get("job_card_open", False))
        st.metric("Open Job Cards", open_jobs)
    
    with col4:
        avg_mileage = sum(t.get("mileage_km", 0) for t in trains) / len(trains)
        st.metric("Avg Mileage", f"{int(avg_mileage):,} km")
    
    # Train List
    st.subheader("📋 Train Inventory")
    df = pd.DataFrame(trains)
    
    # Select columns to display
    display_cols = ["train_id", "fitness_rolling", "fitness_signal", "fitness_telecom", 
                    "job_card_open", "mileage_km", "branding_contract", 
                    "cleaning_hours_required", "home_bay"]
    
    available_cols = [col for col in display_cols if col in df.columns]
    st.dataframe(df[available_cols], use_container_width=True, height=400)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Branding Contracts")
        branding_counts = df["branding_contract"].value_counts()
        fig = px.pie(values=branding_counts.values, 
                    names=branding_counts.index,
                    title="Train Distribution by Branding")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📍 Bay Distribution")
        bay_counts = df["home_bay"].value_counts()
        fig = px.bar(x=bay_counts.index, y=bay_counts.values,
                    labels={"x": "Bay", "y": "Count"},
                    title="Trains per Bay")
        st.plotly_chart(fig, use_container_width=True)
    
    # Mileage Distribution
    st.subheader("🛤️ Mileage Distribution")
    fig = px.histogram(df, x="mileage_km", nbins=20,
                      title="Train Mileage Distribution",
                      labels={"mileage_km": "Mileage (km)"})
    st.plotly_chart(fig, use_container_width=True)

def upload_page():
    """CSV upload page"""
    st.header("📤 Upload Train Data")
    
    # Instructions
    with st.expander("📄 CSV Format Requirements", expanded=True):
        st.markdown("""
        **Required Columns:**
        - `TrainID` - Unique identifier (e.g., T01, T02, ...)
        - `Fitness_Rolling` - Valid/Expired
        - `Fitness_Signal` - Valid/Expired
        - `Fitness_Telecom` - Valid/Expired
        - `JobCard_Open` - Yes/No
        - `Mileage_km` - Numeric
        - `Branding_Contract` - Contract name or "None"
        - `Branding_Hours_Current` - Numeric
        - `Branding_Hours_Required` - Numeric
        - `Cleaning_Hours_Req` - Numeric
        - `Home_Bay` - Bay identifier (e.g., B1, B2, ...)
        
        **Note:** TrainID must be unique. Duplicate IDs will be skipped.
        """)
    
    # Sample CSV Download
    st.subheader("📥 Download Sample CSV")
    sample_csv = """TrainID,Fitness_Rolling,Fitness_Signal,Fitness_Telecom,JobCard_Open,Mileage_km,Branding_Contract,Branding_Hours_Current,Branding_Hours_Required,Cleaning_Hours_Req,Home_Bay
T01,Valid,Valid,Valid,No,10500,CocaCola,20,24,2,B1
T02,Valid,Valid,Expired,Yes,8000,Pepsi,10,12,3,B3
T03,Valid,Valid,Valid,No,12000,Amazon,22,24,1,B2"""
    
    st.download_button(
        label="⬇️ Download Sample CSV",
        data=sample_csv,
        file_name="kmrl_trains_sample.csv",
        mime="text/csv"
    )
    
    # File Upload
    st.subheader("📁 Upload Your CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Preview
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: {len(df)} rows")
            
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Validation
            required_cols = ["TrainID", "Fitness_Rolling", "Fitness_Signal", 
                           "Fitness_Telecom", "JobCard_Open", "Mileage_km",
                           "Branding_Contract", "Branding_Hours_Current", 
                           "Branding_Hours_Required", "Cleaning_Hours_Req", "Home_Bay"]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ All required columns present")
                
                # Check for duplicate TrainIDs in file
                duplicates = df[df.duplicated(subset=["TrainID"], keep=False)]
                if not duplicates.empty:
                    st.warning(f"⚠️ Found {len(duplicates)} duplicate TrainIDs in file:")
                    st.dataframe(duplicates[["TrainID"]])
                
                # Upload button
                if st.button("🚀 Upload to System", type="primary"):
                    with st.spinner("Uploading..."):
                        uploaded_file.seek(0)  # Reset file pointer
                        result = upload_csv(uploaded_file)
                        
                        if "error" in result:
                            st.error(f"❌ Upload failed: {result['error']}")
                        else:
                            st.success(f"✅ Upload completed!")
                            st.json({
                                "Added": result.get("added", 0),
                                "Skipped": result.get("skipped", 0),
                                "Total": len(df)
                            })
                            
                            if result.get("errors"):
                                with st.expander("⚠️ View Errors"):
                                    for error in result["errors"]:
                                        st.text(error)
                            
                            st.balloons()
        
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

def optimization_page():
    """Optimization configuration and execution"""
    st.header("⚙️ Run Induction Optimization")
    
    trains = get_all_trains()
    if not trains:
        st.warning("⚠️ No trains available. Please upload train data first.")
        return
    
    st.info(f"📊 Fleet size: {len(trains)} trains")
    
    # Configuration
    st.subheader("🎛️ Optimization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_service = st.number_input(
            "Maximum Service Trains",
            min_value=1,
            max_value=len(trains),
            value=min(20, len(trains)),
            help="Maximum number of trains that can be in service"
        )
        
        cleaning_capacity = st.number_input(
            "Cleaning Capacity (hours)",
            min_value=0,
            max_value=200,
            value=40,
            help="Total cleaning hours available per night"
        )
    
    with col2:
        solver_time_limit = st.number_input(
            "Solver Time Limit (seconds)",
            min_value=1,
            max_value=60,
            value=10,
            help="Maximum time for optimization solver"
        )
        
        current_date = st.date_input(
            "Planning Date",
            value=datetime.now(),
            help="Date for which to plan induction"
        )
    
    # Run Optimization
    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner("🔄 Running optimization..."):
            params = {
                "max_service_trains": max_service,
                "cleaning_capacity_hours": cleaning_capacity,
                "solver_time_limit": solver_time_limit,
                "current_date": current_date.isoformat()
            }
            
            result = run_optimization(params)
            
            if "error" in result:
                st.error(f"❌ Optimization failed: {result['error']}")
            else:
                st.success("✅ Optimization completed successfully!")
                
                # Store result in session state
                st.session_state['optimization_result'] = result
                
                # Display results
                display_optimization_results(result)

def display_optimization_results(result):
    """Display optimization results"""
    st.subheader("📋 Induction Plan")
    
    plan = result.get("plan", {})
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Service", len(plan.get("service", [])), 
                 delta=None, delta_color="normal")
    
    with col2:
        st.metric("Standby", len(plan.get("standby", [])),
                 delta=None, delta_color="off")
    
    with col3:
        st.metric("IBL", len(plan.get("ibl", [])),
                 delta=None, delta_color="inverse")
    
    with col4:
        solver = plan.get("solver", "unknown")
        st.metric("Solver", solver.split("_")[0].upper())
    
    # Train Lists
    tab1, tab2, tab3 = st.tabs(["🟢 Service", "🟡 Standby", "🔴 IBL"])
    
    with tab1:
        service_trains = plan.get("service", [])
        if service_trains:
            st.success(f"**{len(service_trains)} trains assigned to Service:**")
            cols = st.columns(5)
            for i, train_id in enumerate(service_trains):
                cols[i % 5].button(train_id, key=f"service_{train_id}")
        else:
            st.info("No trains assigned to service")
    
    with tab2:
        standby_trains = plan.get("standby", [])
        if standby_trains:
            st.warning(f"**{len(standby_trains)} trains on Standby:**")
            cols = st.columns(5)
            for i, train_id in enumerate(standby_trains):
                cols[i % 5].button(train_id, key=f"standby_{train_id}")
        else:
            st.info("No trains on standby")
    
    with tab3:
        ibl_trains = plan.get("ibl", [])
        if ibl_trains:
            st.error(f"**{len(ibl_trains)} trains in IBL:**")
            cols = st.columns(5)
            for i, train_id in enumerate(ibl_trains):
                cols[i % 5].button(train_id, key=f"ibl_{train_id}")
        else:
            st.info("No trains in IBL")
    
    # KPIs
    if "kpis" in result:
        st.subheader("📊 Performance Metrics")
        kpis = result["kpis"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fleet Utilization", 
                     f"{kpis.get('fleet_utilization', 0):.1f}%")
            st.metric("Cleaning Utilization",
                     f"{kpis.get('cleaning_utilization', 0):.1f}%")
        
        with col2:
            st.metric("Mileage Imbalance (σ)",
                     f"{kpis.get('mileage_imbalance_std', 0):,.0f} km")
            st.metric("Avg Health Score",
                     f"{kpis.get('avg_service_health_score', 0):.1f}%")
        
        with col3:
            st.metric("Branding Shortfall",
                     f"{kpis.get('total_branding_shortfall_hours', 0)} hrs")
    
    # Scored Trains Table
    if "scored_trains" in result:
        st.subheader("🎯 Train Scores")
        scored_df = pd.DataFrame(result["scored_trains"])
        st.dataframe(scored_df, use_container_width=True, height=400)

def explain_page():
    """Explain individual train decision"""
    st.header("🔍 Explain Train Decision")
    
    trains = get_all_trains()
    if not trains:
        st.warning("⚠️ No trains available.")
        return
    
    train_ids = [t["train_id"] for t in trains]
    
    selected_train = st.selectbox("Select Train", train_ids)
    
    if st.button("🔎 Get Explanation", type="primary"):
        with st.spinner("Analyzing..."):
            explanation = get_train_explanation(selected_train)
            
            if explanation:
                st.success(f"✅ Analysis for **{selected_train}**")
                
                # Overall Status
                status = explanation.get("overall_status", "UNKNOWN")
                if status == "READY":
                    st.success(f"### ✅ {status} FOR SERVICE")
                else:
                    st.error(f"### ❌ {status}")
                
                # Service Ready
                is_ready = explanation.get("service_ready", False)
                reasons = explanation.get("disqualification_reasons", [])
                
                if not is_ready and reasons:
                    st.error("**Disqualification Reasons:**")
                    for reason in reasons:
                        st.markdown(f"- {reason}")
                
                # Detailed Checks
                st.subheader("🔍 Detailed Checks")
                
                checks = explanation.get("detailed_checks", {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fitness = checks.get("fitness", {})
                    if fitness.get("valid"):
                        st.success("✅ **Fitness Certificates**")
                    else:
                        st.error("❌ **Fitness Certificates**")
                    st.json(fitness)
                
                with col2:
                    jobcards = checks.get("job_cards", {})
                    if jobcards.get("valid"):
                        st.success("✅ **Job Cards**")
                    else:
                        st.error("❌ **Job Cards**")
                    st.json(jobcards)
                
                with col3:
                    health = checks.get("health", {})
                    if health.get("valid"):
                        st.success("✅ **Component Health**")
                    else:
                        st.error("❌ **Component Health**")
                    st.json(health)
                
                # Scoring Breakdown
                st.subheader("🎯 Scoring Breakdown")
                scoring = explanation.get("scoring_breakdown", {})
                
                scores_df = pd.DataFrame({
                    "Metric": ["Composite", "Mileage", "Branding", "Cleaning", "Health", "Reliability"],
                    "Score": [
                        scoring.get("composite_score", 0),
                        scoring.get("mileage_score", 0),
                        scoring.get("branding_score", 0),
                        scoring.get("cleaning_score", 0),
                        scoring.get("health_score", 0),
                        scoring.get("reliability_score", 0)
                    ]
                })
                
                fig = px.bar(scores_df, x="Metric", y="Score",
                            title="Score Components",
                            color="Score",
                            color_continuous_scale="RdYlGn")
                st.plotly_chart(fig, use_container_width=True)
                
                # Operational Data
                st.subheader("📊 Operational Data")
                operational = explanation.get("operational_data", {})
                st.json(operational)
                
            else:
                st.error("Failed to get explanation")

def whatif_page():
    """What-if analysis page"""
    st.header("🔮 What-If Analysis")
    
    trains = get_all_trains()
    if not trains:
        st.warning("⚠️ No trains available.")
        return
    
    st.info("💡 Test scenarios by modifying train parameters")
    
    train_ids = [t["train_id"] for t in trains]
    selected_train = st.selectbox("Select Train", train_ids)
    
    st.subheader("🎛️ Modify Parameters")
    
    col1, col2 = st.columns(2)
    
    changes = {}
    
    with col1:
        new_mileage = st.number_input("New Mileage (km)", min_value=0, value=10000, step=100)
        changes["mileage_km"] = new_mileage
        
        new_fitness_rolling = st.selectbox("Fitness Rolling", ["Valid", "Expired"])
        changes["fitness_rolling"] = new_fitness_rolling
        
        new_fitness_signal = st.selectbox("Fitness Signal", ["Valid", "Expired"])
        changes["fitness_signal"] = new_fitness_signal
    
    with col2:
        new_fitness_telecom = st.selectbox("Fitness Telecom", ["Valid", "Expired"])
        changes["fitness_telecom"] = new_fitness_telecom
        
        new_jobcard = st.selectbox("Job Card Open", ["Yes", "No"])
        changes["job_card_open"] = new_jobcard == "Yes"
        
        new_cleaning = st.number_input("Cleaning Hours", min_value=1, max_value=10, value=2)
        changes["cleaning_hours_required"] = new_cleaning
    
    if st.button("🔮 Run What-If Analysis", type="primary"):
        with st.spinner("Analyzing scenario..."):
            result = whatif_analysis(selected_train, changes)
            
            if result:
                st.success("✅ Analysis completed")
                
                comparison = result.get("comparison", {})
                original = comparison.get("original", {})
                scenario = comparison.get("scenario", {})
                impact = comparison.get("impact", {})
                
                # Comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📍 Original")
                    if original.get("service_ready"):
                        st.success("✅ Service Ready")
                    else:
                        st.error("❌ Not Ready")
                    st.metric("Score", f"{original.get('composite_score', 0):.3f}")
                    if original.get("reasons"):
                        st.write("**Issues:**")
                        for reason in original["reasons"]:
                            st.text(f"- {reason}")
                
                with col2:
                    st.subheader("🔮 Scenario")
                    if scenario.get("service_ready"):
                        st.success("✅ Service Ready")
                    else:
                        st.error("❌ Not Ready")
                    st.metric("Score", f"{scenario.get('composite_score', 0):.3f}",
                             delta=f"{impact.get('score_delta', 0):.3f}")
                    if scenario.get("reasons"):
                        st.write("**Issues:**")
                        for reason in scenario["reasons"]:
                            st.text(f"- {reason}")
                
                # Impact Analysis
                st.subheader("📊 Impact Analysis")
                if impact.get("improvement"):
                    st.success("✅ Scenario shows improvement")
                else:
                    st.warning("⚠️ No improvement in scenario")
                
                if impact.get("readiness_changed"):
                    st.info("ℹ️ Service readiness status changed")
            else:
                st.error("❌ Analysis failed")

def kpis_page():
    """Fleet KPIs dashboard"""
    st.header("📈 Fleet KPIs & Analytics")
    
    kpis = get_kpis()
    
    if not kpis:
        st.warning("⚠️ Unable to fetch KPIs")
        return
    
    # Service Readiness
    st.subheader("🚦 Service Readiness")
    readiness = kpis.get("service_readiness", {})
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Ready Trains", readiness.get("ready_count", 0))
    col2.metric("Not Ready", readiness.get("not_ready_count", 0))
    col3.metric("Readiness Rate", f"{readiness.get('readiness_rate', 0):.1f}%")
    
    # Mileage Statistics
    st.subheader("🛤️ Mileage Statistics")
    mileage = kpis.get("mileage", {})
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average", f"{mileage.get('avg', 0):,.0f} km")
    col2.metric("Std Dev", f"{mileage.get('std', 0):,.0f} km")
    col3.metric("Min", f"{mileage.get('min', 0):,.0f} km")
    col4.metric("Max", f"{mileage.get('max', 0):,.0f} km")
    
    # Health Statistics
    st.subheader("💊 Component Health")
    health = kpis.get("health", {})
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Health Score", f"{health.get('avg', 0):.1f}%")
    col2.metric("Min Score", f"{health.get('min', 0):.1f}%")
    col3.metric("Below Threshold", health.get("below_threshold_count", 0))
    
    # Branding Summary
    st.subheader("🎨 Branding Summary")
    branding = kpis.get("branding", {})
    st.metric("Total Shortfall", f"{branding.get('total_shortfall', 0)} hours")
    
    contracts = branding.get("contracts", {})
    if contracts:
        contract_df = pd.DataFrame([
            {
                "Contract": name,
                "Trains": data.get("count", 0),
                "Shortfall": data.get("total_shortfall", 0),
                "Current": data.get("total_current", 0),
                "Required": data.get("total_required", 0)
            }
            for name, data in contracts.items()
        ])
        st.dataframe(contract_df, use_container_width=True)
    
    # Maintenance
    st.subheader("🔧 Maintenance Status")
    maintenance = kpis.get("maintenance", {})
    
    col1, col2 = st.columns(2)
    col1.metric("Open Job Cards", maintenance.get("open_jobcards", 0))
    col2.metric("Critical Job Cards", maintenance.get("critical_jobcards", 0))
    
    # Fitness Certificates
    st.subheader("📜 Fitness Certificate Status")
    fitness = kpis.get("fitness_certificates", {})
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rolling Expired", fitness.get("rolling_expired", 0))
    col2.metric("Signal Expired", fitness.get("signal_expired", 0))
    col3.metric("Telecom Expired", fitness.get("telecom_expired", 0))

def manage_trains_page():
    """Manage individual trains"""
    st.header("🗂️ Manage Trains")
    
    trains = get_all_trains()
    
    if not trains:
        st.warning("⚠️ No trains available.")
        return
    
    st.info(f"Total trains: {len(trains)}")
    
    # Tabs for different operations
    tab1, tab2, tab3 = st.tabs(["📋 View All", "➕ Add Single", "🗑️ Delete Train"])
    
    with tab1:
        st.subheader("📋 All Trains")
        
        # Convert to DataFrame
        df = pd.DataFrame(trains)
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fitness_filter = st.multiselect(
                "Filter by Rolling Fitness",
                options=df["fitness_rolling"].unique().tolist() if "fitness_rolling" in df.columns else [],
                default=None
            )
        
        with col2:
            bay_filter = st.multiselect(
                "Filter by Bay",
                options=df["home_bay"].unique().tolist() if "home_bay" in df.columns else [],
                default=None
            )
        
        with col3:
            jobcard_filter = st.selectbox(
                "Filter by Job Card",
                ["All", "Open Only", "Closed Only"]
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if fitness_filter:
            filtered_df = filtered_df[filtered_df["fitness_rolling"].isin(fitness_filter)]
        
        if bay_filter:
            filtered_df = filtered_df[filtered_df["home_bay"].isin(bay_filter)]
        
        if jobcard_filter == "Open Only":
            filtered_df = filtered_df[filtered_df["job_card_open"] == True]
        elif jobcard_filter == "Closed Only":
            filtered_df = filtered_df[filtered_df["job_card_open"] == False]
        
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Export option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Export to CSV",
            data=csv,
            file_name=f"kmrl_trains_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("➕ Add Single Train")
        
        with st.form("add_train_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                train_id = st.text_input("Train ID *", placeholder="T26")
                fitness_rolling = st.selectbox("Fitness Rolling *", ["Valid", "Expired"])
                fitness_signal = st.selectbox("Fitness Signal *", ["Valid", "Expired"])
                fitness_telecom = st.selectbox("Fitness Telecom *", ["Valid", "Expired"])
                jobcard_open = st.selectbox("Job Card Open *", ["No", "Yes"])
                mileage = st.number_input("Mileage (km) *", min_value=0, value=5000)
            
            with col2:
                branding = st.text_input("Branding Contract", value="None")
                branding_current = st.number_input("Branding Hours Current", min_value=0, value=0)
                branding_required = st.number_input("Branding Hours Required", min_value=0, value=0)
                cleaning = st.number_input("Cleaning Hours Required", min_value=1, max_value=10, value=2)
                home_bay = st.text_input("Home Bay *", value="B1")
            
            submitted = st.form_submit_button("➕ Add Train", type="primary")
            
            if submitted:
                if not train_id:
                    st.error("❌ Train ID is required")
                else:
                    try:
                        train_data = {
                            "train_id": train_id.strip().upper(),
                            "fitness_rolling": fitness_rolling,
                            "fitness_signal": fitness_signal,
                            "fitness_telecom": fitness_telecom,
                            "job_card_open": jobcard_open == "Yes",
                            "mileage_km": mileage,
                            "branding_contract": branding,
                            "branding_hours_current": branding_current,
                            "branding_hours_required": branding_required,
                            "cleaning_hours_required": cleaning,
                            "home_bay": home_bay
                        }
                        
                        response = requests.post(f"{API_BASE_URL}/trains", json=train_data)
                        
                        if response.status_code == 201:
                            st.success(f"✅ Train {train_id} added successfully!")
                            st.balloons()
                            st.rerun()
                        elif response.status_code == 409:
                            st.error(f"❌ Train ID {train_id} already exists!")
                        else:
                            st.error(f"❌ Failed to add train: {response.json().get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    with tab3:
        st.subheader("🗑️ Delete Train")
        
        st.warning("⚠️ This action cannot be undone!")
        
        train_ids = [t["train_id"] for t in trains]
        train_to_delete = st.selectbox("Select Train to Delete", train_ids)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Delete Train", type="primary"):
                if delete_train(train_to_delete):
                    st.success(f"✅ Train {train_to_delete} deleted successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to delete train {train_to_delete}")
        
        with col2:
            if st.button("🔄 Refresh List"):
                st.rerun()

# Footer
def add_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🚆 <strong>KMRL AI-Driven Train Induction Planning System</strong></p>
        <p>Smart India Hackathon 2025 | Kochi Metro Rail Limited</p>
        <p>Developed for optimization of train induction planning with AI-driven decision support</p>
    </div>
    """, unsafe_allow_html=True)

# Run App
if __name__ == "__main__":
    main()
    add_footer()
