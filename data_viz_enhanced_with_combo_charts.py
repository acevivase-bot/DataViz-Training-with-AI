import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import openai
import requests
import json
import hashlib
import time
from io import StringIO

# ============= KONFIGURASI HALAMAN =============
st.set_page_config(
    page_title="Training Data Visualization - AI Enhanced",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CHAT HISTORY MANAGEMENT =============

def manage_chat_history(uploaded_file):
    """Clear chat history when dataset changes"""
    if uploaded_file is not None:
        dataset_key = f"{uploaded_file.name}_{uploaded_file.size}"
    else:
        dataset_key = "no_dataset"

    # Check if dataset changed
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = dataset_key
        st.session_state.messages = []
    elif st.session_state.current_dataset != dataset_key:
        # Dataset changed - clear chat history
        st.session_state.current_dataset = dataset_key
        st.session_state.messages = []
        st.info("💬 Chat history cleared for new dataset")

def add_clear_chat_button():
    """Add manual clear chat button"""
    if st.session_state.get("messages", []):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.success("Chat history cleared!")
                st.rerun()
        with col3:
            chat_count = len(st.session_state.messages)
            st.info(f"💬 {chat_count} messages")

def create_session_isolation():
    """Create session isolation per browser"""
    if "session_id" not in st.session_state:
        timestamp = str(time.time())
        session_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        st.session_state.session_id = session_hash

    # Display session info in sidebar
    st.sidebar.markdown(f"🔐 Session: `{st.session_state.session_id}`")

def auto_clear_on_inactivity():
    """Auto clear chat after 30 minutes of inactivity"""
    current_time = time.time()
    idle_timeout = 1800  # 30 minutes

    if "last_activity" not in st.session_state:
        st.session_state.last_activity = current_time

    # Check if idle timeout exceeded
    if current_time - st.session_state.last_activity > idle_timeout:
        if st.session_state.get("messages", []):
            st.session_state.messages = []
            st.info("💬 Chat cleared due to 30 minutes inactivity")

    # Always update last activity
    st.session_state.last_activity = current_time

# ============= CSS STYLING =============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-msg {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .session-info {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
        margin: 1rem 0;
    }
    .creator-info {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============= AI RESPONSE FUNCTIONS =============
def get_openai_response_v1(prompt, data_context="", api_key="", model="gpt-3.5-turbo"):
    """Generate response from OpenAI API - Updated for v1.0+"""
    try:
        if not api_key:
            return "⚠️ OpenAI API key tidak dikonfigurasi. Silakan konfigurasi di sidebar."

        openai.api_key = api_key

        system_prompt = """Anda adalah AI Data Analyst Expert yang membantu menganalisis data. 
        Berikan insights yang mendalam, actionable recommendations, dan saran visualisasi yang sesuai.
        Jawab dalam bahasa Indonesia dengan gaya profesional namun mudah dipahami."""

        full_prompt = f"""
        Context Data: {data_context}

        User Query: {prompt}

        Berikan analisis yang komprehensif dengan:
        1. Ringkasan
        2. Key insights
        3. Actionable recommendations  
        4. Saran visualisasi jika diperlukan
        """

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )

            return response.choices[0].message.content

        except ImportError:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )

            return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        if "no longer supported" in error_msg:
            return """❌ Error OpenAI API: Versi library openai yang terinstall tidak kompatibel.

Solusi:
1. Update requirements.txt dengan: openai>=1.0.0
2. Atau gunakan versi lama: openai==0.28.0

Untuk sementara, silakan gunakan Perplexity API sebagai alternatif."""
        else:
            return f"❌ Error OpenAI: {error_msg}"

def get_perplexity_response(prompt, data_context="", api_key="", model="llama-3.1-sonar-small-128k-online"):
    """Generate response from Perplexity API"""
    try:
        if not api_key:
            return "⚠️ Perplexity API key tidak dikonfigurasi. Silakan konfigurasi di sidebar."

        url = "https://api.perplexity.ai/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        system_prompt = """Anda adalah AI Data Analyst Expert yang membantu menganalisis data. 
        Berikan insights yang mendalam, actionable recommendations, dan saran visualisasi yang sesuai.
        Jawab dalam bahasa Indonesia dengan gaya profesional namun mudah dipahami."""

        full_prompt = f"""
        Context Data: {data_context}

        User Query: {prompt}

        Berikan analisis yang komprehensif dengan:
        1. Ringkasan findings
        2. Key insights  
        3. Actionable recommendations
        4. Saran visualisasi jika diperlukan
        """

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"❌ Error Perplexity: {response.status_code} - {response.text}"

    except Exception as e:
        return f"❌ Error Perplexity: {str(e)}"

# ============= VISUALIZATION FUNCTIONS - ENHANCED =============
def create_visualization(data, graph_type, x_axis, y_axis, color=None, size=None, facet_col=None):
    """Create visualization based on parameters"""
    try:
        if graph_type == 'line':
            fig = px.line(data_frame=data, x=x_axis, y=y_axis, color=color, markers=True, 
                         template='plotly_white', title=f"{graph_type.title()} Chart: {x_axis} vs {y_axis}")
        elif graph_type == 'bar':
            fig = px.bar(data_frame=data, x=x_axis, y=y_axis, color=color, facet_col=facet_col,
                        barmode='group', template='plotly_white', title=f"{graph_type.title()} Chart: {x_axis} vs {y_axis}")
        elif graph_type == 'scatter':
            fig = px.scatter(data_frame=data, x=x_axis, y=y_axis, color=color, size=size,
                           template='plotly_white', title=f"{graph_type.title()} Plot: {x_axis} vs {y_axis}")
        elif graph_type == 'pie':
            fig = px.pie(data_frame=data, values=y_axis, names=x_axis, 
                        template='plotly_white', title=f"{graph_type.title()} Chart: {x_axis}")
        elif graph_type == 'histogram':
            fig = px.histogram(data_frame=data, x=x_axis, color=color, 
                             template='plotly_white', title=f"Histogram: {x_axis}")
        elif graph_type == 'box':
            fig = px.box(data_frame=data, x=x_axis, y=y_axis, color=color,
                        template='plotly_white', title=f"Box Plot: {x_axis} vs {y_axis}")
        else:
            return None

        # Update layout
        fig.update_layout(
            font=dict(size=12),
            title_font_size=16,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

def create_combination_chart(data, chart1_type, chart1_x, chart1_y, chart1_color, 
                           chart2_type, chart2_x, chart2_y, chart2_color):
    """Create combination chart with two different chart types"""
    try:
        # Create subplots with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Chart 1 (Primary y-axis)
        if chart1_type == 'bar':
            fig.add_trace(
                go.Bar(
                    x=data[chart1_x], 
                    y=data[chart1_y], 
                    name=f"Bar: {chart1_y}",
                    marker_color='#1f77b4' if not chart1_color else None,
                    opacity=0.8
                ),
                secondary_y=False
            )
        elif chart1_type == 'line':
            fig.add_trace(
                go.Scatter(
                    x=data[chart1_x], 
                    y=data[chart1_y], 
                    mode='lines+markers',
                    name=f"Line: {chart1_y}",
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=6)
                ),
                secondary_y=False
            )
        elif chart1_type == 'scatter':
            fig.add_trace(
                go.Scatter(
                    x=data[chart1_x], 
                    y=data[chart1_y], 
                    mode='markers',
                    name=f"Scatter: {chart1_y}",
                    marker=dict(color='#1f77b4', size=8, opacity=0.7)
                ),
                secondary_y=False
            )

        # Chart 2 (Secondary y-axis) 
        if chart2_type == 'line':
            fig.add_trace(
                go.Scatter(
                    x=data[chart2_x], 
                    y=data[chart2_y], 
                    mode='lines+markers',
                    name=f"Line: {chart2_y}",
                    line=dict(color='#ff7f0e', width=3, dash='dot'),
                    marker=dict(size=6)
                ),
                secondary_y=True
            )
        elif chart2_type == 'bar':
            fig.add_trace(
                go.Bar(
                    x=data[chart2_x], 
                    y=data[chart2_y], 
                    name=f"Bar: {chart2_y}",
                    marker_color='#ff7f0e',
                    opacity=0.6
                ),
                secondary_y=True
            )
        elif chart2_type == 'scatter':
            fig.add_trace(
                go.Scatter(
                    x=data[chart2_x], 
                    y=data[chart2_y], 
                    mode='markers',
                    name=f"Scatter: {chart2_y}",
                    marker=dict(color='#ff7f0e', size=10, opacity=0.6, symbol='diamond')
                ),
                secondary_y=True
            )

        # Update layout
        fig.update_layout(
            title=f"Combination Chart: {chart1_type.title()} + {chart2_type.title()}",
            template='plotly_white',
            font=dict(size=12),
            title_font_size=16,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )

        # Update y-axes labels
        fig.update_yaxes(title_text=f"{chart1_y}", secondary_y=False)
        fig.update_yaxes(title_text=f"{chart2_y}", secondary_y=True)
        fig.update_xaxes(title_text=f"{chart1_x}")

        return fig

    except Exception as e:
        st.error(f"Error creating combination chart: {e}")
        return None

# ============= MAIN APPLICATION =============
def main():
    # Initialize session management
    create_session_isolation()
    auto_clear_on_inactivity()

    # Header
    st.markdown('<h1 class="main-header">Training Data Visualization- AI Enhanced</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore your data with AI-powered insights and advanced visualizations</p>', unsafe_allow_html=True)

    # ============= SIDEBAR KONFIGURASI API =============
    st.sidebar.title("AI Configuration")

    # API Provider Selection
    api_provider = st.sidebar.selectbox(
        "Choose AI Provider:",
        ["OpenAI", "Perplexity"]
    )

    # API Key Configuration
    with st.sidebar.expander("🔑 API Configuration", expanded=True):
        if api_provider == "OpenAI":
            api_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                placeholder="sk-...",
                help="Masukkan OpenAI API key Anda"
            )
            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
            model = st.selectbox("Model:", model_options)

        else:  # Perplexity
            api_key = st.text_input(
                "Perplexity API Key:",
                type="password", 
                placeholder="pplx-...",
                help="Masukkan Perplexity API key Anda"
            )
            model_options = [
                "llama-3.1-sonar-small-128k-online",
                "llama-3.1-sonar-large-128k-online", 
                "llama-3.1-sonar-huge-128k-online"
            ]
            model = st.selectbox("Model:", model_options)

    # API Status Indicator
    if api_key:
        st.sidebar.success("✅ API Key configured!")
    else:
        st.sidebar.warning("⚠️ Please configure API key to use AI features")

    st.sidebar.markdown("---")

    # Creator information at bottom of sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="creator-info">
    <h4>👨‍💻 Creator Information</h4>
    <p><strong>Created by:</strong> Vito Devara</p>
    <p><strong>Phone:</strong> 081259795994</p>
    </div>
    """, unsafe_allow_html=True)

    # ============= FILE UPLOAD =============
    st.markdown("### 📤 Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Drop your CSV or Excel File here",
        type=['csv', 'xlsx'],
        help="Supported formats: CSV, Excel (XLSX). Maximum file size: 200MB"
    )

    # Manage chat history based on dataset
    manage_chat_history(uploaded_file)

    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            # Success message
            st.markdown('<div class="success-msg">✅ The file has been uploaded successfully!</div>', unsafe_allow_html=True)

            # Display data preview
            st.markdown("### Data Preview")
            st.dataframe(data, use_container_width=True, height=300)

            # ============= BASIC INFORMATION TABS =============
            st.markdown("### Basic Information about the Dataset")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                '📈 Summary', 'Top and Bottom Rows', 
                '🏷️ Data Types', '📋 Columns', 'Null Values'
            ])

            with tab1:
                # Dataset overview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f'<div class="metric-card"><h3>{data.shape[0]:,}</h3><p>Total Rows</p></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h3>{data.shape[1]}</h3><p>Total Columns</p></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card"><h3>{data.isnull().sum().sum()}</h3><p>Missing Values</p></div>', unsafe_allow_html=True)
                with col4:
                    numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
                    st.markdown(f'<div class="metric-card"><h3>{numeric_cols}</h3><p>Numeric Columns</p></div>', unsafe_allow_html=True)

                st.markdown("#### Statistical Summary about the Dataset")
                if len(data.select_dtypes(include=[np.number]).columns) > 0:
                    st.dataframe(data.describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found for statistical summary.")

            with tab2:
                st.markdown("#### Top Rows")
                top_rows = st.slider('Number of top rows:', 1, min(data.shape[0], 100), 5, key='top')
                st.dataframe(data.head(top_rows), use_container_width=True)

                st.markdown("#### Bottom Rows") 
                bottom_rows = st.slider('Number of bottom rows:', 1, min(data.shape[0], 100), 5, key='bottom')
                st.dataframe(data.tail(bottom_rows), use_container_width=True)

            with tab3:
                st.markdown("#### 🏷️ Data Types of Columns")
                dtype_df = pd.DataFrame({
                    'Column': data.dtypes.index,
                    'Data Type': data.dtypes.values,
                    'Non-Null Count': data.count().values,
                    'Null Count': data.isnull().sum().values
                })
                st.dataframe(dtype_df, use_container_width=True)

            with tab4:
                st.markdown("#### 📋 Column Names")
                cols_df = pd.DataFrame({
                    'Index': range(len(data.columns)),
                    'Column Name': data.columns,
                    'Data Type': data.dtypes.values
                })
                st.dataframe(cols_df, use_container_width=True)

            with tab5:
                st.markdown("#### Null Values in Columns")
                null_df = pd.DataFrame({
                    'Column': data.columns,
                    'Null Count': data.isnull().sum().values,
                    'Null Percentage': (data.isnull().sum() / len(data) * 100).round(2).values
                }).sort_values('Null Count', ascending=False)
                st.dataframe(null_df, use_container_width=True)

            # ============= COLUMN VALUE COUNTS =============
            st.markdown("### 📊 Column Values Analysis")
            with st.expander('🔍 Value Count Analysis', expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    column = st.selectbox('Choose Column:', options=list(data.columns))
                with col2:
                    top_rows = st.number_input('Top rows to show:', min_value=1, value=10, step=1)

                if st.button("📊 Analyze Values", use_container_width=True):
                    if data[column].dtype == 'object' or data[column].nunique() < 50:
                        result = data[column].value_counts().reset_index().head(top_rows)
                        result.columns = [column, 'count']

                        st.dataframe(result, use_container_width=True)

                        # Quick visualizations
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("#### 📊 Bar Chart")
                            fig_bar = px.bar(result, x=column, y='count', template='plotly_white')
                            st.plotly_chart(fig_bar, use_container_width=True)

                        with col2:
                            st.markdown("#### 📈 Line Chart")  
                            fig_line = px.line(result, x=column, y='count', markers=True, template='plotly_white')
                            st.plotly_chart(fig_line, use_container_width=True)

                        with col3:
                            st.markdown("#### 🥧 Pie Chart")
                            fig_pie = px.pie(result, names=column, values='count', template='plotly_white')
                            st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info(f"Column '{column}' has too many unique values ({data[column].nunique()}). Showing distribution instead.")
                        fig_hist = px.histogram(data, x=column, template='plotly_white')
                        st.plotly_chart(fig_hist, use_container_width=True)

            # ============= ADVANCED DATA VISUALIZATION SECTION - ENHANCED =============
            st.markdown("### 📈 Advanced Data Visualization")

            # Pilihan mode visualisasi
            viz_mode = st.radio(
                "Choose Visualization Mode:",
                ["Single Chart", "Combination Chart (Bar + Line)"],
                horizontal=True,
                help="Single Chart: One chart only | Combination Chart: Combine two charts in one figure"
            )

            if viz_mode == "Single Chart":
                # Manual Visualization Builder (ORIGINAL)
                with st.expander("🎨 Custom Visualization Builder", expanded=True):
                    # Graph type selection
                    col1, col2 = st.columns(2)

                    with col1:
                        graph_type = st.selectbox(
                            'Choose your graph:',
                            options=['line', 'bar', 'scatter', 'pie', 'histogram', 'box'],
                            format_func=lambda x: f"📊 {x.title()} Chart" if x != 'scatter' else "🔍 Scatter Plot"
                        )

                    with col2:
                        if graph_type in ['line', 'bar', 'scatter', 'box']:
                            x_axis = st.selectbox('Choose X axis:', options=list(data.columns))
                        elif graph_type == 'histogram':
                            x_axis = st.selectbox('Choose Column:', options=list(data.columns))
                        else:  # pie
                            x_axis = st.selectbox('Choose Category Column:', options=list(data.columns))

                    # Y-axis selection (when needed)
                    if graph_type in ['line', 'bar', 'scatter', 'box']:
                        col1, col2 = st.columns(2)
                        with col1:
                            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                            y_axis = st.selectbox('Choose Y axis:', options=numeric_columns if numeric_columns else list(data.columns))

                        with col2:
                            color_options = [None] + list(data.columns)
                            color = st.selectbox('Color Information:', options=color_options)
                    elif graph_type == 'pie':
                        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                        y_axis = st.selectbox('Choose Values Column:', options=numeric_columns if numeric_columns else list(data.columns))
                        color = None
                    else:  # histogram
                        y_axis = None
                        color_options = [None] + list(data.columns)
                        color = st.selectbox('Color Information:', options=color_options)

                    # Generate visualization
                    if st.button("🚀 Generate Single Visualization", use_container_width=True):
                        with st.spinner("Creating visualization..."):
                            if graph_type in ['line', 'bar', 'scatter', 'box'] and y_axis is None:
                                st.error("Please select a Y-axis column.")
                            elif graph_type == 'pie' and y_axis is None:
                                st.error("Please select a values column for pie chart.")
                            else:
                                # Prepare data for visualization
                                viz_data = data.copy()

                                # For categorical analysis, use value counts
                                if graph_type in ['bar', 'line', 'pie'] and data[x_axis].dtype == 'object':
                                    if graph_type == 'pie':
                                        viz_data = data[x_axis].value_counts().reset_index()
                                        viz_data.columns = [x_axis, 'count']
                                        y_axis = 'count'

                                fig = create_visualization(viz_data, graph_type, x_axis, y_axis, color)

                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error("Could not generate visualization. Please check your data and selections.")

            else:  # Combination Chart Mode
                with st.expander("🎨 Combination Chart Builder", expanded=True):
                    st.info("💡 **Perfect for business reporting!** Combine different chart types to show multiple metrics. Example: Bar chart for volume + Line chart for trend over time.")

                    # Chart 1 Configuration
                    st.markdown("#### 📊 Chart 1 Configuration")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        chart1_type = st.selectbox(
                            'Chart 1 Type:',
                            options=['bar', 'line', 'scatter'],
                            format_func=lambda x: f"📊 {x.title()} Chart" if x != 'scatter' else "🔍 Scatter Plot",
                            key="chart1_type"
                        )

                    with col2:
                        chart1_x = st.selectbox('Chart 1 X-axis:', options=list(data.columns), key="chart1_x")

                    with col3:
                        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                        chart1_y = st.selectbox('Chart 1 Y-axis:', options=numeric_columns if numeric_columns else list(data.columns), key="chart1_y")

                    # Chart 1 Color (optional)
                    chart1_color_options = ["None"] + list(data.columns)
                    chart1_color = st.selectbox('Chart 1 Color (optional):', options=chart1_color_options, key="chart1_color")
                    chart1_color = None if chart1_color == "None" else chart1_color

                    st.markdown("---")

                    # Chart 2 Configuration
                    st.markdown("#### 📈 Chart 2 Configuration")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        chart2_type = st.selectbox(
                            'Chart 2 Type:',
                            options=['None'] + ['line', 'bar', 'scatter'],
                            format_func=lambda x: "➖ None (Single Chart Only)" if x == 'None' else f"📈 {x.title()} Chart" if x != 'scatter' else "🔍 Scatter Plot",
                            key="chart2_type"
                        )

                    if chart2_type != 'None':
                        with col2:
                            chart2_x = st.selectbox('Chart 2 X-axis:', options=list(data.columns), key="chart2_x")

                        with col3:
                            chart2_y = st.selectbox('Chart 2 Y-axis:', options=numeric_columns if numeric_columns else list(data.columns), key="chart2_y")

                        # Chart 2 Color (optional)
                        chart2_color_options = ["None"] + list(data.columns)
                        chart2_color = st.selectbox('Chart 2 Color (optional):', options=chart2_color_options, key="chart2_color")
                        chart2_color = None if chart2_color == "None" else chart2_color
                    else:
                        chart2_x = chart2_y = chart2_color = None

                    # Business Use Case Examples
                    with st.expander("💼 Business Use Case Examples", expanded=False):
                        st.markdown("""
                        **Typical Business Combinations:**
                        - **Sales Analysis**: Bar chart (monthly sales volume) + Line chart (profit margin trend)  
                        - **Risk Management**: Bar chart (number of risks) + Line chart (average risk rating over time)
                        - **Performance Monitoring**: Bar chart (transaction count) + Line chart (success rate %)
                        - **Financial Reporting**: Bar chart (revenue by quarter) + Line chart (growth rate %)
                        - **Operational Metrics**: Bar chart (incident count) + Line chart (resolution time trend)
                        """)

                    # Generate combination visualization
                    if st.button("🚀 Generate Combination Chart", use_container_width=True):
                        with st.spinner("Creating combination visualization..."):
                            if chart1_y is None:
                                st.error("Please select Y-axis for Chart 1.")
                            elif chart2_type != 'None' and chart2_y is None:
                                st.error("Please select Y-axis for Chart 2.")
                            else:
                                # Prepare data for visualization
                                viz_data = data.copy()

                                # Handle categorical X-axis data
                                if data[chart1_x].dtype == 'object' and chart1_type in ['bar', 'line']:
                                    if chart2_type == 'None':
                                        # Single chart with categorical X
                                        viz_data = data[chart1_x].value_counts().reset_index()
                                        viz_data.columns = [chart1_x, 'count']
                                        chart1_y = 'count'

                                        fig = create_visualization(viz_data, chart1_type, chart1_x, chart1_y, chart1_color)
                                    else:
                                        st.warning("⚠️ For combination charts, please ensure X-axis columns contain numeric or date data for best results.")
                                        fig = create_combination_chart(viz_data, chart1_type, chart1_x, chart1_y, chart1_color,
                                                                     chart2_type, chart2_x, chart2_y, chart2_color)
                                else:
                                    # Create combination or single chart
                                    if chart2_type == 'None':
                                        fig = create_visualization(viz_data, chart1_type, chart1_x, chart1_y, chart1_color)
                                    else:
                                        fig = create_combination_chart(viz_data, chart1_type, chart1_x, chart1_y, chart1_color,
                                                                     chart2_type, chart2_x, chart2_y, chart2_color)

                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Show insights about the chart
                                    if chart2_type != 'None':
                                        st.success(f"✅ **Combination Chart Created!** {chart1_type.title()} ({chart1_y}) + {chart2_type.title()} ({chart2_y})")
                                        st.info("💡 **Chart Reading Tips**: Left Y-axis shows Chart 1 values, Right Y-axis shows Chart 2 values. Hover over data points for detailed information.")
                                    else:
                                        st.success(f"✅ **Single Chart Created!** {chart1_type.title()} showing {chart1_y}")
                                else:
                                    st.error("Could not generate visualization. Please check your data and selections.")

            # ============= AI CHATBOT SECTION - WITH IMPROVED HISTORY MANAGEMENT =============
            st.markdown("### 🤖 AI Data Assistant")

            # Add clear chat button and session info
            add_clear_chat_button()

            if not api_key:
                st.warning("⚠️ Please configure your API key in the sidebar to use AI features.")
            else:
                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Chat input
                if prompt := st.chat_input("Ask anything about your data..."):
                    # Update activity timestamp
                    st.session_state.last_activity = time.time()

                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    # Generate AI response
                    with st.spinner(f"🤖 {api_provider} is analyzing your data..."):
                        # Prepare data context
                        data_context = f"""
                        Dataset: {uploaded_file.name}
                        Shape: {data.shape}
                        Columns: {', '.join(data.columns[:10])}{'...' if len(data.columns) > 10 else ''}

                        Sample Data:
                        {data.head(3).to_string()}

                        Statistical Summary:
                        {data.describe().to_string() if len(data.select_dtypes(include=[np.number]).columns) > 0 else 'No numeric columns'}
                        """

                        if api_provider == "OpenAI":
                            response = get_openai_response_v1(prompt, data_context, api_key, model)
                        else:
                            response = get_perplexity_response(prompt, data_context, api_key, model)

                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})

                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Quick prompt buttons
                st.markdown("#### 💡 Quick Analysis Prompts")
                col1, col2, col3, col4 = st.columns(4)

                quick_prompts = [
                    "Berikan ringkasan dari dataset ini",
                    "Identifikasi pola menarik dalam data",
                    "Analisis anomali atau outlier",
                    "Rekomendasikan visualisasi terbaik"
                ]

                for i, (col, quick_prompt) in enumerate(zip([col1, col2, col3, col4], quick_prompts)):
                    with col:
                        if st.button(f"💬 {quick_prompt}", key=f"quick_{i}", use_container_width=True):
                            st.session_state.messages.append({"role": "user", "content": quick_prompt})
                            st.session_state.last_activity = time.time()
                            st.rerun()

        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.info("Please make sure your file is a valid CSV or Excel file.")

    # ============= FOOTER =============
    st.markdown("---")
    st.markdown("**Created for Data Analysis Training**")

    # Show session info in footer
    if "session_id" in st.session_state:
        st.markdown(f'<div class="session-info">🔐 Your Session: {st.session_state.session_id} | 💬 Chat Messages: {len(st.session_state.get("messages", []))}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
