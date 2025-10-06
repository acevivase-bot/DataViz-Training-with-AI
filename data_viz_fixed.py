
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import openai
import requests
import json
from io import StringIO

# ============= KONFIGURASI HALAMAN =============
st.set_page_config(
    page_title="DataIQ Pro - AI Enhanced",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .error-msg {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem 1.25rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============= FUNGSI UTILITAS =============

def get_openai_response_v1(prompt, data_context="", api_key="", model="gpt-3.5-turbo"):
    """Generate response from OpenAI API - Updated for v1.0+"""
    try:
        if not api_key:
            return "⚠️ OpenAI API key tidak dikonfigurasi. Silakan konfigurasi di sidebar."

        # Set API key
        openai.api_key = api_key

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

        # Try new v1.0+ API first
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
            # Fallback to old API style if new version not available
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

# ============= APLIKASI UTAMA =============

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 DataIQ Pro - AI Enhanced</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore your data with AI-powered insights and advanced visualizations</p>', unsafe_allow_html=True)

    # ============= SIDEBAR KONFIGURASI API =============
    st.sidebar.title("🤖 AI Configuration")

    # API Provider Selection
    api_provider = st.sidebar.selectbox(
        "🎯 Choose AI Provider:",
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

    # ============= FILE UPLOAD =============
    st.markdown("### 📤 Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Drop your CSV or Excel File here",
        type=['csv', 'xlsx'],
        help="Supported formats: CSV, Excel (XLSX). Maximum file size: 200MB"
    )

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
            st.markdown("### 📋 Data Preview")
            st.dataframe(data, use_container_width=True, height=300)

            # ============= BASIC INFORMATION TABS =============
            st.markdown("### 📊 Basic Information about the Dataset")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                '📈 Summary', '⬆️⬇️ Top and Bottom Rows', 
                '🏷️ Data Types', '📋 Columns', '❓ Null Values'
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

                st.markdown("#### 📊 Statistical Summary about the Dataset")
                if len(data.select_dtypes(include=[np.number]).columns) > 0:
                    st.dataframe(data.describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found for statistical summary.")

            with tab2:
                st.markdown("#### ⬆️ Top Rows")
                top_rows = st.slider('Number of top rows:', 1, min(data.shape[0], 100), 5, key='top')
                st.dataframe(data.head(top_rows), use_container_width=True)

                st.markdown("#### ⬇️ Bottom Rows") 
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
                st.markdown("#### ❓ Null Values in Columns")
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

            # ============= ADVANCED VISUALIZATION SECTION =============
            st.markdown("### 📈 Advanced Data Visualization")

            # Manual Visualization Builder
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
                if st.button("🚀 Generate Visualization", use_container_width=True):
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

            # ============= AI CHATBOT SECTION - MOVED TO BOTTOM =============
            st.markdown("### 🤖 AI Data Assistant")

            if not api_key:
                st.warning("⚠️ Please configure your API key in the sidebar to use AI features.")
            else:
                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Chat input at the bottom
                if prompt := st.chat_input("Ask anything about your data..."):
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
                    "Berikan ringkasan insights dari dataset ini",
                    "Identifikasi pola menarik dalam data",
                    "Analisis anomali atau outlier",
                    "Rekomendasikan visualisasi terbaik"
                ]

                for i, (col, quick_prompt) in enumerate(zip([col1, col2, col3, col4], quick_prompts)):
                    with col:
                        if st.button(f"💬 {quick_prompt}", key=f"quick_{i}", use_container_width=True):
                            st.session_state.messages.append({"role": "user", "content": quick_prompt})
                            st.rerun()

        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.info("Please make sure your file is a valid CSV or Excel file.")

    # ============= FOOTER =============
    st.markdown("---")
    st.markdown("**Created with ❤️ for Data Analysis Training**")

if __name__ == "__main__":
    main()
