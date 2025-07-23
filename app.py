import subprocess
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import requests
import folium
from pygments.unistring import No
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    # initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .nav-link {
        text-decoration: none;
        color: black;
        padding: 10px;
        border-radius: 5px;
    }
    .nav-link:hover {
        background-color: #f0f2f6;
    }
    .active-nav {
        background-color: #f0f2f6;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Crop Recommendation"

# Weather API key
WEATHER_API_KEY = "c5cbf69ebab449a8a44cd06422de3cf4"


# Load the crop recommendation models
@st.cache_resource
def load_crop_models():
    try:
        with open('crop_models_comparison.pkl', 'rb') as f:
            results, scaler, crop_dict = pickle.load(f)
        return results, scaler, crop_dict
    except Exception as e:
        st.error("Error loading crop models. Please ensure 'crop_models_comparison.pkl' is in the same directory.")
        return None, None, None


# Weather data fetching function
@st.cache_data
def get_agweather(lat, lon):
    url = f"https://api.weatherbit.io/v2.0/forecast/agweather?lat={lat}&lon={lon}&key={WEATHER_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()['data'][0]
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None


def crop_recommendation_page():

    # Custom CSS to improve appearance
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton > button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px;
        }
        .stAlert {
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    # Load the saved models
    @st.cache_resource
    def load_models():
        with open('crop_models_comparison.pkl', 'rb') as f:
            results, scaler, crop_dict = pickle.load(f)
        return results, scaler, crop_dict

    # Load models
    try:
        results, scaler, crop_dict = load_models()
        inverse_crop_dict = {v: k for k, v in crop_dict.items()}
    except Exception as e:
        st.error("Error loading models. Please ensure 'crop_models_comparison.pkl' is in the same directory.")
        st.stop()

    # Create DataFrame for model comparison
    def create_comparison_df():
        comparison_df = pd.DataFrame({
            'Test Accuracy (%)': [results[name]['accuracy'] * 100 for name in results.keys()],
            'CV Mean Accuracy (%)': [results[name]['cv_scores'].mean() * 100 for name in results.keys()]
        }).round(2)
        comparison_df.index = results.keys()
        return comparison_df

    # Create bar graph for model comparison
    def create_accuracy_comparison_plot(comparison_df):
        fig = go.Figure()

        # Add bars for Test Accuracy
        fig.add_trace(go.Bar(
            x=comparison_df.index,
            y=comparison_df['Test Accuracy (%)'],
            name='Test Accuracy',
            text=comparison_df['Test Accuracy (%)'].round(1),
            textposition='auto',
        ))

        # Add bars for CV Mean Accuracy
        fig.add_trace(go.Bar(
            x=comparison_df.index,
            y=comparison_df['CV Mean Accuracy (%)'],
            name='CV Mean Accuracy',
            text=comparison_df['CV Mean Accuracy (%)'].round(1),
            textposition='auto',
        ))

        # Update layout
        fig.update_layout(
            title="Model Comparison: Test Accuracy vs CV Mean Accuracy",
            xaxis_title="Models",
            yaxis_title="Accuracy (%)",
            barmode='group',
            height=400,
            showlegend=True,
            xaxis_tickangle=-45
        )

        return fig

    # Title
    st.title("🌾 Crop Recommendation System")

    # Display model comparison at the top
    st.subheader("Model Performance Comparison")
    comparison_df = create_comparison_df()
    fig_accuracy = create_accuracy_comparison_plot(comparison_df)
    st.plotly_chart(fig_accuracy, use_container_width=True)

    st.markdown("""
    This application uses multiple machine learning models to recommend suitable crops based on soil and climate parameters.
    Enter the parameters in the sidebar to get crop recommendations.
    """)

    # Sidebar inputs
    st.sidebar.title("Input Parameters")

    # Soil Parameters in sidebar
    st.sidebar.subheader("Soil Parameters")
    n = st.sidebar.number_input("Nitrogen (N) content in soil", 0, 150, 90)
    p = st.sidebar.number_input("Phosphorous (P) content in soil", 0, 150, 42)
    k = st.sidebar.number_input("Potassium (K) content in soil", 0, 150, 43)
    ph = st.sidebar.slider("Soil pH value", 0.0, 14.0, 6.5, 0.1)

    # Climate Parameters in sidebar
    st.sidebar.subheader("Climate Parameters")
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 20.87, 0.01)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 82.0, 0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 202.93, 0.01)

    # Function to make predictions
    def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        predictions = {}
        confidence_scores = {}

        for name, model_info in results.items():
            model = model_info['model']
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)
                prediction = model.predict(input_scaled)[0]
                confidence = proba[0][prediction - 1] * 100  # -1 because labels start from 1
            else:
                prediction = model.predict(input_scaled)[0]
                confidence = None

            predictions[name] = inverse_crop_dict[prediction]
            confidence_scores[name] = confidence

        return predictions, confidence_scores

    # Create pie chart for consensus
    def create_consensus_pie_chart(predictions):
        from collections import Counter
        prediction_counts = Counter(predictions.values())
        labels = list(prediction_counts.keys())
        values = list(prediction_counts.values())

        hover_text = [f"{label}: {count} models" for label, count in zip(labels, values)]
        colors = px.colors.qualitative.Set3[:len(labels)]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hovertext=hover_text,
            textinfo='percent',
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(colors=colors)
        )])

        fig.update_layout(
            title="Consensus Distribution",
            showlegend=True,
            height=400,
            margin=dict(t=40, b=0, l=0, r=0)
        )

        return fig

    # Make prediction when user clicks the button
    if st.sidebar.button("Get Crop Recommendations"):
        with st.spinner("Analyzing parameters..."):
            predictions, confidence_scores = predict_crop(n, p, k, temperature, humidity, ph, rainfall)

            # Calculate consensus
            from collections import Counter

            consensus = Counter(predictions.values()).most_common(1)[0]

            # Create two columns for main content
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Individual Model Predictions")

                # Color mapping for confidence levels
                def get_confidence_color(confidence):
                    if confidence is None:
                        return "gray"
                    elif confidence >= 90:
                        return "green"
                    elif confidence >= 70:
                        return "orange"
                    else:
                        return "red"

                # Create a DataFrame for model predictions
                pred_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Predicted Crop': list(predictions.values()),
                    'Confidence': [confidence_scores[model] for model in predictions.keys()]
                })

                # Display predictions in a styled table
                for idx, row in pred_df.iterrows():
                    confidence = row['Confidence']
                    confidence_color = get_confidence_color(confidence)

                    st.markdown(f"""
                        <div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 5px 0;'>
                            <strong>{row['Model']}</strong><br/>
                            Prediction: <strong>{row['Predicted Crop'].title()}</strong><br/>
                            {f"Confidence: <span style='color:{confidence_color}'>{confidence:.1f}%</span>" if confidence is not None else ""}
                        </div>
                    """, unsafe_allow_html=True)

            with col2:
                # Display pie chart
                fig = create_consensus_pie_chart(predictions)
                st.plotly_chart(fig, use_container_width=True)

                # Display consensus prediction
                st.subheader("Consensus Prediction")
                st.markdown(f"""
                    <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                        <h3 style='color: #1f77b4; margin: 0'>{consensus[0].title()}</h3>
                        <p style='color: #1f77b4; margin: 10px 0 0 0;'>{consensus[1]}/{len(predictions)} models agree</p>
                    </div>
                """, unsafe_allow_html=True)

            # Display input parameters summary
            st.subheader("Input Parameters Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Soil Parameters:**")
                st.markdown(f"- Nitrogen (N): {n}")
                st.markdown(f"- Phosphorous (P): {p}")
                st.markdown(f"- Potassium (K): {k}")
                st.markdown(f"- pH: {ph}")
            with col2:
                st.markdown("**Climate Parameters:**")
                st.markdown(f"- Temperature: {temperature}°C")
                st.markdown(f"- Humidity: {humidity}%")
                st.markdown(f"- Rainfall: {rainfall}mm")

    # Add footer with information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>This crop recommendation system uses multiple machine learning models including Random Forest, SVM, SGD, Decision Tree, and Logistic Regression.</p>
        <p>The recommendations are based on soil composition (N, P, K, pH) and environmental factors (temperature, humidity, rainfall).</p>
    </div>
    """, unsafe_allow_html=True)

def smart_farmer_page():
    # Function to check and install spacy and the required model
    def install_spacy_model():
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
                return nlp
            except OSError:
                st.warning("Installing required language model... This may take a moment.")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "en_core_web_sm"])
                import spacy
                return spacy.load("en_core_web_sm")
        except Exception as e:
            st.error(f"""
            Error setting up NLP components. Please ensure you have the required dependencies:
            1. Run: pip install spacy
            2. Run: python -m spacy download en_core_web_sm

            Error details: {str(e)}
            """)
            return No

    # Load Spacy NLP model and cache it using st.cache_resource
    @st.cache_resource
    def load_nlp_model():
        return install_spacy_model()

    # Initialize NLP
    nlp = load_nlp_model()

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        .metric-card {
            background-color: #342e37;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .css-1v0mbdj.e115fcil1 {
            max-width: 100%;
        }
    }
        </style>
        """, unsafe_allow_html=True)

    # API key (replace with your actual API key)
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

    @st.cache_data
    def get_agweather(lat, lon):
        url = f"https://api.weatherbit.io/v2.0/forecast/agweather?lat={lat}&lon={lon}&key={WEATHER_API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data['data'][0]
        except requests.RequestException as e:
            st.error(f"Error fetching agricultural weather data: {e}")
            return None
        except (KeyError, IndexError) as e:
            st.error(f"Error processing weather data: {e}")
            return None

    def recommend_crop(weather_data):
        if weather_data is None:
            return "I'm sorry, I couldn't fetch the weather data. Please try again later.", {}

        try:
            temp = weather_data['temp_2m_avg']
            soil_moisture = weather_data['v_soilm_0_10cm']
            precipitation = weather_data['precip']

            recommendations = []

            if temp > 25 and soil_moisture > 0.3 and precipitation < 5:
                recommendations.extend(["Corn", "Soybeans", "Sunflowers", "Sorghum", "Cotton"])
            elif 15 <= temp <= 25 and 0.2 <= soil_moisture <= 0.3 and 5 <= precipitation < 10:
                recommendations.extend(["Wheat", "Barley", "Oats", "Rye", "Canola"])
            elif temp < 15 and soil_moisture > 0.3 and precipitation >= 10:
                recommendations.extend(["Lettuce", "Spinach", "Peas", "Broccoli", "Carrots"])
            else:
                recommendations.extend(["Millet", "Quinoa", "Amaranth", "Buckwheat", "Chickpeas"])

            selected_crops = random.sample(recommendations, min(3, len(recommendations)))

            conditions = {
                "Temperature": f"{temp}°C",
                "Soil Moisture": f"{soil_moisture:.2f}",
                "Precipitation": f"{precipitation}mm"
            }

            return selected_crops, conditions
        except KeyError as e:
            st.error(f"Error processing weather data for crop recommendations: Missing {e}")
            return [], {}
        except Exception as e:
            st.error(f"Unexpected error in crop recommendations: {e}")
            return [], {}

    def recommend_fertilizer(weather_data, crop):
        if weather_data is None:
            return ["I'm sorry, I couldn't fetch the weather data. Please try again later."]

        try:
            soil_moisture = weather_data['v_soilm_0_10cm']
            precipitation = weather_data['precip']

            npk_ratios = {
                "corn": "10-5-5",
                "soybeans": "3-8-5",
                "wheat": "20-10-10",
                "cotton": "5-10-15",
                "rice": "15-15-15"
            }

            recommendations = []

            # Base recommendation
            if crop.lower() in npk_ratios:
                recommendations.append(f"✓ Use a {npk_ratios[crop.lower()]} NPK fertilizer")
            else:
                recommendations.append("✓ Use a balanced NPK fertilizer")

            # Moisture-based recommendations
            if soil_moisture < 0.2:
                recommendations.append("✓ Apply slow-release fertilizer to prevent nutrient loss")
            elif soil_moisture > 0.4:
                recommendations.append("✓ Reduce nitrogen application due to high soil moisture")

            # Precipitation-based recommendations
            if precipitation > 10:
                recommendations.append("✓ Split fertilizer application to reduce runoff")
            elif precipitation < 5:
                recommendations.append("✓ Irrigate after fertilizer application")

            return recommendations
        except KeyError as e:
            st.error(f"Error processing weather data for fertilizer recommendations: Missing {e}")
            return ["Unable to process weather data for fertilizer recommendations"]
        except Exception as e:
            st.error(f"Unexpected error in fertilizer recommendations: {e}")
            return ["Error generating fertilizer recommendations"]

    def get_weather_info(weather_data):
        if weather_data is None:
            return None, None

        try:
            metrics = {
                "Temperature": f"{weather_data['temp_2m_avg']:.1f}°C",
                "Soil Moisture": f"{weather_data['v_soilm_0_10cm']:.2f}",
                "Precipitation": f"{weather_data['precip']:.1f} mm",
                "Wind Speed": f"{weather_data['wind_10m_spd_avg']:.1f} m/s",
                "Evapotranspiration": f"{weather_data['evapotranspiration']:.1f} mm"
            }

            alerts = []
            if weather_data['temp_2m_avg'] > 30:
                alerts.append("⚠️ High temperature alert - Ensure adequate irrigation")
            elif weather_data['temp_2m_avg'] < 10:
                alerts.append("❄️ Low temperature alert - Watch for frost damage")

            if weather_data['precip'] > 20:
                alerts.append("🌧️ Heavy rainfall alert - Check drainage systems")
            elif weather_data['precip'] < 1:
                alerts.append("☀️ Dry conditions alert - Consider irrigation")

            if weather_data['wind_10m_spd_avg'] > 8:
                alerts.append("💨 Strong wind alert - Protect sensitive crops")

            return metrics, alerts
        except KeyError as e:
            st.error(f"Error processing weather data for weather info: Missing {e}")
            return None, None
        except Exception as e:
            st.error(f"Unexpected error in weather info: {e}")
            return None, None

    # Main UI
    st.title("🌾 Smart Farmer Assistant")
    st.markdown("### Your AI-powered farming companion")

    # Create two columns for the main layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📍 Location Settings")
        lat = st.number_input("Latitude", value=19.0760, format="%.4f")
        lon = st.number_input("Longitude", value=72.8777, format="%.4f")

        st.markdown("<div style='margin-top: 40px'></div>", unsafe_allow_html=True)

    with col2:
        # Fetch weather data
        weather_data = get_agweather(lat, lon)

        # Weather Information Card
        st.markdown("### 🌤️ Current Agricultural Conditions")
        if weather_data:
            metrics, alerts = get_weather_info(weather_data)
            if metrics:
                cols = st.columns(len(metrics))
                for col, (metric, value) in zip(cols, metrics.items()):
                    col.metric(metric, value)

            if alerts:
                st.markdown("### ⚠️ Weather Alerts")
                for alert in alerts:
                    st.warning(alert)

    try:
        # Create a map
        m = folium.Map(location=[lat, lon], zoom_start=10)
        folium.Marker([lat, lon]).add_to(m)
        folium_static(m,width=900)
    except Exception as e:
        st.error(f"Error displaying map: {e}")

    # Create tabs for different features
    tab1, tab2 = st.tabs(["🌱 Crop Recommendations", "🧪 Fertilizer Advice"])

    with tab1:
        if weather_data:
            recommended_crops, conditions = recommend_crop(weather_data)

            if recommended_crops:
                st.markdown("### 🌱 Recommended Crops")
                crop_cols = st.columns(len(recommended_crops))
                for col, crop in zip(crop_cols, recommended_crops):
                    col.markdown(f"""
                    <div class="metric-card">
                        <h3 style="text-align: center;">{crop}</h3>
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🧪 Fertilizer Recommendations")
        crop = st.selectbox(
            "Select your crop",
            ["Corn", "Soybeans", "Wheat", "Cotton", "Rice", "Other"]
        )

        if crop and weather_data:
            recommendations = recommend_fertilizer(weather_data, crop)
            st.markdown("#### Recommended Fertilizer Strategy:")
            for rec in recommendations:
                st.markdown(f"<div class='metric-card'>{rec}</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>💡 Remember to consult with local agricultural experts for personalized advice.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def smart_tracker_page():
    # Add custom CSS
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                width: 100%;
            }
            .stSelectbox {
                margin-bottom: 20px;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    """, unsafe_allow_html=True)

    def get_location_details():
        """Get location details from user input"""
        col1, col2 = st.columns(2)

        with col1:
            state = st.selectbox(
                "Select State",
                options=["Maharashtra", "Karnataka", "Gujarat", "Punjab"],
                key="state_select"
            )

        with col2:
            district = st.selectbox(
                "Select District",
                get_districts_for_state(state),
                key="district_select"
            )

        return state, district

    def get_districts_for_state(state):
        """Return districts for selected state"""
        districts_map = {
            "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
            "Karnataka": ["Bangalore", "Mysore", "Hubli"],
            "Gujarat": ["Ahmedabad", "Surat", "Vadodara"],
            "Punjab": ["Amritsar", "Ludhiana", "Jalandhar"]
        }
        return districts_map.get(state, [])

    def fetch_commodity_data(state, district, commodity, date_range):
        """
        Fetch commodity price data from the API
        """
        try:
            # API endpoint from the documentation
            url = "https://api.agmarknet.gov.in/get-commodity-prices/"

            payload = {
                "commodity": commodity,
                "date_from": date_range[0].strftime("%Y-%m-%d"),
                "date_to": date_range[1].strftime("%Y-%m-%d"),
                "states": [state],
                "districts": [district],
                "markets": []
            }

            # For demonstration, using sample data
            # In production, uncomment the following:
            # response = requests.post(url, json=payload)
            # data = response.json()

            # Sample data for demonstration
            data = {
                "prices": [
                    {
                        "date": (date_range[0] + timedelta(days=x)).strftime("%Y-%m-%d"),
                        "price": 1000 + (x * 50) + (random.randint(-100, 100)),
                        "market": "Sample Market"
                    } for x in range((date_range[1] - date_range[0]).days + 1)
                ]
            }

            return pd.DataFrame(data["prices"])

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None

    def display_metrics(df):
        """Display key metrics"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Average Price",
                f"₹{df['price'].mean():.2f}",
                f"{((df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0] * 100):.1f}%"
            )

        with col2:
            st.metric(
                "Current Price",
                f"₹{df['price'].iloc[-1]:.2f}"
            )

        with col3:
            st.metric(
                "Lowest Price",
                f"₹{df['price'].min():.2f}"
            )

        with col4:
            st.metric(
                "Highest Price",
                f"₹{df['price'].max():.2f}"
            )

    def main():
        # Page title
        st.title("📈 Commodity Price Tracker")
        st.markdown("Track commodity prices across different locations in India")

        # Sidebar filters
        st.sidebar.header("Filters")

        # Date range selection
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            key="date_range",
            max_value=datetime.now()
        )

        # Commodity selection
        commodity = st.sidebar.selectbox(
            "Select Commodity",
            options=["Wheat", "Rice", "Maize", "Potato", "Onion", "Tomato"],
            key="commodity_select"
        )

        # Main content
        state, district = get_location_details()

        # Fetch data button
        if st.button("Fetch Price Data", key="fetch_button"):
            with st.spinner("Fetching data..."):
                df = fetch_commodity_data(state, district, commodity, date_range)

                if df is not None:
                    # Display metrics
                    st.subheader("Price Metrics")
                    display_metrics(df)

                    # Display price trend
                    st.subheader("Price Trend")
                    fig = px.line(
                        df,
                        x="date",
                        y="price",
                        title=f"{commodity} Price Trend in {district}, {state}",
                        labels={"date": "Date", "price": "Price (₹)"}
                    )
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display data table
                    st.subheader("Detailed Price Data")
                    st.dataframe(
                        df.style.format({
                            "price": "₹{:.2f}"
                        })
                    )

                    # Download button
                    st.download_button(
                        label="Download Data",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name=f'{commodity}_prices_{state}_{district}.csv',
                        mime='text/csv'
                    )

    if __name__ == "__main__":
        main()

# Navigation bar
st.sidebar.title("Navigation")
pages = {
    "Crop Recommendation": crop_recommendation_page,
    "Smart Farmer Assistant": smart_farmer_page,
    "Commodity Price Tracker": smart_tracker_page
}

# Create navigation buttons
for page_name in pages:
    if st.sidebar.button(
            page_name,
            key=page_name,
            help=f"Navigate to {page_name}",
            type="primary" if st.session_state.current_page == page_name else "secondary"
    ):
        st.session_state.current_page = page_name
        st.rerun()

# Display current page
pages[st.session_state.current_page]()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>💡 For best results, consult with local agricultural experts.</p>
    </div>
""", unsafe_allow_html=True)
