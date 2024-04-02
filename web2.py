import streamlit as st
import ee
import folium
import folium.plugins
from streamlit_folium import folium_static
import pandas as pd
import plotly.express as px
import json
import numpy as np
import calendar
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import google.generativeai as genai

st.set_page_config(layout="wide")
st.title("🛰️ Sentinel-2 Imagery: AI-Powered Time Series Forecasting Extravaganza 🌍🌱")

genai.configure(api_key="AIzaSyB-C5_2h5nQfIBUYKjBxKs_m55lWTRDnRg")
# Authenticate and initialize Earth Engine
# ee.Authenticate()  # Uncomment this if you haven't authenticated Earth Engine in your environment
ee.Initialize(project='ee-aashishkawade')

# Map initialization
m = folium.Map(location=[20, 0], zoom_start=2)
folium.plugins.Draw(export=True).add_to(m)
folium_static(m,width=1000, height=500)

# Function to add NDVI, EVI, SAVI, and GNDVI bands
def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}).rename('EVI')
    savi = image.expression(
        '(1 + L) * (NIR - RED) / (NIR + RED + L)',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'L': 0.5}).rename('SAVI')
    gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
    return image.addBands([ndvi, evi, savi, gndvi])
def mask_clouds(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).Or(qa.bitwiseAnd(1 << 11))
    return image.updateMask(cloud_mask.Not())
# Load and process Sentinel-2 data
def load_s2_data(start_date, end_date, aoi_rectangle):
    collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterDate(ee.Date(start_date), ee.Date(end_date))
                  .filterBounds(aoi_rectangle)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_percentage))
                  .map(add_indices))

    if apply_cloud_masking:
        collection = collection.map(mask_clouds)  # Apply cloud masking only if chosen

    return collection
# Function to extract mean NDVI, EVI, SAVI, and GNDVI
def extract_mean(image, aoi_rectangle):
    mean_dict = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi_rectangle, scale=30, maxPixels=1e9,
                                   bestEffort=True)
    return image.set('date', image.date().format()).set(mean_dict)

# Function to calculate statistical measures
def calculate_statistics(df):
    stats = df.describe().loc[['mean', 'std']]
    return stats
def create_pie_chart(df, index_name):
    monthly_data = df.groupby(df['date'].dt.month)[index_name].mean()
    monthly_data.index = [calendar.month_abbr[i] for i in monthly_data.index]
    fig = px.pie(monthly_data, values=index_name, names=monthly_data.index, title=f'Monthly Average {index_name}')
    return fig

def create_bar_chart(df, index_name):
    fig = px.bar(df, x='date', y=index_name, title=f'Daily {index_name}')
    return fig
# Streamlit app layout



def merge_past_future_data(df, future_df):
    # Concatenate past and future data
    combined_df = pd.concat([df, future_df], ignore_index=True)
    return combined_df

def train_and_predict_future_trends(df, days_to_predict=30):
    # Convert dates to ordinal numbers for regression
    df['date_ordinal'] = pd.to_datetime(df['date']).apply(lambda date: date.toordinal())

    # Prepare training data
    X = df[['date_ordinal']]
    future_dates = [df['date_ordinal'].iloc[-1] + i for i in range(1, days_to_predict + 1)]

    # Train models and predict
    trends = {}
    for index in ['NDVI', 'EVI']:
        y = df[index]
        model = LinearRegression().fit(X, y)
        future_predictions = model.predict(np.array(future_dates).reshape(-1, 1))
        trends[index] = future_predictions

    return pd.DataFrame({'date': [df['date'].iloc[-1] + timedelta(days=i) for i in range(1, days_to_predict + 1)],
                         'NDVI': trends['NDVI'],
                         'EVI': trends['EVI']})

def get_tile_layer_url(ee_image, vis_params):
    map_id_dict = ee.Image(ee_image).getMapId(vis_params)
    tile_url = map_id_dict['tile_fetcher'].url_format
    return tile_url

# Sidebar for user inputs and GeoJSON file upload
with st.sidebar:
    st.title("NDVI, EVI, SAVI, and GNDVI Time Series Analysis")
    st.title("User Inputs")
    start_date = st.date_input('Start date', value=pd.to_datetime('2023-06-01')).strftime('%Y-%m-%d')
    end_date = st.date_input('End date', value=pd.to_datetime('2023-12-31')).strftime('%Y-%m-%d')
    max_cloud_percentage = st.slider("Maximum Cloud Coverage (%)", 0, 100, 20)
    apply_cloud_masking = st.checkbox("Apply Cloud Masking", True)
    st.subheader("Upload ROI GeoJSON")
    uploaded_file = st.file_uploader("Upload GeoJSON", type=["geojson"])



if uploaded_file is not None:
    st.header("🔍 Dive into Sentinel Imagery Layer Visualization 🔍")
    geojson = json.load(uploaded_file)
    if geojson['features'][0]['geometry']['type'] == 'Polygon':
        aoi_coordinates = geojson['features'][0]['geometry']['coordinates']
        aoi_rectangle = ee.Geometry.Polygon(aoi_coordinates)

        # Process Sentinel-2 data
        s2 = load_s2_data(start_date, end_date, aoi_rectangle)
        s2_with_indices = add_indices(s2.mean()).clip(aoi_rectangle)

        # Create Folium map at the location of the ROI
        m = folium.Map(location=[aoi_coordinates[0][0][1], aoi_coordinates[0][0][0]], zoom_start=12,)
        folium.GeoJson(geojson, name="Area of Interest").add_to(m)

        # Visualization parameters
        # Define your color palettes as before
        ndvi_palette = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
        evi_palette = ['#91cf60', '#d9ef8b', '#789669', '#4a613c', '#1a9850', '#00441B']
        savi_palette = ['#F5F5DC', '#FFEBCD', '#eefa96', '#d6e57b', '#789669', '#4a613c']
        gndvi_palette = ['#d9ef8b', '#91cf60', '#1a9850', '#006400', '#00441B']

        # Construct visualization parameters with palettes
        ndvi_vis_params = {'min': 0, 'max': 1, 'palette': ndvi_palette}
        evi_vis_params = {'min': 0, 'max': 1, 'palette': evi_palette}
        savi_vis_params = {'min': 0, 'max': 1, 'palette': savi_palette}
        gndvi_vis_params = {'min': 0, 'max': 1, 'palette': gndvi_palette}

        # Get tile layer URLs with the appropriate palettes
        ndvi_tile_url = get_tile_layer_url(s2_with_indices.select('NDVI'), ndvi_vis_params)
        evi_tile_url = get_tile_layer_url(s2_with_indices.select('EVI'), evi_vis_params)
        savi_tile_url = get_tile_layer_url(s2_with_indices.select('SAVI'), savi_vis_params)
        gndvi_tile_url = get_tile_layer_url(s2_with_indices.select('GNDVI'), gndvi_vis_params)

        # Add tile layers to the map
        folium.TileLayer(tiles=ndvi_tile_url, name='NDVI', attr='NDVI').add_to(m)
        folium.TileLayer(tiles=evi_tile_url, name='EVI', attr='EVI').add_to(m)
        folium.TileLayer(tiles=savi_tile_url, name='SAVI', attr='SAVI').add_to(m)
        folium.TileLayer(tiles=gndvi_tile_url, name='GNDVI', attr='GNDVI').add_to(m)

        folium.LayerControl().add_to(m)
        folium_static(m,width=1000, height=500)

        # Time series data processing
        time_series = s2.map(lambda image: extract_mean(image, aoi_rectangle))
        data = time_series.reduceColumns(ee.Reducer.toList(5), ['date', 'NDVI', 'EVI', 'SAVI', 'GNDVI']).getInfo()[
            'list']
        df = pd.DataFrame(data, columns=['date', 'NDVI', 'EVI', 'SAVI', 'GNDVI'])
        df['date'] = pd.to_datetime(df['date'])

        # Interactive Plotting with Plotly
        fig = px.line(df, x='date', y=df.columns[1:], title='Vegetation Indices Time Series',
                      labels={'value': 'Index Value', 'variable': 'Vegetation Index'})
        fig.update_layout(hovermode='x')
        st.plotly_chart(fig, use_container_width=True)

        # Convert the DataFrame to a CSV string for download
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='vegetation_indices_time_series.csv',
            mime='text/csv',
        )

        prediction_days = st.number_input("Number of Days for Prediction", min_value=1, max_value=365, value=30)

        # Data Visualization

        st.plotly_chart(fig, use_container_width=True)
        # Data Visualization
        st.header("📈 Time Series Analysis Charts: A Journey Through Data 📉")
        # NDVI Time Series Plot
        st.subheader("NDVI Time Series")
        fig_ndvi = px.line(df, x='date', y='NDVI', title='NDVI Time Series', labels={'date': 'Date', 'NDVI': 'NDVI'})
        fig_ndvi.update_layout(hovermode='x')
        st.plotly_chart(fig_ndvi, use_container_width=True)

        # EVI Time Series Plot
        st.subheader("EVI Time Series")
        fig_evi = px.line(df, x='date', y='EVI', title='EVI Time Series', labels={'date': 'Date', 'EVI': 'EVI'})
        fig_evi.update_layout(hovermode='x')
        st.plotly_chart(fig_evi, use_container_width=True)

        # SAVI Time Series Plot
        st.subheader("SAVI Time Series")
        fig_savi = px.line(df, x='date', y='SAVI', title='SAVI Time Series', labels={'date': 'Date', 'SAVI': 'SAVI'})
        fig_savi.update_layout(hovermode='x')
        st.plotly_chart(fig_savi, use_container_width=True)

        # GNDVI Time Series Plot
        st.subheader("GNDVI Time Series")
        fig_gndvi = px.line(df, x='date', y='GNDVI', title='GNDVI Time Series',
                            labels={'date': 'Date', 'GNDVI': 'GNDVI'})
        fig_gndvi.update_layout(hovermode='x')
        st.plotly_chart(fig_gndvi, use_container_width=True)
        # Statistical Measures Visualization
        stats = calculate_statistics(df)
        st.subheader("Statistical Measures")
        st.table(stats)
        index_for_visualization = st.selectbox("Choose an index for detailed visualization",
                                               ['NDVI', 'EVI', 'SAVI', 'GNDVI'])

        # Pie Chart for Monthly Averages
        st.subheader(f"Monthly Average {index_for_visualization}")
        st.plotly_chart(create_pie_chart(df, index_for_visualization), use_container_width=True)

        # Bar Chart for Daily Values
        st.subheader(f"Daily {index_for_visualization}")
        st.plotly_chart(create_bar_chart(df, index_for_visualization), use_container_width=True)

        # Additional statistics
        st.subheader("Additional Statistical Insights")
        st.text(f"Overall Average {index_for_visualization}: {df[index_for_visualization].mean():.2f}")
        st.text(
            f"Highest Daily {index_for_visualization}: {df[index_for_visualization].max():.2f} on {df[df[index_for_visualization] == df[index_for_visualization].max()]['date'].iloc[0].strftime('%Y-%m-%d')}")
        st.text(
            f"Lowest Daily {index_for_visualization}: {df[index_for_visualization].min():.2f} on {df[df[index_for_visualization] == df[index_for_visualization].min()]['date'].iloc[0].strftime('%Y-%m-%d')}")

        if st.button('Predict Future Trends for NDVI and EVI And AI Summarys',use_container_width=True):

            future_trend_df = train_and_predict_future_trends(df, days_to_predict=prediction_days)

            # Merge past and future trends
            combined_df = merge_past_future_data(df, future_trend_df)

            # Plotting combined past and future trends
            st.subheader("Combined Past and Future Trends for NDVI and EVI")
            fig_combined_trends = px.line(combined_df, x='date', y=['NDVI', 'EVI'],
                                          title='Combined Past and Future Trends for NDVI and EVI',
                                          labels={'value': 'Index Value', 'variable': 'Vegetation Index'},
                                          color_discrete_map={'NDVI': 'red', 'EVI': 'blue'})

            # Highlighting future trends
            fig_combined_trends.add_scatter(x=future_trend_df['date'], y=future_trend_df['NDVI'], mode='lines',
                                            name='Future NDVI', line=dict(color='yellow', dash='dash'))
            fig_combined_trends.add_scatter(x=future_trend_df['date'], y=future_trend_df['EVI'], mode='lines',
                                            name='Future EVI', line=dict(color='yellow', dash='dash'))

            st.plotly_chart(fig_combined_trends, use_container_width=True)

            st.title(" 🌟✨ AI-Driven Insights Summary ✨🌟 ")

            stats = df[['NDVI', 'EVI', 'SAVI', 'GNDVI']].describe()

            def format_stats_for_prompt(stats):
                formatted_stats = []
                for index in ['NDVI', 'EVI', 'SAVI', 'GNDVI']:
                    mean_val = stats.at['mean', index]
                    max_val = stats.at['max', index]
                    min_val = stats.at['min', index]
                    formatted_stats.append(f"{index} - Mean: {mean_val:.2f}, Max: {max_val:.2f}, Min: {min_val:.2f}")
                return ' '.join(formatted_stats)


            # Assuming you have calculated 'stats' from 'df'
            formatted_statistics = format_stats_for_prompt(stats)


            # --- Google GenerativeAI for Summary ---
            def get_gemini_response(input_text):
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(input_text)
                return response.text


            future_trend_insights = """
                    Future trends predict a change in NDVI and EVI compared to past data. This change could be due to various environmental factors and indicates the dynamic nature of vegetation health and density.
                    """
            # Generate a detailed prompt
            input_text = f"""
            **Detailed Vegetation Index Analysis Summary:**
            The analysis covers the period from {start_date} to {end_date} and focuses on key vegetation indices: NDVI, EVI, SAVI, and GNDVI.
            Key Statistical Findings:
            {formatted_statistics}

            Additionally, an advanced regression model was used to predict future trends in NDVI and EVI, offering insights into expected changes in vegetation health and density. According to the model's predictions:

            {future_trend_insights}

            Based on this comprehensive analysis, a detailed summary is needed that covers both the historical data analysis and the implications of the predicted future trends on environmental monitoring and agricultural planning.

            **Instructions:**
            Please generate a comprehensive and insightful summary report, encompassing both the statistical analysis of the past vegetation index data and the predictions of future trends. Highlight key findings, their significance, and potential implications.
            """
            summary_report = get_gemini_response(input_text)
            st.write(summary_report)
