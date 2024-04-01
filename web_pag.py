import streamlit as st
import ee
import geemap
import pandas as pd
import matplotlib.pyplot as plt

# Authenticate and initialize Earth Engine
ee.Authenticate()  # Follow the prompts for authorization
ee.Initialize(project='ee-aashishkawade')

# Function to add NDVI and EVI bands
def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = image.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                           {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}).rename('EVI')
    return image.addBands([ndvi, evi])

# Load and process Sentinel-2 data
def load_s2_data(start_date, end_date, aoi_rectangle):
    return (ee.ImageCollection('COPERNICUS/S2')
            .filterDate(ee.Date(start_date), ee.Date(end_date))
            .filterBounds(aoi_rectangle)
            .map(add_indices))

# Function to extract mean NDVI and EVI
def extract_mean(image, aoi_rectangle):
    mean_dict = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi_rectangle, scale=30, maxPixels=1e9, bestEffort=True)
    return image.set('date', image.date().format()).set(mean_dict)

# Streamlit code
st.title('NDVI and EVI Time Series Analysis')

start_date = st.date_input('Start date', value=pd.to_datetime('2023-06-01'))
end_date = st.date_input('End date', value=pd.to_datetime('2023-12-31'))

# Convert pandas.Timestamp to string
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')

aoi_rectangle = ee.Geometry.Polygon([[[75.15, 19.75], [75.15, 20.05], [75.55, 20.05], [75.55, 19.75], [75.15, 19.75]]])
s2 = load_s2_data(start_date, end_date, aoi_rectangle)

# Extract time series data
time_series = s2.map(lambda image: extract_mean(image, aoi_rectangle))
data = time_series.reduceColumns(ee.Reducer.toList(3), ['date', 'NDVI', 'EVI']).getInfo()['list']
df = pd.DataFrame(data, columns=['date', 'NDVI', 'EVI'])
df['date'] = pd.to_datetime(df['date'])

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].plot(df['date'], df['NDVI'], label='NDVI', color='green')
ax[0].set_xlabel('Date')
ax[0].set_ylabel('NDVI')
ax[0].set_title('NDVI Time Series')
ax[0].grid(True)
ax[0].legend()

ax[1].plot(df['date'], df['EVI'], label='EVI', color='blue')
ax[1].set_xlabel('Date')
ax[1].set_ylabel('EVI')
ax[1].set_title('EVI Time Series')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
st.pyplot(fig)





