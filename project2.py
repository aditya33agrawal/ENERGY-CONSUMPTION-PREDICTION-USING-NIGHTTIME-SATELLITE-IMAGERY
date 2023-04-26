import ee
import datetime

# Initialize the Earth Engine Python API
ee.Initialize()

# Define the Jaipur region of interest (ROI)
jaipur_roi = ee.Geometry.Rectangle(75.6384, 26.7285, 76.0125, 27.0934)

# Define the start and end dates
start_date = datetime.datetime(2018, 1, 1)
end_date = datetime.datetime(2018, 12, 31)

# Define the image collection
nighttime = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')

# Filter the image collection by date and region of interest
nighttime_filtered = nighttime.filterDate(start_date, end_date).filterBounds(jaipur_roi)

# Reduce the image collection to monthly averages
nighttime_monthly = nighttime_filtered.reduce(ee.Reducer.mean())

# Download the monthly nighttime imagery for Jaipur region for 2018
for i in range(1, 13):
    image = nighttime_monthly.select('avg_rad').filter(ee.Filter.calendarRange(i, i, 'month')).first()
    file_name = 'jaipur_nighttime_' + str(i) + '.tif'
    task = ee.batch.Export.image.toDrive(image=image, description=file_name, folder='jaipur_nighttime', region=jaipur_roi, scale=30)
    task.start()
