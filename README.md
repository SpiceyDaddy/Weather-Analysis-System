# Weather-Analysis-System 
A comprehensive meteorological analysis tool that provides real-time weather data, satellite imagery, and forecast modeling for professional weather analysis.

## Features

- **Real-time Weather Data**: Current conditions, severe weather alerts, and hourly forecasts from the National Weather Service
- **Model Forecasts**: Open-Meteo weather model data with searchable Zulu hour functionality
- **Satellite Imagery**: Animated GOES-16 national satellite imagery (GeoColor and Infrared)
- **Regional IR Zoom**: Automatically cropped infrared satellite imagery focused on your specific location
- **Weather Analysis Charts**: Temperature, pressure, wind, and humidity trend visualizations
- **Location Support**: Geocoding for city/state names or direct coordinate input

## Installation

### Required Dependencies

```bash
pip install requests pillow matplotlib numpy tkinter
```

### Optional Dependencies

For enhanced HTTP reliability:
```bash
pip install urllib3
```

### Python Version
- Python 3.7 or higher required
- Tested on Python 3.10

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/weather-analysis-system.git
cd weather-analysis-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python weather_analysis_system.py
```

## Interface Overview

### Location Input
- Enter city and state (e.g., "Nashville, TN") or coordinates (e.g., "36.1627,-86.7816")
- Automatic geocoding for US locations

### Tabs
- **Current Conditions**: Live weather observations and severe weather alerts
- **Model Data**: Open-Meteo forecast data with Zulu hour search
- **Satellite & IR**: National satellite imagery and regional IR zoom
- **Analysis**: Weather trend charts and graphs

### Key Features
- **Zulu Hour Jump**: Search model data for specific forecast hours (e.g., "12Z", "00")
- **Dual Satellite View**: Side-by-side national and regional satellite displays
- **Automatic Updates**: Weather data refreshes on location changes
- **Professional Format**: Meteorologist-friendly data presentation

## Data Sources

- **National Weather Service**: Current observations, forecasts, and severe weather alerts
- **Open-Meteo**: High-resolution weather model data
- **NOAA GOES-16**: Satellite imagery (GeoColor and Infrared)
- **Nominatim**: Geocoding services for location lookup

## Technical Details

- Built with Python Tkinter for cross-platform compatibility
- Threaded network operations to maintain responsive UI
- Automatic retry logic for network requests
- Memory-efficient satellite image processing
- Real-time coordinate-to-pixel mapping for regional satellite cropping

## System Requirements

- Internet connection required for weather data and satellite imagery
- Minimum 1400x950 screen resolution recommended
- 100MB+ available memory for satellite image processing

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome. Please ensure all weather data sources comply with their respective terms of service.

## Disclaimer

This tool is for educational and research purposes. For official weather warnings and forecasts, always consult the National Weather Service or your local meteorological authority.
