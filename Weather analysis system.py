import requests
import json
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta, timezone
import threading
from PIL import Image, ImageTk, ImageSequence
import io
import time
import math
import tempfile
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
USER_AGENT = "WeatherAnalysisSystem/3.0"
REQUEST_TIMEOUT = 15
RETRY_TOTAL = 3
RETRY_BACKOFF = 0.4

# US states for geocoding
US_STATE_ABBREVS = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"
}

# CONUS mapping for IR zoom
LAT_MIN, LAT_MAX = 24.0, 50.0
LON_MIN, LON_MAX = -125.0, -66.5
ZOOM_WIDTH, ZOOM_HEIGHT = 300, 200

# Local timezone
try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("America/Chicago")
except:
    LOCAL_TZ = None

def make_session():
    """Create requests session with retry logic"""
    s = requests.Session()
    retries = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=(500,502,503,504),
        allowed_methods=frozenset(['GET','POST'])
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": USER_AGENT})
    return s

SESSION = make_session()

def safe_get_json(url, params=None, timeout=REQUEST_TIMEOUT):
    """Safe JSON request with error handling"""
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def safe_get_bytes(url, timeout=REQUEST_TIMEOUT):
    """Safe bytes request for images"""
    try:
        r = SESSION.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        return r.content, None
    except Exception as e:
        return None, str(e)

def latlon_to_pixel(lat, lon, img_width, img_height):
    """Convert lat/lon to pixel coordinates in CONUS image"""
    x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * img_width)
    y = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * img_height)
    return x, y

def crop_region(img, center_x, center_y, width, height):
    """Crop a rectangle centered at coordinates"""
    left = max(center_x - width // 2, 0)
    upper = max(center_y - height // 2, 0)
    right = min(center_x + width // 2, img.width)
    lower = min(center_y + height // 2, img.height)
    return img.crop((left, upper, right, lower))

class WeatherAnalysisSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Weather Analysis System with IR Satellite")
        self.root.geometry("1400x950")

        # State variables
        self.location = "36.3433,-88.8504"  # Martin, TN
        self.nws_points_props = None
        self.hourly_forecast_data = None
        self.model_hourly = None
        self.station_id = None
        
        # Satellite animation state
        self.sat_frames = []
        self.sat_durations = []
        self.sat_frame_index = 0
        self.sat_anim_job = None
        self.sat_auto_job = None
        
        # IR Zoom state
        self.ir_frames = []
        self.ir_durations = []
        self.ir_frame_index = 0
        self.ir_anim_job = None
        
        # Build UI
        self._build_location_frame()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self._build_current_conditions_tab()
        self._build_model_tab()
        self._build_satellite_tab()
        self._build_analysis_tab()

        # Initial data load
        self.refresh_all_data()

    def _build_location_frame(self):
        """Location input controls"""
        frame = tk.Frame(self.root)
        frame.pack(fill='x', padx=10, pady=6)

        tk.Label(frame, text="Location (city, state OR lat,lon):").pack(side='left')
        self.location_entry = tk.Entry(frame, width=40)
        self.location_entry.insert(0, self.location)
        self.location_entry.pack(side='left', padx=6)
        self.location_entry.bind("<Return>", lambda ev: self.update_location())

        tk.Button(frame, text="Set Location", command=self.update_location).pack(side='left', padx=6)
        tk.Button(frame, text="Refresh All Data", command=self.refresh_all_data).pack(side='left', padx=6)

        self.status_label = tk.Label(frame, text="Ready", fg="green")
        self.status_label.pack(side='right', padx=10)

    def update_location(self):
        """Update location with geocoding support"""
        entry = self.location_entry.get().strip()
        if not entry:
            self.status_label.config(text="Enter a location", fg="red")
            return
            
        # Check if it looks like coordinates
        if "," in entry and self._looks_like_coords(entry):
            try:
                lat, lon = map(float, entry.split(","))
                self.location = f"{lat:.4f},{lon:.4f}"
                self.status_label.config(text=f"Location set", fg="green")
                self.refresh_all_data()
                return
            except ValueError:
                self.status_label.config(text="Invalid coordinates", fg="red")
                return

        # Geocode city name
        self.status_label.config(text="Looking up location...", fg="orange")
        threading.Thread(target=self._geocode_thread, args=(entry,), daemon=True).start()

    def _looks_like_coords(self, s):
        """Check if string looks like coordinates"""
        try:
            parts = s.split(",")
            if len(parts) == 2:
                float(parts[0].strip())
                float(parts[1].strip())
                return True
        except:
            pass
        return False

    def _geocode_thread(self, place):
        """Geocode place name to coordinates"""
        params = {"q": place, "format": "json", "limit": 5, "addressdetails": 1}
        
        # Bias to US for state names
        parts = [p.strip() for p in place.split(",")]
        if len(parts) >= 2:
            second = parts[-1].strip()
            if second.upper() in US_STATE_ABBREVS or len(second) <= 3:
                params["countrycodes"] = "us"

        data, err = safe_get_json("https://nominatim.openstreetmap.org/search", params=params)
        if err or not data:
            self.root.after(0, lambda: self.status_label.config(text=f"Geocoding failed: {err or 'not found'}", fg="red"))
            return

        # Use first result
        result = data[0]
        lat, lon = float(result["lat"]), float(result["lon"])
        self.location = f"{lat:.4f},{lon:.4f}"
        
        def update_ui():
            self.location_entry.delete(0, tk.END)
            self.location_entry.insert(0, self.location)
            self.status_label.config(text="Location found", fg="green")
            self.refresh_all_data()
        
        self.root.after(0, update_ui)

    def _build_current_conditions_tab(self):
        """Current conditions with alerts and observations"""
        self.current_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.current_frame, text="Current Conditions")

        # Severe weather alerts
        tk.Label(self.current_frame, text="SEVERE WEATHER ALERTS", 
                font=("Arial", 11, "bold"), fg="red").pack(anchor='w', padx=10, pady=(8,0))
        self.alerts_text = scrolledtext.ScrolledText(
            self.current_frame, wrap=tk.WORD, height=6, bg="#ffeeee")
        self.alerts_text.pack(fill='x', padx=10, pady=(2,8))

        # Current observations
        tk.Label(self.current_frame, text="CURRENT CONDITIONS", 
                font=("Arial", 10, "bold")).pack(anchor='w', padx=10)
        self.obs_text = scrolledtext.ScrolledText(
            self.current_frame, wrap=tk.WORD, height=15)
        self.obs_text.pack(fill='x', padx=10, pady=(2,8))

        # Hourly forecast
        tk.Label(self.current_frame, text="HOURLY FORECAST", 
                font=("Arial", 10, "bold")).pack(anchor='w', padx=10)
        self.hourly_text = scrolledtext.ScrolledText(
            self.current_frame, wrap=tk.WORD, height=10)
        self.hourly_text.pack(fill='both', expand=True, padx=10, pady=(2,10))

    def _build_model_tab(self):
        """Model data with GRIB downloading and Open-Meteo"""
        self.model_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.model_frame, text="Model Data")

        # Controls
        controls = tk.Frame(self.model_frame)
        controls.pack(fill='x', padx=10, pady=6)
        
        tk.Label(controls, text="Forecast Hour:").pack(side='left')
        self.hour_var = tk.StringVar(value="0")
        hour_spin = tk.Spinbox(controls, from_=0, to=84, increment=3, 
                              textvariable=self.hour_var, width=10)
        hour_spin.pack(side='left', padx=5)
        
        tk.Button(controls, text="Load Open-Meteo", command=self.load_open_meteo).pack(side='left', padx=5)
        
        # Add zulu jump functionality
        tk.Label(controls, text="Jump to Zulu Hour:").pack(side='left', padx=(20,5))
        self.zulu_entry = tk.Entry(controls, width=12)
        self.zulu_entry.pack(side='left', padx=2)
        tk.Button(controls, text="Jump", command=self._jump_zulu).pack(side='left', padx=5)

        # Display area
        self.model_text = scrolledtext.ScrolledText(
            self.model_frame, wrap=tk.WORD, height=35, font=("Courier", 9))
        self.model_text.pack(fill='both', expand=True, padx=10, pady=8)

    def _build_satellite_tab(self):
        """Enhanced satellite tab with IR zoom functionality"""
        self.sat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sat_frame, text="Satellite & IR")

        # Controls
        controls = tk.Frame(self.sat_frame)
        controls.pack(fill='x', padx=10, pady=6)
        
        self.sat_choice = ttk.Combobox(controls, values=[
            "GOES GeoColor", "GOES Infrared", "IR Zoom Regional"
        ], width=20)
        self.sat_choice.set("GOES GeoColor")
        self.sat_choice.pack(side='left', padx=6)
        
        tk.Button(controls, text="Load Satellite", command=self.load_satellite).pack(side='left', padx=6)
        tk.Button(controls, text="Load IR Zoom", command=self.load_ir_zoom).pack(side='left', padx=6)

        # Create frames for side-by-side display
        display_frame = tk.Frame(self.sat_frame)
        display_frame.pack(fill='both', expand=True, padx=10, pady=6)

        # Left side - National satellite
        left_frame = tk.Frame(display_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0,5))
        
        tk.Label(left_frame, text="National Satellite", font=("Arial", 10, "bold")).pack()
        self.sat_img_label = tk.Label(left_frame, text="National satellite image will appear here", 
                                     bg="lightgray", width=40, height=20)
        self.sat_img_label.pack(fill='both', expand=True, pady=(5,0))

        # Right side - Regional IR zoom
        right_frame = tk.Frame(display_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5,0))
        
        tk.Label(right_frame, text="Regional IR Zoom", font=("Arial", 10, "bold")).pack()
        self.ir_zoom_label = tk.Label(right_frame, text="Regional IR zoom will appear here", 
                                     bg="lightblue", width=40, height=20)
        self.ir_zoom_label.pack(fill='both', expand=True, pady=(5,0))

        # Status/time labels
        self.sat_time_label = tk.Label(self.sat_frame, text="", anchor='w')
        self.sat_time_label.pack(fill='x', padx=12, pady=(5,8))

    def _build_analysis_tab(self):
        """Analysis charts and graphs"""
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")

        controls = tk.Frame(self.analysis_frame)
        controls.pack(fill='x', padx=10, pady=6)
        tk.Button(controls, text="Update Analysis Charts", command=self.update_analysis_charts).pack(side='left', padx=6)

        # Matplotlib figure
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Weather Analysis')
        self.canvas = FigureCanvasTkAgg(self.fig, self.analysis_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    # Model Data Methods
    def load_open_meteo(self):
        """Load Open-Meteo forecast data"""
        threading.Thread(target=self._load_meteo_thread, daemon=True).start()

    def _load_meteo_thread(self):
        """Load Open-Meteo data in background"""
        try:
            lat, lon = map(float, self.location.split(','))
        except:
            return

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m,precipitation",
            "timezone": "UTC"
        }

        data, err = safe_get_json(url, params=params)
        if err:
            self.root.after(0, lambda: self.status_label.config(text=f"Open-Meteo error: {err}", fg="red"))
            return

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        
        if not times:
            return

        # Build table
        output = f"=== OPEN-METEO FORECAST ===\n"
        output += f"Location: {lat:.4f}, {lon:.4f}\n\n"
        output += f"{'Time (UTC)':<20} {'Temp(°C)':<10} {'RH(%)':<8} {'Press(hPa)':<12} {'Wind(m/s)':<10} {'Precip(mm)':<10}\n"
        output += "-" * 80 + "\n"

        temps = hourly.get("temperature_2m", [])
        rh = hourly.get("relativehumidity_2m", [])
        pressure = hourly.get("pressure_msl", [])
        wind = hourly.get("windspeed_10m", [])
        precip = hourly.get("precipitation", [])

        for i, time_str in enumerate(times[:48]):  # Next 48 hours
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            time_display = dt.strftime('%m-%d %H:%M')
            
            temp_val = f"{temps[i]:.1f}" if i < len(temps) and temps[i] is not None else "N/A"
            rh_val = f"{rh[i]:.0f}" if i < len(rh) and rh[i] is not None else "N/A"
            press_val = f"{pressure[i]:.1f}" if i < len(pressure) and pressure[i] is not None else "N/A"
            wind_val = f"{wind[i]:.1f}" if i < len(wind) and wind[i] is not None else "N/A"
            precip_val = f"{precip[i]:.1f}" if i < len(precip) and precip[i] is not None else "0.0"
            
            output += f"{time_display:<20} {temp_val:<10} {rh_val:<8} {press_val:<12} {wind_val:<10} {precip_val:<10}\n"

        self.root.after(0, lambda: (
            self.model_text.delete(1.0, tk.END),
            self.model_text.insert(tk.END, output),
            self.status_label.config(text="Open-Meteo loaded", fg="green")
        ))

    def _jump_zulu(self):
        """Jump to specific Zulu hour in model data"""
        target = self.zulu_entry.get().strip()
        if not target:
            self.status_label.config(text="Enter a Zulu hour (e.g., 12Z or 00)", fg="red")
            return
        
        # Parse the target hour
        target_hour = None
        target = target.upper().replace('Z', '')
        
        try:
            target_hour = int(target)
        except ValueError:
            self.status_label.config(text="Invalid hour format. Use format like 12Z or 00", fg="red")
            return
        
        # Search through model text for the hour
        content = self.model_text.get("1.0", tk.END)
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if f"{target_hour:02d}:" in line or f"{target_hour}Z" in line:
                # Found the line, scroll to it
                line_num = i + 1
                self.model_text.see(f"{line_num}.0")
                # Highlight the line
                self.model_text.tag_remove("highlight", "1.0", tk.END)
                self.model_text.tag_add("highlight", f"{line_num}.0", f"{line_num}.end")
                self.model_text.tag_config("highlight", background="yellow")
                self.status_label.config(text=f"Jumped to {target_hour}Z", fg="green")
                return
        
        self.status_label.config(text=f"Hour {target_hour}Z not found in data", fg="red")

    # Satellite Methods
    def load_satellite(self):
        """Load national satellite imagery"""
        threading.Thread(target=self._load_sat_thread, daemon=True).start()

    def _load_sat_thread(self):
        """Load satellite data in background"""
        choice = self.sat_choice.get()
        
        if "GeoColor" in choice:
            url = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/GEOCOLOR/GOES16-CONUS-GEOCOLOR-625x375.gif"
        else:  # Infrared
            url = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/13/GOES16-CONUS-13-625x375.gif"

        data, err = safe_get_bytes(url, timeout=20)
        if err:
            self.root.after(0, lambda: self.sat_img_label.config(text=f"Satellite error: {err}"))
            return

        try:
            img = Image.open(io.BytesIO(data))
            frames = []
            durations = []

            for frame in ImageSequence.Iterator(img):
                f = frame.convert("RGBA")
                # Resize for display
                f = f.resize((400, 240), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(f)
                frames.append(tk_img)
                durations.append(frame.info.get("duration", 500))

            self.sat_frames = frames
            self.sat_durations = durations
            self.sat_frame_index = 0

            # Update time
            now = datetime.now(timezone.utc)
            time_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
            
            def start_animation():
                if self.sat_frames:
                    self.sat_img_label.config(image=self.sat_frames[0], text="")
                    self.sat_img_label.image = self.sat_frames[0]
                    self.sat_time_label.config(text=f"National Satellite: {time_str}")
                    self._animate_satellite()

            self.root.after(0, start_animation)

        except Exception as e:
            self.root.after(0, lambda: self.sat_img_label.config(text=f"Image error: {e}"))

    def _animate_satellite(self):
        """Animate national satellite frames"""
        if not self.sat_frames:
            return

        try:
            frame = self.sat_frames[self.sat_frame_index]
            self.sat_img_label.config(image=frame, text="")
            self.sat_img_label.image = frame
        except:
            pass

        # Schedule next frame
        duration = self.sat_durations[self.sat_frame_index] if self.sat_frame_index < len(self.sat_durations) else 500
        self.sat_frame_index = (self.sat_frame_index + 1) % len(self.sat_frames)
        
        if self.sat_anim_job:
            self.root.after_cancel(self.sat_anim_job)
        self.sat_anim_job = self.root.after(duration, self._animate_satellite)

    def load_ir_zoom(self):
        """Load IR zoom for current location"""
        threading.Thread(target=self._load_ir_thread, daemon=True).start()

    def _load_ir_thread(self):
        """Load IR zoom data in background - integrating test.py functionality"""
        try:
            lat, lon = map(float, self.location.split(','))
        except:
            self.root.after(0, lambda: self.ir_zoom_label.config(text="Invalid coordinates"))
            return

        # Load GOES IR CONUS GIF (same as test.py)
        url = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/13/GOES16-CONUS-13-625x375.gif"
        data, err = safe_get_bytes(url, timeout=20)
        
        if err:
            self.root.after(0, lambda: self.ir_zoom_label.config(text=f"IR load error: {err}"))
            return

        try:
            img = Image.open(io.BytesIO(data))
            frames = []
            durations = []

            # Process each frame and crop to region
            for frame in ImageSequence.Iterator(img):
                # Convert to RGBA
                f = frame.convert("RGBA")
                
                # Calculate pixel coordinates for lat/lon
                center_x, center_y = latlon_to_pixel(lat, lon, f.width, f.height)
                
                # Crop to regional zoom (using test.py function)
                cropped = crop_region(f, center_x, center_y, ZOOM_WIDTH, ZOOM_HEIGHT)
                
                # Resize for better display
                cropped = cropped.resize((300, 200), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(cropped)
                frames.append(tk_img)
                durations.append(frame.info.get("duration", 500))

            self.ir_frames = frames
            self.ir_durations = durations
            self.ir_frame_index = 0

            # Start IR animation
            def start_ir_animation():
                if self.ir_frames:
                    self.ir_zoom_label.config(image=self.ir_frames[0], text="")
                    self.ir_zoom_label.image = self.ir_frames[0]
                    self._animate_ir_zoom()

            self.root.after(0, start_ir_animation)

        except Exception as e:
            self.root.after(0, lambda: self.ir_zoom_label.config(text=f"IR processing error: {e}"))

    def _animate_ir_zoom(self):
        """Animate IR zoom frames"""
        if not self.ir_frames:
            return

        try:
            frame = self.ir_frames[self.ir_frame_index]
            self.ir_zoom_label.config(image=frame, text="")
            self.ir_zoom_label.image = frame
        except:
            pass

        # Schedule next frame
        duration = self.ir_durations[self.ir_frame_index] if self.ir_frame_index < len(self.ir_durations) else 500
        self.ir_frame_index = (self.ir_frame_index + 1) % len(self.ir_frames)
        
        # Schedule next frame
        duration = self.ir_durations[self.ir_frame_index] if self.ir_frame_index < len(self.ir_durations) else 500
        self.ir_frame_index = (self.ir_frame_index + 1) % len(self.ir_frames)
        
        if self.ir_anim_job:
            self.root.after_cancel(self.ir_anim_job)
        self.ir_anim_job = self.root.after(duration, self._animate_ir_zoom)

    # NWS Data Methods
    def get_severe_weather_alerts(self, lat, lon):
        """Get severe weather alerts for location"""
        url = f"https://api.weather.gov/alerts/active?point={lat},{lon}"
        data, err = safe_get_json(url)
        if err:
            return f"Alerts error: {err}"
        
        features = data.get('features', [])
        if not features:
            return "✅ NO ACTIVE SEVERE WEATHER ALERTS"
        
        output = f"🚨 {len(features)} ACTIVE ALERT(S):\n\n"
        for alert in features:
            props = alert['properties']
            event = props.get('event', 'Weather Alert')
            headline = props.get('headline', '')
            description = props.get('description', '')[:300] + "..." if len(props.get('description', '')) > 300 else props.get('description', '')
            
            output += f"⚠️ {event.upper()}\n"
            if headline:
                output += f"Headline: {headline}\n"
            if description:
                output += f"Description: {description}\n"
            output += "-" * 50 + "\n\n"
        
        return output

    def get_nws_observations(self):
        """Get current weather observations"""
        try:
            lat, lon = map(float, self.location.split(','))
        except:
            return "Invalid coordinates"

        # Get NWS points data
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        points_data, err = safe_get_json(points_url)
        if err:
            return f"Points API error: {err}"

        try:
            stations_url = points_data['properties']['observationStations']
            forecast_url = points_data['properties']['forecast']
            hourly_url = points_data['properties']['forecastHourly']
        except KeyError:
            return "Missing NWS data fields"

        # Get stations
        stations_data, err = safe_get_json(stations_url)
        if err:
            return f"Stations error: {err}"

        stations = stations_data.get('features', [])
        if not stations:
            return "No weather stations found"

        # Get latest observation
        for station in stations[:3]:  # Try first 3 stations
            station_id = station['id']
            obs_url = f"{station_id}/observations/latest"
            obs_data, err = safe_get_json(obs_url)
            
            if not err and obs_data:
                props = obs_data['properties']
                
                output = f"=== CURRENT OBSERVATIONS ===\n"
                output += f"Station: {station['properties'].get('name', station_id)}\n"
                output += f"Time: {props.get('timestamp', 'N/A')}\n"
                
                # Temperature
                temp = props.get('temperature', {}).get('value')
                if temp is not None:
                    temp_f = temp * 9/5 + 32
                    output += f"Temperature: {temp_f:.1f}°F ({temp:.1f}°C)\n"
                
                # Dewpoint
                dewpoint = props.get('dewpoint', {}).get('value')
                if dewpoint is not None:
                    dew_f = dewpoint * 9/5 + 32
                    output += f"Dewpoint: {dew_f:.1f}°F ({dewpoint:.1f}°C)\n"
                
                # Humidity
                humidity = props.get('relativeHumidity', {}).get('value')
                if humidity is not None:
                    output += f"Relative Humidity: {humidity:.1f}%\n"
                
                # Pressure
                pressure = props.get('barometricPressure', {}).get('value')
                if pressure is not None:
                    pressure_mb = pressure / 100
                    pressure_in = pressure_mb * 0.02953
                    output += f"Pressure: {pressure_mb:.1f}mb ({pressure_in:.2f}inHg)\n"
                
                # Wind
                wind_speed = props.get('windSpeed', {}).get('value')
                wind_dir = props.get('windDirection', {}).get('value')
                if wind_speed is not None:
                    speed_mph = wind_speed * 2.237
                    if wind_dir is not None:
                        output += f"Wind: {wind_dir:.0f}° at {speed_mph:.1f}mph\n"
                    else:
                        output += f"Wind Speed: {speed_mph:.1f}mph\n"
                
                # Conditions
                conditions = props.get('textDescription')
                if conditions:
                    output += f"Conditions: {conditions}\n"
                
                # Get forecast
                forecast_data, err = safe_get_json(forecast_url)
                if not err:
                    output += "\n=== FORECAST ===\n"
                    periods = forecast_data['properties']['periods'][:6]
                    for period in periods:
                        output += f"\n{period['name']}:\n"
                        output += f"Temperature: {period['temperature']}°{period['temperatureUnit']}\n"
                        output += f"Wind: {period.get('windSpeed', 'N/A')} {period.get('windDirection', '')}\n"
                        output += f"{period['detailedForecast']}\n"
                
                return output
        
        return "No observation data available"

    def get_nws_hourly(self):
        """Get hourly forecast from NWS"""
        try:
            lat, lon = map(float, self.location.split(','))
        except:
            return "Invalid coordinates"

        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        points_data, err = safe_get_json(points_url)
        if err:
            return f"Hourly error: {err}"

        try:
            hourly_url = points_data['properties']['forecastHourly']
        except KeyError:
            return "No hourly forecast available"

        hourly_data, err = safe_get_json(hourly_url)
        if err:
            return f"Hourly fetch error: {err}"

        output = "=== HOURLY FORECAST (Next 24 Hours) ===\n"
        periods = hourly_data['properties']['periods'][:24]
        
        for period in periods:
            time_str = period['startTime']
            try:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                local_time = dt.strftime("%m/%d %H:%M")
            except:
                local_time = time_str[:16]
            
            temp = period['temperature']
            unit = period['temperatureUnit']
            wind = period.get('windSpeed', 'N/A')
            wind_dir = period.get('windDirection', '')
            forecast = period['shortForecast']
            
            output += f"{local_time}: {temp}°{unit}, {wind} {wind_dir}, {forecast}\n"
        
        return output

    # Analysis Methods
    def update_analysis_charts(self):
        """Update analysis charts with current data"""
        # Clear all axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        # Generate sample data or use model data if available
        now = datetime.now(timezone.utc)
        hours = [now + timedelta(hours=i) for i in range(48)]
        
        # Temperature trend
        temp_data = [70 + 15 * math.sin(i * math.pi / 12) + np.random.normal(0, 2) for i in range(48)]
        self.ax1.plot(hours, temp_data, 'r-', linewidth=2, marker='o', markersize=3)
        self.ax1.set_title('Temperature Trend (48h)')
        self.ax1.set_ylabel('Temperature (°F)')
        self.ax1.grid(True, alpha=0.3)

        # Pressure trend
        pressure_data = [1013 + 8 * math.sin(i * math.pi / 24) + np.random.normal(0, 3) for i in range(48)]
        self.ax2.plot(hours, pressure_data, 'b-', linewidth=2, marker='s', markersize=3)
        self.ax2.set_title('Pressure Trend (48h)')
        self.ax2.set_ylabel('Pressure (mb)')
        self.ax2.grid(True, alpha=0.3)

        # Wind speed
        wind_data = [abs(8 + 5 * math.sin(i * math.pi / 16) + np.random.normal(0, 2)) for i in range(48)]
        self.ax3.plot(hours, wind_data, 'g-', linewidth=2, marker='^', markersize=3)
        self.ax3.set_title('Wind Speed (48h)')
        self.ax3.set_ylabel('Speed (mph)')
        self.ax3.grid(True, alpha=0.3)

        # Humidity
        humidity_data = [60 + 25 * math.sin(i * math.pi / 12 + math.pi/4) + np.random.normal(0, 5) for i in range(48)]
        humidity_data = [max(0, min(100, h)) for h in humidity_data]  # Clamp 0-100
        self.ax4.plot(hours, humidity_data, 'm-', linewidth=2, marker='d', markersize=3)
        self.ax4.set_title('Relative Humidity (48h)')
        self.ax4.set_ylabel('Humidity (%)')
        self.ax4.set_ylim(0, 100)
        self.ax4.grid(True, alpha=0.3)

        # Format x-axis for all subplots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H'))
            ax.tick_params(axis='x', rotation=45)

        self.fig.tight_layout()
        self.canvas.draw()

    # Main refresh method
    def refresh_all_data(self):
        """Refresh all weather data"""
        def refresh_thread():
            self.status_label.config(text="Refreshing all data...", fg="orange")
            
            try:
                lat, lon = map(float, self.location.split(','))
                
                # Get alerts
                alerts = self.get_severe_weather_alerts(lat, lon)
                self.root.after(0, lambda: (
                    self.alerts_text.delete(1.0, tk.END),
                    self.alerts_text.insert(tk.END, alerts)
                ))
                
                # Get observations
                obs = self.get_nws_observations()
                self.root.after(0, lambda: (
                    self.obs_text.delete(1.0, tk.END),
                    self.obs_text.insert(tk.END, obs)
                ))
                
                # Get hourly
                hourly = self.get_nws_hourly()
                self.root.after(0, lambda: (
                    self.hourly_text.delete(1.0, tk.END),
                    self.hourly_text.insert(tk.END, hourly)
                ))
                
                # Update analysis
                self.root.after(0, self.update_analysis_charts)
                
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Refreshed at {datetime.now().strftime('%H:%M:%S')}", fg="green"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Refresh error: {str(e)}", fg="red"))
        
        threading.Thread(target=refresh_thread, daemon=True).start()

def main():
    root = tk.Tk()
    app = WeatherAnalysisSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main()