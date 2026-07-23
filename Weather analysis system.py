import requests
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
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
SAT_DISPLAY_WIDTH, SAT_DISPLAY_HEIGHT = 840, 520
IR_DISPLAY_WIDTH, IR_DISPLAY_HEIGHT = 840, 520

# Local timezone
try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("America/Chicago")
except:
    LOCAL_TZ = None

# Rain timing animation configuration
RAIN_EXTEND_INTO_NEXT_DAY = True
RAIN_FADE_MINUTES = 45
RAIN_FPS = 12
RAIN_SIM_MINUTES_PER_FRAME = 15
RAIN_WEST, RAIN_EAST = -90.6, -87.2
RAIN_SOUTH, RAIN_NORTH = 34.9, 36.7
RAIN_LONS = np.linspace(RAIN_WEST, RAIN_EAST, 120)
RAIN_LATS = np.linspace(RAIN_SOUTH, RAIN_NORTH, 120)
RAIN_LON_GRID, RAIN_LAT_GRID = np.meshgrid(RAIN_LONS, RAIN_LATS)
RAIN_LAT_PTS = [35.05, 35.45, 35.9, 36.5]
RAIN_START_PTS = [21*60, 20*60, 18*60, 17*60]
clear_extra = 240 if RAIN_EXTEND_INTO_NEXT_DAY else 0


def rain_start_time_at_lat(lat):
    return np.interp(lat, RAIN_LAT_PTS, RAIN_START_PTS)


def rain_clear_time_at_lat(lat):
    base = rain_start_time_at_lat(lat) + 240
    south_weight = np.clip((35.5 - lat) / (35.5 - 35.05), 0, 1)
    return base + clear_extra * south_weight


start_grid = rain_start_time_at_lat(RAIN_LAT_GRID)
clear_grid = rain_clear_time_at_lat(RAIN_LAT_GRID)

TOTAL_MINUTES = int(np.ceil((clear_grid.max() + RAIN_FADE_MINUTES) / 60) * 60)
n_frames = int(TOTAL_MINUTES / RAIN_SIM_MINUTES_PER_FRAME) + 1


def rain_intensity_field(now_min):
    op = np.zeros_like(start_grid)
    fading_in = (now_min >= start_grid - RAIN_FADE_MINUTES) & (now_min < start_grid)
    op[fading_in] = (now_min - (start_grid[fading_in] - RAIN_FADE_MINUTES)) / RAIN_FADE_MINUTES
    raining = (now_min >= start_grid) & (now_min < clear_grid)
    op[raining] = 1.0
    fading_out = (now_min >= clear_grid) & (now_min < clear_grid + RAIN_FADE_MINUTES)
    op[fading_out] = 1.0 - (now_min - clear_grid[fading_out]) / RAIN_FADE_MINUTES
    return op


def rain_fmt_clock(total_min):
    day_offset = int(total_min // 1440)
    m = int(total_min % 1440)
    h, mm = divmod(m, 60)
    ampm = "PM" if h >= 12 else "AM"
    h12 = h % 12 or 12
    tag = " (next day)" if day_offset > 0 else ""
    return f"{h12}:{mm:02d} {ampm}{tag}"


def make_session():
    """Create requests session with retry logic"""
    warnings.filterwarnings("ignore", category=UserWarning, module=r"cartopy\.io")
    warnings.filterwarnings("ignore", category=Warning, module=r"cartopy\.io")
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

def get_moon_phase_name(dt):
    """Return an approximate moon phase name for a date."""
    if isinstance(dt, datetime):
        dt = dt.astimezone(timezone.utc)
    else:
        dt = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)

    diff = dt - datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
    days = diff.total_seconds() / 86400.0
    lunations = days / 29.53058867
    phase = lunations - math.floor(lunations)
    index = int((phase * 8) + 0.5) % 8
    names = [
        "New Moon",
        "Waxing Crescent",
        "First Quarter",
        "Waxing Gibbous",
        "Full Moon",
        "Waning Gibbous",
        "Last Quarter",
        "Waning Crescent",
    ]
    return names[index]

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
        self.current_display_mode = None
        self.hourly_forecast_data = None
        self.rain_frame_times = []
        self.rain_frame_arrays = []
        self.rain_frame_index = 0
        self.rain_preview_job = None
        
        # Build UI
        self._build_location_frame()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self._build_current_conditions_tab()
        self._build_satellite_tab()
        self._build_weather_maps_tab()
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

    def _build_satellite_tab(self):
        """Enhanced satellite tab with switchable national satellite and IR zoom."""
        self.sat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sat_frame, text="Satellite & IR")

        controls = tk.Frame(self.sat_frame)
        controls.pack(fill='x', padx=10, pady=6)

        self.sat_choice = ttk.Combobox(controls, values=[
            "GOES GeoColor", "GOES Infrared", "IR Zoom Regional"
        ], width=22, state="readonly")
        self.sat_choice.set("GOES GeoColor")
        self.sat_choice.pack(side='left', padx=6)

        tk.Button(controls, text="Load Selected View", command=self.load_selected_view).pack(side='left', padx=6)

        self.sat_title_label = tk.Label(self.sat_frame, text="National Satellite", font=("Arial", 10, "bold"))
        self.sat_title_label.pack(anchor='w', padx=10, pady=(4,0))

        self.sat_display_frame = tk.Frame(self.sat_frame, bg="black")
        self.sat_display_frame.configure(width=SAT_DISPLAY_WIDTH, height=SAT_DISPLAY_HEIGHT)
        self.sat_display_frame.pack(fill='both', expand=True, padx=10, pady=(5,0))

        self.sat_display_label = tk.Label(
            self.sat_display_frame,
            text="Satellite or IR zoom image will appear here",
            bg="black",
            fg="white",
            anchor='center',
            justify='center'
        )
        self.sat_display_label.pack(fill='both', expand=True)
        self.sat_display_frame.pack_propagate(False)

        self.sat_time_label = tk.Label(self.sat_frame, text="", anchor='w')
        self.sat_time_label.pack(fill='x', padx=12, pady=(5,8))

    def _build_weather_maps_tab(self):
        """Weather maps tab showing forecast temperature contours and rain timing"""
        self.weather_maps_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.weather_maps_frame, text="Weather Maps")

        self.weather_maps_notebook = ttk.Notebook(self.weather_maps_frame)
        self.weather_maps_notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Temperature map subtab
        self.temp_tab = ttk.Frame(self.weather_maps_notebook)
        self.weather_maps_notebook.add(self.temp_tab, text="Temperature Map")

        temp_controls = tk.Frame(self.temp_tab)
        temp_controls.pack(fill='x', padx=10, pady=6)
        tk.Button(temp_controls, text="Load Temperature Map", command=self.load_weather_map).pack(side='left', padx=6)
        self.weather_maps_status = tk.Label(temp_controls, text="", fg="green")
        self.weather_maps_status.pack(side='left', padx=12)

        self.weather_maps_fig = plt.Figure(figsize=(12, 8), tight_layout=True)
        self.weather_maps_ax = self.weather_maps_fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        self.weather_maps_canvas = FigureCanvasTkAgg(self.weather_maps_fig, self.temp_tab)
        self.weather_maps_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        # Rain timing subtab
        self.rain_tab = ttk.Frame(self.weather_maps_notebook)
        self.weather_maps_notebook.add(self.rain_tab, text="Rain Timing")

        rain_controls = tk.Frame(self.rain_tab)
        rain_controls.pack(fill='x', padx=10, pady=6)
        tk.Button(rain_controls, text="Render Rain Timing", command=self.load_rain_timing).pack(side='left', padx=6)
        self.rain_save_button = tk.Button(rain_controls, text="Save Rain Animation...", command=self.save_rain_timing, state='disabled')
        self.rain_save_button.pack(side='left', padx=6)
        self.rain_status_label = tk.Label(rain_controls, text="", fg="green")
        self.rain_status_label.pack(side='left', padx=12)

        self.rain_fig = plt.Figure(figsize=(12, 8), tight_layout=True)
        self.rain_ax = self.rain_fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        self.rain_canvas = FigureCanvasTkAgg(self.rain_fig, self.rain_tab)
        self.rain_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def load_selected_view(self):
        choice = self.sat_choice.get()
        if "IR Zoom" in choice:
            self.load_ir_zoom()
        else:
            self.load_satellite()

    def load_weather_map(self):
        """Load temperature contour map using Open-Meteo forecast data"""
        self.weather_maps_status.config(text="Loading temperature map...", fg="orange")
        threading.Thread(target=self._load_weather_map_thread, daemon=True).start()

    def _load_weather_map_thread(self):
        lats = np.arange(25, 50, 3)
        lons = np.arange(-125, -65, 3)
        lat_grid, lon_grid = np.meshgrid(lats, lons)
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()

        temps = []
        for lat, lon in zip(lat_flat, lon_flat):
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": float(lat),
                "longitude": float(lon),
                "hourly": "temperature_2m",
                "forecast_days": 1,
                "temperature_unit": "fahrenheit"
            }
            data, err = safe_get_json(url, params=params)
            if err or not data:
                self.root.after(0, lambda: self.weather_maps_status.config(text=f"Map load failed: {err or 'no data'}", fg="red"))
                return
            temps.append(data.get("hourly", {}).get("temperature_2m", [None])[12])

        temps = np.array(temps, dtype=float)

        grid_lon, grid_lat = np.meshgrid(
            np.linspace(lons.min(), lons.max(), 200),
            np.linspace(lats.min(), lats.max(), 200)
        )
        grid_temp = griddata(
            (lon_flat, lat_flat), temps,
            (grid_lon, grid_lat), method='cubic'
        )

        def plot_map():
            self.weather_maps_ax.clear()
            self.weather_maps_ax.set_title('Interpolated Temperature Forecast')
            self.weather_maps_ax.set_extent([-125, -65, 25, 50], crs=ccrs.PlateCarree())
            self.weather_maps_ax.add_feature(cfeature.COASTLINE)
            self.weather_maps_ax.add_feature(cfeature.BORDERS)
            self.weather_maps_ax.add_feature(cfeature.STATES, linestyle=':')
            self.weather_maps_ax.add_feature(cfeature.OCEAN, zorder=10, facecolor='white')

            contour = self.weather_maps_ax.contourf(
                grid_lon, grid_lat, grid_temp, levels=20,
                cmap='coolwarm', transform=ccrs.PlateCarree(), alpha=0.85
            )
            self.weather_maps_fig.colorbar(contour, orientation='horizontal', pad=0.05, label='Temperature (°F)')
            self.weather_maps_canvas.draw()
            self.weather_maps_status.config(text='Temperature map loaded', fg='green')

        self.root.after(0, plot_map)

    def _cancel_rain_preview(self):
        if self.rain_preview_job:
            try:
                self.root.after_cancel(self.rain_preview_job)
            except tk.TclError:
                pass
            self.rain_preview_job = None

    def load_rain_timing(self):
        """Prepare rain timing preview and enable saving."""
        self.rain_status_label.config(text="Rendering rain timing preview...", fg="orange")
        self.rain_save_button.config(state='disabled')
        self._cancel_rain_preview()
        threading.Thread(target=self._load_rain_timing_thread, daemon=True).start()

    def _load_rain_timing_thread(self):
        try:
            self.rain_frame_times = [i * RAIN_SIM_MINUTES_PER_FRAME for i in range(n_frames)]
            self.rain_frame_arrays = [rain_intensity_field(t) for t in self.rain_frame_times]
            self.rain_frame_index = 0
            self.root.after(0, self._start_rain_preview)
        except Exception as e:
            self.root.after(0, lambda: self.rain_status_label.config(text=f"Rain preview failed: {e}", fg="red"))

    def _start_rain_preview(self):
        self.rain_save_button.config(state='normal')
        self.rain_status_label.config(text="Rain timing preview ready", fg="green")
        self._draw_rain_frame_preview()

    def _draw_rain_frame_preview(self):
        if not self.rain_frame_arrays:
            return

        now_min = self.rain_frame_times[self.rain_frame_index]
        self.rain_ax.clear()
        self.rain_ax.set_extent([RAIN_WEST, RAIN_EAST, RAIN_SOUTH, RAIN_NORTH], crs=ccrs.PlateCarree())
        self.rain_ax.add_feature(cfeature.COASTLINE)
        self.rain_ax.add_feature(cfeature.BORDERS)
        self.rain_ax.add_feature(cfeature.STATES, linestyle=':')

        self.rain_ax.contourf(
            RAIN_LON_GRID,
            RAIN_LAT_GRID,
            self.rain_frame_arrays[self.rain_frame_index],
            levels=20,
            cmap='Blues',
            vmin=0,
            vmax=1,
            alpha=0.85,
            transform=ccrs.PlateCarree()
        )
        self.rain_ax.set_title(f'Rain Timing — West Tennessee   |   {rain_fmt_clock(now_min)}')
        self.rain_canvas.draw_idle()

        self.rain_frame_index = (self.rain_frame_index + 1) % len(self.rain_frame_arrays)
        self.rain_preview_job = self.root.after(int(1000 / RAIN_FPS), self._draw_rain_frame_preview)

    def save_rain_timing(self):
        if not self.rain_frame_arrays:
            self.rain_status_label.config(text="Render the rain timing preview first", fg="red")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("GIF animation", "*.gif")],
            title="Save Rain Timing Animation"
        )
        if not path:
            return

        self.rain_status_label.config(text="Saving rain animation...", fg="orange")
        threading.Thread(target=self._save_rain_timing_thread, args=(path,), daemon=True).start()

    def _save_rain_timing_thread(self, path):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ext = os.path.splitext(path)[1].lower()
        primary_writer = FFMpegWriter(fps=RAIN_FPS, bitrate=1800) if ext == '.mp4' else PillowWriter(fps=RAIN_FPS)

        def build_frame(now_min):
            ax.clear()
            ax.set_extent([RAIN_WEST, RAIN_EAST, RAIN_SOUTH, RAIN_NORTH], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(cfeature.STATES, linestyle=':')
            ax.contourf(
                RAIN_LON_GRID,
                RAIN_LAT_GRID,
                rain_intensity_field(now_min),
                levels=20,
                cmap='Blues',
                vmin=0,
                vmax=1,
                alpha=0.85,
                transform=ccrs.PlateCarree()
            )
            ax.set_title(f'Rain Timing — West Tennessee   |   {rain_fmt_clock(now_min)}')

        try:
            with primary_writer.saving(fig, path, dpi=100):
                for now_min in self.rain_frame_times:
                    build_frame(now_min)
                    primary_writer.grab_frame()
            plt.close(fig)
            self.root.after(0, lambda: self.rain_status_label.config(text=f"Saved animation to {os.path.basename(path)}", fg='green'))
            return
        except Exception as e:
            if ext == '.mp4':
                gif_path = os.path.splitext(path)[0] + '.gif'
                try:
                    fallback_writer = PillowWriter(fps=RAIN_FPS)
                    with fallback_writer.saving(fig, gif_path, dpi=100):
                        for now_min in self.rain_frame_times:
                            build_frame(now_min)
                            fallback_writer.grab_frame()
                    plt.close(fig)
                    self.root.after(0, lambda: self.rain_status_label.config(
                        text=f"MP4 save failed, saved GIF instead: {os.path.basename(gif_path)}", fg='green'))
                    return
                except Exception as e2:
                    plt.close(fig)
                    self.root.after(0, lambda: self.rain_status_label.config(text=f"Save failed: {e2}", fg='red'))
                    return
            plt.close(fig)
            self.root.after(0, lambda: self.rain_status_label.config(text=f"Save failed: {e}", fg='red'))

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
        if not hourly.get("time"):
            self.root.after(0, lambda: self.status_label.config(text="Open-Meteo returned no data", fg="red"))
            return

        self.hourly_forecast_data = hourly
        self.root.after(0, lambda: self.status_label.config(text="Open-Meteo loaded", fg="green"))
        self.root.after(0, self.update_analysis_charts)

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
        self._cancel_all_animation()
        self.current_display_mode = 'sat'
        self.sat_title_label.config(text="National Satellite")
        self.sat_display_label.config(text="Loading satellite imagery...", image="", bg="black")
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
            self.root.after(0, lambda: self.sat_display_label.config(text=f"Satellite error: {err}", image=""))
            return

        try:
            img = Image.open(io.BytesIO(data))
            frames = []
            durations = []

            for frame in ImageSequence.Iterator(img):
                f = frame.convert("RGBA")
                # Resize for display
                f = f.resize((SAT_DISPLAY_WIDTH, SAT_DISPLAY_HEIGHT), Image.Resampling.LANCZOS)
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
                    self.current_display_mode = 'sat'
                    self.sat_display_label.config(image=self.sat_frames[0], text="")
                    self.sat_display_label.image = self.sat_frames[0]
                    self.sat_time_label.config(text=f"National Satellite: {time_str}")
                    self._animate_satellite()

            self.root.after(0, start_animation)

        except Exception as e:
            self.root.after(0, lambda: self.sat_display_label.config(text=f"Image error: {e}", image=""))

    def _animate_satellite(self):
        """Animate national satellite frames"""
        if not self.sat_frames:
            return

        try:
            frame = self.sat_frames[self.sat_frame_index]
            self.sat_display_label.config(image=frame, text="")
            self.sat_display_label.image = frame
        except Exception:
            pass

        # Schedule next frame
        duration = self.sat_durations[self.sat_frame_index] if self.sat_frame_index < len(self.sat_durations) else 500
        self.sat_frame_index = (self.sat_frame_index + 1) % len(self.sat_frames)
        
        self.sat_anim_job = self.root.after(duration, self._animate_satellite)

    def load_ir_zoom(self):
        """Load IR zoom for current location"""
        self._cancel_all_animation()
        self.current_display_mode = 'ir'
        self.sat_title_label.config(text="Regional IR Zoom")
        self.sat_display_label.config(text="Loading IR zoom...", image="", bg="black")
        threading.Thread(target=self._load_ir_thread, daemon=True).start()

    def _cancel_sat_animation(self):
        if self.sat_anim_job:
            try:
                self.root.after_cancel(self.sat_anim_job)
            except tk.TclError:
                pass
            self.sat_anim_job = None

    def _cancel_ir_animation(self):
        if self.ir_anim_job:
            try:
                self.root.after_cancel(self.ir_anim_job)
            except tk.TclError:
                pass
            self.ir_anim_job = None

    def _cancel_all_animation(self):
        self._cancel_sat_animation()
        self._cancel_ir_animation()
        self.sat_frames = []
        self.ir_frames = []

    def _load_ir_thread(self):
        """Load IR zoom data in background - integrating test.py functionality"""
        try:
            lat, lon = map(float, self.location.split(','))
        except:
            self.root.after(0, lambda: self.sat_display_label.config(text="Invalid coordinates", image=""))
            return

        # Load GOES IR CONUS GIF (same as test.py)
        url = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/13/GOES16-CONUS-13-625x375.gif"
        data, err = safe_get_bytes(url, timeout=20)
        
        if err:
            self.root.after(0, lambda: self.sat_display_label.config(text=f"IR load error: {err}", image=""))
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
                cropped = cropped.resize((IR_DISPLAY_WIDTH, IR_DISPLAY_HEIGHT), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(cropped)
                frames.append(tk_img)
                durations.append(frame.info.get("duration", 500))

            self.ir_frames = frames
            self.ir_durations = durations
            self.ir_frame_index = 0

                # Start IR animation
            def start_ir_animation():
                if self.ir_frames:
                    self.sat_display_label.config(image=self.ir_frames[0], text="")
                    self.sat_display_label.image = self.ir_frames[0]
                    self._animate_ir_zoom()

            self.root.after(0, start_ir_animation)

        except Exception as e:
            self.root.after(0, lambda: self.sat_display_label.config(text=f"IR processing error: {e}", image=""))

    def _animate_ir_zoom(self):
        """Animate IR zoom frames"""
        if not self.ir_frames:
            return

        try:
            frame = self.ir_frames[self.ir_frame_index]
            self.sat_display_label.config(image=frame, text="")
            self.sat_display_label.image = frame
        except:
            pass

        # Schedule next frame
        duration = self.ir_durations[self.ir_frame_index] if self.ir_frame_index < len(self.ir_durations) else 500
        self.ir_frame_index = (self.ir_frame_index + 1) % len(self.ir_frames)
        
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
                if not err and forecast_data:
                    output += "\n=== 8-DAY FORECAST ===\n"
                    periods = forecast_data['properties'].get('periods', [])
                    daily = {}

                    for period in periods:
                        start_time = period.get('startTime')
                        try:
                            dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        except Exception:
                            continue

                        date_key = dt.date()
                        if date_key not in daily:
                            daily[date_key] = {
                                'day': None,
                                'night': None,
                                'unit': period.get('temperatureUnit', 'F'),
                                'moon': get_moon_phase_name(dt),
                            }

                        entry = {
                            'temp': period.get('temperature'),
                            'wind': period.get('windSpeed', 'N/A'),
                            'precip': period.get('probabilityOfPrecipitation', {}).get('value')
                                     if isinstance(period.get('probabilityOfPrecipitation'), dict) else period.get('probabilityOfPrecipitation'),
                            'conds': period.get('shortForecast', period.get('detailedForecast', 'N/A')),
                        }

                        if period.get('isDaytime'):
                            daily[date_key]['day'] = entry
                        else:
                            daily[date_key]['night'] = entry

                    dates = sorted(daily.keys())[:8]
                    for date_key in dates:
                        entry = daily[date_key]
                        date_str = date_key.strftime('%a %m/%d')
                        output += f"\n{date_str} — Moon: {entry['moon']}\n"

                        day = entry['day']
                        night = entry['night']
                        unit = entry['unit']

                        if day:
                            day_precip = f"{int(day['precip'])}%" if isinstance(day['precip'], (int, float)) else str(day['precip'] or 'N/A')
                            output += f"Day: High {day['temp']}°{unit}, Wind {day['wind']}, Precip {day_precip}, {day['conds']}\n"
                        else:
                            output += "Day: No data available\n"

                        if night:
                            night_precip = f"{int(night['precip'])}%" if isinstance(night['precip'], (int, float)) else str(night['precip'] or 'N/A')
                            output += f"Night: Low {night['temp']}°{unit}, Wind {night['wind']}, Precip {night_precip}, {night['conds']}\n"
                        else:
                            output += "Night: No data available\n"

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

        hours = []
        temp_data = []
        pressure_data = []
        wind_data = []
        humidity_data = []

        if self.hourly_forecast_data and self.hourly_forecast_data.get('time'):
            times = self.hourly_forecast_data.get('time', [])
            temp_data_c = self.hourly_forecast_data.get('temperature_2m', [])
            pressure_data_hpa = self.hourly_forecast_data.get('pressure_msl', [])
            wind_data_ms = self.hourly_forecast_data.get('windspeed_10m', [])
            humidity_data_pct = self.hourly_forecast_data.get('relativehumidity_2m', [])

            for i, time_str in enumerate(times[:48]):
                try:
                    dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                except Exception:
                    continue
                hours.append(dt)
                temp_data.append(temp_data_c[i] * 9/5 + 32 if i < len(temp_data_c) and temp_data_c[i] is not None else None)
                pressure_data.append(pressure_data_hpa[i] if i < len(pressure_data_hpa) and pressure_data_hpa[i] is not None else None)
                wind_data.append(wind_data_ms[i] * 2.237 if i < len(wind_data_ms) and wind_data_ms[i] is not None else None)
                humidity_data.append(humidity_data_pct[i] if i < len(humidity_data_pct) and humidity_data_pct[i] is not None else None)

        if not hours:
            now = datetime.now(timezone.utc)
            hours = [now + timedelta(hours=i) for i in range(48)]
            temp_data = [70 + 15 * math.sin(i * math.pi / 12) + np.random.normal(0, 2) for i in range(48)]
            pressure_data = [1013 + 8 * math.sin(i * math.pi / 24) + np.random.normal(0, 3) for i in range(48)]
            wind_data = [abs(8 + 5 * math.sin(i * math.pi / 16) + np.random.normal(0, 2)) for i in range(48)]
            humidity_data = [max(0, min(100, 60 + 25 * math.sin(i * math.pi / 12 + math.pi/4) + np.random.normal(0, 5))) for i in range(48)]

        # Temperature trend
        self.ax1.plot(hours, temp_data, 'r-', linewidth=2, marker='o', markersize=3)
        self.ax1.set_title('Temperature Trend (48h)')
        self.ax1.set_ylabel('Temperature (°F)')
        self.ax1.grid(True, alpha=0.3)

        # Pressure trend
        self.ax2.plot(hours, pressure_data, 'b-', linewidth=2, marker='s', markersize=3)
        self.ax2.set_title('Pressure Trend (48h)')
        self.ax2.set_ylabel('Pressure (hPa)')
        self.ax2.grid(True, alpha=0.3)

        # Wind speed
        self.ax3.plot(hours, wind_data, 'g-', linewidth=2, marker='^', markersize=3)
        self.ax3.set_title('Wind Speed (48h)')
        self.ax3.set_ylabel('Speed (mph)')
        self.ax3.grid(True, alpha=0.3)

        # Humidity
        humidity_data = [max(0, min(100, h)) if h is not None else None for h in humidity_data]
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
                self.load_open_meteo()
                
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
