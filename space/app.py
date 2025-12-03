from flask import Flask, render_template, request, send_file
from skyfield.api import load, Topos, EarthSatellite
import math
from datetime import datetime, timedelta, timezone
from sgp4.api import WGS72, Satrec, jday, days2mdhms
import numpy as np
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Store last prediction so the skyplot route can reuse it
last_tle_result = None
last_gs_params = None
last_pass_list = None


def create_app() -> Flask:
    flask_app = Flask(__name__)

    @flask_app.route("/", methods=["GET", "POST"])
    def home():
        hero = {
            "title": "Ground System Simulator",
            "subtitle": "Coordinate satellite-to-ground station communications with real-time status, pass planning, and anomaly tracking.",
            "primary_cta": {"label": "Initiate Pass Simulation", "href": "#planner"},
        }

        tle_result = None
        pass_list = None
        errors = []
        form_values = {}

        if request.method == "POST":
            form_values = request.form.to_dict()
            use_sample = request.form.get("use_sample") == "on"

            validator = TLEValidator()
            try:
                if use_sample:
                    tle_result = validator.load_sample_tle()
                else:
                    sat_name = request.form.get("sat_name", "").strip()
                    line1 = request.form.get("tle_line1", "").strip()
                    line2 = request.form.get("tle_line2", "").strip()
                    if not line1 or not line2:
                        raise ValueError("Both TLE lines are required, or select 'Use sample TLE'.")

                    tle_errors = validator.validate_tle_format(line1, line2)
                    if tle_errors:
                        errors.extend(tle_errors)
                    else:
                        line1_data = validator.parse_tle_line1(line1)
                        line2_data = validator.parse_tle_line2(line2)
                        satellite_name = sat_name if sat_name else f"SAT-{line1_data['catalog_number']}"
                        tle_result = {
                            "name": satellite_name,
                            "line1": line1,
                            "line2": line2,
                            "elements": {**line1_data, **line2_data},
                        }

                # Ground station parameters with defaults
                def _get_float(name, default):
                    raw = request.form.get(name, "").strip()
                    if not raw:
                        return default
                    try:
                        return float(raw)
                    except ValueError:
                        errors.append(f"Invalid value for {name.replace('_', ' ')}; using default.")
                        return default

                gs_params = {
                    "gain_dBi": _get_float("gain_dBi", 16.0),
                    "freq_mhz": _get_float("freq_mhz", 425.0),
                    "system_noise_temp": _get_float("system_noise_temp", 350.0),
                    "hpbw_deg": _get_float("hpbw_deg", 30.0),
                    "impl_loss_dB": _get_float("impl_loss_dB", 3.0),
                    "latitude": _get_float("latitude", 34.05),
                    "longitude": _get_float("longitude", -118.25),
                    "altitude_km": _get_float("altitude_km", 0.1),
                }

                if tle_result:
                    predictor = PassPredictor(
                        tle_result["line1"],
                        tle_result["line2"],
                        gs_params["latitude"],
                        gs_params["longitude"],
                        gs_params["altitude_km"],
                    )
                    pass_list = predictor.find_passes(duration_days=2)

                    # remember for skyplot route
                    global last_tle_result, last_gs_params, last_pass_list
                    last_tle_result = tle_result
                    last_gs_params = gs_params
                    last_pass_list = pass_list
            except Exception as exc:
                errors.append(str(exc))

        return render_template(
            "index.html",
            hero=hero,
            tle_result=tle_result,
            pass_list=pass_list,
            errors=errors,
            form_values=form_values,
        )

    @flask_app.route("/skyplot.png")
    def skyplot():
        global last_tle_result, last_gs_params, last_pass_list
        if not last_tle_result or not last_gs_params or not last_pass_list:
            return ("No prediction available for skyplot", 400)

        # Build a predictor with the stored config
        predictor = PassPredictor(
            last_tle_result["line1"],
            last_tle_result["line2"],
            last_gs_params["latitude"],
            last_gs_params["longitude"],
            last_gs_params["altitude_km"],
        )

        # Create figure with dark background theme
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(8, 8), facecolor='#0a0e1a')
        ax = fig.add_subplot(111, projection="polar", facecolor='#0a0e1a')
        
        # Set up polar plot
        ax.set_theta_zero_location("N")   # 0° at North
        ax.set_theta_direction(-1)        # clockwise azimuth
        ax.set_rlim(0, 90)                # 0 = zenith, 90 = horizon
        ax.set_rlabel_position(225)
        
        # Add elevation circles (horizon, 30°, 60°, zenith)
        elevation_levels = [0, 30, 60, 90]
        for el in elevation_levels:
            r_circle = 90 - el
            if el == 0:
                ax.plot(np.linspace(0, 2*np.pi, 100), 
                       [r_circle]*100, 'w-', linewidth=1.5, alpha=0.3, label='Horizon' if el == 0 else None)
            else:
                circle = plt.Circle((0, 0), r_circle, transform=ax.transData._b, 
                                   fill=False, color='white', linewidth=1, alpha=0.2, linestyle='--')
                ax.add_patch(circle)
        
        # Add cardinal direction labels
        ax.text(0, 95, 'N', ha='center', va='center', fontsize=14, fontweight='bold', color='#3be4ff')
        ax.text(np.pi/2, 95, 'E', ha='center', va='center', fontsize=14, fontweight='bold', color='#3be4ff')
        ax.text(np.pi, 95, 'S', ha='center', va='center', fontsize=14, fontweight='bold', color='#3be4ff')
        ax.text(3*np.pi/2, 95, 'W', ha='center', va='center', fontsize=14, fontweight='bold', color='#3be4ff')
        
        # Add elevation labels
        for el in [30, 60]:
            r_label = 90 - el
            ax.text(0.1, r_label, f'{el}°', fontsize=9, color='white', alpha=0.6)
        
        # Title with satellite name
        sat_name = last_tle_result.get("name", "Satellite")
        ax.set_title(f"Skyplot: {sat_name}\nGround Station: {last_gs_params['latitude']:.2f}°N, {last_gs_params['longitude']:.2f}°E", 
                    pad=25, fontsize=13, fontweight="bold", color='#eff6ff')

        # Use vibrant color palette for passes
        colors = ['#3be4ff', '#ff6b9d', '#ffd93d', '#6bcf7f', '#ff8c42', '#9d4edd', '#ff006e', '#06ffa5']
        
        has_any_data = False
        
        # Plot each pass with improved accuracy (10 second steps)
        for idx, pass_data in enumerate(last_pass_list):
            if pass_data["LOS"] == "END OF WINDOW":
                continue
            
            az_list, el_list = predictor.get_pass_track(
                pass_data["AOS"], pass_data["LOS"], time_step_sec=10  # More accurate: 10 sec steps
            )
            
            if az_list and el_list:
                az = np.radians(az_list)
                r = 90 - np.array(el_list)
                color = colors[idx % len(colors)]
                
                # Plot the pass track
                ax.plot(az, r, marker='o', lw=2.5, label=f"Pass {idx+1}", 
                       color=color, markersize=4, alpha=0.9, markevery=5)
                
                # Mark AOS (start) and LOS (end) points
                ax.plot(az[0], r[0], marker='s', markersize=10, color=color, 
                       markeredgecolor='white', markeredgewidth=1.5, label=f'AOS {idx+1}' if idx == 0 else '')
                ax.plot(az[-1], r[-1], marker='^', markersize=10, color=color, 
                       markeredgecolor='white', markeredgewidth=1.5, label=f'LOS {idx+1}' if idx == 0 else '')
                
                # Mark maximum elevation point
                max_el_idx = np.argmax(el_list)
                ax.plot(az[max_el_idx], r[max_el_idx], marker='*', markersize=15, 
                       color=color, markeredgecolor='white', markeredgewidth=1.5,
                       label=f'Max El {idx+1}' if idx == 0 else '')
                
                has_any_data = True

        if not has_any_data:
            plt.close(fig)
            return ("No elevation above threshold for skyplot", 400)

        # Customize grid and labels
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.3, color='white')
        ax.set_thetagrids(np.arange(0, 360, 30), labels=None)  # Azimuth grid every 30°
        
        # Add legend with better positioning
        if len([p for p in last_pass_list if p.get("LOS") != "END OF WINDOW"]) > 1:
            legend = ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1.0), 
                             fontsize=9, framealpha=0.9, facecolor='#16233a', 
                             edgecolor='#3be4ff', labelcolor='#eff6ff')
            legend.get_frame().set_linewidth(1.5)

        # Add info text box
        info_text = f"Total Passes: {len([p for p in last_pass_list if p.get('LOS') != 'END OF WINDOW'])}\n"
        info_text += f"Min Elevation: {predictor.MIN_ELEVATION_DEG}°"
        ax.text(1.15, 0.95, info_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#16233a', 
               alpha=0.8, edgecolor='#3be4ff', linewidth=1), color='#eff6ff')

        # Save with high DPI for crispness
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor='#0a0e1a', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    @flask_app.context_processor
    def inject_meta():
        env_label = flask_app.config.get("ENV", "production").title()
        return {"environment_label": env_label}

    return flask_app


# ----------------------------------------------------------------------
# --- TLEValidator Class (adapted from spaceconnect.py)
# ----------------------------------------------------------------------


class TLEValidator:
    """
    Validates and parses Two-Line Element (TLE) data.
    """

    def __init__(self):
        self.tle_data = None
        self.satellite_name = None
        self.line1 = None
        self.line2 = None
        self.orbital_elements = {}

    def validate_line_length(self, line, expected_length=69):
        return len(line) == expected_length

    def validate_line_number(self, line, expected_number):
        if len(line) < 1:
            return False
        return line[0] == str(expected_number)

    def calculate_checksum(self, line):
        checksum = 0
        for char in line[:-1]:
            if char.isdigit():
                checksum += int(char)
            elif char == "-":
                checksum += 1
        return checksum % 10

    def validate_checksum(self, line):
        if len(line) < 69:
            return False
        calculated = self.calculate_checksum(line)
        provided = int(line[68])
        return calculated == provided

    def validate_tle_format(self, line1, line2):
        errors = []
        if not self.validate_line_length(line1):
            errors.append(f"Line 1: Invalid length ({len(line1)} chars, expected 69)")
        if not self.validate_line_number(line1, 1):
            errors.append("Line 1: Must start with '1'")
        if not self.validate_checksum(line1):
            errors.append("Line 1: Checksum validation failed")

        if not self.validate_line_length(line2):
            errors.append(f"Line 2: Invalid length ({len(line2)} chars, expected 69)")
        if not self.validate_line_number(line2, 2):
            errors.append("Line 2: Must start with '2'")
        if not self.validate_checksum(line2):
            errors.append("Line 2: Checksum validation failed")

        try:
            cat_num1 = line1[2:7].strip()
            cat_num2 = line2[2:7].strip()
            if cat_num1 != cat_num2:
                errors.append(f"Catalog numbers don't match (Line1: {cat_num1}, Line2: {cat_num2})")
        except Exception:
            errors.append("Error reading catalog numbers")

        return errors

    def parse_tle_line1(self, line1):
        try:
            data = {}
            data["line_number"] = int(line1[0])
            data["catalog_number"] = line1[2:7].strip()
            data["classification"] = line1[7]
            data["international_designator"] = line1[9:17].strip()
            data["epoch_year"] = int(line1[18:20])
            data["epoch_day"] = float(line1[20:32].strip())
            mm_deriv_str = line1[33:43].strip().replace("+", "")
            data["mean_motion_derivative"] = float(mm_deriv_str)
            data["mean_motion_sec_derivative"] = self._parse_scientific(line1[44:52])
            data["bstar_drag"] = self._parse_scientific(line1[53:61])
            data["ephemeris_type"] = int(line1[62])
            data["element_set_number"] = int(line1[64:68])
            data["checksum_line1"] = int(line1[68])
            return data
        except Exception as exc:
            raise ValueError(f"Error parsing Line 1: {str(exc)}") from exc

    def parse_tle_line2(self, line2):
        try:
            data = {}
            data["line_number"] = int(line2[0])
            data["catalog_number"] = line2[2:7].strip()
            data["inclination"] = float(line2[8:16])
            data["raan"] = float(line2[17:25])
            data["eccentricity"] = float("0." + line2[26:33])
            data["argument_of_perigee"] = float(line2[34:42])
            data["mean_anomaly"] = float(line2[43:51])
            data["mean_motion"] = float(line2[52:63])
            data["revolution_number"] = int(line2[63:68])
            data["checksum_line2"] = int(line2[68])
            return data
        except Exception as exc:
            raise ValueError(f"Error parsing Line 2: {str(exc)}") from exc

    def _parse_scientific(self, value_str):
        value_str = value_str.strip()
        if len(value_str) < 2:
            return 0.0
        sign = 1 if value_str[0] != "-" else -1
        mantissa = value_str[1:6]
        exp_sign = value_str[6] if len(value_str) > 6 else "+"
        exponent = value_str[7] if len(value_str) > 7 else "0"
        try:
            value = sign * float("0." + mantissa) * (10 ** (int(exp_sign + exponent)))
        except ValueError:
            return 0.0
        return value

    def load_sample_tle(self):
        name = "ISS (ZARYA)"
        line1 = "1 25544U 98067A   24336.50000000  .00000000+  00000+0  41420-4 0  9990"
        line2 = "2 25544  51.6461 339.8014 0001745  92.3456 267.8123 15.49500000000000"
        line1 = line1[:-1] + str(self.calculate_checksum(line1))
        line2 = line2[:-1] + str(self.calculate_checksum(line2))
        line1_data = self.parse_tle_line1(line1)
        line2_data = self.parse_tle_line2(line2)
        self.satellite_name = name
        self.line1 = line1
        self.line2 = line2
        self.orbital_elements = {**line1_data, **line2_data}
        return {"name": name, "line1": line1, "line2": line2, "elements": self.orbital_elements}


# ----------------------------------------------------------------------
# --- PassPredictor Class (adapted from spaceconnect.py)
# ----------------------------------------------------------------------


class PassPredictor:
    """
    Calculates satellite passes using SGP4 and geometric transformations.
    """

    def __init__(self, tle_line1, tle_line2, gs_lat_deg, gs_lon_deg, gs_alt_km):
        self.satellite = Satrec.twoline2rv(tle_line1, tle_line2, WGS72)
        self.gs_lat_rad = math.radians(gs_lat_deg)
        self.gs_lon_rad = math.radians(gs_lon_deg)
        self.gs_alt_km = gs_alt_km
        self._compute_station_ecef()
        self.tle_epoch_dt = self._compute_tle_epoch_datetime()
        self.MIN_ELEVATION_DEG = 10.0

    def _compute_station_ecef(self):
        Re = 6378.137
        f = 1.0 / 298.257223563
        e2 = f * (2.0 - f)

        sin_lat = math.sin(self.gs_lat_rad)
        cos_lat = math.cos(self.gs_lat_rad)
        sin_lon = math.sin(self.gs_lon_rad)
        cos_lon = math.cos(self.gs_lon_rad)

        N = Re / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        alt = self.gs_alt_km

        self.gs_x = (N + alt) * cos_lat * cos_lon
        self.gs_y = (N + alt) * cos_lat * sin_lon
        self.gs_z = (N * (1.0 - e2) + alt) * sin_lat

    def _compute_tle_epoch_datetime(self):
        year = self.satellite.epochyr
        if year >= 57:
            year += 1900
        else:
            year += 2000

        day_of_year = self.satellite.epochdays
        month, day, hour, minute, sec = days2mdhms(year, day_of_year)
        sec_int = int(sec)
        micro = int(round((sec - sec_int) * 1e6))

        return datetime(year, month, day, hour, minute, sec_int, microsecond=micro, tzinfo=timezone.utc)

    def _get_look_angles(self, t_utc):
        jd, fr = jday(t_utc.year, t_utc.month, t_utc.day, t_utc.hour, t_utc.minute, t_utc.second + t_utc.microsecond / 1e6)
        jd_ut1 = jd + fr
        error_code, r_teme, v_teme = self.satellite.sgp4(jd, fr)
        if error_code != 0:
            raise RuntimeError(f"SGP4 error code {error_code} at {t_utc.isoformat()}")
        r_ecef = self._teme_to_ecef(r_teme, jd_ut1)
        az_deg, el_deg, range_km = self._ecef_to_az_el_range(r_ecef)
        return az_deg, el_deg, range_km

    def _teme_to_ecef(self, r_teme, jd_ut1):
        T = (jd_ut1 - 2451545.0) / 36525.0
        gmst_deg = (
            280.46061837
            + 360.98564736629 * (jd_ut1 - 2451545.0)
            + 0.000387933 * T * T
            - (T**3) / 38710000.0
        )
        gmst_rad = math.radians(gmst_deg % 360.0)
        cos_g = math.cos(gmst_rad)
        sin_g = math.sin(gmst_rad)
        x_t, y_t, z_t = r_teme
        x_e = cos_g * x_t + sin_g * y_t
        y_e = -sin_g * x_t + cos_g * y_t
        z_e = z_t
        return x_e, y_e, z_e

    def _ecef_to_az_el_range(self, r_ecef):
        rx = r_ecef[0] - self.gs_x
        ry = r_ecef[1] - self.gs_y
        rz = r_ecef[2] - self.gs_z

        sin_lat = math.sin(self.gs_lat_rad)
        cos_lat = math.cos(self.gs_lat_rad)
        sin_lon = math.sin(self.gs_lon_rad)
        cos_lon = math.cos(self.gs_lon_rad)

        s = -sin_lat * cos_lon * rx - sin_lat * sin_lon * ry + cos_lat * rz
        e = -sin_lon * rx + cos_lon * ry
        z = cos_lat * cos_lon * rx + cos_lat * sin_lon * ry + sin_lat * rz

        range_km = math.sqrt(s * s + e * e + z * z)
        el_rad = math.asin(z / range_km)
        az_rad = math.atan2(e, -s)
        if az_rad < 0.0:
            az_rad += 2.0 * math.pi

        return math.degrees(az_rad), math.degrees(el_rad), range_km

    def find_passes(self, start_time_utc=None, duration_days=2, time_step_sec=10):
        passes = []

        if start_time_utc is None:
            tle_epoch_dt = self.tle_epoch_dt
            current_system_time = datetime.now(timezone.utc)
            if abs((tle_epoch_dt - current_system_time).days) > 30:
                start_time_utc = tle_epoch_dt.replace(microsecond=0)
            else:
                start_time_utc = current_system_time.replace(microsecond=0)

        if start_time_utc.tzinfo is None or start_time_utc.tzinfo.utcoffset(start_time_utc) is None:
            start_time_utc = start_time_utc.replace(tzinfo=timezone.utc)

        current_time = start_time_utc
        end_time = start_time_utc + timedelta(days=duration_days)

        in_pass = False
        aos_time = None
        max_el_deg = 0.0
        max_el_az_deg = 0.0

        while current_time < end_time:
            try:
                az_deg, el_deg, range_km = self._get_look_angles(current_time)
            except Exception:
                current_time += timedelta(seconds=time_step_sec)
                continue

            is_visible = el_deg >= self.MIN_ELEVATION_DEG

            if is_visible and not in_pass:
                in_pass = True
                aos_time = current_time
                max_el_deg = el_deg
                max_el_az_deg = az_deg
            elif is_visible and in_pass:
                if el_deg > max_el_deg:
                    max_el_deg = el_deg
                    max_el_az_deg = az_deg
            elif not is_visible and in_pass:
                in_pass = False
                los_time = current_time
                duration = los_time - aos_time

                passes.append(
                    {
                        "AOS": aos_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "LOS": los_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "Duration": str(duration).split(".")[0],
                        "Max El": f"{max_el_deg:.2f}°",
                        "Azimuth at Max El": f"{max_el_az_deg:.2f}°",
                        "Range at LOS": f"{range_km:.1f} km",
                    }
                )

                aos_time = None
                max_el_deg = 0.0
                max_el_az_deg = 0.0

            current_time += timedelta(seconds=time_step_sec)

        if in_pass:
            passes.append(
                {
                    "AOS": aos_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "LOS": "END OF WINDOW",
                    "Duration": "N/A",
                    "Max El": f"{max_el_deg:.2f}°",
                    "Azimuth at Max El": f"{max_el_az_deg:.2f}°",
                    "Range at LOS": "N/A",
                }
            )

        return passes

    def get_pass_track(self, aos_str, los_str, time_step_sec=30):
        """
        Return lists of azimuth and elevation (degrees) between AOS and LOS
        for a single pass.
        """
        if los_str == "END OF WINDOW":
            return [], []

        aos = datetime.strptime(aos_str, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        los = datetime.strptime(los_str, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)

        az_list, el_list = [], []
        t = aos
        while t <= los:
            az_deg, el_deg, _ = self._get_look_angles(t)
            if el_deg >= self.MIN_ELEVATION_DEG:
                az_list.append(az_deg)
                el_list.append(el_deg)
            t += timedelta(seconds=time_step_sec)

        return az_list, el_list


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
