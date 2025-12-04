s# -*- coding: utf-8 -*-

import math
from datetime import datetime, timedelta, timezone
from sgp4.api import WGS72, Satrec, jday, days2mdhms  # sgp4.api.jday is required for time conversion
import numpy as np

# --- TLEValidator Class ---
class TLEValidator:
    """
    Validates and parses Two-Line Element (TLE) data
    """

    def __init__(self):
        self.tle_data = None
        self.satellite_name = None
        self.line1 = None
        self.line2 = None
        self.orbital_elements = {}

    def validate_line_length(self, line, expected_length=69):
        """Check if line has correct length"""
        return len(line) == expected_length

    def validate_line_number(self, line, expected_number):
        """Check if line starts with correct line number"""
        if len(line) < 1:
            return False
        return line[0] == str(expected_number)

    def calculate_checksum(self, line):
        """
        Calculate TLE checksum
        Rules: - Add all digits (0-9) - Minus sign (-) counts as 1
        - Plus sign, space, letter, and period are ignored - Take modulo 10
        """
        checksum = 0
        for char in line[:-1]:  # Exclude the checksum digit itself
            if char.isdigit():
                checksum += int(char)
            elif char == '-':
                checksum += 1
        return checksum % 10

    def validate_checksum(self, line):
        """Verify TLE line checksum"""
        if len(line) < 69:
            return False

        calculated = self.calculate_checksum(line)
        provided = int(line[68])

        return calculated == provided

    def validate_tle_format(self, line1, line2):
        """
        Comprehensive TLE format validation
        """
        errors = []

        # Line 1 validation
        if not self.validate_line_length(line1):
            errors.append(f"Line 1: Invalid length ({len(line1)} chars, expected 69)")
        if not self.validate_line_number(line1, 1):
            errors.append("Line 1: Must start with '1'")
        if not self.validate_checksum(line1):
            errors.append("Line 1: Checksum validation failed")

        # Line 2 validation
        if not self.validate_line_length(line2):
            errors.append(f"Line 2: Invalid length ({len(line2)} chars, expected 69)")
        if not self.validate_line_number(line2, 2):
            errors.append("Line 2: Must start with '2'")
        if not self.validate_checksum(line2):
            errors.append("Line 2: Checksum validation failed")

        # Catalog number consistency
        try:
            cat_num1 = line1[2:7].strip()
            cat_num2 = line2[2:7].strip()
            if cat_num1 != cat_num2:
                errors.append(f"Catalog numbers don't match (Line1: {cat_num1}, Line2: {cat_num2})")
        except:
            errors.append("Error reading catalog numbers")

        return errors

    def parse_tle_line1(self, line1):
        """Parse Line 1 of TLE."""
        try:
            data = {}
            data['line_number'] = int(line1[0])
            data['catalog_number'] = line1[2:7].strip()
            data['classification'] = line1[7]
            data['international_designator'] = line1[9:17].strip()
            data['epoch_year'] = int(line1[18:20])
            data['epoch_day'] = float(line1[20:32].strip())
            mm_deriv_str = line1[33:43].strip().replace('+', '')
            data['mean_motion_derivative'] = float(mm_deriv_str)
            data['mean_motion_sec_derivative'] = self._parse_scientific(line1[44:52])
            data['bstar_drag'] = self._parse_scientific(line1[53:61])
            data['ephemeris_type'] = int(line1[62])
            data['element_set_number'] = int(line1[64:68])
            data['checksum_line1'] = int(line1[68])
            return data
        except Exception as e:
            raise ValueError(f"Error parsing Line 1: {str(e)}")

    def parse_tle_line2(self, line2):
        """Parse Line 2 of TLE"""
        try:
            data = {}
            data['line_number'] = int(line2[0])
            data['catalog_number'] = line2[2:7].strip()
            data['inclination'] = float(line2[8:16])
            data['raan'] = float(line2[17:25])
            data['eccentricity'] = float('0.' + line2[26:33])
            data['argument_of_perigee'] = float(line2[34:42])
            data['mean_anomaly'] = float(line2[43:51])
            data['mean_motion'] = float(line2[52:63])
            data['revolution_number'] = int(line2[63:68])
            data['checksum_line2'] = int(line2[68])
            return data
        except Exception as e:
            raise ValueError(f"Error parsing Line 2: {str(e)}")

    def _parse_scientific(self, value_str):
        """
        Parse TLE scientific notation.
        Format: ±.ddddd±d where last digit is exponent (e.g., -12345-3 = -0.12345e-3)
        """
        value_str = value_str.strip()
        if len(value_str) < 2: return 0.0
        sign = 1 if value_str[0] != '-' else -1
        mantissa = value_str[1:6]
        exp_sign = value_str[6] if len(value_str) > 6 else '+'
        exponent = value_str[7] if len(value_str) > 7 else '0'
        try:
            value = sign * float('0.' + mantissa) * (10 ** (int(exp_sign + exponent)))
        except ValueError:
            return 0.0
        return value

    def get_tle_input(self):
        """Interactive TLE input with validation"""
        print("=" * 70)
        print("TLE (Two-Line Element) Input")
        print("=" * 70)
        print("\nTLE Format Requirements:")
        print("- Line 0: Satellite name (optional)")
        print("- Line 1: Must start with '1', exactly 69 characters")
        print("- Line 2: Must start with '2', exactly 69 characters")
        print("- Both lines must have valid checksums")
        print("\nExample TLE:")
        print("ISS (ZARYA)")
        print("1 25544U 98067A   24123.45678901  .00002182  00000+0  41420-4 0  9992")
        print("2 25544  51.6461 339.8014 0001745  92.3456 267.8123 15.48919393123456")
        print("\nYou can also enter 'sample' to use a default ISS TLE")
        print("=" * 70)

        while True:
            print("\n" + "-" * 70)

            name = input("Enter Satellite Name (or press Enter to skip): ").strip()
            if name.lower() == 'sample': return self.load_sample_tle()
            line1 = input("Enter TLE Line 1: ").strip()
            if line1.lower() == 'sample': return self.load_sample_tle()
            line2 = input("Enter TLE Line 2: ").strip()

            errors = self.validate_tle_format(line1, line2)
            if errors:
                print("\n❌ TLE VALIDATION FAILED:")
                for error in errors: print(f" \u00A0• {error}")
                retry = input("\nWould you like to try again? (yes/no): ").strip().lower()
                if retry not in ['yes', 'y', '']: return None
                continue

            try:
                line1_data = self.parse_tle_line1(line1)
                line2_data = self.parse_tle_line2(line2)
                self.satellite_name = name if name else f"SAT-{line1_data['catalog_number']}"
                self.line1 = line1
                self.line2 = line2
                self.orbital_elements = {**line1_data, **line2_data}
                print("\n✅ TLE VALIDATED SUCCESSFULLY!")
                self.display_parsed_data()
                return {'name': self.satellite_name, 'line1': line1, 'line2': line2, 'elements': self.orbital_elements}
            except Exception as e:
                print(f"\n❌ ERROR: {str(e)}")
                retry = input("\nWould you like to try again? (yes/no): ").strip().lower()
                if retry not in ['yes', 'y', '']: return None

    def load_sample_tle(self):
        print("\n📡 Loading sample ISS TLE...")
        name = "ISS (ZARYA)"
        # Note: This is a placeholder TLE structure and may not be current/real
        line1 = "1 25544U 98067A   24336.50000000  .00000000+  00000+0  41420-4 0  9990"
        line2 = "2 25544  51.6461 339.8014 0001745  92.3456 267.8123 15.49500000000000"

        # Ensure checksum is correct for modified line 1
        line1 = line1[:-1] + str(self.calculate_checksum(line1))
        # Ensure checksum is correct for modified line 2
        line2 = line2[:-1] + str(self.calculate_checksum(line2))

        line1_data = self.parse_tle_line1(line1)
        line2_data = self.parse_tle_line2(line2)
        self.satellite_name = name
        self.line1 = line1
        self.line2 = line2
        self.orbital_elements = {**line1_data, **line2_data}
        print("✅ Sample TLE loaded successfully!")
        self.display_parsed_data()
        return {'name': name, 'line1': line1, 'line2': line2, 'elements': self.orbital_elements}

    def display_parsed_data(self):
        """Display parsed orbital elements"""
        print("\n" + "=" * 70)
        print("PARSED ORBITAL ELEMENTS")
        print("=" * 70)
        print(f"Satellite Name:         {self.satellite_name}")
        print(f"Catalog Number:         {self.orbital_elements['catalog_number']}")
        print(f"Epoch Year:             20{self.orbital_elements['epoch_year']}")
        print(f"Epoch Day:              {self.orbital_elements['epoch_day']:.8f}")
        print("-" * 70)
        print(f"Inclination:            {self.orbital_elements['inclination']:.4f}°")
        print(f"RAAN:                   {self.orbital_elements['raan']:.4f}°")
        print(f"Eccentricity:           {self.orbital_elements['eccentricity']:.7f}")
        print(f"Arg of Perigee:         {self.orbital_elements['argument_of_perigee']:.4f}°")
        print(f"Mean Anomaly:           {self.orbital_elements['mean_anomaly']:.4f}°")
        print(f"Mean Motion:            {self.orbital_elements['mean_motion']:.8f} rev/day")
        print(f"Revolution Number:      {self.orbital_elements['revolution_number']}")
        print("=" * 70)

# ----------------------------------------------------------------------
# --- GroundStationValidator Class ---
# ----------------------------------------------------------------------

class GroundStationValidator:
    """
    Handles input and validation for Ground Station Parameters
    """
    def __init__(self):
        self.station_data = {}

    def get_float(self, prompt, default=None):
        """Utility to safely read a float with optional default."""
        while True:
            x = input(prompt)
            if x.strip() == "" and default is not None:
                return default
            try:
                return float(x)
            except ValueError:
                print("❌ Please enter a valid number.")

    def _get_float_input(self, prompt, default, units):
        """Helper for geographical coordinates with units in prompt"""
        full_prompt = f"{prompt} [{default}] {units}: "
        return self.get_float(full_prompt, default)

    def get_station_input(self):
        """Collects all ground station parameters"""
        print("\n" + "=" * 70)
        print("GROUND STATION PARAMETERS")
        print("=" * 70)

        self.station_data['gain_dBi'] = self.get_float("Enter antenna gain (dBi) [default 16]: ", default=16.0)
        print("\n--- Frequency band (UHF: 400–450 MHz) ---")
        freq = self.get_float("Enter center frequency in MHz [default 425]: ", default=425.0)
        self.station_data['freq_mhz'] = freq
        self.station_data['freq_hz'] = freq * 1e6
        self.station_data['system_noise_temp'] = self.get_float("Enter system noise temperature (K) [default 350]: ", default=350.0)
        self.station_data['hpbw_deg'] = self.get_float("Enter HPBW (degrees) [default 30]: ", default=30.0)
        self.station_data['impl_loss_dB'] = self.get_float("Enter implementation loss (dB) [default 3]: ", default=3.0)

        print("\n--- Geographical Location ---")
        self.station_data['latitude'] = self._get_float_input("Enter Latitude", 34.05, "deg (N is +, S is -)")
        self.station_data['longitude'] = self._get_float_input("Enter Longitude", -118.25, "deg (E is +, W is -)")
        self.station_data['altitude_km'] = self._get_float_input("Enter Altitude", 0.1, "km (above sea level)")

        return self.station_data

# ----------------------------------------------------------------------
# --- NEW PASS PREDICTOR CLASS ---
# ----------------------------------------------------------------------

class PassPredictor:
    """
    Calculates satellite passes using SGP4 and geometric transformations.
    """

    def __init__(self, tle_line1, tle_line2, gs_lat_deg, gs_lon_deg, gs_alt_km):
        # 1. Initialize SGP4 Propagator (Satrec)
        self.satellite = Satrec.twoline2rv(tle_line1, tle_line2, WGS72)

        # 2. Store Ground Station Location (Convert to radians for SGP4 functions)
        self.gs_lat_rad = math.radians(gs_lat_deg)
        self.gs_lon_rad = math.radians(gs_lon_deg)
        self.gs_alt_km = gs_alt_km

        # Pre-compute ground station ECEF position
        self._compute_station_ecef()

        # Compute TLE epoch as a timezone-aware datetime in UTC
        self.tle_epoch_dt = self._compute_tle_epoch_datetime()

        self.MIN_ELEVATION_DEG = 10.0

    def _compute_station_ecef(self):
        """
        Compute the ground-station position in ECEF coordinates (km)
        using the WGS-84 ellipsoid.
        """
        # WGS-84 ellipsoid constants
        Re = 6378.137  # Equatorial radius [km]
        f = 1.0 / 298.257223563  # Flattening
        e2 = f * (2.0 - f)  # Eccentricity squared

        sin_lat = math.sin(self.gs_lat_rad)
        cos_lat = math.cos(self.gs_lat_rad)
        sin_lon = math.sin(self.gs_lon_rad)
        cos_lon = math.cos(self.gs_lon_rad)

        N = Re / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        alt = self.gs_alt_km

        # Ground station ECEF coordinates
        self.gs_x = (N + alt) * cos_lat * cos_lon
        self.gs_y = (N + alt) * cos_lat * sin_lon
        self.gs_z = (N * (1.0 - e2) + alt) * sin_lat

    def _compute_tle_epoch_datetime(self):
        """
        Build a datetime for the TLE epoch from the Satrec fields.
        """
        # Satrec.epochyr is two-digit year, Satrec.epochdays is day-of-year
        year = self.satellite.epochyr
        # Standard TLE rule: >=57 => 1900s, else 2000s
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
        """
        Calculates Azimuth, Elevation, and Range for the given time and ground station.
        """
        # 1. Convert UTC time to Julian Day (jd) and fractional day (fr)
        jd, fr = jday(t_utc.year, t_utc.month, t_utc.day, t_utc.hour, t_utc.minute, t_utc.second + t_utc.microsecond / 1e6)
        jd_ut1 = jd + fr

        # 2. Propagate the satellite to this time in the TEME frame
        error_code, r_teme, v_teme = self.satellite.sgp4(jd, fr)
        if error_code != 0:
            # Non-zero means SGP4 error (e.g., satellite has decayed or time out of range)
            raise RuntimeError(f"SGP4 error code {error_code} at {t_utc.isoformat()}")

        # 3. Convert TEME position to ECEF
        r_ecef = self._teme_to_ecef(r_teme, jd_ut1)

        # 4. Convert ECEF position to topocentric (Az, El, Range)
        az_deg, el_deg, range_km = self._ecef_to_az_el_range(r_ecef)

        return az_deg, el_deg, range_km

    def _teme_to_ecef(self, r_teme, jd_ut1):
        """
        Convert TEME position vector to ECEF using a simple GMST rotation.
        This is sufficient for pass prediction accuracy.
        """
        # Compute GMST (Greenwich Mean Sidereal Time) in radians
        T = (jd_ut1 - 2451545.0) / 36525.0
        gmst_deg = (
            280.46061837
            + 360.98564736629 * (jd_ut1 - 2451545.0)
            + 0.000387933 * T * T
            - (T ** 3) / 38710000.0
        )
        gmst_rad = math.radians(gmst_deg % 360.0)

        cos_g = math.cos(gmst_rad)
        sin_g = math.sin(gmst_rad)

        x_t, y_t, z_t = r_teme

        # Rotate about Earth's Z-axis
        x_e = cos_g * x_t + sin_g * y_t
        y_e = -sin_g * x_t + cos_g * y_t
        z_e = z_t

        return x_e, y_e, z_e

    def _ecef_to_az_el_range(self, r_ecef):
        """
        Convert ECEF coordinates of the satellite to Azimuth, Elevation, and Range
        as seen from the ground station.
        """
        # Vector from station to satellite in ECEF
        rx = r_ecef[0] - self.gs_x
        ry = r_ecef[1] - self.gs_y
        rz = r_ecef[2] - self.gs_z

        sin_lat = math.sin(self.gs_lat_rad)
        cos_lat = math.cos(self.gs_lat_rad)
        sin_lon = math.sin(self.gs_lon_rad)
        cos_lon = math.cos(self.gs_lon_rad)

        # ECEF to SEZ (South-East-Zenith) transformation
        s = -sin_lat * cos_lon * rx - sin_lat * sin_lon * ry + cos_lat * rz
        e = -sin_lon * rx + cos_lon * ry
        z = cos_lat * cos_lon * rx + cos_lat * sin_lon * ry + sin_lat * rz

        range_km = math.sqrt(s * s + e * e + z * z)
        el_rad = math.asin(z / range_km)
        az_rad = math.atan2(e, -s)  # Az measured from North, positive towards East
        if az_rad < 0.0:
            az_rad += 2.0 * math.pi

        return math.degrees(az_rad), math.degrees(el_rad), range_km

    def find_passes(self, start_time_utc=None, duration_days=2, time_step_sec=10):
        """
        Calculates upcoming passes over the ground station.
        """
        passes = []

        # Determine prediction start time, considering TLE epoch validity
        if start_time_utc is None:
            tle_epoch_dt = self.tle_epoch_dt
            current_system_time = datetime.now(timezone.utc)

            # If TLE epoch is more than 30 days different from current time, use TLE epoch
            # Otherwise, use current system time.
            if abs((tle_epoch_dt - current_system_time).days) > 30:
                print(f"⚠️ Warning: TLE epoch ({tle_epoch_dt.date()}) is significantly different from current system time ({current_system_time.date()}).")
                print(f"   Using TLE epoch as the prediction start time for better accuracy.")
                start_time_utc = tle_epoch_dt.replace(microsecond=0)  # Use TLE epoch for start
            else:
                start_time_utc = current_system_time.replace(microsecond=0)

        # Ensure the start_time_utc is timezone-aware and UTC
        if start_time_utc.tzinfo is None or start_time_utc.tzinfo.utcoffset(start_time_utc) is None:
            start_time_utc = start_time_utc.replace(tzinfo=timezone.utc)

        current_time = start_time_utc
        end_time = start_time_utc + timedelta(days=duration_days)

        in_pass = False
        aos_time = None
        max_el_deg = 0.0
        max_el_az_deg = 0.0

        print("\n" + "-" * 70)
        print(f"SEARCHING FOR PASSES ({duration_days} day window, Min El: {self.MIN_ELEVATION_DEG}°) ...")
        print(f"  Prediction window: {start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} to {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("-" * 70)

        while current_time < end_time:
            # Propagate satellite position and get look angles
            try:
                az_deg, el_deg, range_km = self._get_look_angles(current_time)
            except Exception as e:
                # Catch SGP4 errors (e.g., satellite decay, position outside bounds)
                # Print an error and skip the problematic time step
                # print(f"Error propagating at {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}: {e}")
                current_time += timedelta(seconds=time_step_sec)
                continue

            is_visible = el_deg >= self.MIN_ELEVATION_DEG

            if is_visible and not in_pass:
                # Acquisition of Signal (AOS) - Pass starts
                in_pass = True
                aos_time = current_time
                max_el_deg = el_deg
                max_el_az_deg = az_deg

            elif is_visible and in_pass:
                # Still in pass - check for max elevation
                if el_deg > max_el_deg:
                    max_el_deg = el_deg
                    max_el_az_deg = az_deg

            elif not is_visible and in_pass:
                # Loss of Signal (LOS) - Pass ends
                in_pass = False
                los_time = current_time

                # Use a slightly less precise time for max elevation to simplify code
                max_time = aos_time + (los_time - aos_time) / 2 # Approximation for TCA

                # Find TCA by checking around the rough max time for the peak elevation
                tca_time = None
                tca_range = 0.0

                # For simplicity, we stick to the Max El recorded during the step cycle

                duration = los_time - aos_time

                passes.append({
                    'AOS': aos_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'LOS': los_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'Duration': str(duration).split('.')[0],
                    'Max El': f"{max_el_deg:.2f}°",
                    'Azimuth at Max El': f"{max_el_az_deg:.2f}°",
                    'Range at LOS': f"{range_km:.1f} km"
                })

                # Reset tracking variables
                aos_time = None
                max_el_deg = 0.0
                max_el_az_deg = 0.0

            current_time += timedelta(seconds=time_step_sec)

        # Handle case where pass extends past end_time
        if in_pass:
            passes.append({
                'AOS': aos_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'LOS': "END OF WINDOW",
                'Duration': "N/A",
                'Max El': f"{max_el_deg:.2f}°",
                'Azimuth at Max El': f"{max_el_az_deg:.2f}°",
                'Range at LOS': "N/A"
            })

        return passes

# ----------------------------------------------------------------------
# --- Main Execution Block ---
# ----------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Get TLE Data
    tle_validator = TLEValidator()
    tle_data = tle_validator.get_tle_input()

    if not tle_data:
        print("\n❌ TLE input cancelled. Exiting.")
        exit()

    # 2. Get Ground Station Data
    gs_validator = GroundStationValidator()
    gs_data = gs_validator.get_station_input()

    # 3. Predict Passes
    try:
        predictor = PassPredictor(
            tle_data['line1'],
            tle_data['line2'],
            gs_data['latitude'],
            gs_data['longitude'],
            gs_data['altitude_km']
        )

        # Predict passes for the next 2 days
        pass_list = predictor.find_passes(duration_days=2)

        if pass_list:
            print("\n" + "=" * 85)
            print(f"🛰️ UPCOMING PASSES FOR {tle_data['name']}")
            print("=" * 85)

            # Print a neat table of passes
            print(f"{'AOS Time (UTC)':<20} | {'LOS Time (UTC)':<20} | {'Duration':<10} | {'Max El':<8} | {'Azimuth at Max El'}")
            print("-" * 85)
            for p in pass_list:
                print(f"{p['AOS']:<20} | {p['LOS']:<20} | {p['Duration']:<10} | {p['Max El']:<8} | {p['Azimuth at Max El']}")
            print("-" * 85)
            print(f"\nTotal passes found: {len(pass_list)}")

        else:
            print("\n" + "=" * 70)
            print("❌ NO PASSES FOUND ABOVE 10° ELEVATION IN THE NEXT 2 DAYS.")
            print("=" * 70)

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during Pass Prediction: {e}")

    print("\n✅ Execution finished.")

