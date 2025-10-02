"""GPS Prior Loader voor MegaSaM/DROID-SLAM integratie.

Dit module laadt GPS data uit normalized manifest bestanden en converteert
ze naar DROID-SLAM compatibele pose priors.
"""

import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class GPSPriorLoader:
    """Laadt en verwerkt GPS priors uit normalized manifest."""

    def __init__(self, manifest_path: str, reference_frame: int = 0):
        """
        Args:
            manifest_path: Path naar normalized_manifest_camera_0.json
            reference_frame: Frame index voor origin (default: 0)
        """
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"GPS manifest niet gevonden: {manifest_path}")

        self.reference_frame = reference_frame
        self.manifest_data = self._load_manifest()
        self.frames_data = self.manifest_data['frames']
        self.n_frames = len(self.frames_data)

        # Extract Lambert2008 coördinaten
        self.lambert_coords = self._extract_lambert_coords()

        # Extract GPS metadata
        self.gps_metadata = self._extract_gps_metadata()

        print(f"GPS Prior Loader geïnitialiseerd:")
        print(f"  Manifest: {self.manifest_path.name}")
        print(f"  Aantal frames: {self.n_frames}")
        print(f"  GPS beschikbaar: {self.manifest_data.get('gps_available', False)}")
        print(f"  Referentie frame: {reference_frame}")

    def _load_manifest(self) -> Dict:
        """Laad JSON manifest bestand."""
        with open(self.manifest_path, 'r') as f:
            return json.load(f)

    def _extract_lambert_coords(self) -> np.ndarray:
        """Extract Lambert2008 coördinaten voor alle frames.

        Returns:
            numpy array (N, 2) met [x, y] coördinaten
        """
        coords = []
        for frame in self.frames_data:
            lambert = frame.get('lambert2008', {})
            x = lambert.get('x', 0.0)
            y = lambert.get('y', 0.0)
            coords.append([x, y])
        return np.array(coords, dtype=np.float64)

    def _extract_gps_metadata(self) -> Dict:
        """Extract aanvullende GPS metadata."""
        metadata = {
            'speed_kmh': [],
            'track': [],  # Heading/richting in graden
            'altitude': [],
            'lat': [],
            'lon': []
        }

        for frame in self.frames_data:
            gps = frame.get('gps', {})
            metadata['speed_kmh'].append(gps.get('speed_kmh', 0.0))
            metadata['track'].append(gps.get('track', 0.0))
            metadata['altitude'].append(gps.get('alt', 0.0))
            metadata['lat'].append(gps.get('lat', 0.0))
            metadata['lon'].append(gps.get('lon', 0.0))

        # Converteer naar numpy arrays
        for key in metadata:
            metadata[key] = np.array(metadata[key], dtype=np.float64)

        return metadata

    def compute_relative_positions(self) -> np.ndarray:
        """Bereken relatieve posities t.o.v. referentie frame.

        Returns:
            numpy array (N, 3) met [x, y, z] posities in meters
            Lambert X → DROID X (rechts)
            Lambert Y → DROID -Z (vooruit, noord is negatieve Z)
            Hoogte → DROID Y (verticaal, relatief t.o.v. start)
        """
        ref_x, ref_y = self.lambert_coords[self.reference_frame]
        ref_alt = self.gps_metadata['altitude'][self.reference_frame]

        positions = np.zeros((self.n_frames, 3), dtype=np.float64)

        for i in range(self.n_frames):
            x, y = self.lambert_coords[i]
            alt = self.gps_metadata['altitude'][i]

            # Relatieve posities
            dx = x - ref_x
            dy = y - ref_y
            dz = alt - ref_alt

            # Transform naar DROID coördinaten
            positions[i, 0] = dx       # X blijft X (oost = rechts)
            positions[i, 1] = dz       # Y = hoogte verschil
            positions[i, 2] = -dy      # Z = -noord (vooruit is negatieve Z)

        return positions

    def compute_orientations_from_track(self) -> np.ndarray:
        """Bereken camera orientaties uit GPS track (heading).

        GPS track geeft heading in graden (0=noord, 90=oost).
        We converteren dit naar quaternions voor DROID-SLAM.

        Returns:
            numpy array (N, 4) met quaternions [qx, qy, qz, qw]
        """
        orientations = np.zeros((self.n_frames, 4), dtype=np.float64)

        for i in range(self.n_frames):
            # GPS track: 0° = noord, 90° = oost (clockwise)
            track_deg = self.gps_metadata['track'][i]

            # Converteer naar radianen
            # DROID: camera kijkt in -Z richting (vooruit)
            # GPS Noord (0°) = DROID -Z
            # GPS Oost (90°) = DROID +X
            # Dus: DROID heading = -(GPS_track - 90°) = 90° - GPS_track

            heading_rad = np.deg2rad(90.0 - track_deg)

            # Rotatie om Y-as (verticaal)
            # Quaternion: [qx, qy, qz, qw] voor rotatie rond Y-as
            qx = 0.0
            qy = np.sin(heading_rad / 2.0)
            qz = 0.0
            qw = np.cos(heading_rad / 2.0)

            orientations[i] = [qx, qy, qz, qw]

        return orientations

    def compute_orientations_from_motion(self, positions: np.ndarray,
                                        smooth_window: int = 5) -> np.ndarray:
        """Bereken camera orientaties uit bewegingsrichting.

        Dit is een alternatief voor GPS track data, geschat uit positie veranderingen.

        Args:
            positions: (N, 3) array met posities
            smooth_window: Venster grootte voor smoothing

        Returns:
            numpy array (N, 4) met quaternions [qx, qy, qz, qw]
        """
        orientations = np.zeros((self.n_frames, 4), dtype=np.float64)

        for i in range(self.n_frames):
            # Bepaal forward direction uit nabije frames
            if i == 0:
                forward = positions[min(i + smooth_window, self.n_frames - 1)] - positions[i]
            elif i == self.n_frames - 1:
                forward = positions[i] - positions[max(0, i - smooth_window)]
            else:
                # Average van forward en backward
                forward = (positions[min(i + smooth_window, self.n_frames - 1)] -
                          positions[max(0, i - smooth_window)])

            # Normaliseer
            forward_norm = np.linalg.norm(forward)
            if forward_norm < 1e-6:
                # Geen beweging: gebruik vorige orientatie of identiteit
                if i > 0:
                    orientations[i] = orientations[i-1]
                else:
                    orientations[i] = [0, 0, 0, 1]  # Identiteit
                continue

            forward = forward / forward_norm

            # Bereken yaw (rotatie om Y-as)
            # forward = [fx, fy, fz], we willen heading uit fx en fz
            yaw = np.arctan2(forward[0], -forward[2])  # -Z is vooruit

            # Quaternion voor rotatie om Y-as
            qx = 0.0
            qy = np.sin(yaw / 2.0)
            qz = 0.0
            qw = np.cos(yaw / 2.0)

            orientations[i] = [qx, qy, qz, qw]

        return orientations

    def get_pose_priors(self, use_gps_heading: bool = True) -> torch.Tensor:
        """Bereken complete pose priors voor DROID-SLAM.

        Args:
            use_gps_heading: Gebruik GPS track voor orientatie (True) of
                           schat uit beweging (False)

        Returns:
            torch.Tensor (N, 7) met [tx, ty, tz, qx, qy, qz, qw]
        """
        # Bereken posities
        positions = self.compute_relative_positions()

        # Bereken orientaties
        if use_gps_heading:
            orientations = self.compute_orientations_from_track()
        else:
            orientations = self.compute_orientations_from_motion(positions)

        # Combineer tot 7-DoF poses
        poses = np.concatenate([positions, orientations], axis=1)

        # Converteer naar torch tensor
        poses_tensor = torch.from_numpy(poses).float()

        print(f"GPS pose priors gegenereerd:")
        print(f"  Posities bereik X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
        print(f"  Posities bereik Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m")
        print(f"  Posities bereik Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m")
        print(f"  Totale afgelegde afstand: {np.linalg.norm(np.diff(positions, axis=0), axis=1).sum():.2f} m")
        print(f"  Orientatie bron: {'GPS track' if use_gps_heading else 'Bewegingsrichting'}")

        return poses_tensor

    def get_gps_confidence_weights(self, speed_threshold: float = 1.0) -> torch.Tensor:
        """Bereken confidence weights voor GPS data.

        Lagere confidence bij:
        - Lage snelheid (GPS drift is groter bij stilstand)
        - Grote heading changes (mogelijk GPS glitches)

        Args:
            speed_threshold: Minimum snelheid (km/h) voor volle confidence

        Returns:
            torch.Tensor (N,) met weights tussen 0.0 en 1.0
        """
        speeds = self.gps_metadata['speed_kmh']

        # Speed-based confidence (sigmoid)
        speed_confidence = 1.0 / (1.0 + np.exp(-2.0 * (speeds - speed_threshold)))

        # Detect grote heading jumps
        tracks = self.gps_metadata['track']
        track_diffs = np.abs(np.diff(tracks, prepend=tracks[0]))
        # Wrap around 360°
        track_diffs = np.minimum(track_diffs, 360.0 - track_diffs)

        # Lagere confidence bij grote heading changes (> 45°)
        heading_confidence = 1.0 / (1.0 + np.exp(0.1 * (track_diffs - 45.0)))

        # Combineer
        total_confidence = speed_confidence * heading_confidence

        # Clamp tussen 0.1 en 1.0 (minimum confidence om GPS niet volledig te negeren)
        total_confidence = np.clip(total_confidence, 0.1, 1.0)

        return torch.from_numpy(total_confidence).float()

    def get_lambert_offset(self) -> Dict:
        """Haal Lambert2008 offset metadata op voor terug-conversie.

        Returns:
            Dict met origin coördinaten en metadata voor conversie naar absolute Lambert2008
        """
        ref_x, ref_y = self.lambert_coords[self.reference_frame]
        ref_alt = self.gps_metadata['altitude'][self.reference_frame]
        ref_lat = self.gps_metadata['lat'][self.reference_frame]
        ref_lon = self.gps_metadata['lon'][self.reference_frame]

        return {
            'x_origin': float(ref_x),              # Lambert2008 X van referentie frame
            'y_origin': float(ref_y),              # Lambert2008 Y van referentie frame
            'altitude_origin': float(ref_alt),     # Altitude van referentie frame
            'lat_origin': float(ref_lat),          # WGS84 latitude van referentie frame
            'lon_origin': float(ref_lon),          # WGS84 longitude van referentie frame
            'reference_frame': self.reference_frame,  # Welk frame als origin
            'crs': 'EPSG:3812',                    # Lambert2008 Belgian CRS code
            'crs_name': 'Lambert 2008',
            'n_frames': self.n_frames
        }

    def get_statistics(self) -> Dict:
        """Bereken statistieken over GPS trajectory."""
        positions = self.compute_relative_positions()

        # Bereken afgelegde afstand
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = distances.sum()

        # Snelheid statistieken
        speeds = self.gps_metadata['speed_kmh']

        # Altitude variatie
        altitudes = self.gps_metadata['altitude']

        return {
            'n_frames': self.n_frames,
            'total_distance_m': total_distance,
            'avg_speed_kmh': speeds.mean(),
            'max_speed_kmh': speeds.max(),
            'altitude_range_m': (altitudes.min(), altitudes.max()),
            'altitude_variation_m': altitudes.max() - altitudes.min(),
            'bbox_x_m': (positions[:, 0].min(), positions[:, 0].max()),
            'bbox_y_m': (positions[:, 1].min(), positions[:, 1].max()),
            'bbox_z_m': (positions[:, 2].min(), positions[:, 2].max())
        }


def test_gps_loader():
    """Test functie voor GPS loader."""
    manifest_path = "/data/2025-09-05-wagen-gopro13/normalized_frames/metadata/normalized_manifest_camera_0.json"

    print("=" * 60)
    print("GPS Prior Loader Test")
    print("=" * 60)

    try:
        loader = GPSPriorLoader(manifest_path)

        # Statistieken
        stats = loader.get_statistics()
        print("\nGPS Trajectory Statistieken:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Genereer pose priors
        print("\n" + "-" * 60)
        poses = loader.get_pose_priors(use_gps_heading=True)
        print(f"\nPose priors shape: {poses.shape}")
        print(f"Pose priors dtype: {poses.dtype}")

        # Confidence weights
        weights = loader.get_gps_confidence_weights()
        print(f"\nConfidence weights shape: {weights.shape}")
        print(f"Confidence range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"Mean confidence: {weights.mean():.3f}")

        print("\n" + "=" * 60)
        print("Test succesvol!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test gefaald: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_gps_loader()