#!/usr/bin/env python3
"""Converteer MegaSaM output met relatieve coördinaten naar absolute Lambert2008.

Dit script leest een MegaSaM output npz bestand met:
- Relatieve camera posities (t.o.v. eerste frame)
- GPS Lambert2008 offset metadata

En schrijft een nieuw npz bestand met:
- Absolute camera posities in Lambert2008 coördinaten
- Behoud van alle andere data (images, depths, intrinsic)

Gebruik:
    python convert_to_lambert2008.py input.npz output_lambert2008.npz

Voorbeeld:
    python convert_to_lambert2008.py \
        outputs/awv_moge2_gps_droid.npz \
        outputs/awv_moge2_gps_droid_lambert2008.npz
"""

import numpy as np
import argparse
from pathlib import Path


def convert_poses_to_lambert2008(cam_c2w, gps_offset):
    """Converteer relatieve camera poses naar absolute Lambert2008 coördinaten.

    Args:
        cam_c2w: (N, 4, 4) camera-to-world matrices met relatieve posities
        gps_offset: Dict met Lambert2008 offset metadata

    Returns:
        (N, 4, 4) camera-to-world matrices met absolute Lambert2008 posities
    """
    # Haal offset metadata op
    x_origin = gps_offset['x_origin']
    y_origin = gps_offset['y_origin']
    altitude_origin = gps_offset['altitude_origin']

    print(f"\nLambert2008 offset:")
    print(f"  X origin: {x_origin:.2f} m")
    print(f"  Y origin: {y_origin:.2f} m")
    print(f"  Altitude origin: {altitude_origin:.2f} m")
    print(f"  CRS: {gps_offset.get('crs', 'EPSG:3812')}")

    # Copy poses
    cam_c2w_absolute = cam_c2w.copy()

    # Relatieve DROID coördinaten:
    #   X = dx (relatief t.o.v. start)
    #   Y = dz (altitude verschil)
    #   Z = -dy (noord is negatieve Z)
    #
    # Terug naar absolute Lambert2008:
    #   Lambert X = X_origin + dx
    #   Lambert Y = Y_origin - dz  (want Z = -dy, dus dy = -Z)
    #   Altitude = Altitude_origin + dy

    for i in range(len(cam_c2w_absolute)):
        # Relatieve posities uit pose matrix
        dx = cam_c2w[i, 0, 3]  # DROID X
        dy = cam_c2w[i, 1, 3]  # DROID Y (altitude verschil)
        dz = cam_c2w[i, 2, 3]  # DROID Z (negatieve noord)

        # Converteer naar absolute Lambert2008
        lambert_x = x_origin + dx
        lambert_y = y_origin - dz  # Z = -dy, dus Lambert_Y = Y_origin - dZ
        altitude = altitude_origin + dy

        # Schrijf terug naar pose matrix
        cam_c2w_absolute[i, 0, 3] = lambert_x
        cam_c2w_absolute[i, 1, 3] = altitude
        cam_c2w_absolute[i, 2, 3] = lambert_y

    print(f"\nCoördinaten geconverteerd:")
    print(f"  Relatief X bereik: [{cam_c2w[:, 0, 3].min():.2f}, {cam_c2w[:, 0, 3].max():.2f}] m")
    print(f"  Relatief Y bereik: [{cam_c2w[:, 1, 3].min():.2f}, {cam_c2w[:, 1, 3].max():.2f}] m")
    print(f"  Relatief Z bereik: [{cam_c2w[:, 2, 3].min():.2f}, {cam_c2w[:, 2, 3].max():.2f}] m")
    print(f"")
    print(f"  Absoluut Lambert X: [{cam_c2w_absolute[:, 0, 3].min():.2f}, {cam_c2w_absolute[:, 0, 3].max():.2f}] m")
    print(f"  Absoluut Altitude:  [{cam_c2w_absolute[:, 1, 3].min():.2f}, {cam_c2w_absolute[:, 1, 3].max():.2f}] m")
    print(f"  Absoluut Lambert Y: [{cam_c2w_absolute[:, 2, 3].min():.2f}, {cam_c2w_absolute[:, 2, 3].max():.2f}] m")

    return cam_c2w_absolute


def convert_npz_to_lambert2008(input_path, output_path):
    """Converteer MegaSaM npz naar absolute Lambert2008 coördinaten.

    Args:
        input_path: Path naar input npz bestand met relatieve coördinaten
        output_path: Path voor output npz met absolute Lambert2008
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Check input bestand
    if not input_path.exists():
        raise FileNotFoundError(f"Input bestand niet gevonden: {input_path}")

    print("="*60)
    print("MegaSaM → Lambert2008 Conversie")
    print("="*60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Laad input data
    print(f"\nLaden input data...")
    data = np.load(input_path, allow_pickle=True)

    # Check voor GPS offset metadata
    if 'gps_lambert_offset' not in data:
        raise ValueError(
            f"Geen GPS Lambert2008 offset metadata gevonden in {input_path}.\n"
            "Dit bestand is waarschijnlijk niet gegenereerd met GPS priors.\n"
            "Run MegaSaM met --gps_manifest parameter om GPS metadata op te nemen."
        )

    # Haal GPS offset op (dict is opgeslagen als numpy object)
    gps_offset = data['gps_lambert_offset'].item()

    print(f"✅ GPS metadata gevonden")
    print(f"   Frames: {gps_offset['n_frames']}")
    print(f"   Reference frame: {gps_offset['reference_frame']}")

    # Check camera poses
    if 'cam_c2w' not in data:
        raise ValueError(f"Geen camera poses (cam_c2w) gevonden in {input_path}")

    cam_c2w = data['cam_c2w']
    print(f"✅ Camera poses gevonden: {cam_c2w.shape}")

    # Converteer poses
    print(f"\nConverteren naar Lambert2008...")
    cam_c2w_lambert = convert_poses_to_lambert2008(cam_c2w, gps_offset)

    # Bereid output data voor (copy alle velden, overschrijf cam_c2w)
    output_data = {key: data[key] for key in data.files}
    output_data['cam_c2w'] = cam_c2w_lambert
    output_data['coordinate_system'] = 'Lambert2008_absolute'  # Markeer als absolute
    output_data['crs'] = gps_offset.get('crs', 'EPSG:3812')

    # Schrijf output
    print(f"\nSchrijven output...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **output_data)

    print(f"✅ Output geschreven naar: {output_path}")
    print(f"")
    print("="*60)
    print("Conversie compleet!")
    print("="*60)
    print(f"\nDe output bevat:")
    print(f"  - Absolute Lambert2008 X coördinaten (oost, meters)")
    print(f"  - Absolute altitudes (meters boven WGS84 ellipsoid)")
    print(f"  - Absolute Lambert2008 Y coördinaten (noord, meters)")
    print(f"  - CRS: {gps_offset.get('crs', 'EPSG:3812')} ({gps_offset.get('crs_name', 'Lambert 2008')})")
    print(f"\nGeschikt voor:")
    print(f"  - GIS export (QGIS, ArcGIS)")
    print(f"  - Overlay op kaarten")
    print(f"  - Georeferenced point clouds")


def main():
    parser = argparse.ArgumentParser(
        description="Converteer MegaSaM output naar absolute Lambert2008 coördinaten",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Voorbeeld gebruik:

  # Converteer sensor depths output
  python convert_to_lambert2008.py \\
      outputs/awv_moge2_gps_droid.npz \\
      outputs/awv_moge2_gps_droid_lambert2008.npz

  # Converteer optimized depths output
  python convert_to_lambert2008.py \\
      outputs/awv_moge2_gps_estimated_droid.npz \\
      outputs/awv_moge2_gps_estimated_droid_lambert2008.npz

LET OP: Input bestand moet gegenereerd zijn met GPS priors!
        (run MegaSaM met --gps_manifest parameter)
        """
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input npz bestand met relatieve coördinaten"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output npz bestand voor absolute Lambert2008 coördinaten"
    )

    args = parser.parse_args()

    try:
        convert_npz_to_lambert2008(args.input, args.output)
    except Exception as e:
        print(f"\n❌ Fout: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())