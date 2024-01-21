import numpy as np

def normalize_angles(angles):
    """
    Args:
        angles (float or numpy.array): angles in radian (= [a1, a2, ...], shape of [n,])
    Returns:
        numpy.array or float: angles in radians normalized b/w/ -pi and +pi (same shape w/ angles)
    """
    angles = (angles + np.pi) % (2 * np.pi ) - np.pi
    return angles
    
def rotation_matrix(theta, axis):
    """Generalized rotation matrix function"""
    c = np.cos(theta)
    s = np.sin(theta)

    # Define dictionaries to map axes to rotation matrices
    rotation_matrices = {
        'x': np.array([
            [1, 0, 0],
            [0, c, s],
            [0, -s, c]
        ]),
        'y': np.array([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ]),
        'z': np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])
    }

    # Check if the specified axis is valid, otherwise raise an error
    if axis not in rotation_matrices:
        raise ValueError("Invalid axis. Please use 'x', 'y', or 'z'.")

    return rotation_matrices[axis]


def lla_to_ecef(points_lla):
    """
    Convert coordinates from [longitude(deg), latitude(deg), altitude(m)]
    into Earth-Centered-Earth-Fixed (ECEF) frame coordinates [x, y, z].
    """
    longitude_deg = points_lla[0]  # Longitude in degrees
    latitude_deg = points_lla[1]  # Latitude in degrees
    altitude = points_lla[2]  # Altitude in meters

    # Convert degrees to radians
    longitude_rad = np.radians(longitude_deg)
    latitude_rad = np.radians(latitude_deg)

    # Constant parameters defined in [1]
    _a = 6378137.0  # Semi-major axis of the Earth (meters)
    _e = 8.1819190842622e-2  # Eccentricity of Earth

    # Calculate N (radius of curvature in the prime vertical)
    N = _a / np.sqrt(1. - (_e * np.sin(latitude_rad)) ** 2.)  # Radius of curvature

    # Calculate ECEF coordinates
    x = (N + altitude) * np.cos(latitude_rad) * np.cos(longitude_rad)
    y = (N + altitude) * np.cos(latitude_rad) * np.sin(longitude_rad)
    z = (N * (1. - _e ** 2.) + altitude) * np.sin(latitude_rad)

    # Stack x, y, z to form the ECEF coordinates matrix
    points_ecef = np.stack([x, y, z], axis=0)  # [3, N]
    
    return points_ecef

def ecef_to_enu(points_ecef, ref_lla):
    """
    Convert coordinates from Earth-Centered-Earth-Fixed (ECEF) frame
    into a local East-North-Up (ENU) frame.
    """
    # Reference point in latitude, longitude, altitude
    ref_longitude_deg = ref_lla[0]  # Reference Longitude in degrees
    ref_latitude_deg = ref_lla[1]   # Reference Latitude in degrees
    ref_altitude = ref_lla[2]       # Reference Altitude in meters

    # Convert reference point degrees to radians
    ref_longitude_rad = np.radians(ref_longitude_deg)
    ref_latitude_rad = np.radians(ref_latitude_deg)

    # Convert reference point to ECEF
    ref_ecef = lla_to_ecef(ref_lla)  # ECEF coordinates of the reference point [3,]

    # Calculate relative ECEF coordinates
    relative = points_ecef - ref_ecef[:, np.newaxis]  # Difference between points_ecef and ref_ecef [3, N]

    # Calculate rotation matrix for conversion to ENU frame
    R_z = rotation_matrix(np.pi / 2.0, 'z')
    R_y = rotation_matrix(np.pi / 2.0 - ref_latitude_rad, 'y')
    R_x = rotation_matrix(ref_longitude_rad, 'z')
    R = R_z @ R_y @ R_x  # Combined rotation matrix [3, 3]

    # Convert ECEF coordinates to ENU coordinates using rotation matrix
    points_enu = R @ relative  # ENU coordinates [3, N]

    return points_enu

def lla_to_enu(points_lla, ref_lla):
    """
    Convert coordinates from [longitude(deg), latitude(deg), altitude(m)]
    into a local East-North-Up (ENU) frame [x, y, z].
    """
    # Convert points from LLA to ECEF coordinates
    points_ecef = lla_to_ecef(points_lla)

    # Convert ECEF coordinates to ENU coordinates using reference LLA
    points_enu = ecef_to_enu(points_ecef, ref_lla)

    return points_enu

