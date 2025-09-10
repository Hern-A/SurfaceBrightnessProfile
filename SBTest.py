#!/usr/bin/env python3
"""
FITS Surface Brightness Profile Generator

This script reads a FITS file and creates a radial surface brightness profile plot.
It calculates the surface brightness as a function of distance from the center
(or a specified center) of the image.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAnnulus, aperture_photometry, CircularAperture
import argparse
import sys
from pathlib import Path


def load_fits_image(fits_file):
    """
    Load FITS file and return image data and header.
    
    Parameters:
    -----------
    fits_file : str
        Path to FITS file
        
    Returns:
    --------
    data : numpy.ndarray
        Image data
    header : astropy.io.fits.Header
        FITS header
    """
    try:
        with fits.open(fits_file) as hdul:
            # Get the primary HDU or first extension with data
            data = None
            header = None
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data.astype(float)
                    header = hdu.header
                    break
            
            if data is None:
                raise ValueError("No image data found in FITS file")
                
            # Handle 3D or 4D data cubes (take first slice)
            while data.ndim > 2:
                data = data[0]
                
            return data, header
            
    except Exception as e:
        print(f"Error loading FITS file: {e}")
        sys.exit(1)


def calculate_surface_brightness_profile(data, center=None, max_radius=None, 
                                       annulus_width=1.0, background_subtract=True):
    """
    Calculate radial surface brightness profile.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D image data
    center : tuple, optional
        (x, y) center coordinates. If None, uses image center.
    max_radius : float, optional
        Maximum radius for profile. If None, uses half the minimum dimension.
    annulus_width : float, optional
        Width of annuli in pixels (default: 1.0)
    background_subtract : bool, optional
        Whether to subtract background (default: True)
        
    Returns:
    --------
    radii : numpy.ndarray
        Radial distances
    surface_brightness : numpy.ndarray
        Surface brightness values
    brightness_err : numpy.ndarray
        Uncertainties in surface brightness
    """
    # Set default center
    if center is None:
        center = (data.shape[1] / 2, data.shape[0] / 2)
    
    # Set default max radius
    if max_radius is None:
        max_radius = min(data.shape) / 2 - 10
    
    # Estimate background if requested
    background = 0
    if background_subtract:
        # Use sigma-clipped statistics on outer regions
        y, x = np.ogrid[:data.shape[0], :data.shape[1]]
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        outer_mask = distance > max_radius * 0.8
        if np.any(outer_mask):
            background = sigma_clipped_stats(data[outer_mask])[1]  # median
        print(f"Estimated background: {background:.4f}")
    
    # Create annuli
    radii = []
    surface_brightness = []
    brightness_err = []
    
    current_radius = annulus_width / 2
    while current_radius < max_radius:
        inner_radius = current_radius - annulus_width / 2
        outer_radius = current_radius + annulus_width / 2
        
        # Skip if inner radius is negative
        if inner_radius < 0:
            inner_radius = 0
        
        # Create annulus
        annulus = CircularAnnulus(center, r_in=inner_radius, r_out=outer_radius)
        phot_table = aperture_photometry(data - background, annulus)
        
        # Calculate surface brightness (flux per unit area)
        annulus_area = annulus.area
        total_flux = phot_table['aperture_sum'][0]
        
        if annulus_area > 0 and not np.isnan(total_flux):
            sb = total_flux / annulus_area
            sb_err = np.sqrt(np.abs(total_flux)) / annulus_area  # Poisson error
            
            radii.append(current_radius)
            surface_brightness.append(sb)
            brightness_err.append(sb_err)
        
        current_radius += annulus_width
    
    return np.array(radii), np.array(surface_brightness), np.array(brightness_err)


def plot_surface_brightness_profile(radii, surface_brightness, brightness_err,
                                   fits_file, center, log_scale=True, 
                                   magnitude_scale=False, save_plot=None):
    """
    Create surface brightness profile plot.
    
    Parameters:
    -----------
    radii : numpy.ndarray
        Radial distances
    surface_brightness : numpy.ndarray
        Surface brightness values
    brightness_err : numpy.ndarray
        Uncertainties
    fits_file : str
        Original FITS filename for plot title
    center : tuple
        Center coordinates used
    log_scale : bool
        Whether to use log scale for y-axis
    magnitude_scale : bool
        Whether to convert to magnitude scale
    save_plot : str, optional
        Filename to save plot
    """
    plt.figure(figsize=(10, 8))
    
    # Convert to magnitudes if requested
    if magnitude_scale:
        # Convert to magnitude per arcsec^2 (assuming pixel scale of 1 arcsec/pixel)
        with np.errstate(divide='ignore', invalid='ignore'):
            mag_sb = -2.5 * np.log10(surface_brightness)
            mag_err = 2.5 * brightness_err / (surface_brightness * np.log(10))
        
        # Filter out invalid values
        valid_mask = np.isfinite(mag_sb) & np.isfinite(mag_err)
        radii_plot = radii[valid_mask]
        sb_plot = mag_sb[valid_mask]
        err_plot = mag_err[valid_mask]
        
        plt.errorbar(radii_plot, sb_plot, yerr=err_plot, 
                    fmt='o-', markersize=4, capsize=3, alpha=0.7)
        plt.ylabel('Surface Brightness (mag/arcsec²)', fontsize=12)
        plt.gca().invert_yaxis()  # Invert y-axis for magnitudes
        
    else:
        # Filter out non-positive values for log scale
        if log_scale:
            valid_mask = (surface_brightness > 0) & np.isfinite(surface_brightness)
            radii_plot = radii[valid_mask]
            sb_plot = surface_brightness[valid_mask]
            err_plot = brightness_err[valid_mask]
        else:
            valid_mask = np.isfinite(surface_brightness)
            radii_plot = radii[valid_mask]
            sb_plot = surface_brightness[valid_mask]
            err_plot = brightness_err[valid_mask]
        
        plt.errorbar(radii_plot, sb_plot, yerr=err_plot, 
                    fmt='o-', markersize=4, capsize=3, alpha=0.7)
        plt.ylabel('Surface Brightness (counts/pixel)', fontsize=12)
        
        if log_scale:
            plt.yscale('log')
    
    plt.xlabel('Radius (pixels)', fontsize=12)
    plt.title(f'Surface Brightness Profile\n{Path(fits_file).name}\n'
              f'Center: ({center[0]:.1f}, {center[1]:.1f})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {save_plot}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Create surface brightness profile from FITS file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fits_profile.py image.fits
  python fits_profile.py image.fits --center 256 256 --max-radius 100
  python fits_profile.py image.fits --magnitude --save profile.png
        """
    )
    
    parser.add_argument('fits_file', help='Input FITS file')
    parser.add_argument('--center', nargs=2, type=float, metavar=('X', 'Y'),
                       help='Center coordinates (default: image center)')
    parser.add_argument('--max-radius', type=float,
                       help='Maximum radius in pixels (default: auto)')
    parser.add_argument('--annulus-width', type=float, default=1.0,
                       help='Width of annuli in pixels (default: 1.0)')
    parser.add_argument('--no-background', action='store_true',
                       help='Skip background subtraction')
    parser.add_argument('--linear', action='store_true',
                       help='Use linear scale instead of log scale')
    parser.add_argument('--magnitude', action='store_true',
                       help='Convert to magnitude scale')
    parser.add_argument('--save', type=str,
                       help='Save plot to file (e.g., profile.png)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.fits_file).exists():
        print(f"Error: File '{args.fits_file}' not found")
        sys.exit(1)
    
    # Load FITS file
    print(f"Loading FITS file: {args.fits_file}")
    data, header = load_fits_image(args.fits_file)
    print(f"Image shape: {data.shape}")
    
    # Set center
    center = tuple(args.center) if args.center else None
    
    # Calculate surface brightness profile
    print("Calculating surface brightness profile...")
    radii, sb, sb_err = calculate_surface_brightness_profile(
        data, 
        center=center,
        max_radius=args.max_radius,
        annulus_width=args.annulus_width,
        background_subtract=not args.no_background
    )
    
    # Determine actual center used
    if center is None:
        center = (data.shape[1] / 2, data.shape[0] / 2)
    
    print(f"Profile calculated with {len(radii)} data points")
    
    # Create plot
    plot_surface_brightness_profile(
        radii, sb, sb_err, args.fits_file, center,
        log_scale=not args.linear,
        magnitude_scale=args.magnitude,
        save_plot=args.save
    )


if __name__ == "__main__":
    main()