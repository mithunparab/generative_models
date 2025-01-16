
import torch
import matplotlib.pyplot as plt
import numpy as np


# Function to generate points for a Greek PSI (Ψ) symbol
def generate_psi(num_points_per_segment=500, noise_std=0.02):
    """
    Generate a Greek PSI (Ψ) symbol in 2D with added Gaussian noise.

    Args:
        num_points_per_segment: Number of points per segment of the Ψ.
        noise_std: Standard deviation of Gaussian noise added to the points.

    Returns:
        torch.Tensor: 2D tensor containing points in the Ψ pattern.
    """
    # Define the segments of the Ψ symbol
    segments = [
        # Left vertical line
        torch.stack([
            torch.full((num_points_per_segment,), -2.0),  # X-coordinates
            torch.linspace(-2.0, 2.0, num_points_per_segment)  # Y-coordinates
        ], dim=1),
        
        # Right vertical line
        torch.stack([
            torch.full((num_points_per_segment,), 2.0),  # X-coordinates
            torch.linspace(-2.0, 2.0, num_points_per_segment)  # Y-coordinates
        ], dim=1),
        
        # Middle vertical line
        torch.stack([
            torch.full((num_points_per_segment,), 0.0),  # X-coordinates
            torch.linspace(-5.0, 2.0, num_points_per_segment)  # Y-coordinates
        ], dim=1),
        
        # Bottom curved segment (semi-circle connecting the three vertical lines)
        torch.stack([
            torch.cos(torch.linspace(-np.pi, 0, num_points_per_segment)) * 2.0,  # X-coordinates (semi-circle)
            torch.sin(torch.linspace(-np.pi, 0, num_points_per_segment)) - 2.0  # Y-coordinates
        ], dim=1)
    ]
    
    # Combine all segments
    points = torch.cat(segments, dim=0)
    
    # Add Gaussian noise to make it a distribution
    noise = torch.randn_like(points) * noise_std
    noisy_points = points + noise
    
    return noisy_points

# Function to generate points for a spiral pattern
def generate_spiral(num_points_per_segment=3000, noise_std=0.02, num_turns=3):
    """
    Generate a 2D spiral pattern with added Gaussian noise.

    Args:
        num_points_per_segment: Total number of points in the spiral.
        noise_std: Standard deviation of Gaussian noise added to the points.
        num_turns: Number of turns in the spiral.

    Returns:
        torch.Tensor: 2D tensor containing points in the spiral pattern.
    """
    theta = torch.linspace(0, 2 * np.pi * num_turns, num_points_per_segment)  # Angle
    r = torch.linspace(0, 2.6, num_points_per_segment)  # Radius
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    # Combine into a 2D tensor
    points = torch.stack([x, y], dim=1)

    # Add Gaussian noise
    noise = torch.randn_like(points) * noise_std
    noisy_points = points + noise

    return noisy_points

# Function to generate points for a swastika pattern
def generate_swastika(num_points_per_segment=500, noise_std=0.02):
    """
    Generate a swastika pattern in 2D with added Gaussian noise.
    Args:
        num_points_per_segment: Number of points per line segment.
        noise_std: Standard deviation of Gaussian noise added to the points.
    Returns:
        torch.Tensor: 2D tensor containing points in the swastika pattern.
    """
    # Define segments of the swastika (centered at origin)
    segments = [
        # Horizontal top-right and bottom-left bars
        torch.stack([torch.linspace(0.0, 2.6, num_points_per_segment), torch.full((num_points_per_segment,), 2.6)], dim=1),
        torch.stack([torch.linspace(0.0, -2.6, num_points_per_segment), torch.full((num_points_per_segment,), -2.6)], dim=1),
        
        # Vertical top-left and bottom-right bars
        torch.stack([torch.full((num_points_per_segment,), -2.6), torch.linspace(0.0, 2.6, num_points_per_segment)], dim=1),
        torch.stack([torch.full((num_points_per_segment,), 2.6), torch.linspace(0.0, -2.6, num_points_per_segment)], dim=1),
        
        # Horizontal crossing bar
        torch.stack([torch.linspace(-2.6, 2.6, num_points_per_segment), torch.full((num_points_per_segment,), 0.0)], dim=1),
        
        # Vertical crossing bar
        torch.stack([torch.full((num_points_per_segment,), 0.0), torch.linspace(-2.6, 2.6, num_points_per_segment)], dim=1),
    ]
    
    # Combine all segments
    points = torch.cat(segments, dim=0)
    
    # Add Gaussian noise to make it a distribution
    noise = torch.randn_like(points) * noise_std
    noisy_points = points + noise
    
    return noisy_points

# Function to generate points for a swastika pattern
def generate_swastika(num_points_per_segment=500, noise_std=0.02):
    """
    Generate a swastika pattern in 2D with added Gaussian noise.
    Args:
        num_points_per_segment: Number of points per line segment.
        noise_std: Standard deviation of Gaussian noise added to the points.
    Returns:
        torch.Tensor: 2D tensor containing points in the swastika pattern.
    """
    # Define segments of the swastika (centered at origin)
    segments = [
        # Horizontal top-right and bottom-left bars
        torch.stack([torch.linspace(0.0, 2.6, num_points_per_segment), torch.full((num_points_per_segment,), 2.6)], dim=1),
        torch.stack([torch.linspace(0.0, -2.6, num_points_per_segment), torch.full((num_points_per_segment,), -2.6)], dim=1),
        
        # Vertical top-left and bottom-right bars
        torch.stack([torch.full((num_points_per_segment,), -2.6), torch.linspace(0.0, 2.6, num_points_per_segment)], dim=1),
        torch.stack([torch.full((num_points_per_segment,), 2.6), torch.linspace(0.0, -2.6, num_points_per_segment)], dim=1),
        
        # Horizontal crossing bar
        torch.stack([torch.linspace(-2.6, 2.6, num_points_per_segment), torch.full((num_points_per_segment,), 0.0)], dim=1),
        
        # Vertical crossing bar
        torch.stack([torch.full((num_points_per_segment,), 0.0), torch.linspace(-2.6, 2.6, num_points_per_segment)], dim=1),
    ]
    
    # Combine all segments
    points = torch.cat(segments, dim=0)
    
    # Add Gaussian noise to make it a distribution
    noise = torch.randn_like(points) * noise_std
    noisy_points = points + noise
    
    return noisy_points
