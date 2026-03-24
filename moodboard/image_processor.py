import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from typing import List, Tuple

class ColorPaletteExtractor:
    '''Extracts dominant color palettes from images using K-means clustering'''
    
    def __init__(self, n_colors: int = 5):
        self.n_colors = n_colors
        self.kmeans = KMeans(n_clusters=n_colors, random_state=42)
        
    def extract_colors(self, image_path: str) -> List[Tuple[int, int, int]]:
        '''
        Extract dominant colors from an image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of RGB color tuples representing the dominant colors
        '''
        # Load and preprocess image
        img = Image.open(image_path)
        img = img.convert('RGB')
        
        # Reshape image data for clustering
        pixels = np.float32(img).reshape(-1, 3)
        
        # Fit K-means clustering
        self.kmeans.fit(pixels)
        
        # Get cluster centers and convert to RGB integers
        colors = self.kmeans.cluster_centers_
        colors = np.uint8(colors)
        
        # Convert to list of RGB tuples
        color_palette = [tuple(color) for color in colors]
        
        return color_palette
    
    def get_color_percentages(self, image_path: str) -> List[float]:
        '''
        Get the percentage of pixels closest to each dominant color
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of percentages for each dominant color
        '''
        # Load and preprocess image
        img = Image.open(image_path)
        img = img.convert('RGB')
        pixels = np.float32(img).reshape(-1, 3)
        
        # Get cluster labels for all pixels
        labels = self.kmeans.predict(pixels)
        
        # Calculate percentage of pixels in each cluster
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100
        
        return percentages.tolist()
    
    def create_palette_image(self, colors: List[Tuple[int, int, int]], 
                           size: Tuple[int, int] = (500, 100)) -> Image:
        '''
        Create an image showing the extracted color palette
        
        Args:
            colors: List of RGB color tuples
            size: Tuple of (width, height) for output image
            
        Returns:
            PIL Image object showing the color palette
        '''
        palette_img = Image.new('RGB', size)
        draw = palette_img.load()
        
        # Calculate width of each color band
        band_width = size[0] // len(colors)
        
        # Draw color bands
        for i, color in enumerate(colors):
            for x in range(i * band_width, (i + 1) * band_width):
                for y in range(size[1]):
                    draw[x, y] = color
                    
        return palette_img

def analyze_image_colors(image_path: str) -> dict:
    '''
    Analyze an image and return color information
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary containing color palette and percentages
    '''
    extractor = ColorPaletteExtractor()
    colors = extractor.extract_colors(image_path)
    percentages = extractor.get_color_percentages(image_path)
    
    palette_img = extractor.create_palette_image(colors)
    palette_path = image_path.rsplit('.', 1)[0] + '_palette.png'
    palette_img.save(palette_path)
    
    return {
        'colors': colors,
        'percentages': percentages,
        'palette_image': palette_path
    }