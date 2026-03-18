# Visualization System

This module provides a class-based visualization system for JSON data with improved organization and modern styling.

## Structure

### Main Classes

1. **`Visualization`** - Main class for creating visualizations
   - `__init__(json_path, output_html, plot_rooms=True)`
   - `load_data()` - Load JSON data from file
   - `visualize()` - Generate the complete visualization

2. **`HTMLGenerator`** - Handles HTML generation
   - `generate_toc_page()` - Generate table of contents
   - `generate_sample_page()` - Generate individual sample pages
   - `generate_html()` - Generate complete HTML file

3. **`VisualizationHelper`** - Utility class for data processing
   - `extract()` - Extract content from regex patterns
   - `split_into_turns()` - Split messages into conversation turns
   - `dict_to_html()` - Convert dictionaries to HTML
   - `squash_exp_logs()` - Merge exploration logs

4. **`RoomPlotter`** - Handles room visualization
   - `plot_initial_room()` - Plot initial room state

### Files

- `visualization.py` - Main visualization classes and logic
- `html_templates.py` - HTML templates, CSS styles, and JavaScript
- `example_usage.py` - Example usage demonstration

## Usage

### Basic Usage

```python
from ragen.utilities.visualization import Visualization

# Create visualization
viz = Visualization(
    json_path="path/to/data.json",
    output_html="output/dashboard.html",
    plot_rooms=True
)

# Generate visualization
output_path = viz.visualize()
```

### Legacy Usage (Backward Compatibility)

```python
from ragen.utilities.visualization import visualize_json

output_path = visualize_json(
    json_path="path/to/data.json",
    output_html="output/dashboard.html",
    plot_rooms=True
)
```

## Features

### Improved Organization
- **Class-based structure** for better code organization
- **Separated concerns** with dedicated classes for different functionalities
- **Modular design** with HTML templates in separate file

### Enhanced Styling
- **Modern gradient backgrounds** with smooth transitions
- **Improved typography** with better font choices
- **Responsive design** that works on mobile devices
- **Hover effects** and animations for better user experience
- **Better color coding** for different message types

### Better User Experience
- **Smooth page transitions** with fade-in animations
- **Enhanced navigation** with keyboard shortcuts (Home/End keys)
- **Improved visual hierarchy** with better spacing and typography
- **Accessibility improvements** with better contrast and focus states

## Keyboard Navigation

- **Arrow Keys** / **Page Up/Down** - Navigate between pages
- **Home** - Go to first page
- **End** - Go to last page
- **Enter** - Go to specific page number

## Styling Improvements

- **Gradient backgrounds** for visual appeal
- **Card-based layout** with shadows and rounded corners
- **Color-coded message blocks** (User: Blue, Think: Orange, Answer: Green)
- **Smooth animations** and hover effects
- **Responsive design** for mobile devices
- **Better typography** with improved readability

## Backward Compatibility

The original `visualize_json()` function is still available for backward compatibility, so existing code will continue to work without changes.