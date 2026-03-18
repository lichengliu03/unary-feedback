# html_templates.py
# HTML templates, CSS styles, and JavaScript code for visualization

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SpatialGym Dashboard</title>
    <style>
{css_styles}
    </style>
</head>
<body>
    <div id='nav'>
        <button onclick="prevPage({total_pages})">Prev</button>
        <button onclick="nextPage({total_pages})">Next</button>
        <input id="goto" type="number" min="1" max="{total_pages}" placeholder="page" onkeydown="if(event.key==='Enter') gotoPage({total_pages});">
        <button onclick="gotoPage({total_pages})">Go</button>
        <span id='counter'></span>
        <div id='eval-task-selector' style='display:none; margin-left: 20px;'>
            <label for='task-select'>Evaluation Task:</label>
            <select id='task-select' onchange="switchEvaluationTask()">
            </select>
        </div>
    </div>
    
    <h1>Model: {model_name}</h1>
    
    <script>
{javascript_code}
    </script>
"""

CSS_STYLES = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

#nav {
    position: fixed;
    top: 0;
    width: 100%;
    background: rgba(51, 51, 51, 0.95);
    backdrop-filter: blur(10px);
    color: #fff;
    padding: 15px 0;
    text-align: center;
    z-index: 999;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

#nav button {
    margin: 0 8px;
    padding: 8px 16px;
    border: none;
    background: #4CAF50;
    color: #fff;
    cursor: pointer;
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.3s ease;
}

#nav button:hover {
    background: #45a049;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

#nav input {
    width: 80px;
    padding: 8px;
    border-radius: 6px;
    border: 1px solid #ddd;
    margin-left: 12px;
    text-align: center;
    font-size: 14px;
}

#counter {
    margin-left: 12px;
    font-size: 0.9em;
    opacity: 0.9;
    font-weight: 500;
}

.sample-page {
    display: none;
    padding: 80px 16px 16px 16px;
    max-width: 1400px;
    margin: auto;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    margin-top: 10px;
    margin-bottom: 10px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}

.sample-page.active {
    display: block;
    animation: fadeIn 0.5s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Summary sections */
.summary-section {
    margin: 30px 0;
    padding: 20px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 12px;
    border: 1px solid #dee2e6;
}

.summary-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-left: 4px solid;
}

.summary-card.exploration {
    border-left-color: #17a2b8;
}

.summary-card.evaluation {
    border-left-color: #28a745;
}

.summary-card h4 {
    margin: 0 0 15px 0;
    color: #495057;
    font-size: 1.2em;
    font-weight: 600;
}

.config-summaries {
    margin: 30px 0;
}

.config-summary {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border: 1px solid #dee2e6;
}

.config-summary h4 {
    margin: 0 0 15px 0;
    color: #495057;
    font-size: 1.1em;
    font-weight: 600;
}

.config-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 15px;
}

.stat-item {
    background: #e9ecef;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.9em;
    color: #495057;
    font-weight: 500;
}

.group-metrics {
    margin: 10px 0;
    padding: 10px;
    background: #f8f9fa;
    border-radius: 6px;
    border-left: 3px solid #6c757d;
}

.group-metrics strong {
    color: #495057;
    font-weight: 600;
    display: block;
    margin-bottom: 8px;
}

/* Improved dictionary styling */
.dict-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px;
    margin-top: 10px;
}

.dict-item {
    background: #f8f9fa;
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #dee2e6;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9em;
}

.dict-key {
    font-weight: 600;
    color: #495057;
    margin-right: 8px;
}

.dict-value {
    color: #6c757d;
    font-weight: 500;
}

.dict-value.number {
    color: #007bff;
    font-weight: 600;
}

.dict-value.true {
    color: #28a745;
    font-weight: 600;
}

.dict-value.false {
    color: #dc3545;
    font-weight: 600;
}

.dict-value.string {
    color: #6f42c1;
    font-weight: 600;
}

.dict-item.nested {
    flex-direction: column;
    align-items: flex-start;
}

.dict-value.nested-dict {
    width: 100%;
    max-width: 100%;
    margin-top: 8px;
    padding: 8px;
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 4px;
    overflow: hidden;
    word-wrap: break-word;
    box-sizing: border-box;
}

/* Ensure nested dict containers are properly constrained */
.dict-value.nested-dict .dict-container {
    width: 100%;
    max-width: 100%;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    overflow: hidden;
}

/* Ensure deeply nested dict items don't overflow */
.dict-value.nested-dict .dict-item {
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    overflow: hidden;
    word-wrap: break-word;
    margin: 0;
}

/* Handle deeply nested dict values */
.dict-value.nested-dict .dict-value.nested-dict {
    width: calc(100% - 16px);
    max-width: calc(100% - 16px);
    margin: 4px 0 0 0;
    padding: 6px;
}

/* Fix nested dict overlapping in group-metrics sections (main page) */
.group-metrics .dict-container {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    overflow: hidden;
}

.group-metrics .dict-item {
    box-sizing: border-box;
    overflow: hidden;
    word-wrap: break-word;
    margin: 0;
}

.group-metrics .dict-value.nested-dict {
    margin-top: 8px;
    padding: 8px;
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 4px;
    overflow: hidden;
    word-wrap: break-word;
    box-sizing: border-box;
}

.group-metrics .dict-value.nested-dict .dict-container {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    overflow: hidden;
}

.group-metrics .dict-value.nested-dict .dict-item {
    box-sizing: border-box;
    overflow: hidden;
    word-wrap: break-word;
    margin: 0;
}

.group-metrics .dict-value.nested-dict .dict-value.nested-dict {
    margin: 4px 0 0 0;
    padding: 6px;
}

.empty-dict {
    color: #6c757d;
    font-style: italic;
    padding: 8px 12px;
    background: #f8f9fa;
    border-radius: 6px;
    border: 1px dashed #dee2e6;
}

.turn {
    background: #fff;
    border: 1px solid #e1e5e9;
    border-radius: 12px;
    margin: 20px 0;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}

.turn:hover {
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}

.turn h3 {
    margin: 0 0 15px 0;
    font-size: 18px;
    color: #2c3e50;
    font-weight: 600;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px;
}

.turn-split {
    background: #fff;
    border: 1px solid #e1e5e9;
    border-radius: 12px;
    margin: 20px 0;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
    display: flex;
    flex-direction: column;
    min-height: 350px;
}

.turn-split .turn-content {
    display: flex;
    flex-direction: row;
    gap: 20px;
    margin-top: 15px;
}

.turn-split:hover {
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}

.turn-split h3 {
    margin: 20px 0 15px 0;
    padding-bottom: 8px;
    font-size: 18px;
    color: #2c3e50;
    font-weight: 600;
    border-bottom: 2px solid #3498db;
}

.turn-left {
    flex: 2;
    margin-right: 16px;
}

.turn-right {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 8px;
}

.room-plot {
    max-width: 100%;
    max-height: 400px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.message-image {
    max-width: 100%;
    max-height: 300px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.block {
    padding: 12px;
    border-radius: 8px;
    margin: 10px 0;
    font-size: 14px;
    line-height: 1.5;
    border-left: 4px solid;
    position: relative;
    overflow: hidden;
}

.block::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    opacity: 0.05;
    background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.3) 50%, transparent 70%);
    transform: translateX(-100%);
    transition: transform 0.6s ease;
}

.block:hover::before {
    transform: translateX(100%);
}

.block.user {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left-color: #2196F3;
    color: #1565c0;
}

.block.user.expandable {
    cursor: pointer;
    transition: all 0.2s ease;
}


.expand-hint {
    font-size: 0.8em;
    color: #1976d2;
    font-weight: normal;
}


.block.think {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border-left-color: #ff9800;
    color: #e65100;
    font-style: italic;
}

.block.think.expandable {
    cursor: pointer;
    transition: all 0.2s ease;
}

.block.answer {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
    border-left-color: #4caf50;
    color: #2e7d32;
}

.block.evaluation {
    background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
    border-left-color: #9c27b0;
    color: #6a1b9a;
}

.block.cogmap {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left-color: #2196f3;
    color: #0d47a1;
}

.block.cogmap-response {
    background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    border-left-color: #ffa000;
    color: #e65100;
}

.block.cogmap-response.expandable {
    cursor: pointer;
    transition: all 0.2s ease;
}


/* Cognitive map layout */
.cogmap-compare {
    display: grid;
    grid-template-columns: 75% 25%;
    gap: 0;
    margin-top: 10px;
}

/* Handle single column case when right side is empty */
.cogmap-compare:has(.cogmap-box.side:only-child) {
    grid-template-columns: 1fr;
}

/* Alternative for browsers that don't support :has() */
.cogmap-compare.single-column {
    grid-template-columns: 1fr;
}
.cogmap-box {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 0;
    box-shadow: none;
    overflow: hidden;
    margin: 0;
    padding: 0;
}
.cogmap-box.side {
    display: flex;
    flex-direction: column;
    min-height: 100%;
    border-left: none;
}

.cogmap-box.side:first-child {
    border-right: none;
}

.cogmap-box.side:last-child {
    border-left: 1px solid #dee2e6;
}

/* Add minimal padding to prevent text clipping in right panel */
.cogmap-box.side .dict-container {
    padding: 0 6px;
    margin: 0;
}

.cogmap-box-title {
    background: linear-gradient(135deg, #e9f3ff 0%, #d7ebff 100%);
    color: #0d47a1;
    font-weight: 600;
    padding: 10px 12px;
    border-bottom: 1px solid #d0e2ff;
}

.cogmap-box-title.expandable {
    cursor: pointer;
    transition: all 0.2s ease;
}

.cogmap-box-title.expandable:hover {
    background: linear-gradient(135deg, #d7ebff 0%, #c5dbff 100%);
}
/* Ground Truth framed panel */
.groundtruth-box.framed {
    border: 1px solid #bcd4f6;
    background: linear-gradient(135deg, #eff6ff 0%, #e7f0ff 100%);
    box-shadow: 0 4px 14px rgba(13, 71, 161, 0.10);
}
/* Section chunks inside GT */
.cogmap-gt-section {
    padding: 10px 12px;
    border-bottom: 1px dashed #dbe4f3;
    margin: 0;
}
.cogmap-gt-section:last-child {
    border-bottom: none;
}
.cogmap-section-title {
    font-weight: 600;
    color: #1b4b91;
    margin-bottom: 6px;
}
/* Content body with readable mono + soft background */
.cogmap-box-body {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 13px;
    line-height: 1.5;
    background: #f8fbff;
    border: none;
    border-radius: 0;
    padding: 8px 10px;
    color: #0d47a1;
    margin: 0;
}
/* Room chunk blocks inside the Rooms section */
.room-chunk {
    margin-bottom: 8px;
    padding: 6px 8px;
    background: #f1f7ff;
    border: 1px solid #e1ecfb;
    border-radius: 6px;
}

.metrics {
    margin-top: 12px;
    font-size: 13px;
    color: #555;
    background: #f8f9fa;
    padding: 10px;
    border-radius: 6px;
    border-left: 3px solid #6c757d;
}

.metrics div {
    margin: 4px 0;
    padding: 2px 0;
}

.metrics strong {
    color: #495057;
    font-weight: 600;
}

img.room {
    max-width: 300px;
    height: auto;
    border: 2px solid #e1e5e9;
    border-radius: 8px;
    margin: 15px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

h1 {
    margin-top: 60px;
    color: #fff;
    text-align: center;
    font-size: 2.5em;
    font-weight: 300;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    margin-bottom: 30px;
}

h2 {
    color: #2c3e50;
    margin-top: 30px;
    font-size: 1.8em;
    font-weight: 500;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}

h3 {
    color: #34495e;
    margin: 25px 0 0 0;
    padding: 0 0 8px 0;
    font-size: 1.4em;
    font-weight: 500;
    border-bottom: 1px solid #bdc3c7;
    text-align: left;
}

h4 {
    color: #495057;
    margin: 20px 0 0 0;
    padding: 0 0 6px 0;
    font-size: 1.2em;
    font-weight: 500;
    border-bottom: 1px solid #dee2e6;
    text-align: left;
}

h5 {
    color: #6c757d;
    margin-top: 15px;
    font-size: 1.1em;
    font-weight: 500;
}

h6 {
    color: #868e96;
    margin-top: 10px;
    font-size: 1.0em;
    font-weight: 500;
}

ul {
    list-style: none;
    padding-left: 0;
}

ul ul {
    padding-left: 20px;
}

li {
    margin: 8px 0;
    padding: 8px 12px;
    border-radius: 6px;
    transition: all 0.3s ease;
}

li:hover {
    background: rgba(52, 152, 219, 0.1);
}

a {
    color: #3498db;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.3s ease;
}

a:hover {
    color: #2980b9;
    text-decoration: underline;
}

/* Summary sections */
.summary-section {
    margin: 30px 0;
    padding: 20px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 12px;
    border: 1px solid #dee2e6;
}

.summary-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-left: 4px solid;
}

.summary-card.exploration {
    border-left-color: #17a2b8;
}

.summary-card.evaluation {
    border-left-color: #28a745;
}

.summary-card h4 {
    margin: 0 0 15px 0;
    color: #495057;
    font-size: 1.2em;
    font-weight: 600;
}

.config-summaries {
    margin: 30px 0;
}

.config-summary {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border: 1px solid #dee2e6;
}

.config-summary h4 {
    margin: 0 0 15px 0;
    color: #495057;
    font-size: 1.1em;
    font-weight: 600;
}

.config-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 15px;
}

.stat-item {
    background: #e9ecef;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.9em;
    color: #495057;
    font-weight: 500;
}

/* Responsive design */
@media (max-width: 768px) {
    .sample-page {
        padding: 80px 16px 16px 16px;
        margin: 10px;
    }

    .combo-container {
        /* Mobile styling matching sample-page */
        padding: 15px;
        margin: 10px;
    }

    .combo-content-inner {
        /* Mobile inner content */
    }
    
    h1 {
        font-size: 2em;
        margin-top: 50px;
    }
    
    .turn {
        padding: 15px;
    }
    
    img.room {
        max-width: 100%;
    }
    
    #nav input {
        width: 60px;
    }
    
    .dict-container {
        grid-template-columns: 1fr;
    }
    
    .config-stats {
        flex-direction: column;
    }
}

/* Cognitive Map Charts Styling */
.cognitive-map-charts {
    background: white;
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border: 1px solid #dee2e6;
}

.cognitive-map-charts h3 {
    color: #2c3e50;
    margin: 0 0 20px 0;
    font-size: 1.4em;
    font-weight: 600;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px;
}

.cogmap-grid {
    display: flex;
    flex-direction: column;
    gap: 30px;
}

.cogmap-row {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    border: 1px solid #e9ecef;
}

.cogmap-row h4 {
    color: #495057;
    margin: 0 0 15px 0;
    font-size: 1.1em;
    font-weight: 600;
    text-align: center;
    padding: 8px;
    background: rgba(52, 152, 219, 0.1);
    border-radius: 6px;
}

.cogmap-columns {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}

.cogmap-column {
    background: white;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border: 1px solid #dee2e6;
}

.cogmap-column.empty {
    background: #f8f9fa;
    border: 1px dashed #dee2e6;
}

.cogmap-column h5 {
    color: #495057;
    margin: 0 0 10px 0;
    font-size: 1em;
    font-weight: 600;
}

.cogmap-plot {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.no-data {
    color: #6c757d;
    font-style: italic;
    margin: 20px 0;
}

/* Turn averages section */
.turn-averages-section {
    margin: 30px 0;
    padding: 20px;
    background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
    border-radius: 12px;
    border: 1px solid #b3d9ff;
}

.turn-averages-section h3 {
    color: #0d47a1;
    margin: 0 0 20px 0;
    font-size: 1.4em;
    font-weight: 600;
    border-bottom: 2px solid #1976d2;
    padding-bottom: 8px;
    text-align: center;
}

.turn-averages-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
}

.turn-average-plot {
    background: white;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border: 1px solid #cce7ff;
}

.turn-average-plot h4 {
    color: #1565c0;
    margin: 0 0 15px 0;
    font-size: 1.1em;
    font-weight: 600;
}

/* Responsive design for cognitive map charts */
@media (max-width: 768px) {
    .cogmap-columns {
        grid-template-columns: 1fr;
    }

    .turn-averages-grid {
        grid-template-columns: 1fr;
    }

    .cognitive-map-charts,
    .turn-averages-section {
        margin: 15px 0;
        padding: 15px;
    }

    .three-plots-grid {
        grid-template-columns: 1fr;
    }

    .plots-grid {
        grid-template-columns: 1fr;
    }

    /* Stack cognitive map layout vertically on mobile */
    .cogmap-compare {
        grid-template-columns: 1fr;
    }
}

/* Separate text and plots sections */
.text-metrics-section {
    margin-bottom: 20px;
}

.plots-section {
    margin-top: 20px;
}

.single-plot {
    margin: 20px 0;
    text-align: center;
}

.plots-row {
    margin: 30px 0;
}

.three-plots-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-top: 15px;
}

.plots-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-top: 15px;
}

.plot-item {
    text-align: center;
    background: white;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border: 1px solid #dee2e6;
}

.plot-image {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Combination Selector Styles */
.combination-selector {
    background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
    padding: 20px;
    margin: 20px 0;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.combination-selector h3 {
    color: white;
    margin: 0 0 15px 0;
    font-size: 1.2em;
    font-weight: 600;
}

.combo-buttons {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.combo-btn {
    padding: 10px 20px;
    border: 2px solid transparent;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.2);
    color: white;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 0.9em;
}

.combo-btn:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}

.combo-btn.active {
    background: white;
    color: #0984e3;
    border-color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.combo-container {
    /* Single container with consistent background for all configs */
    padding: 20px;
    max-width: 1400px;
    margin: 20px auto;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}

.combo-content-inner {
    /* Inner content that gets replaced, no background changes */
    animation: fadeIn 0.3s ease-in-out;
}

/* Single combination info styles */
.single-combo-info {
    background: linear-gradient(135deg, #00cec9 0%, #00b894 100%);
    padding: 15px;
    margin: 20px 0;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.single-combo-info h3 {
    color: white;
    margin: 0;
    font-size: 1.1em;
    font-weight: 600;
}

/* Question section styles for evaluation tasks - moved to later section */

.question-section h4 {
    color: #495057;
    margin: 0 0 15px 0;
    font-size: 1.1em;
    font-weight: 600;
    border-bottom: 1px solid #dee2e6;
    padding-bottom: 8px;
}

/* Question content layout for side-by-side display */
.question-content {
    display: flex;
    gap: 20px;
    margin-top: 20px; /* Increased margin from title */
}

.question-left {
    flex: 2;
    min-width: 0; /* Prevent overflow */
    overflow: hidden; /* Contain content */
    position: relative; /* For z-index context */
}

.question-right {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    min-width: 0; /* Prevent overflow */
    overflow: hidden; /* Contain content */
    position: relative; /* For z-index context */
}

/* Fix expandable content to prevent container expansion */
.block.expandable .content-text {
    word-wrap: break-word;
    overflow-wrap: break-word;
    max-width: 100%;
}

.block.expandable[data-expanded='true'] .content-text {
    max-height: 300px;
    overflow-y: auto;
    overflow-x: hidden;
    word-break: break-word;
    white-space: pre-wrap;
}



/* Add spacing for turn sections */
.turn-split {
    margin-top: 40px; /* Much more space between turns */
    margin-bottom: 30px; /* Space after each turn */
    clear: both; /* Clear any floating elements */
}

.turn-split:first-child {
    margin-top: 20px; /* Less space for first turn */
}

.turn-split:first-child h3 {
    margin-top: 5px; /* Less top margin for first turn title */
}

/* Ensure turn content doesn't overflow, but allow hover expansion */
.turn-content {
    overflow: hidden;
    position: relative;
}

.turn-left {
    overflow: hidden;
    position: relative;
}


/* Evaluation task specific styling */
.eval-task {
    margin-top: 50px; /* Much more space for evaluation sections */
    margin-bottom: 40px; /* Space after evaluation sections */
    clear: both; /* Clear any floating elements */
}

.eval-task h3 {
    margin: 0 0 30px 0; /* Remove top margin, increase bottom */
    padding: 20px 0; /* More padding around evaluation title */
    border-bottom: 2px solid #2196f3;
    color: #1565c0;
    font-size: 1.4em; /* Larger font for evaluation titles */
    font-weight: 600;
    line-height: 1.4; /* Better line height */
    word-wrap: break-word; /* Allow title to wrap if needed */
    overflow-wrap: break-word;
}

/* Question section improvements */
.question-section {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    margin: 20px 0; /* Increased margin between questions */
    padding: 18px; /* Slightly more padding */
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    overflow: hidden; /* Prevent content from breaking out */
    position: relative;
}

/* Simple question section styling - no special hover effects */

/* Responsive design for question layout */
@media (max-width: 768px) {
    .question-content {
        flex-direction: column;
        gap: 15px;
    }

    .question-left,
    .question-right {
        flex: 1;
    }
}

/* JSON Display Styles with Collapsible Functionality */
.json-container {
    margin: 15px 0;
    background: #f8f9fa;
    border-radius: 8px;
    border: 1px solid #dee2e6;
    overflow: hidden;
}

.json-header {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 12px 15px;
    border-bottom: 1px solid #dee2e6;
}

.json-header strong {
    color: #1565c0;
    font-weight: 600;
    display: block;
}

.json-content {
    /* Always visible, no transition needed */
}

/* Global layout - three columns */
.json-compare.global {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0;
}

/* Local layout - two columns */
.json-compare.local {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
}

.json-box {
    background: white;
    border: none;
    padding: 15px;
    margin: 0;
    min-height: 200px;
}

.json-box.left {
    border-right: 1px solid #dee2e6;
}

.json-box.middle {
    border-right: 1px solid #dee2e6;
    border-left: none;
}

.json-box.right {
    border-left: none;
}

.json-box strong {
    display: block;
    color: #495057;
    font-weight: 600;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid #e9ecef;
    font-size: 0.9em;
}

.json-content-inner {
    background: #f8f9fa;
    border-radius: 4px;
    padding: 10px;
    border: 1px solid #e9ecef;
    max-height: 400px;
    overflow-y: auto;
}

.json-content-inner pre {
    margin: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 11px;
    line-height: 1.4;
    color: #495057;
    background: transparent;
    border: none;
    padding: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-x: auto;
}

.empty-json {
    color: #6c757d;
    font-style: italic;
    padding: 10px;
    background: #f8f9fa;
    border-radius: 4px;
    border: 1px dashed #dee2e6;
    text-align: center;
    font-size: 0.9em;
}

/* Color coding for different JSON types */
.json-box:has(strong:contains("Predicted")) .json-content-inner {
    border-left: 3px solid #17a2b8;
}

.json-box:has(strong:contains("Ground Truth (Observed)")) .json-content-inner {
    border-left: 3px solid #28a745;
}

.json-box:has(strong:contains("Ground Truth (Full)")) .json-content-inner {
    border-left: 3px solid #ffc107;
}

.json-box:has(strong:contains("Ground Truth")) .json-content-inner {
    border-left: 3px solid #28a745;
}

/* Fallback for browsers that don't support :has() */
.json-box.predicted .json-content-inner {
    border-left: 3px solid #17a2b8;
}

.json-box.gt-observed .json-content-inner {
    border-left: 3px solid #28a745;
}

.json-box.gt-full .json-content-inner {
    border-left: 3px solid #ffc107;
}

.json-box.gt .json-content-inner {
    border-left: 3px solid #28a745;
}

/* Responsive design for JSON displays */
@media (max-width: 1024px) {
    .json-compare.global {
        grid-template-columns: 1fr;
    }

    .json-compare.local {
        grid-template-columns: 1fr;
    }

    .json-box.left,
    .json-box.middle {
        border-right: none;
        border-bottom: 1px solid #dee2e6;
    }

    .json-box.right {
        border-left: none;
        border-top: none;
    }

    .json-box:last-child {
        border-bottom: none;
    }
}

@media (max-width: 768px) {
    .json-content-inner {
        max-height: 300px;
        font-size: 10px;
    }

    .json-box {
        padding: 12px;
        min-height: 150px;
    }
}

/* Metrics Section */
.metrics-section {
    margin: 20px 0;
    padding: 20px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 12px;
    border: 1px solid #dee2e6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.metrics-section h3 {
    margin-top: 0;
    margin-bottom: 20px;
    color: #495057;
    font-weight: 600;
    text-align: center;
    font-size: 1.3em;
}

.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
    align-items: start;
}

.metrics-grid.four-columns {
    grid-template-columns: 1fr 1fr 1fr 1fr;
}

@media (max-width: 1024px) {
    .metrics-grid {
        grid-template-columns: 1fr;
        gap: 15px;
    }
    .metrics-grid.four-columns {
        grid-template-columns: 1fr;
    }
}

@media (min-width: 1025px) and (max-width: 1400px) {
    .metrics-grid {
        grid-template-columns: 1fr 1fr;
    }
    .metrics-grid.four-columns {
        grid-template-columns: 1fr 1fr;
    }
}

@media (min-width: 1401px) and (max-width: 1600px) {
    .metrics-grid.four-columns {
        grid-template-columns: 1fr 1fr 1fr;
    }
}

.metrics-box {
    background: #ffffff;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    border: 1px solid #e9ecef;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metrics-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.metrics-box h4 {
    margin-top: 0;
    margin-bottom: 12px;
    font-size: 1.1em;
    font-weight: 600;
    color: #495057;
    border-bottom: 2px solid #e9ecef;
    padding-bottom: 8px;
}

.metrics-box.exploration h4 {
    color: #28a745;
    border-bottom-color: #28a745;
}

.metrics-box.evaluation h4 {
    color: #007bff;
    border-bottom-color: #007bff;
}

.metrics-box.cogmap h4 {
    color: #6f42c1;
    border-bottom-color: #6f42c1;
}

.metrics-box.correlation h4 {
    color: #fd7e14;
    border-bottom-color: #fd7e14;
}

.metrics-box .dict-container {
    margin: 0;
    padding: 0;
}

.metrics-box .dict-item {
    margin: 6px 0;
    padding: 4px 0;
    border-bottom: 1px solid #f8f9fa;
}

.metrics-box .dict-item:last-child {
    border-bottom: none;
}

.metrics-box .dict-key {
    font-weight: 500;
    color: #495057;
}

.metrics-box .dict-value {
    margin-left: 8px;
}

.metrics-box .dict-value.number {
    font-weight: 600;
    color: #495057;
}

.metrics-box .dict-value.true {
    color: #28a745;
    font-weight: 600;
}

.metrics-box .dict-value.false {
    color: #dc3545;
    font-weight: 600;
}

"""

JAVASCRIPT_CODE = """
let currentPage = 0;

function showPage(n, total) {
    currentPage = Math.max(0, Math.min(total-1, n));
    const pages = document.querySelectorAll('.sample-page');
    pages.forEach((p, i) => {
        p.classList.toggle('active', i === currentPage);
    });
    document.getElementById('counter').innerText = (currentPage+1) + ' / ' + total;
    document.getElementById('goto').value = currentPage+1;
    location.hash = '#p' + (currentPage+1);

    // Smooth scroll to top
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });

    // CRITICAL: Force reset all combo states on ALL pages before initializing
    resetAllComboStates();

    // Initialize combination display for current page
    setTimeout(() => {
        initCombinationDisplay();
        initEvaluationTaskSelector();
    }, 50);
}

function nextPage(total) {
    showPage(currentPage+1, total);
}

function prevPage(total) {
    showPage(currentPage-1, total);
}

function gotoPage(total) {
    const v = parseInt(document.getElementById('goto').value, 10);
    if (!isNaN(v)) {
        showPage(v-1, total);
    }
}

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === 'PageDown') {
        nextPage({total_pages});
    }
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        prevPage({total_pages});
    }
    if (e.key === 'Home') {
        showPage(0, {total_pages});
    }
    if (e.key === 'End') {
        showPage({total_pages}-1, {total_pages});
    }
});

// Initialize on page load
window.addEventListener('load', () => {
    const m = location.hash.match(/#p(\\d+)/);
    if (m) {
        showPage(parseInt(m[1], 10)-1, {total_pages});
    } else {
        showPage(0, {total_pages});
    }
});

// Add smooth transitions and initialize on load
document.addEventListener('DOMContentLoaded', () => {
    const style = document.createElement('style');
    style.textContent = `
        .sample-page {
            transition: opacity 0.3s ease, transform 0.3s ease;
        }
    `;
    document.head.appendChild(style);

    // CRITICAL: Global reset then initialize
    setTimeout(() => {
        resetAllComboStates();
        setTimeout(() => {
            initCombinationDisplay();
            initEvaluationTaskSelector();
        }, 50);
    }, 100);
});

// Toggle observation functionality
function toggleObservation(obsId) {
    const mainDiv = document.getElementById(obsId);
    const fullContentDiv = document.getElementById(obsId + '_full');
    const shortContentDiv = document.getElementById(obsId + '_short');
    const contentSpan = mainDiv.querySelector('.content-text');
    
    if (mainDiv && fullContentDiv && shortContentDiv && contentSpan) {
        const isExpanded = mainDiv.getAttribute('data-expanded') === 'true';
        
        if (isExpanded) {
            // Switch to short content
            contentSpan.innerHTML = shortContentDiv.innerHTML;
            mainDiv.setAttribute('data-expanded', 'false');
        } else {
            // Switch to full content
            contentSpan.innerHTML = fullContentDiv.innerHTML;
            mainDiv.setAttribute('data-expanded', 'true');
        }
    }
}

// Expand thinking functionality
function expandThinking(thinkId) {
    const fullContent = document.getElementById(thinkId).innerHTML;
    
    // Create modal if it doesn't exist
    let modal = document.getElementById('think-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'think-modal';
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="modal-close" onclick="closeThinking()">&times;</span>
                <h3>🤔 Full Assistant Thinking</h3>
                <div id="modal-think-content"></div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Close on background click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeThinking();
        });
    }
    
    // Set content and show modal
    document.getElementById('modal-think-content').innerHTML = fullContent;
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeThinking() {
    const modal = document.getElementById('think-modal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
}

// Toggle cognitive map response functionality
function toggleCogmapResponse(cogmapId) {
    const mainDiv = document.getElementById(cogmapId);
    const fullContentDiv = document.getElementById(cogmapId + '_full');
    const shortContentDiv = document.getElementById(cogmapId + '_short');
    const contentSpan = mainDiv.querySelector('.content-text');
    
    if (mainDiv && fullContentDiv && shortContentDiv && contentSpan) {
        const isExpanded = mainDiv.getAttribute('data-expanded') === 'true';
        
        if (isExpanded) {
            // Switch to short content
            contentSpan.innerHTML = shortContentDiv.innerHTML;
            mainDiv.setAttribute('data-expanded', 'false');
        } else {
            // Switch to full content
            contentSpan.innerHTML = fullContentDiv.innerHTML;
            mainDiv.setAttribute('data-expanded', 'true');
        }
    }
}

// Toggle ground truth visibility
function toggleGroundTruth(gtId) {
    const content = document.getElementById(gtId);
    if (content) {
        // Check both style.display and computed style to handle initial inline styles
        const isHidden = content.style.display === 'none' ||
                        getComputedStyle(content).display === 'none';

        if (isHidden) {
            content.style.display = 'block';
        } else {
            content.style.display = 'none';
        }
    }
}


// Reset all combo states across ALL pages
function resetAllComboStates() {
    console.log('Resetting all combo states globally');

    // Reset ALL combo buttons across all pages
    const allComboButtons = document.querySelectorAll('.combo-btn');
    allComboButtons.forEach(btn => {
        btn.classList.remove('active');
    });

    console.log('Global combo reset complete:', allComboButtons.length, 'buttons reset');
}

// Initialize combination display for current page
function initCombinationDisplay() {
    const currentPageDiv = document.querySelector('.sample-page.active');
    if (!currentPageDiv) {
        console.log('No active page found for initialization');
        return;
    }

    // Find combo buttons on this page
    const comboButtons = currentPageDiv.querySelectorAll('.combo-btn');
    if (comboButtons.length === 0) {
        console.log('No combo buttons found on this page');
        return;
    }

    console.log('Initializing combo display for page:', currentPageDiv.id || 'unknown');

    // Initialize with first combo
    const firstButton = comboButtons[0];
    const sampleId = firstButton.dataset.sample;
    const firstCombo = firstButton.dataset.combo;

    if (sampleId && firstCombo) {
        // Switch to the first combination
        switchCombination(firstCombo, sampleId);
        console.log('Initialized with first combination:', firstCombo);
    }
}

// Initialize evaluation task selector
function initEvaluationTaskSelector() {
    const currentPageDiv = document.querySelector('.sample-page.active');
    if (!currentPageDiv) return;

    const taskSelector = document.getElementById('eval-task-selector');
    const taskSelect = document.getElementById('task-select');

    // Look for evaluation tasks in the current page's active content
    const activeContent = currentPageDiv.querySelector('.combo-content-inner');
    let evalSections = [];

    if (activeContent) {
        // Look for eval tasks in the active combo content
        evalSections = activeContent.querySelectorAll('.eval-task');
    } else {
        // Fallback: look in the entire page
        evalSections = currentPageDiv.querySelectorAll('.eval-task');
    }

    if (evalSections.length > 1) {
        // Show selector and populate options
        taskSelector.style.display = 'inline-block';
        taskSelect.innerHTML = '';

        evalSections.forEach((section, index) => {
            const taskName = section.getAttribute('data-task-name');
            const option = document.createElement('option');
            option.value = taskName;
            option.textContent = taskName;
            if (index === 0) option.selected = true;
            taskSelect.appendChild(option);
        });

        // Show first task by default
        switchEvaluationTask();
    } else {
        taskSelector.style.display = 'none';
    }
}

// Switch evaluation task display
function switchEvaluationTask() {
    const taskSelect = document.getElementById('task-select');
    const selectedTask = taskSelect.value;
    const currentPageDiv = document.querySelector('.sample-page.active');

    if (!currentPageDiv) return;

    // Look for evaluation tasks in the current page's active content
    const activeContent = currentPageDiv.querySelector('.combo-content-inner');
    let evalSections = [];

    if (activeContent) {
        // Look for eval tasks in the active combo content
        evalSections = activeContent.querySelectorAll('.eval-task');
    } else {
        // Fallback: look in the entire page
        evalSections = currentPageDiv.querySelectorAll('.eval-task');
    }

    // Hide all evaluation tasks
    evalSections.forEach(section => {
        section.style.display = 'none';
    });

    // Show selected task
    const selectedSection = activeContent ?
        activeContent.querySelector(`.eval-task[data-task-name="${selectedTask}"]`) :
        currentPageDiv.querySelector(`.eval-task[data-task-name="${selectedTask}"]`);

    if (selectedSection) {
        selectedSection.style.display = 'block';
    }
}

// Switch combination for a sample
function switchCombination(selectedCombo, sampleId) {
    // Get current active page only
    const currentPage = document.querySelector('.sample-page.active');
    if (!currentPage) return;

    console.log('Switching to combo:', selectedCombo, 'for sample:', sampleId);

    // Get the data script containing all combo HTML
    const dataScript = document.getElementById(`${sampleId}-data`);
    if (!dataScript) {
        console.error('No data found for sample:', sampleId);
        return;
    }

    let comboData;
    try {
        comboData = JSON.parse(dataScript.textContent);
    } catch (e) {
        console.error('Failed to parse combo data:', e);
        return;
    }

    // Get the content container
    const contentContainer = document.getElementById(`${sampleId}-content`);
    if (!contentContainer || !comboData[selectedCombo]) {
        console.error('Content container or combo data not found');
        return;
    }

    // Seamlessly replace the inner content without changing the container
    contentContainer.innerHTML = comboData[selectedCombo].html;
    console.log('Seamlessly switched content to:', selectedCombo);

    // Update button states for this sample in current page only
    const comboButtons = currentPage.querySelectorAll('.combo-btn');
    comboButtons.forEach(btn => {
        if (btn.dataset.sample === sampleId) {
            if (btn.dataset.combo === selectedCombo) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        }
    });

    // Reinitialize evaluation task selector for the new combination
    initEvaluationTaskSelector();
}
"""