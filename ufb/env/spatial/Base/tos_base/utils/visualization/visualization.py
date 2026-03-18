# visualization.py
import json
import os
from pathlib import Path
from html import escape
from typing import List, Dict, Optional
from .html_templates import HTML_TEMPLATE, CSS_STYLES, JAVASCRIPT_CODE

from ..utils import parse_llm_response
from .charts import create_infogain_plot, create_cogmap_metrics_plot, create_correlation_plot



class VisualizationHelper:
    """Helper class for data processing and HTML generation"""
    
    @staticmethod
    def dict_to_html(d: Dict) -> str:
        """Convert dictionary to HTML format with better styling"""
        if not d:
            return "<div class='empty-dict'>(none)</div>"
        
        html = "<div class='dict-container'>"
        for k, v in d.items():
            if isinstance(v, (int, float)):
                # Format numbers nicely
                if isinstance(v, float):
                    if v != v:  # NaN check
                        formatted_v = "NaN"
                    else:
                        formatted_v = f"{v:.3f}" if v != int(v) else str(int(v))
                else:
                    formatted_v = str(v)
                html += f"<div class='dict-item'><span class='dict-key'>{escape(str(k))}:</span> <span class='dict-value number'>{formatted_v}</span></div>"
            elif isinstance(v, bool):
                # Color-code booleans
                color_class = "true" if v else "false"
                html += f"<div class='dict-item'><span class='dict-key'>{escape(str(k))}:</span> <span class='dict-value {color_class}'>{str(v)}</span></div>"
            elif isinstance(v, dict):
                # Handle nested dictionaries
                nested_html = VisualizationHelper.dict_to_html(v)
                html += f"<div class='dict-item nested'><span class='dict-key'>{escape(str(k))}:</span> <div class='dict-value nested-dict'>{nested_html}</div></div>"
            else:
                # String values with consistent styling
                html += f"<div class='dict-item'><span class='dict-key'>{escape(str(k))}:</span> <span class='dict-value string'>{escape(str(v))}</span></div>"
        html += "</div>"
        return html
    
    








class HTMLGenerator:
    """Handles HTML generation for the visualization"""
    
    def __init__(self, data: Dict, output_html: str, show_images: bool = True):
        self.data = data
        self.output_html = output_html
        self.show_images = show_images
        self.out_dir = os.path.dirname(output_html)
        self.base = Path(output_html).stem

        # Extract data
        self.meta = data.get("meta_info", {})
        self.samples = data.get("samples", {})
        self.total_samples = len(self.samples)

        # Extract summary data
        self.exp_summary = data.get("exp_summary", {})
        self.eval_summary = data.get("eval_summary", {})
        self.cogmap_summary = data.get("cogmap_summary", {})
        self.correlation_summary = data.get("correlation", {})

        # Calculate statistics - each sample becomes one page
        self.total_pages = 1 + self.total_samples  # page 0 = TOC

        # Build flat list for samples (sample_id, sample_data)
        self.flat = []
        for sample_id, sample_data in self.samples.items():
            self.flat.append((sample_id, sample_data))

        # Extract available combinations from sample data keys
        self.combinations = self._extract_combinations_from_samples()

    def _extract_combinations_from_samples(self) -> List[str]:
        """Extract unique combination keys from all samples"""
        combination_keys = set()

        for sample_data in self.samples.values():
            for key in sample_data.keys():
                combination_keys.add(key)

        # Return as sorted list for consistent ordering
        return sorted(list(combination_keys))

    def generate_config_summaries(self, f) -> None:
        """Generate summaries for each config combination"""
        f.write("<div class='config-summaries'>\n")
        f.write("<h3>📋 Configuration Summaries</h3>\n")

        for gname in self.combinations:
            # Count samples that have this combination
            sample_count = sum(1 for sample_data in self.samples.values()
                             if gname in sample_data and sample_data[gname] is not None)
            f.write(f"<div class='config-summary'>\n")
            f.write(f"<h4>⚙️ {escape(gname)}</h4>\n")
            f.write(f"<div class='config-stats'>\n")
            f.write(f"<div class='stat-item'>📊 Samples: {sample_count}</div>\n")
            f.write("</div>\n")

            # Generate plot data first (to pop values before metrics display)
            infogain_plot = None
            cogmap_update_plot = None
            cogmap_full_plot = None

            # Exploration infogain plot
            if self.exp_summary.get("group_performance", {}).get(gname):
                exp_group = self.exp_summary["group_performance"][gname]
                infogain_per_turn = exp_group.pop("infogain_per_turn", [])
                if infogain_per_turn:
                    infogain_plot = create_infogain_plot(infogain_per_turn, gname)

            # Cognitive map plots (only global now)
            if self.cogmap_summary.get("group_performance", {}).get(gname):
                cogmap_group = self.cogmap_summary["group_performance"][gname]
                update_data = cogmap_group.pop("cogmap_update_per_turn", {})
                full_data = cogmap_group.pop("cogmap_full_per_turn", {})

                # Only accept new shape (metric -> list)
                global_update = update_data if isinstance(update_data, dict) else {}
                if global_update and any(global_update.values()):
                    title = f"{gname} - Global (Update)"
                    cogmap_update_plot = create_cogmap_metrics_plot(global_update, title)

                # Full mode plot (global only)
                global_full = full_data if isinstance(full_data, dict) else {}
                if global_full and any(global_full.values()):
                    title = f"{gname} - Global (Full)"
                    cogmap_full_plot = create_cogmap_metrics_plot(global_full, title)

            # Generate correlation plots
            correlation_plots = {}
            if self.correlation_summary.get("group_performance", {}).get(gname):
                correlation_data = self.correlation_summary["group_performance"][gname]
                cogmap_values = correlation_data.pop('last_global_vs_gt_fulls', [])
                acc_values = correlation_data.pop('avg_acc_metrics', [])
                infogain_values = correlation_data.pop('last_infogains', [])

                # Call twice to generate two scatter plots using existing correlation_info
                if cogmap_values and acc_values:
                    acc_correlation = correlation_data.get('cogmap_acc_correlations', {}).get('avg_accuracy', None)
                    correlation_plots['cogmap_vs_accuracy'] = create_correlation_plot(
                        cogmap_values, acc_values,
                        'Cognitive Map Score (Last Global vs GT Full)',
                        'Average Accuracy',
                        'Cognitive Map Score vs Average Accuracy',
                        acc_correlation
                    )

                if cogmap_values and infogain_values:
                    infogain_correlation = correlation_data.get('cogmap_infogain_correlation', None)
                    correlation_plots['cogmap_vs_infogain'] = create_correlation_plot(
                        cogmap_values, infogain_values,
                        'Cognitive Map Score (Last Global vs GT Full)',
                        'Information Gain',
                        'Cognitive Map Score vs Information Gain',
                        infogain_correlation
                    )

            # Config metrics section with four-column layout (display metrics first)
            f.write("<div class='metrics-section'>\n")
            f.write("<div class='metrics-grid four-columns'>\n")

            # Group exploration performance
            exp_group = self.exp_summary.get("group_performance", {}).get(gname)
            if exp_group:
                exp_group_filtered = {k: v for k, v in exp_group.items() if k != "infogain_per_turn"}
                if exp_group_filtered:
                    f.write("<div class='metrics-box exploration'>\n")
                    f.write("<h4>🔍 Exploration</h4>\n")
                    f.write(VisualizationHelper.dict_to_html(exp_group_filtered))
                    f.write("</div>\n")

            # Group evaluation performance
            eval_group = self.eval_summary.get("group_performance", {}).get(gname)
            if eval_group:
                f.write("<div class='metrics-box evaluation'>\n")
                f.write("<h4>✅ Evaluation</h4>\n")
                f.write(VisualizationHelper.dict_to_html(eval_group))
                f.write("</div>\n")

            # Group cognitive map performance
            cogmap_group = self.cogmap_summary.get("group_performance", {}).get(gname)
            if cogmap_group:
                # Display main cognitive map metrics (exclude per_turn data)
                main_metrics = {k: v for k, v in cogmap_group.items()
                               if k not in ["cogmap_update_per_turn", "cogmap_full_per_turn"]}
                if main_metrics:
                    f.write("<div class='metrics-box cogmap'>\n")
                    f.write("<h4>🧠 Cognitive Map</h4>\n")
                    f.write(VisualizationHelper.dict_to_html(main_metrics))
                    f.write("</div>\n")

            # Group correlation performance
            correlation_summary = getattr(self, 'correlation_summary', {})
            correlation_group = correlation_summary.get("group_performance", {}).get(gname)
            if correlation_group:
                f.write("<div class='metrics-box correlation'>\n")
                f.write("<h4>📈 Correlation</h4>\n")
                f.write(VisualizationHelper.dict_to_html(correlation_group))
                f.write("</div>\n")

            f.write("</div>\n")  # End metrics-grid
            f.write("</div>\n")  # End metrics-section

            # Plots section (display after metrics, but plots were generated earlier)
            f.write("<div class='plots-section'>\n")

            # Display plots in a single row (up to 5 plots now)
            available_plots = []
            if infogain_plot:
                available_plots.append(("Information Gain per Turn", infogain_plot, "Information Gain per Turn"))
            if cogmap_update_plot:
                available_plots.append(("Cognitive Map (Update)", cogmap_update_plot, "Cognitive Map Update Turn Averages"))
            if cogmap_full_plot:
                available_plots.append(("Cognitive Map (Full)", cogmap_full_plot, "Cognitive Map Full Turn Averages"))

            # Add correlation plots
            if correlation_plots.get('cogmap_vs_accuracy'):
                available_plots.append(("CogMap vs Accuracy", correlation_plots['cogmap_vs_accuracy'], "Cognitive Map vs Accuracy Correlation"))
            if correlation_plots.get('cogmap_vs_infogain'):
                available_plots.append(("CogMap vs InfoGain", correlation_plots['cogmap_vs_infogain'], "Cognitive Map vs Information Gain Correlation"))

            if available_plots:
                f.write("<div class='plots-row'>")
                f.write("<h5>Performance Charts</h5>")
                # Use flexible grid that can handle more plots
                f.write("<div class='plots-grid'>")
                for title, plot_uri, alt_text in available_plots:
                    f.write(f"<div class='plot-item'>")
                    f.write(f"<h6>{title}</h6>")
                    f.write(f"<img src='{plot_uri}' alt='{alt_text}' class='plot-image'>")
                    f.write("</div>")
                f.write("</div>")
                f.write("</div>")

            f.write("</div>\n")  # End plots-section

            f.write("</div>\n")  # End config-summary

        f.write("</div>\n")


    def generate_toc_page(self, f) -> None:
        """Generate table of contents page with summaries"""
        f.write("<section class='sample-page active' id='page0'>\n")
        f.write("<h2>📋 Dashboard Overview</h2>\n")

        self.generate_config_summaries(f)

        f.write("<h3>📖 Sample Navigation</h3>\n")
        f.write("<ul>\n")
        running_page = 1

        for sample_id, sample_data in self.samples.items():
            # Count available combinations for this sample
            available_combos = [combo for combo in self.combinations
                               if combo in sample_data and sample_data[combo] is not None]
            combo_count = len(available_combos)

            f.write(
                f"<li>"
                f"<a href='#' onclick=\"showPage({running_page}, {self.total_pages});return false;\">"
                f"{escape(sample_id)} ({combo_count} combinations)</a>"
                f"</li>\n"
            )
            running_page += 1
        f.write("</ul>\n</section>\n")

    def generate_sample_metrics(self, f, entry: Dict, sample_name: str) -> None:
        """Generate sample-level metrics visualization"""
        metrics = entry.get("metrics", {})
        if not metrics:
            return

        f.write("<div class='metrics-section'>\n")
        f.write("<h3>📊 Sample Metrics</h3>\n")

        # Create a three-column layout for exploration, evaluation, and cogmap metrics
        f.write("<div class='metrics-grid'>\n")

        # Helper function to filter out per_turn keys
        def filter_per_turn_keys(data):
            if not isinstance(data, dict):
                return data
            return {k: v for k, v in data.items() if "per_turn" not in k}

        # Exploration metrics
        exploration_metrics = metrics.get("exploration", {})
        if exploration_metrics:
            filtered_exploration = filter_per_turn_keys(exploration_metrics)
            if filtered_exploration:
                f.write("<div class='metrics-box exploration'>\n")
                f.write("<h4>🔍 Exploration</h4>\n")
                f.write(VisualizationHelper.dict_to_html(filtered_exploration))
                f.write("</div>\n")

        # Evaluation metrics
        evaluation_metrics = metrics.get("evaluation", {})
        if evaluation_metrics:
            filtered_evaluation = filter_per_turn_keys(evaluation_metrics)
            if filtered_evaluation:
                f.write("<div class='metrics-box evaluation'>\n")
                f.write("<h4>✅ Evaluation</h4>\n")
                f.write(VisualizationHelper.dict_to_html(filtered_evaluation))
                f.write("</div>\n")

        # Cognitive map metrics
        cogmap_metrics = metrics.get("cogmap", {})
        if cogmap_metrics:
            filtered_cogmap = filter_per_turn_keys(cogmap_metrics)
            if filtered_cogmap:
                f.write("<div class='metrics-box cogmap'>\n")
                f.write("<h4>🧠 Cognitive Map</h4>\n")
                f.write(VisualizationHelper.dict_to_html(filtered_cogmap))
                f.write("</div>\n")

        f.write("</div>\n")  # End metrics-grid
        f.write("</div>\n")  # End metrics-section

    def generate_cognitive_map_charts(self, f, entry: Dict, sample_name: str) -> None:
        """Generate cognitive map charts and information gain chart in a single row"""
        # Extract information gain data from exploration turns
        infogain_per_turn = entry['metrics'].get('exploration', {}).pop('infogain_per_turn', [])
        cogmap_update_data = entry['metrics'].get('cogmap', {}).pop('cogmap_update_per_turn', {})
        cogmap_full_data = entry['metrics'].get('cogmap', {}).pop('cogmap_full_per_turn', {})
        
        # Generate plots
        infogain_plot = None
        update_plot = None
        full_plot = None

        # Information gain plot
        if infogain_per_turn:
            infogain_plot = create_infogain_plot(infogain_per_turn, sample_name)

        # Cognitive map plots
        if any(cogmap_update_data.values()):
            title = f"{sample_name} - Global (Update)"
            update_plot = create_cogmap_metrics_plot(cogmap_update_data, title)

        if any(cogmap_full_data.values()):
            title = f"{sample_name} - Global (Full)"
            full_plot = create_cogmap_metrics_plot(cogmap_full_data, title)

        # Display all plots in horizontal layout (3 plots for samples)
        available_plots = []
        if infogain_plot:
            available_plots.append(("Information Gain per Turn", infogain_plot, "Information Gain per Turn"))
        if update_plot:
            available_plots.append(("Cognitive Map (Update)", update_plot, "Global Update Metrics"))
        if full_plot:
            available_plots.append(("Cognitive Map (Full)", full_plot, "Global Full Metrics"))

        if available_plots:
            f.write("<div class='cognitive-map-charts'>\n")
            f.write("<h3>📊 Performance Charts</h3>\n")
            f.write("<div class='plots-row'>\n")
            f.write("<div class='three-plots-grid'>\n")

            for title, plot_uri, alt_text in available_plots:
                f.write("<div class='plot-item'>\n")
                f.write(f"<h6>{title}</h6>\n")
                f.write(f"<img src='{plot_uri}' alt='{alt_text}' class='plot-image'>\n")
                f.write("</div>\n")

            f.write("</div>\n")  # End three-plots-grid
            f.write("</div>\n")  # End plots-row
            f.write("</div>\n")  # End cognitive-map-charts

    def generate_sample_page(self, f, page_idx: int, sample_id: str, sample_data: Dict) -> None:
        """Generate a single sample page with combination selector"""
        f.write(f"<section class='sample-page' id='page{page_idx}'>\n")
        f.write(f"<h2>{escape(sample_id)}</h2>\n")

        # Add combination selector
        available_combos = [combo for combo in self.combinations
                           if combo in sample_data and sample_data[combo] is not None]

        # Always show combination selector for consistency, regardless of count
        if len(available_combos) >= 1:
            f.write("<div class='combination-selector'>\n")
            f.write("<h3>Select Configuration:</h3>\n")
            f.write("<div class='combo-buttons'>\n")
            for i, combo in enumerate(available_combos):
                active_class = "active" if i == 0 else ""
                f.write(f"<button class='combo-btn {active_class}' data-combo='{combo}' data-sample='{sample_id}' onclick=\"switchCombination('{combo}', '{sample_id}')\">{combo.replace('_', ' ').title()}</button>\n")
            f.write("</div>\n")
            f.write("</div>\n")

        # Single container for all combinations with seamless background
        f.write(f"<div class='combo-container' id='{sample_id}-container'>\n")

        # Store all combo data as JSON for dynamic switching
        f.write(f"<script type='application/json' id='{sample_id}-data'>\n")
        combo_data = {}
        for combo in available_combos:
            if combo in sample_data and sample_data[combo] is not None:
                combo_data[combo] = {
                    'html': self._generate_combo_html(sample_data[combo], combo, sample_id, page_idx)
                }
        f.write(json.dumps(combo_data, ensure_ascii=False))
        f.write("</script>\n")

        # Generate initial content (first combo)
        if available_combos:
            initial_combo = available_combos[0]
            entry = sample_data[initial_combo]

            # Generate content that will be replaced dynamically
            f.write(f"<div class='combo-content-inner' id='{sample_id}-content'>\n")
            f.write(self._generate_combo_html(entry, initial_combo, sample_id, page_idx))
            f.write("</div>\n")

        f.write("</div>\n")  # End combo-container
        f.write("</section>\n")

    def _generate_combo_html(self, entry: Dict, combo: str, sample_id: str, page_idx: int) -> str:
        """Generate HTML content for a single combo configuration"""
        from io import StringIO
        output = StringIO()

        # Generate Sample Metrics at the top
        self.generate_sample_metrics(output, entry, f"{combo} {sample_id}")

        # Generate Performance Charts (Information Gain + Cognitive Map plots)
        self.generate_cognitive_map_charts(output, entry, f"{combo} {sample_id}")

        # Display initial room image if available
        if self.show_images and entry.get("initial_room_image"):
            img_name = entry["initial_room_image"]
            output.write(f"<img src='{img_name}' class='room' alt='Initial room state'>\n")

        # Environment config
        # cfg = entry["env_info"]["config"]
        # output.write("<div class='metrics'><strong>🔧 Environment Configuration</strong>")
        # output.write(VisualizationHelper.dict_to_html(cfg))
        # output.write("</div>\n")

        # Generate exploration turns and evaluation tasks
        self.generate_exploration_turns(output, entry, page_idx)

        return output.getvalue()

    def _render_expandable_block(self, f, content: str, block_id: str, title: str, block_class: str = "user") -> None:
        """Helper to render expandable content blocks"""
        if not content:
            return
        content_short = escape(content[:300]).replace("\n", "<br>")
        content_full = escape(content).replace("\n", "<br>")
        f.write(f"<div id='{block_id}' class='block {block_class} expandable' onclick='toggleObservation(\"{block_id}\")' data-expanded='false'><strong>{title} <span class='expand-hint'>(click to toggle)</span></strong><br><span class='content-text'>{content_short}...</span></div>\n")
        f.write(f"<div id='{block_id}_full' style='display:none'>{content_full}</div>\n")
        f.write(f"<div id='{block_id}_short' style='display:none'>{content_short}...</div>\n")

    def _render_simple_block(self, f, content: str, title: str, block_class: str) -> None:
        """Helper to render simple content blocks"""
        if not content:
            return
        content_escaped = escape(content).replace("\n", "<br>")
        f.write(f"<div class='block {block_class}'><strong>{title}</strong><br>{content_escaped}</div>\n")

    def _render_cogmap_responses(self, f, cogmap_log: Dict, page_idx: int, t_idx: int) -> None:
        """Helper to render cognitive map responses"""
        cogmap_types = [
            ('global', '🗺️ Global Cognitive Map Response'),
            ('local', '🗺️ Local Cognitive Map Response'),
            ('relations', '🗺️ Relations Cognitive Map Response'),
            ('rooms', '🗺️ Rooms Cognitive Map Response'),
            ('false_belief', '🧭 False Belief Cognitive Map Response')
        ]

        for map_type, title in cogmap_types:
            data = cogmap_log.get(map_type, {})
            if data.get('original_response'):
                response_id = f"cogmap_{map_type}_{page_idx}_{t_idx}"
                self._render_expandable_block(f, data['original_response'], response_id, title, "cogmap-response")

                # Add JSON display for global and local
                if map_type == 'global':
                    # For global, display pred_json, gt_json, and gt_json_full in three columns
                    pred_json = data.get('pred_json', {})
                    gt_json = data.get('gt_json', {})
                    gt_json_full = data.get('gt_json_full', {})

                    if pred_json or gt_json or gt_json_full:
                        f.write("<div class='json-container global'>\n")
                        f.write("<div class='json-header'>")
                        f.write("<strong>📊 Cognitive Map JSONs</strong>")
                        f.write("</div>\n")
                        f.write("<div class='json-content'>\n")
                        f.write("<div class='json-compare global'>\n")

                        # Left - pred_json
                        f.write("<div class='json-box left predicted'>\n")
                        f.write("<strong>🤖 Predicted</strong>\n")
                        if pred_json:
                            f.write("<div class='json-content-inner'>\n")
                            f.write(f"<pre>{escape(json.dumps(pred_json, indent=2))}</pre>\n")
                            f.write("</div>\n")
                        else:
                            f.write("<div class='empty-json'>(no data)</div>\n")
                        f.write("</div>\n")

                        # Middle - gt_json
                        f.write("<div class='json-box middle gt-observed'>\n")
                        f.write("<strong>🎯 Ground Truth (Observed)</strong>\n")
                        if gt_json:
                            f.write("<div class='json-content-inner'>\n")
                            f.write(f"<pre>{escape(json.dumps(gt_json, indent=2))}</pre>\n")
                            f.write("</div>\n")
                        else:
                            f.write("<div class='empty-json'>(no data)</div>\n")
                        f.write("</div>\n")

                        # Right - gt_json_full
                        f.write("<div class='json-box right gt-full'>\n")
                        f.write("<strong>🎯 Ground Truth (Full)</strong>\n")
                        if gt_json_full:
                            f.write("<div class='json-content-inner'>\n")
                            f.write(f"<pre>{escape(json.dumps(gt_json_full, indent=2))}</pre>\n")
                            f.write("</div>\n")
                        else:
                            f.write("<div class='empty-json'>(no data)</div>\n")
                        f.write("</div>\n")

                        f.write("</div>\n")  # End json-compare
                        f.write("</div>\n")  # End json-content
                        f.write("</div>\n")  # End json-container

                elif map_type == 'local':
                    # For local, display pred_json and gt_json in two columns
                    pred_json = data.get('pred_json', {})
                    gt_json = data.get('gt_json', {})

                    if pred_json or gt_json:
                        f.write("<div class='json-container local'>\n")
                        f.write("<div class='json-header'>")
                        f.write("<strong>📊 Cognitive Map JSONs</strong>")
                        f.write("</div>\n")
                        f.write("<div class='json-content'>\n")
                        f.write("<div class='json-compare local'>\n")

                        # Left - pred_json
                        f.write("<div class='json-box left predicted'>\n")
                        f.write("<strong>🤖 Predicted</strong>\n")
                        if pred_json:
                            f.write("<div class='json-content-inner'>\n")
                            f.write(f"<pre>{escape(json.dumps(pred_json, indent=2))}</pre>\n")
                            f.write("</div>\n")
                        else:
                            f.write("<div class='empty-json'>(no data)</div>\n")
                        f.write("</div>\n")

                        # Right - gt_json
                        f.write("<div class='json-box right gt'>\n")
                        f.write("<strong>🎯 Ground Truth</strong>\n")
                        if gt_json:
                            f.write("<div class='json-content-inner'>\n")
                            f.write(f"<pre>{escape(json.dumps(gt_json, indent=2))}</pre>\n")
                            f.write("</div>\n")
                        else:
                            f.write("<div class='empty-json'>(no data)</div>\n")
                        f.write("</div>\n")

                        f.write("</div>\n")  # End json-compare
                        f.write("</div>\n")  # End json-content
                        f.write("</div>\n")  # End json-container

                elif map_type == 'false_belief':
                    # For false_belief, display only pred_json in single column
                    pred_json = data.get('pred_json', {})

                    if pred_json:
                        f.write("<div class='json-container false-belief'>\n")
                        f.write("<div class='json-header'>")
                        f.write("<strong>📊 False Belief Prediction JSON</strong>")
                        f.write("</div>\n")
                        f.write("<div class='json-content'>\n")
                        f.write("<div class='json-compare false-belief'>\n")

                        # Single column - pred_json
                        f.write("<div class='json-box single predicted'>\n")
                        f.write("<strong>🤖 Predicted</strong>\n")
                        f.write("<div class='json-content-inner'>\n")
                        f.write(f"<pre>{escape(json.dumps(pred_json, indent=2))}</pre>\n")
                        f.write("</div>\n")
                        f.write("</div>\n")

                        f.write("</div>\n")  # End json-compare
                        f.write("</div>\n")  # End json-content
                        f.write("</div>\n")  # End json-container

    def _render_cogmap_metrics(self, f, cogmap_log: Dict) -> None:
        """Helper to render cognitive map metrics"""
        if not cogmap_log:
            return

        # Extract metrics
        global_log = cogmap_log.get("global", {})
        local_log = cogmap_log.get("local", {})
        rooms_log = cogmap_log.get("rooms", {})
        relations_log = cogmap_log.get("relations", {})

        metrics_block = {
            "Global": global_log.get("metrics", {}) if global_log else {},
            "Global (Full)": global_log.get("metrics_full", {}) if global_log else {},
            "Local": local_log.get("metrics", {}) if local_log else {},
            "Rooms": rooms_log.get("metrics", {}) if rooms_log else {},
            "Relations": relations_log.get("metrics", {}) if relations_log else {}
        }

        if any(metrics_block.values()):
            f.write("<div class='block cogmap'><strong>🧠 Cognitive Map Metrics</strong>")
            f.write(VisualizationHelper.dict_to_html(metrics_block))
            f.write("</div>\n")

    def _render_turn_metrics(self, f, env_log: Dict) -> None:
        """Helper to render turn metrics"""
        metrics = {}

        if env_log['is_exploration_phase'] and env_log.get('exploration_log'):
            exp_log = env_log['exploration_log']
            metrics.update({
                "node_coverage": exp_log.get('node_coverage'),
                "edge_coverage": exp_log.get('edge_coverage'),
                "is_action_fail": exp_log.get('is_action_fail'),
                "step": exp_log.get('step'),
                "action_counts": exp_log.get('action_counts'),
                "information_gain": exp_log.get('information_gain')
            })

        if env_log.get('info'):
            metrics.update(env_log['info'])

        if metrics:
            f.write("<div class='metrics'><strong>📈 Turn Metrics</strong>")
            f.write(VisualizationHelper.dict_to_html(metrics))
            f.write("</div>\n")

    def _render_turn_images(self, f, env_log: Dict, env_turn_logs: List, t_idx: int) -> None:
        """Helper to render turn images"""
        f.write("<div class='turn-right'>\n")

        if self.show_images:
            # Previous image (initial if first turn)
            if t_idx > 0:
                prev_img = env_turn_logs[t_idx-1].get('room_image')
                if prev_img:
                    f.write(f"<figure><img src='{prev_img}' class='room-plot' alt='Previous state'><figcaption>State before Turn {t_idx+1}</figcaption></figure>\n")

            # Current image
            curr_img = env_log.get('room_image')
            if curr_img:
                f.write(f"<figure><img src='{curr_img}' class='room-plot' alt='Current state'><figcaption>State at Turn {t_idx+1}</figcaption></figure>\n")

            # Message images
            if 'message_images' in env_log:
                for img_idx, img_path in enumerate(env_log['message_images']):
                    if isinstance(img_path, str):
                        f.write(f"<figure><img src='{img_path}' class='room-plot' alt='Environment image {img_idx + 1}'><figcaption>Observation {img_idx + 1}</figcaption></figure>\n")

        f.write("</div>\n")  # End turn-right


    def generate_exploration_turns(self, f, entry: Dict, page_idx: int) -> None:
        """Generate exploration turn logs and evaluation tasks"""
        env_turn_logs = entry.get("env_turn_logs", [])
        evaluation_tasks = entry.get("evaluation_tasks", {})

        if not env_turn_logs and not evaluation_tasks:
            f.write("<div class='metrics'><strong>⚠️ No turns available</strong></div>\n")
            return

        # Generate exploration turns
        for t_idx, env_log in enumerate(env_turn_logs):
            f.write("<div class='turn-split'>\n")
            f.write(f"<h3>🔄 Turn {t_idx+1}</h3>\n")
            f.write("<div class='turn-content'>\n")

            # Left side: conversation and metrics
            f.write("<div class='turn-left'>\n")

            # Display user message (environment observation)
            if env_log.get('user_message'):
                obs_id = f"obs_{page_idx}_{t_idx}"
                self._render_expandable_block(f, env_log['user_message'], obs_id, "👤 Environment Observation")

            # Display assistant thinking and action
            if env_log.get('assistant_think_message'):
                think_id = f"think_{page_idx}_{t_idx}"
                self._render_expandable_block(f, env_log['assistant_think_message'], think_id, "🤔 Assistant Thinking", "think")
            self._render_simple_block(f, env_log.get('assistant_parsed_message', ''), "💬 Assistant Action", "answer")

            # Display cognitive map original responses if available
            if env_log.get('cogmap_log'):
                self._render_cogmap_responses(f, env_log['cogmap_log'], page_idx, t_idx)
                self._render_cogmap_metrics(f, env_log['cogmap_log'])

            # Display turn metrics
            self._render_turn_metrics(f, env_log)
            f.write("</div>\n")  # End turn-left

            # Right side: room and message images
            self._render_turn_images(f, env_log, env_turn_logs, t_idx)
            f.write("</div>\n")  # End turn-content
            f.write("</div>\n")  # End turn-split

        # Generate evaluation turns if available
        if evaluation_tasks:
            # Handle new nested structure: {task_type: {question_id: eval_data}}
            for eval_idx, (task_type, task_questions) in enumerate(evaluation_tasks.items()):
                t_idx = len(env_turn_logs) + eval_idx

                # Create evaluation turn section with task selector data
                f.write(f"<div class='turn-split eval-task' data-task-name='{escape(task_type)}'")
                if eval_idx > 0:  # Hide all but first task by default
                    f.write(" style='display:none'")
                f.write(">\n")
                f.write(f"<h3>📊 Evaluation: {escape(task_type)}</h3>\n")

                f.write("<div class='turn-content'>\n")

                # Left side: conversation and metrics
                f.write("<div class='turn-left'>\n")

                for question_idx, (question_id, eval_log) in enumerate(task_questions.items()):
                    f.write(f"<div class='question-section' data-question-id='{question_id}'>\n")
                    f.write(f"<h4>Question {question_idx + 1} (ID: {question_id})</h4>\n")

                    # Create a split layout for this question
                    f.write("<div class='question-content'>\n")
                    f.write("<div class='question-left'>\n")

                    # Display evaluation question using helper function
                    if eval_log.get('user_message'):
                        obs_id = f"obs_{page_idx}_{t_idx}_{question_idx}"
                        self._render_expandable_block(f, eval_log['user_message'], obs_id, "❓ Evaluation Question")
                    # Display assistant thinking and action
                    if eval_log.get('assistant_think_message'):
                        think_id = f"think_{page_idx}_{t_idx}_{question_idx}"
                        self._render_expandable_block(f, eval_log['assistant_think_message'], think_id, "🤔 Assistant Thinking", "think")
                    
                    self._render_simple_block(f, eval_log.get('assistant_parsed_message', ''), "💬 Assistant Answer", "answer")
                    if eval_log.get('cogmap_log'):
                        self._render_cogmap_responses(f, eval_log['cogmap_log'], page_idx, t_idx)
                        self._render_cogmap_metrics(f, eval_log['cogmap_log'])
                    # Display evaluation results
                    if eval_log.get('evaluation_log'):
                        eval_info = eval_log['evaluation_log']
                        f.write("<div class='block evaluation'><strong>✅ Evaluation Results</strong>")
                        details = {
                            **eval_info.get("evaluation_data", {}),
                            **eval_info.get("evaluation_info", {}),
                            "Correct": eval_info.get("is_correct"),
                        }
                        f.write(VisualizationHelper.dict_to_html(details))
                        f.write("</div>\n")

                    f.write("</div>\n")  # End question-left

                    # Right side: images for this specific question
                    f.write("<div class='question-right'>\n")
                    if self.show_images:
                        # Current evaluation state image
                        if eval_log.get("room_image"):
                            img_name = eval_log["room_image"]
                            f.write(f"<figure><img src='{img_name}' class='room-plot' alt='Evaluation state'><figcaption>Q{question_idx + 1}: {escape(task_type)}</figcaption></figure>\n")

                        # Display message images if available
                        if 'message_images' in eval_log:
                            for img_idx, img_path in enumerate(eval_log['message_images']):
                                if isinstance(img_path, str):
                                    f.write(f"<figure><img src='{img_path}' class='room-plot' alt='Evaluation image {img_idx + 1}'><figcaption>Q{question_idx + 1} Image {img_idx + 1}</figcaption></figure>\n")

                    f.write("</div>\n")  # End question-right
                    f.write("</div>\n")  # End question-content
                    f.write("</div>\n")  # End question-section

                f.write("</div>\n")  # End turn-left
                f.write("</div>\n")  # End turn-content
                f.write("</div>\n")  # End turn-split

        f.write("</section>\n")

    def generate_html(self) -> str:
        """Generate the complete HTML file"""
        with open(self.output_html, "w") as f:
            # Write HTML header with CSS and JS
            js_code = JAVASCRIPT_CODE.replace('{total_pages}', str(self.total_pages))
            f.write(HTML_TEMPLATE.format(
                model_name=escape(self.meta.get('model_name', 'Unknown Model')),
                total_pages=self.total_pages,
                css_styles=CSS_STYLES,
                javascript_code=js_code
            ))

            # Generate TOC page (now with summaries)
            self.generate_toc_page(f)

            # Generate sample pages
            for page_idx, (sample_id, sample_data) in enumerate(self.flat, start=1):
                self.generate_sample_page(f, page_idx, sample_id, sample_data)

            f.write("</body></html>")

        return self.output_html

