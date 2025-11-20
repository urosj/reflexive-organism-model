#!/usr/bin/env python3
# --------------------------------------------------------------
#  Agentic Super-Organism Visualization System
#  Visualizes state evolution of components and the whole system
# --------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import json

class VisualizationSystem:
    def __init__(self, agent):
        self.agent = agent
        self.fig = None
        self.axs: Dict[str, Any] = {}
        self.turn_history: List[Dict[str, Any]] = []
        self._seen_turn_nodes = set()

    def setup_figure(self):
        """Initialize the visualization figure with multiple subplots"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))

        # Create subplots for different visualization types
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Experience Field (Spatial EI Grid)
        ax_ei = fig.add_subplot(gs[0, :2])

        # Agent Selection & Cooperation
        ax_coop = fig.add_subplot(gs[0, 2])

        # Memory Usage Over Time
        ax_memory = fig.add_subplot(gs[1, 0])

        # System State Overview
        ax_overview = fig.add_subplot(gs[1, 1])

        # EI trend
        ax_ei_trend = fig.add_subplot(gs[1, 2])

        # Assembly graph (new DAG view)
        ax_assembly = fig.add_subplot(gs[2, :2])

        # Network Graph (Self-model)
        ax_graph = fig.add_subplot(gs[2, 2])

        self.axs = {
            "ei": ax_ei,
            "coop": ax_coop,
            "memory": ax_memory,
            "overview": ax_overview,
            "ei_trend": ax_ei_trend,
            "assembly": ax_assembly,
            "self_graph": ax_graph,
        }
        self.fig = fig

        return fig

    def visualize_experience_field(self, ax):
        """Visualize the spatial experience field (EI grid)"""
        if not hasattr(self.agent, 'ei_grid'):
            return

        grid = self.agent.ei_grid.grid
        im = ax.imshow(grid.T, cmap='viridis', origin='lower')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Experience Intensity')

        ax.set_title('Spatial Experience Field (EI Grid)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        # Add grid lines
        for i in range(grid.shape[0]):
            ax.axhline(i - 0.5, color='white', linewidth=0.5)
        for j in range(grid.shape[1]):
            ax.axvline(j - 0.5, color='white', linewidth=0.5)

    def visualize_cooperation_scores(self, ax):
        """Visualize cooperation scores of children"""
        if not hasattr(self.agent, 'children'):
            return

        # Get cooperation scores
        coop_scores = []
        agent_names = []

        for i, child in enumerate(self.agent.children):
            # Use last_used_in_parent as a proxy for cooperation score
            coop_scores.append(child.last_used_in_parent)
            agent_names.append(f"Child {i}")

        if not coop_scores:
            return

        # Create bar chart
        bars = ax.bar(range(len(coop_scores)), coop_scores, color='skyblue')
        ax.set_title('Cooperation Scores (Children)')
        ax.set_xlabel('Agent Index')
        ax.set_ylabel('Cooperation Score')
        ax.set_xticks(range(len(coop_scores)))
        ax.set_xticklabels(agent_names, rotation=45)

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, coop_scores)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.2f}', ha='center', va='bottom')

    def visualize_memory_usage(self, ax):
        """Visualize memory usage over time"""
        if not self.turn_history:
            ax.text(0.5, 0.5, 'No turns recorded yet', ha='center', va='center')
            ax.set_axis_off()
            return

        memory_sizes = [entry["mem"] for entry in self.turn_history]
        turns = list(range(1, len(memory_sizes) + 1))

        ax.plot(turns, memory_sizes, marker='o', linewidth=2, markersize=4)
        ax.set_title('Memory Growth Over Time')
        ax.set_xlabel('Turn Number')
        ax.set_ylabel('Number of Memory Entries')
        ax.grid(alpha=0.2)

    def visualize_system_overview(self, ax):
        """Visualize overall system state"""
        if not hasattr(self.agent, 'children') or not hasattr(self.agent, 'ei_grid'):
            return

        # Create a summary of key metrics
        total_agents = len(self.agent.children) + len(self.agent.grandchildren)
        ei_mean = self.agent.ei_grid.mean() if hasattr(self.agent, 'ei_grid') else 0.0

        if total_agents == 0:
            ax.text(0.5, 0.5, 'No agents yet', ha='center', va='center')
            ax.set_axis_off()
        else:
            labels = ['Children', 'Grandchildren']
            sizes = [len(self.agent.children), len(self.agent.grandchildren)]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%')
            ax.set_title('Agent Hierarchy Distribution\n(Total: {})'.format(total_agents))

        # Annotate EI stats
        if self.turn_history:
            recent_ei = self.turn_history[-1].get("ei", 0.0)
            avg_ei = np.mean([entry.get("ei", 0.0) for entry in self.turn_history])
            ax.text(0.0, -1.2, f"EI mean: {avg_ei:.3f}\nEI latest: {recent_ei:.3f}", ha='left', va='center')

    def visualize_ei_trend(self, ax):
        """Plot EI and lambda history for the last turns."""
        if not self.turn_history:
            ax.text(0.5, 0.5, 'No EI history yet', ha='center', va='center')
            ax.set_axis_off()
            return

        turns = list(range(1, len(self.turn_history) + 1))
        ei_values = np.asarray([entry.get("ei", np.nan) for entry in self.turn_history], dtype=float)
        lam_raw = np.asarray([entry.get("lam", np.nan) for entry in self.turn_history], dtype=float)
        lam_floor = np.asarray([entry.get("lam_floor", np.nan) for entry in self.turn_history], dtype=float)
        lam_effective = np.asarray([entry.get("lam_effective", np.nan) for entry in self.turn_history], dtype=float)

        if not np.any(np.isfinite(ei_values)) and not np.any(np.isfinite(lam_raw)):
            ax.text(0.5, 0.5, 'No EI/λ history yet', ha='center', va='center')
            ax.set_axis_off()
            return

        ax.plot(turns, np.nan_to_num(ei_values, nan=0.0), label='EI', color='purple', marker='o')
        ax.plot(turns, np.nan_to_num(lam_raw, nan=0.5), label='λ raw', color='orange', linestyle='--', marker='x')
        ax.plot(turns, np.nan_to_num(lam_floor, nan=0.5), label='λ floor', color='green', linestyle=':', marker='s')
        ax.plot(turns, np.nan_to_num(lam_effective, nan=0.5), label='λ effective', color='blue', linestyle='-.', marker='d')
        ax.set_title('Experience Index / λ Trend')
        ax.set_xlabel('Turn')
        ax.set_ylabel('Value')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(loc='lower right')

    def visualize_self_model_graph(self, ax):
        """Visualize the self-model network graph"""
        if not hasattr(self.agent, 'graph') or len(self.agent.graph.nodes()) == 0:
            return

        # Draw network graph
        pos = nx.spring_layout(self.agent.graph, k=1.5, iterations=50)

        # Separate nodes by type
        parent_nodes = ['parent']
        child_nodes = [n for n in self.agent.graph.nodes() if n.startswith('child')]
        turn_nodes = [n for n in self.agent.graph.nodes() if n.startswith('t')]

        # Draw different node types with different colors
        nx.draw_networkx_nodes(self.agent.graph, pos,
                              nodelist=parent_nodes,
                              node_color='red', node_size=1000, ax=ax)
        nx.draw_networkx_nodes(self.agent.graph, pos,
                              nodelist=child_nodes,
                              node_color='blue', node_size=800, ax=ax)
        nx.draw_networkx_nodes(self.agent.graph, pos,
                              nodelist=turn_nodes,
                              node_color='green', node_size=600, ax=ax)

        # Draw edges
        nx.draw_networkx_edges(self.agent.graph, pos, alpha=0.5, ax=ax)

        # Add labels
        labels = {n: n[:10] + '...' if len(n) > 10 else n for n in self.agent.graph.nodes()}
        nx.draw_networkx_labels(self.agent.graph, pos, labels, font_size=8, ax=ax)

        ax.set_title('Self-Model Network Graph')
        ax.axis('off')

    def visualize_assembly_graph(self, ax):
        """Render the assembly DAG with memory/child/parent nodes."""
        if not hasattr(self.agent, 'assembly_index'):
            ax.text(0.5, 0.5, 'Assembly index unavailable', ha='center', va='center')
            ax.set_axis_off()
            return

        graph = self.agent.assembly_index.graph
        if graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, 'No assemblies yet', ha='center', va='center')
            ax.set_axis_off()
            return

        # Limit nodes for readability
        max_nodes = 80
        subgraph = graph
        if graph.number_of_nodes() > max_nodes:
            nodes_sorted = sorted(
                graph.nodes(),
                key=lambda n: graph.nodes[n].get('ei', 0.0) if graph.nodes[n].get('kind') == 'turn' else 0.0,
                reverse=True,
            )[:max_nodes]
            subgraph = graph.subgraph(nodes_sorted).copy()

        pos = nx.spring_layout(subgraph, k=0.8, iterations=50)

        memory_nodes = [n for n, data in subgraph.nodes(data=True) if data.get('kind') == 'memory']
        assembly_nodes = [n for n, data in subgraph.nodes(data=True) if data.get('kind') == 'assembly']
        turn_nodes = [n for n, data in subgraph.nodes(data=True) if data.get('kind') == 'turn']

        nx.draw_networkx_nodes(subgraph, pos, nodelist=memory_nodes, node_color='lightblue', node_size=200, ax=ax, label='Memory')
        nx.draw_networkx_nodes(subgraph, pos, nodelist=assembly_nodes, node_color='plum', node_size=300, ax=ax, label='Assemblies')
        nx.draw_networkx_nodes(subgraph, pos, nodelist=turn_nodes, node_color='lightgreen', node_size=250, ax=ax, label='Turns')
        nx.draw_networkx_edges(subgraph, pos, alpha=0.4, ax=ax)

        ax.set_title('Assembly DAG (recent nodes)')
        ax.axis('off')

    def visualize_all_components(self):
        """Visualize all components at once"""
        if not self.fig:
            self.setup_figure()

        self._update_turn_history()

        # Clear axes
        for ax in self.axs.values():
            ax.clear()

        # Update each visualization component
        try:
            self.visualize_experience_field(self.axs["ei"])
        except Exception as e:
            print(f"Error visualizing experience field: {e}")

        try:
            self.visualize_cooperation_scores(self.axs["coop"])
        except Exception as e:
            print(f"Error visualizing cooperation scores: {e}")

        try:
            self.visualize_memory_usage(self.axs["memory"])
        except Exception as e:
            print(f"Error visualizing memory usage: {e}")

        try:
            self.visualize_system_overview(self.axs["overview"])
        except Exception as e:
            print(f"Error visualizing system overview: {e}")

        try:
            self.visualize_ei_trend(self.axs["ei_trend"])
        except Exception as e:
            print(f"Error visualizing EI trend: {e}")

        try:
            self.visualize_assembly_graph(self.axs["assembly"])
        except Exception as e:
            print(f"Error visualizing assembly graph: {e}")

        try:
            self.visualize_self_model_graph(self.axs["self_graph"])
        except Exception as e:
            print(f"Error visualizing self-model graph: {e}")

        # Set overall title
        self.fig.suptitle('Agentic Super-Organism State Visualization', fontsize=16, y=0.98)

    def update_visualization(self):
        """Update visualization for new data"""
        self.visualize_all_components()
        plt.pause(0.01)  # Small pause to allow plot updates

    def save_snapshot(self, filename=None):
        """Save current visualization as image file"""
        if not filename:
            timestamp = int(time.time())
            filename = f"agentic_superorganism_snapshot_{timestamp}.png"

        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved snapshot to {filename}")

    def _update_turn_history(self):
        """Sync turn history from agent graph nodes."""
        if not hasattr(self.agent, 'graph'):
            return

        for node, data in self.agent.graph.nodes(data=True):
            if not isinstance(node, str) or not node.startswith('t'):
                continue
            if node in self._seen_turn_nodes:
                continue

            timestamp = None
            try:
                timestamp = int(node[1:])
            except Exception:
                timestamp = int(time.time())

            ei_val = data.get('ei', data.get('value', 0.0))
            lam_val = data.get('lam', 0.0)
            if 'mem_size' in data:
                mem_size = data['mem_size']
            else:
                mem_store = getattr(self.agent, 'mem', None)
                mem_size = len(mem_store.texts) if hasattr(mem_store, 'texts') else 0

            self.turn_history.append({
                "turn_id": node,
                "ts": timestamp,
                "ei": ei_val,
                "lam": lam_val,
                "lam_floor": data.get('lam_floor', lam_val),
                "lam_effective": data.get('lam_effective', lam_val),
                "mem": mem_size,
            })
            self._seen_turn_nodes.add(node)

        self.turn_history.sort(key=lambda entry: entry["ts"])

class RealTimeVisualizationSystem(VisualizationSystem):
    """Enhanced visualization system with real-time updates"""

    def __init__(self, agent, update_interval=1.0):
        super().__init__(agent)
        self.update_interval = update_interval
        self.animation = None

    def setup_realtime_visualization(self):
        """Setup for animated visualization"""
        fig = self.setup_figure()

        # Create animation
        def animate(frame):
            try:
                self.visualize_all_components()
                return list(self.axs.values())

            except Exception as e:
                print(f"Error in animation: {e}")
                return list(self.axs.values())

        # Start animation - updates every update_interval seconds
        self.animation = FuncAnimation(
            fig, animate, interval=int(self.update_interval * 1000),
            blit=False, repeat=True
        )

        return fig

    def start_realtime_visualization(self):
        """Start the real-time visualization"""
        if not self.fig:
            self.setup_realtime_visualization()

        plt.show()

def create_comprehensive_monitor(agent):
    """
    Create a comprehensive monitoring system that tracks multiple metrics over time
    and allows detailed analysis of system evolution.
    """

    # This function would be called after each turn to collect data
    def update_tracking_data(turn_number):
        tracking_data = {
            'turn': turn_number,
            'ei_grid_mean': agent.ei_grid.mean() if hasattr(agent, 'ei_grid') else 0.0,
            'memory_size': len(agent.mem.texts) if hasattr(agent, 'mem') else 0,
            'num_children': len(agent.children),
            'num_grandchildren': len(agent.grandchildren),
            'cooperation_scores': [child.last_used_in_parent for child in agent.children] if hasattr(agent, 'children') else [],
            'altruism_coefficient': getattr(agent, '_current_lam', 0.5) if hasattr(agent, '_current_lam') else 0.5
        }
        return tracking_data

    return update_tracking_data

# Additional utility functions for detailed analysis

def plot_cooperation_evolution(agent):
    """Plot the evolution of cooperation scores over time"""
    plt.figure(figsize=(12, 8))

    # Get cooperation data from children
    if hasattr(agent, 'children') and len(agent.children) > 0:
        num_children = len(agent.children)

        for i, child in enumerate(agent.children):
            # Plot each child's cooperation scores over time (simplified)
            plt.subplot(2, 3, i+1)
            try:
                coop_scores = [child.last_used_in_parent]
                plt.plot(coop_scores, marker='o', linewidth=2)
                plt.title(f'Child {i} Cooperation Evolution')
                plt.ylabel('Cooperation Score')
                plt.xlabel('Turn Number')
            except Exception as e:
                print(f"Error plotting cooperation for child {i}: {e}")

        plt.tight_layout()
        plt.savefig('cooperation_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_experience_field_evolution(agent):
    """Plot the evolution of experience field over time"""
    if not hasattr(agent, 'ei_grid'):
        return

    # Create a heatmap showing how the EI grid changes
    fig, ax = plt.subplots(figsize=(10, 8))

    try:
        grid = agent.ei_grid.grid
        im = ax.imshow(grid.T, cmap='plasma', origin='lower')

        cbar = plt.colorbar(im)
        cbar.set_label('Experience Intensity')
        ax.set_title('Current Experience Field (Spatial EI Grid)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        plt.savefig('ei_field_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error plotting experience field: {e}")

def analyze_agent_hierarchy(agent):
    """Analyze and report on the agent hierarchy structure"""

    if not hasattr(agent, 'children'):
        return

    print("=== AGENT HIERARCHY ANALYSIS ===")
    print(f"Number of Children: {len(agent.children)}")
    print(f"Number of Grandchildren: {len(agent.grandchildren)}")

    for i, child in enumerate(agent.children):
        print(f"\nChild {i}: {child.name}")
        if hasattr(child, 'last_used_in_parent'):
            print(f"  Cooperation Score: {child.last_used_in_parent:.3f}")
        if hasattr(child, 'recent_emb') and child.recent_emb is not None:
            print(f"  Recent Embedding Length: {len(child.recent_emb)}")

    # Show memory statistics
    if hasattr(agent, 'mem'):
        print(f"\nMemory Statistics:")
        print(f"  Total Entries: {len(agent.mem.texts)}")
        print(f"  Average Entry Length: {np.mean([len(text) for text in agent.mem.texts]) if agent.mem.texts else 0:.1f}")

if __name__ == "__main__":
    # Example usage:
    # This would be called from within the main system

    print("Visualization System Ready")
    print("To use: Create an instance of VisualizationSystem(agent) and call visualize_all_components()")
    print("\nAvailable functions:")
    print("- visualize_all_components(): Show current state")
    print("- save_snapshot(filename): Save current view as PNG")
    print("- plot_cooperation_evolution(agent): Plot cooperation evolution")
    print("- plot_experience_field_evolution(agent): Plot EI field visualization")
    print("- analyze_agent_hierarchy(agent): Analyze agent structure")
