"""
Weighted Directed Graph Visualization for Stock Lead-Lag Relationships
Shows how stocks lead and lag each other based on correlation analysis
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Load the data
df = pd.read_csv('/mnt/user-data/uploads/all_pairs_analysis.csv')

print("="*80)
print("STOCK LEAD-LAG RELATIONSHIP NETWORK ANALYSIS")
print("="*80)
print(f"\nDataset: {len(df)} stock pairs analyzed")
print(f"Unique stocks: {len(set(df['Stock_A'].unique()) | set(df['Stock_B'].unique()))}")
print(f"Lag range: {df['Best_Lag'].min()} to {df['Best_Lag'].max()} days")
print(f"Correlation range: {df['Correlation'].min():.3f} to {df['Correlation'].max():.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 1: FULL NETWORK (All relationships with correlation > 0.3)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "─"*80)
print("GRAPH 1: Full Network (correlation > 0.3)")
print("─"*80)

# Filter for strong correlations and non-synchronous relationships
threshold = 0.3
df_filtered = df[(df['Correlation'] >= threshold) & (df['Best_Lag'] > 0)].copy()

print(f"Showing {len(df_filtered)} relationships (correlation ≥ {threshold}, lag > 0)")

# Create directed graph
G = nx.DiGraph()

# Add edges: Leader → Lagger with weight = correlation
for _, row in df_filtered.iterrows():
    G.add_edge(
        row['Leader'], 
        row['Lagger'],
        weight=row['Correlation'],
        lag=row['Best_Lag']
    )

print(f"Nodes (stocks): {G.number_of_nodes()}")
print(f"Edges (relationships): {G.number_of_edges()}")

# Calculate node metrics
in_degree = dict(G.in_degree())  # How many stocks this one lags
out_degree = dict(G.out_degree())  # How many stocks this one leads
betweenness = nx.betweenness_centrality(G)  # Information flow centrality

print("\nTop 5 Leaders (highest out-degree - stocks that predict many others):")
for stock, degree in sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {stock}: leads {degree} stocks")

print("\nTop 5 Laggers (highest in-degree - stocks predicted by many others):")
for stock, degree in sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {stock}: lags {degree} stocks")

print("\nTop 5 Information Hubs (highest betweenness centrality):")
for stock, centrality in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {stock}: {centrality:.3f}")

# Create visualization
fig, ax = plt.subplots(figsize=(20, 16), facecolor='#0a0e17')
ax.set_facecolor('#0a0e17')

# Layout using spring algorithm (force-directed)
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Node sizes based on total degree (importance)
node_sizes = [300 + (in_degree.get(node, 0) + out_degree.get(node, 0)) * 150 
              for node in G.nodes()]

# Node colors based on net influence (out_degree - in_degree)
net_influence = [out_degree.get(node, 0) - in_degree.get(node, 0) for node in G.nodes()]
node_colors = net_influence

# Draw edges with varying width and opacity based on correlation
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
lags = [G[u][v]['lag'] for u, v in edges]

# Normalize weights for edge width
edge_widths = [1 + w * 4 for w in weights]  # 1-5 pixel width
edge_alphas = [0.3 + w * 0.5 for w in weights]  # 0.3-0.8 alpha

# Draw edges
for (u, v), width, alpha, weight, lag in zip(edges, edge_widths, edge_alphas, weights, lags):
    # Color based on lag
    if lag <= 1:
        color = '#60a5fa'  # Blue for short lags
    elif lag <= 3:
        color = '#a78bfa'  # Purple for medium lags
    else:
        color = '#f472b6'  # Pink for long lags
    
    ax.annotate('',
                xy=pos[v], xycoords='data',
                xytext=pos[u], textcoords='data',
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=color,
                    lw=width,
                    alpha=alpha,
                    connectionstyle='arc3,rad=0.1',
                    mutation_scale=20
                ))

# Draw nodes
nodes = nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap=plt.cm.RdYlBu,
    vmin=-5, vmax=5,
    edgecolors='white',
    linewidths=2,
    ax=ax
)

# Draw labels
labels = nx.draw_networkx_labels(
    G, pos,
    font_size=10,
    font_weight='bold',
    font_color='white',
    font_family='monospace',
    ax=ax
)

# Title and legend
ax.set_title('Stock Lead-Lag Network\nArrow: Leader → Lagger | Size: Influence | Color: Net Leadership',
             fontsize=18, fontweight='bold', color='white', pad=20)

# Create legend
legend_elements = [
    mpatches.Patch(facecolor='#60a5fa', alpha=0.6, label='1-day lag'),
    mpatches.Patch(facecolor='#a78bfa', alpha=0.6, label='2-3 day lag'),
    mpatches.Patch(facecolor='#f472b6', alpha=0.6, label='4-5 day lag'),
    plt.Line2D([0], [0], color='white', linewidth=3, alpha=0.8, label='High correlation'),
    plt.Line2D([0], [0], color='white', linewidth=1, alpha=0.4, label='Low correlation'),
]

legend1 = ax.legend(handles=legend_elements, 
                   loc='upper left',
                   framealpha=0.9,
                   facecolor='#1a1d3a',
                   edgecolor='white',
                   fontsize=10,
                   labelcolor='white')

# Add colorbar for node colors
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, 
                           norm=plt.Normalize(vmin=-5, vmax=5))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Net Leadership (Leaders - Laggers)', 
               rotation=270, labelpad=25, color='white', fontsize=11)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

ax.axis('off')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/full_network_graph.png', 
            dpi=300, bbox_inches='tight', facecolor='#0a0e17')
print("\n✓ Saved: full_network_graph.png")


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 2: TOP INFLUENCERS ONLY (Stocks with highest out-degree)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "─"*80)
print("GRAPH 2: Top Market Leaders (stocks that lead the most others)")
print("─"*80)

# Get top leaders
top_leaders = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:8]
top_leader_stocks = [stock for stock, _ in top_leaders]

print(f"Focusing on top {len(top_leader_stocks)} leaders:")
for stock, degree in top_leaders:
    print(f"  {stock}: leads {degree} stocks")

# Create subgraph with only these leaders and their direct connections
nodes_to_include = set(top_leader_stocks)
for leader in top_leader_stocks:
    nodes_to_include.update(G.successors(leader))  # Add all stocks they lead

G2 = G.subgraph(nodes_to_include).copy()

print(f"\nSubgraph: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")

fig2, ax2 = plt.subplots(figsize=(18, 14), facecolor='#0a0e17')
ax2.set_facecolor('#0a0e17')

# Use hierarchical layout
pos2 = nx.spring_layout(G2, k=3, iterations=50, seed=42)

# Separate node colors: leaders vs laggers
node_colors2 = ['#60a5fa' if node in top_leader_stocks else '#94a3b8' 
                for node in G2.nodes()]
node_sizes2 = [800 if node in top_leader_stocks else 400 for node in G2.nodes()]

# Draw edges
edges2 = G2.edges()
weights2 = [G2[u][v]['weight'] for u, v in edges2]
edge_widths2 = [0.5 + w * 3 for w in weights2]

for (u, v), width, weight in zip(edges2, edge_widths2, weights2):
    ax2.annotate('',
                xy=pos2[v], xycoords='data',
                xytext=pos2[u], textcoords='data',
                arrowprops=dict(
                    arrowstyle='-|>',
                    color='#60a5fa',
                    lw=width,
                    alpha=0.4 + weight * 0.4,
                    connectionstyle='arc3,rad=0.1',
                    mutation_scale=15
                ))

# Draw nodes
nx.draw_networkx_nodes(
    G2, pos2,
    node_size=node_sizes2,
    node_color=node_colors2,
    edgecolors='white',
    linewidths=2,
    ax=ax2
)

# Draw labels with different sizes
for node in G2.nodes():
    x, y = pos2[node]
    fontsize = 12 if node in top_leader_stocks else 9
    fontweight = 'bold' if node in top_leader_stocks else 'normal'
    ax2.text(x, y, node, 
             fontsize=fontsize,
             fontweight=fontweight,
             color='white',
             ha='center',
             va='center',
             family='monospace')

ax2.set_title('Top Market Leaders and Their Influence\nBlue = Leader | Gray = Follower',
             fontsize=16, fontweight='bold', color='white', pad=20)

ax2.axis('off')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/top_leaders_graph.png', 
            dpi=300, bbox_inches='tight', facecolor='#0a0e17')
print("\n✓ Saved: top_leaders_graph.png")


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 3: SECTOR ANALYSIS (Color by sector if we can infer from tickers)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "─"*80)
print("GRAPH 3: Circular Layout by Correlation Strength")
print("─"*80)

# Use only strongest relationships for clarity
strong_threshold = 0.5
df_strong = df[(df['Correlation'] >= strong_threshold) & (df['Best_Lag'] > 0)].copy()

print(f"Showing {len(df_strong)} strongest relationships (correlation ≥ {strong_threshold})")

G3 = nx.DiGraph()
for _, row in df_strong.iterrows():
    G3.add_edge(row['Leader'], row['Lagger'],
                weight=row['Correlation'],
                lag=row['Best_Lag'])

print(f"Nodes: {G3.number_of_nodes()}, Edges: {G3.number_of_edges()}")

fig3, ax3 = plt.subplots(figsize=(16, 16), facecolor='#0a0e17')
ax3.set_facecolor('#0a0e17')

# Circular layout
pos3 = nx.circular_layout(G3)

# Node metrics
in_deg3 = dict(G3.in_degree())
out_deg3 = dict(G3.out_degree())
node_sizes3 = [400 + (in_deg3.get(node, 0) + out_deg3.get(node, 0)) * 200 
               for node in G3.nodes()]

# Draw edges with gradient by lag
edges3 = G3.edges()
for u, v in edges3:
    weight = G3[u][v]['weight']
    lag = G3[u][v]['lag']
    
    # Color gradient from blue to pink based on lag
    lag_normalized = (lag - 1) / 4  # Normalize 1-5 to 0-1
    color = plt.cm.cool(lag_normalized)
    
    ax3.annotate('',
                xy=pos3[v], xycoords='data',
                xytext=pos3[u], textcoords='data',
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=color,
                    lw=1 + weight * 3,
                    alpha=0.5 + weight * 0.3,
                    connectionstyle='arc3,rad=0.2',
                    mutation_scale=15
                ))

# Draw nodes
nx.draw_networkx_nodes(
    G3, pos3,
    node_size=node_sizes3,
    node_color='#60a5fa',
    edgecolors='white',
    linewidths=2,
    ax=ax3
)

# Draw labels
nx.draw_networkx_labels(
    G3, pos3,
    font_size=11,
    font_weight='bold',
    font_color='white',
    font_family='monospace',
    ax=ax3
)

ax3.set_title(f'High-Correlation Network (r ≥ {strong_threshold})\nCircular Layout',
             fontsize=16, fontweight='bold', color='white', pad=20)

# Gradient colorbar for lag
sm3 = plt.cm.ScalarMappable(cmap=plt.cm.cool, 
                            norm=plt.Normalize(vmin=1, vmax=5))
sm3.set_array([])
cbar3 = plt.colorbar(sm3, ax=ax3, fraction=0.03, pad=0.02)
cbar3.set_label('Lag (days)', rotation=270, labelpad=20, color='white', fontsize=11)
cbar3.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color='white')

ax3.axis('off')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/circular_network_graph.png', 
            dpi=300, bbox_inches='tight', facecolor='#0a0e17')
print("\n✓ Saved: circular_network_graph.png")


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 4: LAG-SPECIFIC NETWORKS (Separate graph for each lag time)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "─"*80)
print("GRAPH 4: Lag-Specific Networks (1-day, 2-day, 3+ day lags)")
print("─"*80)

fig4, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='#0a0e17')

lag_ranges = [
    (1, 1, '1-Day Lead'),
    (2, 2, '2-Day Lead'),
    (3, 5, '3-5 Day Lead')
]

for idx, (lag_min, lag_max, title) in enumerate(lag_ranges):
    ax = axes[idx]
    ax.set_facecolor('#0a0e17')
    
    # Filter by lag range
    df_lag = df[(df['Best_Lag'] >= lag_min) & 
                (df['Best_Lag'] <= lag_max) & 
                (df['Correlation'] >= 0.4)].copy()
    
    print(f"\n{title}: {len(df_lag)} relationships")
    
    G_lag = nx.DiGraph()
    for _, row in df_lag.iterrows():
        G_lag.add_edge(row['Leader'], row['Lagger'], weight=row['Correlation'])
    
    if G_lag.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'No relationships in this range',
                ha='center', va='center', color='white', fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold', color='white')
        ax.axis('off')
        continue
    
    print(f"  Nodes: {G_lag.number_of_nodes()}, Edges: {G_lag.number_of_edges()}")
    
    pos_lag = nx.spring_layout(G_lag, k=2, iterations=50, seed=42)
    
    # Node sizes
    degrees = dict(G_lag.degree())
    node_sizes_lag = [300 + degrees[node] * 150 for node in G_lag.nodes()]
    
    # Draw edges
    edges_lag = G_lag.edges()
    weights_lag = [G_lag[u][v]['weight'] for u, v in edges_lag]
    
    for (u, v), weight in zip(edges_lag, weights_lag):
        ax.annotate('',
                   xy=pos_lag[v], xycoords='data',
                   xytext=pos_lag[u], textcoords='data',
                   arrowprops=dict(
                       arrowstyle='-|>',
                       color='#60a5fa',
                       lw=0.5 + weight * 2.5,
                       alpha=0.3 + weight * 0.4,
                       mutation_scale=12
                   ))
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G_lag, pos_lag,
        node_size=node_sizes_lag,
        node_color='#60a5fa',
        edgecolors='white',
        linewidths=1.5,
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G_lag, pos_lag,
        font_size=9,
        font_weight='bold',
        font_color='white',
        font_family='monospace',
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
    ax.axis('off')

plt.suptitle('Lead-Lag Networks by Time Horizon',
            fontsize=18, fontweight='bold', color='white', y=0.98)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/lag_specific_networks.png', 
            dpi=300, bbox_inches='tight', facecolor='#0a0e17')
print("\n✓ Saved: lag_specific_networks.png")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("NETWORK ANALYSIS SUMMARY")
print("="*80)

print(f"\nGraph Density: {nx.density(G):.3f}")
print(f"  (0 = no connections, 1 = fully connected)")

if nx.is_weakly_connected(G):
    print("\nGraph Connectivity: Weakly connected")
    print("  All stocks are connected through some path")
else:
    components = list(nx.weakly_connected_components(G))
    print(f"\nGraph Connectivity: {len(components)} separate components")
    print(f"  Largest component: {len(max(components, key=len))} stocks")

# Average path length (for largest component - use undirected version)
largest_component = max(nx.weakly_connected_components(G), key=len)
G_component = G.subgraph(largest_component)
# Convert to undirected to calculate average path
G_undirected = G_component.to_undirected()
avg_path = nx.average_shortest_path_length(G_undirected)
print(f"\nAverage Shortest Path: {avg_path:.2f} steps")
print(f"  Information travels through ~{avg_path:.1f} stocks on average")

# Reciprocity (how many relationships are bidirectional)
reciprocity = nx.reciprocity(G)
print(f"\nReciprocity: {reciprocity:.3f}")
print(f"  {reciprocity*100:.1f}% of relationships have a reverse relationship")

print("\n" + "="*80)
print("All graphs saved to /mnt/user-data/outputs/")
print("="*80)
print("\nGenerated files:")
print("  • full_network_graph.png - Complete network with all relationships")
print("  • top_leaders_graph.png - Focus on most influential stocks")
print("  • circular_network_graph.png - High-correlation pairs in circular layout")
print("  • lag_specific_networks.png - Separate networks by lag time")
print("\n")
