import os
import numpy as np
import plotly.graph_objects as go
import webbrowser
from scipy.ndimage import zoom, gaussian_filter

def generate_3d_topology(
    conf_matrix: np.ndarray, 
    accuracy: float, 
    labels: list[str], 
    model_title: str, 
    output_filename: str
) -> None:
    """
    Takes a 2D confusion matrix and generates a high-fidelity 3D interactive topology.
    """
    print(f"[SYSTEM] Interpolating Latent Space for {model_title}...")
    
    # 1. INTERPOLATION ENGINE
    scaling = 10
    z_upscaled = zoom(conf_matrix, scaling)
    z_smooth = gaussian_filter(z_upscaled, sigma=2.2)

    x_smooth = np.linspace(0, 9, z_smooth.shape[1])
    y_smooth = np.linspace(0, 9, z_smooth.shape[0])

    # 2. HIGH-FIDELITY INTERACTIVE RENDER
    matrix_green = '#00ff41'
    bg_color = '#0d0d0d'
    floor_level = -150  

    custom_data = np.empty((z_smooth.shape[0], z_smooth.shape[1], 3), dtype=object)

    for i in range(z_smooth.shape[0]):
        for j in range(z_smooth.shape[1]):
            actual_idx = max(0, min(9, int(round(y_smooth[i]))))
            pred_idx = max(0, min(9, int(round(x_smooth[j]))))
            
            custom_data[i, j, 0] = labels[pred_idx]
            custom_data[i, j, 1] = labels[actual_idx]
            custom_data[i, j, 2] = f"{z_smooth[i, j]:.1f}"

    surface_layer = go.Surface(
        z=z_smooth, x=x_smooth, y=y_smooth,
        colorscale='Jet',
        customdata=custom_data,  
        contours=dict(
            x=dict(show=True, color='rgba(0,0,0,0.5)', width=1), 
            y=dict(show=True, color='rgba(0,0,0,0.5)', width=1)
        ),
        lighting=dict(ambient=0.7, diffuse=0.7, specular=0.1, roughness=0.5),
        colorbar=dict(
            title=dict(text='Inference Density', font=dict(color=matrix_green, family='monospace', size=13)),
            tickfont=dict(color=matrix_green, family='monospace'),
            thickness=20, len=0.6, x=0.95
        ),
        hovertemplate=(
            "<b>Predicted:</b> %{customdata[0]}<br>"
            "<b>Actual:</b> %{customdata[1]}<br>"
            "<b>Density:</b> %{customdata[2]}<extra></extra>"
        )
    )

    floor_z = np.full_like(z_smooth, floor_level)
    heatmap_floor_layer = go.Surface(
        z=floor_z, x=x_smooth, y=y_smooth,
        surfacecolor=z_smooth, colorscale='Jet',
        showscale=False, opacity=0.6, hoverinfo='skip'       
    )

    fig = go.Figure(data=[surface_layer, heatmap_floor_layer])

    legend_html = (
        "&gt;&gt; TOPOLOGY LOGIC_<br>"
        "Peaks&nbsp;&nbsp;&nbsp;= High Confidence<br>"
        "Valleys = Low Error<br>"
        "Red&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= Logic Concentration<br>"
        "Blue&nbsp;&nbsp;&nbsp;&nbsp;= Statistical Sparsity"
    )

    fig.update_layout(
        title=dict(
            text=f'3D TOPOLOGICAL ANALYSIS: {model_title}<br><span style="font-size:16px">Calculated Accuracy: {accuracy:.2f}%</span>',
            font=dict(color=matrix_green, family='monospace', size=22),
            x=0.5, y=0.95
        ),
        template="plotly_dark",
        paper_bgcolor=bg_color,
        scene=dict(
            xaxis=dict(title='Predicted Category', ticktext=labels, tickvals=list(range(10)), color=matrix_green, gridcolor='#1a4d1a', backgroundcolor=bg_color),
            yaxis=dict(title='True Category', ticktext=labels, tickvals=list(range(10)), color=matrix_green, gridcolor='#1a4d1a', backgroundcolor=bg_color),
            zaxis=dict(title='Classification Frequency', color=matrix_green, gridcolor='#1a4d1a', backgroundcolor=bg_color, range=[floor_level, np.max(z_smooth)+50]),
            camera=dict(eye=dict(x=2.2, y=-1.5, z=1.2)) 
        ),
        annotations=[dict(
            text=legend_html, align='left',
            x=0.02, y=0.95, xref='paper', yref='paper', showarrow=False, 
            font=dict(color=matrix_green, family='monospace', size=12),
            bgcolor='rgba(13,13,13,0.8)', bordercolor=matrix_green, borderwidth=1, borderpad=8
        )],
        margin=dict(l=0, r=0, t=80, b=0)
    )

    # Export paths inside the 'results/topologies' folder
    html_output_path = f"results/topologies/{output_filename}.html"
    png_output_path = f"results/topologies/{output_filename}.png"

    fig.write_html(html_output_path, include_plotlyjs='cdn')
    fig.write_image(png_output_path, width=1200, height=800, scale=2)

    webbrowser.open('file://' + os.path.abspath(html_output_path))
    print(f"[SUCCESS] Interactive Matrix generated: {html_output_path}")
    print(f"[SUCCESS] Static PNG Preview generated: {png_output_path}\n")