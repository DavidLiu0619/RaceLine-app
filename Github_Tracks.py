import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from shapely.geometry import LineString

# Function to plot the coordinates
def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, '.', color='#999999', zorder=1)

# Function to plot lines
def plot_line(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, color='cyan', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

# Function to print the border and race line
def print_border(ax, waypoints, inner_border_waypoints, outer_border_waypoints):
    line = LineString(waypoints)
    plot_coords(ax, line)
    plot_line(ax, line)

    line = LineString(inner_border_waypoints)
    plot_coords(ax, line)
    plot_line(ax, line)

    line = LineString(outer_border_waypoints)
    plot_coords(ax, line)
    plot_line(ax, line)

# Define the URL structure for GitHub raw content
base_url = "https://raw.githubusercontent.com/aws-deepracer-community/deepracer-race-data/main/raw_data/tracks/npy/"
tracks = [
    "2022_april_open.npy", "2022_april_open_ccw.npy", "2022_april_open_cw.npy",
    "2022_april_pro.npy", "2022_april_pro_ccw.npy", "2022_april_pro_cw.npy",
    "2022_august_open.npy", "2022_august_open_ccw.npy", "2022_august_open_cw.npy",
    "2022_august_pro.npy", "2022_august_pro_ccw.npy", "2022_august_pro_cw.npy",
    "2022_july_open.npy", "2022_july_pro.npy", "2022_july_pro_ccw.npy", "2022_july_pro_cw.npy",
    "2022_june_open.npy", "2022_june_open_ccw.npy", "2022_june_open_cw.npy",
    "2022_june_pro.npy", "2022_june_pro_ccw.npy", "2022_june_pro_cw.npy",
    "2022_march_open.npy", "2022_march_open_ccw.npy", "2022_march_open_cw.npy",
    "2022_march_pro.npy", "2022_march_pro_ccw.npy", "2022_march_pro_cw.npy",
    "2022_may_open.npy", "2022_may_open_ccw.npy", "2022_may_open_cw.npy",
    "2022_may_pro.npy", "2022_may_pro_ccw.npy", "2022_may_pro_cw.npy",
    "2022_october_open.npy", "2022_october_open_ccw.npy", "2022_october_open_cw.npy",
    "2022_october_pro.npy", "2022_october_pro_ccw.npy", "2022_october_pro_cw.npy",
    "2022_reinvent_champ.npy", "2022_reinvent_champ_ccw.npy", "2022_reinvent_champ_cw.npy",
    "2022_september_open.npy", "2022_september_open_ccw.npy", "2022_september_open_cw.npy",
    "2022_september_pro.npy", "2022_september_pro_ccw.npy", "2022_september_pro_cw.npy",
    "2022_summit_speedway.npy", "2022_summit_speedway_ccw.npy", "2022_summit_speedway_cw.npy", "2022_summit_speedway_mini.npy",
    "AWS_track.npy", "Albert.npy", "AmericasGeneratedInclStart.npy",
    "Aragon.npy", "Austin.npy", "Belille.npy",
    "Bowtie_track.npy", "Canada_Training.npy", "China_track.npy",
    "FS_June2020.npy", "H_track.npy", "July_2020.npy",
    "LGSWide.npy", "Mexico_track.npy", "Monaco.npy",
    "Monaco_building.npy", "New_York_Track.npy", "Oval_track.npy",
    "Singapore.npy", "Singapore_building.npy", "Singapore_f1.npy",
    "Spain_track.npy", "Spain_track_f1.npy", "Straight_track.npy",
    "Tokyo_Training_track.npy", "Vegas_track.npy", "Virtual_May19_Train_track.npy",
    "arctic_open.npy", "arctic_open_ccw.npy", "arctic_open_cw.npy",
    "arctic_pro.npy", "arctic_pro_ccw.npy", "arctic_pro_cw.npy",
    "caecer_gp.npy", "caecer_loop.npy", "dubai_open.npy",
    "dubai_open_ccw.npy", "dubai_open_cw.npy", "dubai_pro.npy",
    "hamption_open.npy", "hamption_pro.npy", "jyllandsringen_open.npy",
    "jyllandsringen_open_ccw.npy", "jyllandsringen_open_cw.npy", "jyllandsringen_pro.npy",
    "jyllandsringen_pro_ccw.npy", "jyllandsringen_pro_cw.npy", "morgan_open.npy",
    "morgan_pro.npy", "penbay_open.npy", "penbay_open_ccw.npy",
    "penbay_open_cw.npy", "penbay_pro.npy", "penbay_pro_ccw.npy",
    "penbay_pro_cw.npy", "reInvent2019_track.npy", "reInvent2019_track_ccw.npy",
    "reInvent2019_track_cw.npy", "reInvent2019_wide.npy", "reInvent2019_wide_ccw.npy",
    "reInvent2019_wide_cw.npy", "reInvent2019_wide_mirrored.npy", "red_star_open.npy",
    "red_star_pro.npy", "red_star_pro_ccw.npy", "red_star_pro_cw.npy",
    "reinvent_base.npy", "thunder_hill_open.npy", "thunder_hill_pro.npy",
    "thunder_hill_pro_ccw.npy", "thunder_hill_pro_cw.npy"
]


def load_npy_from_url(url):
    """Load .npy file from a URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return np.load(BytesIO(response.content), allow_pickle=True)
    else:
        st.error("Failed to load the file from URL.")
        return None

st.title('Race Track Visualization')

# Choose the source of the track file
option = st.selectbox("Choose the source of the track file:", ["Upload File", "GitHub"])

if option == "Upload File":
    uploaded_file = st.file_uploader("Upload your track file (.npy)", type="npy")
    if uploaded_file is not None:
        waypoints = np.load(uploaded_file, allow_pickle=True)
elif option == "GitHub":
    selected_track = st.selectbox("Select a track", tracks)
    if st.button("Load Track from GitHub"):
        waypoints = load_npy_from_url(f"{base_url}{selected_track}")

if 'waypoints' in locals():
    center_line = waypoints[:,0:2]
    inner_border = waypoints[:,2:4]
    outer_border = waypoints[:,4:6]

        # Plotting
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='black')
    ax.set_aspect('equal')
    ax.set_facecolor('black')  # Set the axes background color
    fig.patch.set_facecolor('black')  # Set the figure background color

    # Remove axis ticks
    ax.tick_params(axis='both', colors='white')  # Make ticks white

    # Set grid and labels with appropriate colors if necessary
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)  # Optional grid

    print_border(ax, center_line, inner_border, outer_border)
    
    # Use Streamlit's function to display the plot
    st.pyplot(fig)























