import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from shapely.geometry import LineString
import copy
from shapely.geometry import Polygon, Point
import matplotlib.colors as mcolors

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

def menger_curvature(pt1, pt2, pt3, atol=1e-3):
    vec21 = np.array(pt1) - np.array(pt2)
    vec23 = np.array(pt3) - np.array(pt2)
    norm21 = np.linalg.norm(vec21)
    norm23 = np.linalg.norm(vec23)
    theta = np.arccos(np.dot(vec21, vec23) / (norm21 * norm23))
    if np.isclose(theta - np.pi, 0.0, atol=atol):
        theta = 0.0
    dist13 = np.linalg.norm(np.array(pt1) - np.array(pt3))
    return 2 * np.sin(theta) / dist13 if dist13 != 0 else 0


def improve_race_line(old_line, inner_border, outer_border):
    '''Use gradient descent, inspired by K1999, to find the racing line'''
    # start with the center line
    new_line = copy.deepcopy(old_line)
    ls_inner_border = Polygon(inner_border)
    ls_outer_border = Polygon(outer_border)
    for i in range(0,len(new_line)):
        xi = new_line[i]
        npoints = len(new_line)
        prevprev = (i - 2 + npoints) % npoints
        prev = (i - 1 + npoints) % npoints
        nexxt = (i + 1 + npoints) % npoints
        nexxtnexxt = (i + 2 + npoints) % npoints
        #print("%d: %d %d %d %d %d" % (npoints, prevprev, prev, i, nexxt, nexxtnexxt))
        ci = menger_curvature(new_line[prev], xi, new_line[nexxt])
        c1 = menger_curvature(new_line[prevprev], new_line[prev], xi)
        c2 = menger_curvature(xi, new_line[nexxt], new_line[nexxtnexxt])
        target_ci = (c1 + c2) / 2
        #print("i %d ci %f target_ci %f c1 %f c2 %f" % (i, ci, target_ci, c1, c2))

        # Calculate prospective new track position, start at half-way (curvature zero)
        xi_bound1 = copy.deepcopy(xi)
        xi_bound2 = ((new_line[nexxt][0] + new_line[prev][0]) / 2.0, (new_line[nexxt][1] + new_line[prev][1]) / 2.0)
        p_xi = copy.deepcopy(xi)
        for j in range(0,XI_ITERATIONS):
            p_ci = menger_curvature(new_line[prev], p_xi, new_line[nexxt])
            #print("i: {} iter {} p_ci {} p_xi {} b1 {} b2 {}".format(i,j,p_ci,p_xi,xi_bound1, xi_bound2))
            if np.isclose(p_ci, target_ci):
                break
            if p_ci < target_ci:
                # too flat, shrinking track too much
                xi_bound2 = copy.deepcopy(p_xi)
                new_p_xi = ((xi_bound1[0] + p_xi[0]) / 2.0, (xi_bound1[1] + p_xi[1]) / 2.0)
                if Point(new_p_xi).within(ls_inner_border) or not Point(new_p_xi).within(ls_outer_border):
                    xi_bound1 = copy.deepcopy(new_p_xi)
                else:
                    p_xi = new_p_xi
            else:
                # too curved, flatten it out
                xi_bound1 = copy.deepcopy(p_xi)
                new_p_xi = ((xi_bound2[0] + p_xi[0]) / 2.0, (xi_bound2[1] + p_xi[1]) / 2.0)

                # If iteration pushes the point beyond the border of the track,
                # just abandon the refinement at this point.  As adjacent
                # points are adjusted within the track the point should gradually
                # make its way to a new position.  A better way would be to use
                # a projection of the point on the border as the new bound.  Later.
                if Point(new_p_xi).within(ls_inner_border) or not Point(new_p_xi).within(ls_outer_border):
                    xi_bound2 = copy.deepcopy(new_p_xi)
                else:
                    p_xi = new_p_xi
        new_xi = p_xi
        # New point which has mid-curvature of prev and next points but may be outside of track
        #print((new_line[i], new_xi))
        new_line[i] = new_xi
    return new_line

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

def create_download_link(loop_race_line):
    """Create a download link for the numpy array."""
    buffer = BytesIO()
    np.save(buffer, loop_race_line)
    buffer.seek(0)
    return buffer

def circle_radius(coords):
    x1, y1, x2, y2, x3, y3 = [i for sub in coords for i in sub]
    a = x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2
    b = (x1**2+y1**2)*(y3-y2) + (x2**2+y2**2)*(y1-y3) + (x3**2+y3**2)*(y2-y1)
    c = (x1**2+y1**2)*(x2-x3) + (x2**2+y2**2)*(x3-x1) + (x3**2+y3**2)*(x1-x2)
    d = (x1**2+y1**2)*(x3*y2-x2*y3) + (x2**2+y2**2) * (x1*y3-x3*y1) + (x3**2+y3**2)*(x2*y1-x1*y2)
    try:
        r = abs((b**2+c**2-4*a*d) / abs(4*a**2)) ** 0.5
    except:
        r = 999
    return r

def circle_indexes(mylist, index_car, add_index_1=0, add_index_2=0):
    list_len = len(mylist)
    index_1 = (index_car + add_index_1) % list_len
    index_2 = (index_car + add_index_2) % list_len
    return [index_car, index_1, index_2]

def optimal_velocity(track, min_speed, max_speed, look_ahead_points):
    radius = []
    for i in range(len(track)):
        indexes = circle_indexes(track, i, add_index_1=-1, add_index_2=1)
        coords = [track[indexes[0]], track[indexes[1]], track[indexes[2]]]
        radius.append(circle_radius(coords))
    v_min_r = min(radius)**0.5
    constant_multiple = min_speed / v_min_r
    if look_ahead_points == 0:
        max_velocity = [(constant_multiple * i**0.5) for i in radius]
        velocity = [min(v, max_speed) for v in max_velocity]
        return velocity
    else:
        LOOK_AHEAD_POINTS = look_ahead_points
        radius_lookahead = []
        for i in range(len(radius)):
            next_n_radius = []
            for j in range(LOOK_AHEAD_POINTS+1):
                index = circle_indexes(mylist=radius, index_car=i, add_index_1=j)[1]
                next_n_radius.append(radius[index])
            radius_lookahead.append(min(next_n_radius))
        max_velocity_lookahead = [(constant_multiple * i**0.5) for i in radius_lookahead]
        velocity_lookahead = [min(v, max_speed) for v in max_velocity_lookahead]
        return velocity_lookahead
    
def dist_2_points(x1, x2, y1, y2):
    return abs(abs(x1-x2)**2 + abs(y1-y2)**2)**0.5

#####################################################################
st.title('AWS DeepRacer Race Track Visualization')
st.markdown("- Note: Be sure keeeing your computer in NOT SLEEP MODE ") 
st.markdown("- This web app is for calculating and visualizing AWS DeepRacer Optimal Race Line.") 
st.markdown("- The code source is base on the https://github.com/dgnzlz/Capstone_AWS_DeepRacer/tree/master")


# Ensure session state variables are initialized
if 'waypoints' not in st.session_state:
    st.session_state.waypoints = None

# Choose the source of the track file
option = st.selectbox("Choose the source of the track file:", ["Upload File", "GitHub"])

if option == "Upload File":
    uploaded_file = st.file_uploader("Upload your track file (.npy)", type="npy")
    if uploaded_file is not None:
        st.session_state.waypoints = np.load(uploaded_file, allow_pickle=True)
elif option == "GitHub":
    selected_track = st.selectbox("Select a track", tracks)
    if st.button("Load Track from GitHub"):
        # Load the data and store it in session state
        st.session_state.waypoints = load_npy_from_url(f"{base_url}{selected_track}")

# Check if waypoints are loaded
if st.session_state['waypoints'] is not None:
    waypoints = st.session_state['waypoints']
    center_line = waypoints[:, 0:2]
    inner_border = waypoints[:, 2:4]
    outer_border = waypoints[:, 4:6]



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
    # Set default iteration values
    #LINE_ITERATIONS = 1000
    #XI_ITERATIONS = 3
    # Set default values and create sliders for dynamic adjustments
    st.write("## Choose your Hyperparameters:")
    st.markdown("- Number of Line Iterations: Number of times to scan the entire race track to iterate")
    st.markdown("- Xi Iterations: Number of times to iterate each new race line point")

    LINE_ITERATIONS = st.slider('Number of Line Iterations', min_value=100, max_value=2000, value=500, step=100)
    XI_ITERATIONS = st.slider('Xi Iterations', min_value=3, max_value=10, value=5)

    if st.button('Calculate Optimal Race Line'):
        race_line = copy.deepcopy(center_line[:-1])  # Start with a deep copy of the centerline

        # Initialize a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(LINE_ITERATIONS):
            race_line = improve_race_line(race_line, inner_border, outer_border)
            
            # Update progress bar and status text every 20 iterations
            if i % 20 == 0:
                progress_percentage = int(100 * (i / LINE_ITERATIONS))
                progress_bar.progress(progress_percentage)
                status_text.text(f"Computing... Iteration {i} of {LINE_ITERATIONS}")

        # Complete the progress
        progress_bar.progress(100)
        status_text.text("Calculation completed!")

        # Closing the loop to make the race line continuous
        loop_race_line = np.append(race_line, [race_line[0]], axis=0)

        # Display shapes and lengths
        original_length = LineString(center_line).length
        new_length = LineString(loop_race_line).length
        
        st.write(f"Original centerline length: {original_length:.2f}")
        st.write(f"New race line length: {new_length:.2f}")
        st.write("## This is your Optimal Race Line")

        # Plotting the track
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
        # Printing border and race line on the plot
        print_border(ax, loop_race_line, inner_border, outer_border)
        st.pyplot(fig)


        # Provide download button
        buffer = create_download_link(loop_race_line)
        st.download_button(
            label="Download Optimal Race Line as .npy",
            data=buffer,
            file_name="optimal_track.npy",
            mime="application/octet-stream"
        )

