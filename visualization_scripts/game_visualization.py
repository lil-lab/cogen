"""
A simple way to visualize games

Written with help from a cheat sheet created by 
    @daniellewisDL : https://github.com/daniellewisDL
"""
import pickle
import streamlit as st
from pathlib import Path
import base64
from PIL import Image, ImageOps
import numpy as np
import os
import json


DEPLOYMENT_ROUNDS = ["", "all", "1", "2", "3", "4"]
TREATMENTS = ['full', 'no_ji', 'no_ds', 'baseline', 'old_full']
INTERACTION_DATA_DIR = "/home/mog29/cogen/data_and_checkpoints/interaction"
IMAGE_DIR = "/home/mog29/cogen/data_and_checkpoints/kilogram/dataset/square-black-imgs"

st.set_page_config(
    page_title="Reference Game Visualizations",
    layout="wide",
    initial_sidebar_state="auto",
)

### Utility ###

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def svg2png(lst):
    return [t + ".png" for t in lst]

def make_buttons(show_back=True):
    """
    Makes the two buttons at the top of the screen.
    [show_back] is only false when the input player/game doesn't exist
    """
    cols = st.columns([1, 11, 2], gap="small")
    if show_back:
        cols[0].button("Back", on_click=go_back, key="back1")
    if st.session_state.home == "game":
        cols[2].button("Back to home", on_click=go_game_home, key="go game home")
    elif st.session_state.home == "player":
        cols[2].button("Back to home", on_click=go_player_home, key="go player home")

def view_game(game_id, game_info, curr=["game_sum"]):
    """
    Button callback for viewing a game with id [game_id]
    curr denotes what is currently displayed, and curr will be saved to history
     - defaults to game summary
     - if a game, then the first element is "spec_game", followed by [game_id, game_info]

    Precondition: curr must be a non-empty list
    """
    st.session_state.display = "spec_game"
    st.session_state.game_id = game_id
    st.session_state.game_info = game_info
    st.session_state.justClicked = True
    st.session_state.history.append(curr)

def go_back():
    """
    Handles going back whenever called (only called by a button)
    """
    # remove the last element of history and return it
    back_one = st.session_state.history.pop()

    if back_one[0] == "game_sum":
        go_game_home()
    elif back_one[0] == "spec_game":
        # just came from a specific game
        st.session_state.display = "spec_game"
        st.session_state.game_id = back_one[1]
        st.session_state.game_info = back_one[2]
        st.session_state.justClicked = True

def go_game_home():
    """
    Sets display to game summary; resets everything else
    """
    st.session_state.game_id = ""
    st.session_state.game_info = {}
    st.session_state.id = ""
    st.session_state.player_id = ""
    st.session_state.display = "game_sum"
    st.session_state.history = []  # clear history

def make_cols(col_list, col_titles, left_orient, txt_size):
    """
    Gives the columns in col_list titles according to col_titles.
    The titles in left_orient will be left oriented
    """
    for i in range(len(col_list)):
        with col_list[i]:
            if col_titles[i] in left_orient:
                st.write(
                    f'<div style="display: flex; justify-content: left; ">'
                    f'<span style="font-size:{txt_size}px;font-weight:bold">{col_titles[i]}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.write(
                    f'<div style="display: flex; justify-content: center; ">'
                    f'<span style="font-size:{txt_size}px;font-weight:bold">{col_titles[i]}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )

def col_write(col, content, display="flex", orient="center", txt_size=18, color=0):
    """
    Uses markdown to write lines, specifically in columns, according to params
    """
    col.write(
        f'<div style="display: {display}; justify-content: {orient}; ">'
        f'<span style="font-size:{txt_size}px; color:{color}">{content}</span>'
        "</div>",
        unsafe_allow_html=True,
    )

def add_new_states():
    """
    Adds display, game_info, justClicked, history, and home into state
    display is for display purposes (which screens to display)
    game_info and game_id are for figuring out what to display
    home: keeps track of what the "Back to home" button brings you back to.
    """
    if "display" not in st.session_state:
        st.session_state.display = "intro"
    if "game_info" not in st.session_state:
        st.session_state.game_info = {}
    if "justClicked" not in st.session_state:
        st.session_state.justClicked = False
    if "history" not in st.session_state:
        st.session_state.history = []  # start as an empty list
    if "home" not in st.session_state:
        st.session_state.home = "intro"
        
def get_message_length(chat):
    tokens = word_tokenize(chat)
    return len(tokens)

@st.cache_data
def get_all_game_data():
    all_games = {}
    for deployment_round in ["1", "2", "3", "4"]:
        all_games[deployment_round] = {}
        for treatment in ['human_model', 'human_human']:
            # Load the json
            json_path = os.path.join(INTERACTION_DATA_DIR, f'r{deployment_round}_{treatment}_interactions.json')
            with open(json_path, 'r') as f:
                interaction_data = json.load(f)

            for game_id, round_dicts in interaction_data.items():
                players = get_players(treatment, round_dicts)
                accuracy = get_accuracy(round_dicts)
                new_game_id = f'r{deployment_round}_{treatment}_{game_id}'
                all_games[deployment_round][new_game_id] = {
                    'players' : players,
                    'accuracy' : accuracy,
                    'round_dicts' : round_dicts,
                    'treatment' : treatment
                }

    return all_games

def get_players(treatment, round_dicts):
    players = []
    if treatment == 'human_model':
        players.append('bots')

    for curr_round in round_dicts:
        if curr_round['speaker'] not in TREATMENTS and curr_round['speaker'] not in players:
            players.append(f'worker_{curr_round["speaker"]}')
        if curr_round['listener'] not in TREATMENTS and curr_round['listener'] not in players:
            players.append(f'worker_{curr_round["listener"]}')

    return list(players)

def get_accuracy(round_dicts):
    num_rounds = len(round_dicts)
    num_correct = 0
    for curr_round in round_dicts:
        num_correct += 1 if curr_round['reward'] == 1 else 0
    return num_correct / num_rounds * 100

def get_game_from_all(all_games, game_id):
    for _, anno_dict in all_games.items():
        for curr_id, game_dict in anno_dict.items():
            if curr_id == game_id:
                return game_dict
    return ""

def filter_games(filter, all_games):
    # First filter based on annotation
    filtered_games = deployment_round_filter(filter["deployment_round"], all_games) 
    
    # Next filter based on all other parameters
    final_filtered_games = {}
    for game_id, game_dict in filtered_games.items():
        # Filter based on treatment (human-model vs human-human)
        if game_dict["treatment"] not in filter["treatment"]:
            continue
        final_filtered_games[game_id] = game_dict

    return final_filtered_games

def deployment_round_filter(depl_round, all_games):
    filtered_games = {}
    for curr_round, anno_dict in all_games.items():
        if depl_round != "all" and curr_round != depl_round:
            continue

        for game_id, game_dict in anno_dict.items():
            filtered_games[game_id] = game_dict

    return filtered_games

### Set ###


def set_game_id(id, game_info):
    """
    Trying to do stateful button
    """
    st.session_state.game_id = id
    st.session_state.game_info = game_info
    if id == "":
        st.session_state.id = ""

def back_to_game(id, game_info):
    st.session_state.game_id = id
    st.session_state.game_info = game_info

### Display ###

def display_no_game():
    make_buttons(False)
    st.title("This game doesn't exist.")
    image1 = Image.open("tangram-human-color.png")
    image1 = image1.resize((50, 50))
    st.image([image1] * 13)

def display_no_player():
    make_buttons(False)
    st.title("This player doesn't exist.")
    image1 = Image.open("tangram-human-color.png")
    image1 = image1.resize((50, 50))
    st.image([image1] * 13)

def display_title():
    """
    Displays the title screen
    """
    st.title("Please select a game annotation")
    image1 = Image.open("tangram-human-color.png")
    image1 = image1.resize((50, 50))
    st.image([image1] * 13)


def display_game_summary(filter, all_games):
    """
    Displays the games according the filter, which is a dictionary
    Returns the game id of a game we want to see

    [treatment, accuracy, playerA, playerB]
    """
    # create columns:

    cols = st.columns(spec=[2, 1, 1, 1, 1], gap="medium")
    header_lst = [
        "Game Ids",
        "Treatment",
        "Accuracy",
        "Player A",
        "Player B"
    ]
    make_cols(cols, header_lst, ["Game Ids"], 18)
    games = filter_games(filter, all_games)

    accs = []
    for game_id, game_dict in games.items():
        # new columns for alignment purposes
        cols = st.columns(spec=[2, 1, 1, 1, 1], gap="medium")

        # Write remaining columns
        cols[0].button(game_id, on_click=view_game, args=[game_id, game_dict])
        col_write(cols[1], game_dict["treatment"])
        col_write(cols[2], f"{game_dict['accuracy']:.2f}") 
        col_write(cols[3], game_dict["players"][0]) 
        col_write(cols[4], game_dict["players"][1])
        accs.append(game_dict['accuracy'])

    # Add average values here
    cols = st.columns(spec=[2, 1, 1, 1, 1], gap="medium")
    col_write(cols[0], "Average", txt_size=18, orient="left")
    col_write(cols[2], f"{np.mean(accs):.2f}")
    st.write(f"{len(games)} games played")

def display_game(id, game_info):
    """
    Displays game information for game id:
    """
    make_buttons()
    st.header("Game ID: " + id)

    s, l = st.columns(spec=[1, 1], gap="medium")
    s.markdown(f"**Treatment:** {game_info['treatment']}")
    l.markdown(f"**Accuracy:** {game_info['accuracy']:.2f}")

    s.markdown(f"**Player A:** {game_info['players'][0]}")
    l.markdown(f"**Player B:** {game_info['players'][1]}")

    st.divider()
    for curr_round in game_info['round_dicts']:
        st.markdown(f"### Round {curr_round['interaction_round'] + 1}")

        s, l = st.columns(spec=[1, 1], gap="medium")
        speaker = f'worker_{curr_round["speaker"]}' if curr_round['speaker'] not in TREATMENTS else curr_round['speaker']
        listener = f'worker_{curr_round["listener"]}' if curr_round['listener'] not in TREATMENTS else curr_round['listener'] 
        s.markdown(f"**Speaker:** {speaker}")
        l.markdown(f"**Listener:** {listener}")
        
        chat = curr_round['chat']
        st.write(
            f'<span style="font-size: 20px; line-height:1">{chat}',
            unsafe_allow_html=True,
        )

        targets = svg2png([curr_round["gt_target"]])
        context = svg2png(curr_round['speaker_context'])
        clicks = svg2png([curr_round["selection"]])

        display_context(context, targets, clicks)

    st.divider()

def display_context(context, targets, clicks=[]):
    """
    Displays the context with targets and clicks
    """
    # first get all of the tangrams showing correctly
    tangram_list = []
    for img in context:
        image = Image.open(os.path.join(IMAGE_DIR, img)).resize((60, 60)).convert("RGB")
        image = ImageOps.expand(image, border=2, fill="white")
        if img in targets and img in clicks:  # listener selected a target image
            image = ImageOps.expand(image, border=10, fill="green")
        elif img in targets and img not in clicks:  # unselected target:
            image = ImageOps.expand(image, border=10, fill="black")
        elif img in clicks and img not in targets:  # listener selected a wrong image
            image = ImageOps.expand(image, border=10, fill="red")
        else:
            image = ImageOps.expand(image, border=10, fill="white")
        image = ImageOps.expand(image, border=2, fill="white")
        tangram_list.append(image)

    st.image(tangram_list[:10])
    st.image(tangram_list[10:])


### Sidebar ###


def sidebar(filter):
    # title
    st.sidebar.markdown(
        """<img src='data:image/png;base64,{}' class='img-fluid' width=120 height=120>""".format(
            img_to_bytes("tangram-human-color.png")
        ),
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("# Reference Game Visualization")
    filter["deployment_round"] = st.sidebar.radio("Please select a deployment round", DEPLOYMENT_ROUNDS)

    if filter["deployment_round"] != "":
        st.session_state.home = "game"
        filter["game_id"] = st.sidebar.text_input(
            "Game Id", "", max_chars=32, key="id"
        )
        if filter["game_id"] != "":
            st.session_state.display = "spec_game"
        else:
            st.session_state.display = "game_sum"

    # Treatment
    filter["treatment"] = st.sidebar.multiselect(
        "Context treatment: ",
        ["human_model", "human_human"],
    )

    return filter

def main():
    """
    Runs every time a click is made
    """
    add_new_states()
    all_games = get_all_game_data()
    filter = {"deployment_round": "", "game_id" : ""}
    new_filter = sidebar(filter)

    if new_filter["deployment_round"] == "" or st.session_state.display == "intro":
        display_title()
        return None

    if new_filter["game_id"] != "":
        if get_game_from_all(all_games, new_filter["game_id"]) == "":
            st.session_state.history.append(["game_sum"])
            display_no_game()
            return None

    if new_filter["game_id"] != "":
        st.session_state.game_id = new_filter["game_id"]
        # if we want to ga back, it would make sense to just go to summary
        st.session_state.history.append(["game_sum"])
        st.session_state.game_info = get_game_from_all(all_games, new_filter["game_id"])
    elif st.session_state.justClicked:
        st.session_state.display = "spec_game"
        st.session_state.justClicked = False

    # display as necessary
    if st.session_state.display == "intro" or new_filter["deployment_round"] == "":
        display_title()
    if st.session_state.display == "game_sum" and new_filter["deployment_round"] != "":
        display_game_summary(new_filter, all_games)
    elif st.session_state.display == "spec_game":
        display_game(st.session_state.game_id, st.session_state.game_info) 

    return None


if __name__ == "__main__":
    main()
