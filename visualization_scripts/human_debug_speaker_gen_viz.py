"""
A simple way to visualize games

Written with help from a cheat sheet created by 
    @daniellewisDL : https://github.com/daniellewisDL
"""
import pickle
import json
import streamlit as st
from pathlib import Path
import base64
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import math

import numpy as np
import os

import pandas as pd
from datetime import datetime

from nltk.tokenize import word_tokenize

DATA_DIR = "/home/mog29/cogen/data_and_checkpoints/continual_learning"
GAME_TYPES = ["simple_isolated", "simple_similar", "hard_isolated", "hard_similar"]
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
    return [t[:-3] + "png" for t in lst]


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
     - if a player, then the first element is "spec_player", and the second element is the player's ID
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
    elif back_one[0] == "spec_player":
        st.session_state.game_id = ""
        st.session_state.player_id = back_one[1]
        st.session_state.id = ""
        st.session_state.display = "spec_player"
    elif back_one[0] == "player_sum":
        go_player_home()

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

def go_player_home():
    """
    Sets display to player summary; resets everything else
    """
    st.session_state.game_id = ""
    st.session_state.game_info = {}
    st.session_state.id = ""
    st.session_state.player_id = ""
    st.session_state.display = "player_sum"
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
    Adds display, game_info, justClicked, player_id, history, and home into state
    display is for display purposes (which screens to display)
    game_info, player_id, and game_id are for figuring out what to display
    justClicked: tbh not sure why it's needed, but without it, clicking on game buttons from game summary doesn't work
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

def get_game_from_all(game_dict, debug_preds, game_id):
    if game_id in game_dict:
        return game_dict[game_id], debug_preds[game_id]
    return ""

@st.cache_data
def get_debug_preds(base_folder, run_name, file_suffix):
    BASE_FOLDER = "/home/mog29/cogen/data_and_checkpoints/experiments/joint_training"
    save_path = os.path.join(BASE_FOLDER, base_folder, run_name, 'logging', f'human_debug_preds{file_suffix}_gen_preds.pth')
    return torch.load(save_path)

@st.cache_data
def get_all_game_data():
    # First get data for all negatives
    filename = "human_debugging_data.json"
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
            
    return data

def get_game_stats(curr_game, curr_gens):
    gt_lens = []
    model_lens = []

    for round_index, model_dict in curr_gens.items():
        # Start with the human values
        gt_len = get_message_length(curr_game[round_index]["chat"])
        gt_lens.append(gt_len)

        # Move on to the model values
        if "joint_scores" in model_dict:
            max_idx = torch.argmax(model_dict["joint_scores"]).item()
        else:
            max_idx = torch.argmax(model_dict["speaker_scores"]).item()            
        utterance = model_dict["utterances"][max_idx]
        model_len = get_message_length(utterance)
        model_lens.append(model_len)

    return np.mean(gt_lens), np.mean(model_lens)

## END NEW STUFF

### Set ###


def set_game_id(id, game_info):
    """
    Trying to do stateful button
    """
    st.session_state.game_id = id
    st.session_state.game_info = game_info
    if id == "":
        st.session_state.id = ""


def set_player(player_id, curr):
    """
    Brings you to the front page with all games of this player
    curr describes the current state; generally either player summary or a specific game
    """
    st.session_state.history.append(curr)
    st.session_state.game_id = ""
    st.session_state.player_id = player_id
    st.session_state.id = ""
    st.session_state.display = "spec_player"

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


def get_game_type(game_id):
    if "speaker" in game_id:
        return "neg"
    else:
        for difficulty in ["simple", "hard"]:
            for eval_type in ["isolated", "similar"]:
                start = f"{difficulty}_{eval_type}"
                if game_id.startswith(start):
                    return start

def display_game_summary(filter, game_dict, debug_preds):
    """
    Displays the games according the filter, which is a dictionary
    Returns the game id of a game we want to see
    """
    # create columns:

    cols = st.columns(spec=[1.5, 1], gap="medium")
    header_lst = [
        "Game Ids",
        "Game Type"
    ]
    make_cols(cols, header_lst, ["Game Ids"], 18)

    for game_id, curr_game_dict in game_dict.items():
        game_type = get_game_type(game_id)
        if game_type not in filter["valid_game_types"]:
            continue

        # new columns for alignment purposes
        cols = st.columns(spec=[1.5, 1], gap="medium")
        cols[0].button(game_id, on_click=view_game, args=[game_id, (curr_game_dict, debug_preds[game_id])])
        col_write(cols[1], game_type)

def display_game(id, game_info):
    game_dict, model_preds = game_info
    make_buttons()
    st.header("Game ID: " + id)
    st.divider()

    round_indices = sorted(list(model_preds.keys()), key=lambda x: x)
    for round_index in round_indices:
        model_dict = model_preds[round_index]
        round_dict = game_dict[round_index]

        # General info
        try:
            st.markdown(f"### Round {int(round_index) + 1}")
        except:
            st.markdown(f"### {round_index}")            
        targets = svg2png([round_dict["gt_target"] + ".svg"])
        context = svg2png([trg + ".svg" for trg in round_dict['speaker_context']])
        display_context(context, targets)

        # Display the human annotations
        display_model_gens(model_dict)

    st.divider()

def display_model_gens(model_dict):
    if "joint_scores" not in model_dict:
        st.write(
            f'<span style="font-size: 20px; line-height:1">Model outputs',
            unsafe_allow_html=True,
        )

        curr_utt = model_dict['utterances']
        disp_utt = f"Model output: {curr_utt}"
        st.write(
            f'<span style="font-size: 15px; line-height:1">{disp_utt}',
            unsafe_allow_html=True,
        )
    else:
        cols = st.columns(3)
        for col, ranking in zip(cols, ["listener_scores", "speaker_scores", "joint_scores"]):
            with col:
                st.write(
                    f'<span style="font-size: 20px; line-height:1">{ranking}',
                    unsafe_allow_html=True,
                )

                sorted_log, indices = torch.sort(model_dict[ranking], descending=True)
                sorted_prob = F.softmax(sorted_log.float())

                for i in range(len(indices)):
                    curr_prob = sorted_prob[i].item()
                    curr_idx = indices[i].item()
                    curr_utt = model_dict["utterances"][curr_idx]
                    disp_utt = f"Rank {i+1} with probability {curr_prob:.2f}: {curr_utt}"
                    st.write(
                        f'<span style="font-size: 15px; line-height:1">{disp_utt}',
                        unsafe_allow_html=True,
                    )

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

    st.image(tangram_list)


### Sidebar ###


def sidebar(filter):
    # title
    st.sidebar.markdown(
        """<img src='data:image/png;base64,{}' class='img-fluid' width=120 height=120>""".format(
            img_to_bytes("tangram-human-color.png")
        ),
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("# Speaker Model Visualization")

    # Can only view games
    filter["collection"] = "Game"
    st.session_state.home = "game"
    filter["game_id"] = st.sidebar.text_input(
        "Game Id", "", max_chars=32, key="id"
    )
    if filter["game_id"] != "":
        st.session_state.display = "spec_game"
    else:
        st.session_state.display = "game_sum"

    # Model loading arguments
    filter["base_folder"] = st.sidebar.text_input(
        "Name of the folder to load from: ", "", max_chars=32, key="base_folder"
    )
    
    filter["run_name"] = st.sidebar.text_input(
        "Name of the run: ", "", max_chars=64, key="run_name"
    )

    filter["file_suffix"] = st.sidebar.text_input(
        "File Suffix", "", max_chars=32, key="file_suffix"
    )

    filter["valid_game_types"] = st.sidebar.multiselect(
        "Which game_types: ",
        GAME_TYPES 
    )

    return filter

def main():
    """
    Runs every time a click is made
    """
    add_new_states()

    # Get the filter
    filter = {"game_id" : ""}
    new_filter = sidebar(filter)
    game_dict = get_all_game_data()
    debug_preds = get_debug_preds(new_filter["base_folder"], new_filter["run_name"], new_filter["file_suffix"])

    if new_filter["collection"] == "Game" and new_filter["game_id"] != "":
        if get_game_from_all(game_dict, model_gens, new_filter["game_id"]) == "":
            st.session_state.history.append(["game_sum"])
            display_no_game()
            return None

    if new_filter["game_id"] != "":
        st.session_state.game_id = new_filter["game_id"]
        st.session_state.history.append(["game_sum"])
        st.session_state.game_info = get_game_from_all(game_dict, debug_preds, new_filter["game_id"])
    elif st.session_state.justClicked:
        st.session_state.display = "spec_game"
        st.session_state.justClicked = False

    # display as necessary
    if st.session_state.display == "intro":
        display_title()
    if st.session_state.display == "game_sum":
        display_game_summary(new_filter, game_dict, debug_preds)
    elif st.session_state.display == "spec_game":
        display_game(st.session_state.game_id, st.session_state.game_info)

    return None


if __name__ == "__main__":
    main()
