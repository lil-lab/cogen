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
from pymongo import MongoClient
import os

import pandas as pd
from datetime import datetime

from nltk.tokenize import word_tokenize

ANNOTATIONS = ["", "all", "may_1", "may_7", "may_13", "may_17"]

SURVEY_RESPONSES = [
    "veryDissatisfied",
    "dissatisfied",
    "somewhatDissatisfied",
    "somewhatSatisfied",
    "satisfied",
    "verySatisfied",
]


# data base stuff
MONGO_URL = "fill_in"
MYCLIENT = MongoClient(MONGO_URL)
MYDB = MYCLIENT["TangramsCompGen"]
GAME_COLLECTION = MYDB["games"]
ROUND_COLLECTION = MYDB["rounds"]
PLAYERS_COLLECTION = MYDB["players"]
TREATMENT_COLLECTION = MYDB["treatments"]
FACTOR_COLLECTION = MYDB["factors"]
FACTOR_TYPE_COLLECTION = MYDB["factor_types"]

# pathing
TREATMENTS = ['full', 'no_ji', 'no_ds', 'baseline']
IMAGE_DIR = "/Users/mustafaomergul/Desktop/Cornell/Research/kilogram/kilogram/tangram_pngs"

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
    if "player_id" not in st.session_state:
        st.session_state.player_id = ""
    if "history" not in st.session_state:
        st.session_state.history = []  # start as an empty list
    if "home" not in st.session_state:
        st.session_state.home = "intro"


### Get from MONGODB ###


def extract_game_config(doc):
    return doc["data"]["configFile"]

def extract_game_status(doc):
    return "complete" if doc["status"] == "finished" else "premature"

def extract_treatment_info(doc):
    # Get the annotation
    annotation = doc["data"]["annotation"]

    # Get the treatment (bot setting)
    config_name = doc['data']['configFile']
    treatment_name = 'human_model' if 'human_model' in config_name else 'human_human'

    return treatment_name, annotation

def extract_game_performance(doc):
    time_spent = 0
    total_rounds = 0
    total_successes = 0
    idleRounds = 0

    game_rounds = ROUND_COLLECTION.find({"gameId" : doc["_id"]})
    game_rounds = sorted(game_rounds, key=lambda x: x["index"])

    for i, curr_round in enumerate(game_rounds):
        data = curr_round["data"]

        # Skip if incomplete
        if "roundStart" not in data or "roundEnd" not in data:
            continue

        roundStart = pd.to_datetime(data["roundStart"])
        roundEnd = pd.to_datetime(data["roundEnd"])
        time_spent += (roundEnd - roundStart).total_seconds()

        # Skip if attention check
        if is_attention_check(curr_round):
            continue

        # Check if idle
        if data["clickedTangram"] == "no_clicks":
            idleRounds += 1
        else:
            total_rounds += 1
            if data["clickedTangram"] == data["target"]:
                total_successes += 1

    return time_spent, total_successes / (total_rounds + 1e-8), idleRounds

def is_attention_check(round_dict):
    return "isAttnCheck" in round_dict["data"] and round_dict["data"]["isAttnCheck"]

def extract_player_info(doc, treatment_name):
    player_hashes = []
    feedbacks = []
    demographicInfos = []
    bonuses = []

    for player_id in doc["playerIds"]:
        player_doc = PLAYERS_COLLECTION.find({"_id": player_id})[0]
        player_data = player_doc["data"]

        if "bot" in player_doc:
            player_hashes.append(f"{treatment_name}")
            feedbacks.append("")
            demographicInfos.append({
                "english" : "",
                "languages" : "",
                "whereLearn" : ""
            })
            bonuses.append(0.6 + player_data["bonus"])
        else:
            # Get the player hash first
            player_hashes.append(player_doc["urlParams"]["workerId"])
            bonuses.append(0.6 + player_data["bonus"])

            # Get the feedback next
            if "surveyResponses" in player_data and player_data["surveyResponses"] != "didNotSubmit":
                feedbacks.append(player_data["surveyResponses"]["feedback"])
            elif "errorSurveyResponses" in player_data and player_data["errorSurveyResponses"] != "didNotSubmit":
                feedbacks.append(player_data["errorSurveyResponses"]["feedback"])
            else:
                feedbacks.append("No feedback")
            
            # Get the demographic info
            demographic_doc = PLAYERS_COLLECTION.find({"urlParams.workerId" : player_hashes[-1],
                                                       "data.surveyResponses.languages" : {
                                                           "$ne" : "",
                                                           "$exists" : True
                                                       }})
            try:
                demographic_doc = demographic_doc[0]
                demographicInfos.append({
                    "english" : demogprahic_doc["data"]["surveyResponses"]["english"],
                    "languages" : demogprahic_doc["data"]["surveyResponses"]["languages"],
                    "whereLearn" : demogprahic_doc["data"]["surveyResponses"]["whereLearn"],
                })
            except:
                demographicInfos.append({
                    "english" : "",
                    "languages" : "",
                    "whereLearn" : ""
                })

    return player_hashes, max(bonuses), feedbacks, demographicInfos

def extract_round_dict(doc, treatment_name):
    proc_game_rounds = []
    game_rounds = ROUND_COLLECTION.find({"gameId": doc["_id"]})
    game_rounds = sorted(game_rounds, key=lambda x: x["index"])

    msg_lens = []
    for i, curr_round in enumerate(game_rounds):
        # Skip the unnecessary rounds
        data = curr_round["data"]
        attn_check = is_attention_check(curr_round)
        if "roundStart" not in data or "roundEnd" not in data:
            continue
        proc_dict = {"index" : curr_round["index"], "round_id" : curr_round["_id"],
                     "game_id" : doc["_id"]}

        if "botTreatment" not in curr_round['data']:
            proc_dict['bot_treatment'] = 'human'
        else:
            proc_dict['bot_treatment'] = curr_round['data']['botTreatment']

        # Get the speaker and listener info
        for player_role in ["speaker", "listener"]:
            if "bot" in data[player_role]:
                proc_dict[player_role] = curr_round['data']['botTreatment'] 
            else:
                proc_dict[player_role] = data[player_role]["urlParams"]["workerId"]
        proc_dict["idleReason"] = data["idleReason"]

        # Get the various round time informations
        round_time, speaker_time, wait_time, listener_time = extract_round_time(data, curr_round)
        proc_dict["roundTime"] = round_time
        proc_dict["speakerTime"] = speaker_time
        proc_dict["waitTime"] = wait_time
        proc_dict["listenerTime"] = listener_time
        proc_dict["isAttnCheck"] = attn_check

        # Get chat information
        try:
            chat, msg_len = extract_chat_data(data)
        except:
            print("Problem in ", doc["_id"])
            chat = "Speaker failed?"
            msg_len = -1

        proc_dict["chat"] = chat
        proc_dict["msgLen"] = msg_len
        if msg_len != -1:
            msg_lens.append(msg_len)

        # Get context info
        speaker_context, listener_context, target, selection = extract_context_data(data)
        proc_dict["context"] = speaker_context
        proc_dict["speaker_context"] = speaker_context
        proc_dict["listener_context"] = listener_context
        proc_dict["target"] = target
        proc_dict["selection"] = selection

        # Record our failsafe
        if "reportedGameId" in data:
            proc_dict["reportedGameId"] = data["reportedGameId"]
            proc_dict["reportedRoundId"] = data["reportedRoundId"]

        proc_game_rounds.append(proc_dict)

    return proc_game_rounds, np.mean(msg_lens) if len(msg_lens) > 0 else 0

def extract_round_time(data, curr_round):
    roundStart = pd.to_datetime(data["roundStart"])
    roundEnd = pd.to_datetime(data["roundEnd"])
    round_time = (roundEnd - roundStart).total_seconds()

    # Get the speaker time
    if len(data["chat"]) > 0:
        speaker_bot = "bot" in data["speaker"]
        if speaker_bot:
            speakerSent = pd.to_datetime(data["chat"][0]["time"])
            speakerTime = (speakerSent - roundStart).total_seconds()
        else:
            try:
                speakerTime = data["chat"][0]["secUntilSend"]
            except:
                print("Error in non-bot")
                print(curr_round)
                print()
                speakerTime = -1
    else:
        speakerTime = -1
                
    # Get wait time
    if data["listenerObservesMessageSec"] != -1:
        waitTime = data["listenerObservesMessageSec"] - speakerTime
    else:
        waitTime = -1

    # Get the listener time
    if data["clickedTangram"] != "no_clicks":
        listener_bot = "bot" in data["listener"]        
        if listener_bot:
            listenerStart = pd.to_datetime(data["botRequestStartTimestamp"])
            listenerEnd = pd.to_datetime(data["clickedTime"])
            listenerTime = (listenerEnd - listenerStart).total_seconds()
        else:
            secUntilClick = data["secUntilClick"]
            observeMsg = data["listenerObservesMessageSec"]
            listenerTime = secUntilClick - observeMsg
    else:
        listenerTime = -1

    return round_time, speakerTime, waitTime, listenerTime
        
def extract_context_data(data):
    speaker_context = [x["path"] for x in data["tangrams"][0]]
    listener_context = [x["path"] for x in data["tangrams"][1]]
    target = data["target"]
    selection = data["clickedTangram"]
    return speaker_context, listener_context, target, selection

def extract_chat_data(data):
    if len(data["chat"]) > 0:
        chat = data["chat"][0]["text"]
        msg_len = get_message_length(chat)
    else:
        chat = "Speaker idled"
        msg_len = -1
    return chat, msg_len

def get_message_length(chat):
    tokens = word_tokenize(chat)
    return len(tokens)

def get_game(game_id):
    """
    Returns a dictionary containing information of just one game
    """
    try:
        doc = GAME_COLLECTION.find({"_id": game_id})[0]
    except:
        return ""

    config = extract_game_config(doc)
    status = extract_game_status(doc)
    treatmentName, annotation = extract_treatment_info(doc)
    time_spent, accuracy, idleRounds = extract_game_performance(doc)
    players, maxBonus, feedback, demographicInfo = extract_player_info(doc, treatmentName)
    round_dict, avg_msg_len = extract_round_dict(doc, treatmentName)

    # Formatting
    hourlyBonus = 3600 * maxBonus / (time_spent + 1e-8)
    time_msg = f"{time_spent / 60:.2f} mins"

    game_dict = {
        "config" : config,
        "treatment" : treatmentName,
        "anno" : annotation,
        "time_msg" : time_msg,
        "status" : status,
        "accuracy" : accuracy,
        "idleRounds" : idleRounds,
        "players" : players,
        "feedback" : feedback,
        "demographicInfo" : demographicInfo,
        "hourlyPay" : hourlyBonus,
        "roundDicts" : round_dict,
        "msgLen" : avg_msg_len
    }

    return game_dict

def get_game_from_all(all_games, game_id):
    for _, anno_dict in all_games.items():
        for curr_id, game_dict in anno_dict.items():
            if curr_id == game_id:
                return game_dict
    return ""

@st.cache_data
def get_all_cached():
    """
    Returns a nested dictionary containing game ids as keys according filter
    The contents of filter are:
    1. annotation
    2. message length
    3. game status
    4. playerId

    """
    filepath = "all_data_saved.pkl"
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            game_dict, player_dict = pickle.load(f)
    else:
        game_dict = {}
        for annotation in ANNOTATIONS:
            if annotation == "jul_10":
                target_date = datetime(2023, 7, 10)
                start_date = target_date.replace(hour=16, minute=50, second=0)
                end_date = target_date.replace(hour=21, minute=0, second=0)
                query = {"createdAt": {"$gte": start_date, "$lte": end_date}}
            elif annotation == "jul_26":
                query = {"treatmentId" : {"$in" : ["3Axt3QDD2yHGyLxPp", "832mrTQtgEAPy6Hd6"]}}
            elif annotation == "all" or "annotation" == "":
                continue
            elif annotation == "jan_18":
                query = {"data.annotation" : annotation, "batchId" : {"$in" : ["sQPssoDM4BdPi9nXn", "5uxxTfZgySsiRnttC", "Si6nfQg8kEGSvQgiu",
                                                                               "nhJoRztnE7EoMfHP9", "yoSHo6iSPNW5bzKqn"]}}
            elif annotation == "jan_31":
                query = {"data.annotation" : annotation, "batchId" : {"$in" : ["9fovDXEKYeqcDuizD", "XGe6kSn6TGagsFFtg", "FkAuXusJebu8rKtoy"]}}
            elif annotation == "mar_5":
                query = {"data.annotation" : annotation, "batchId" : "CfWAFckzgoqaw6vx2"}
            elif annotation == "mar_7":
                query = {"data.annotation" : annotation, "batchId" : {"$in" : ["iMHBnMW7gdPprR7nx", "wLs4ZFAF8tEGfjd3F"]}}
            elif annotation == "mar_8":
                query = {"data.annotation" : annotation, "batchId" : {"$in" : ["2eGxipZcx43WGZMxr", "LsD7YAoSN2zJFCtp9"]}}
            elif annotation == "may_1":
                query = {"data.annotation" : annotation, "batchId" : {"$in" : ["aqznxYyAjejjLZMqt", "SqMyaxSZLSnuTqktp", "uePyHDd5wQdmbddWY", "hCD349eFR4ZHwf57i"]}}                
            elif annotation == "may_7":
                query = {"data.annotation" : annotation, "batchId" : {"$in" : ["Wt8rDS3JGJkTZobWE", "tDMpLJzwoKFeKiAJ2", "ypaepYoPGLQmzp4Dg", "hPnkRcpW33jCEYCJK",
                                                                               "TioSW8zRGyuxfwK5Q", "TJKknfNqgTvbrgE8z", "dAHctpb9CNoRdqqHo", "RThvi8ajsiXGcvfJ3"]}}                
            elif annotation == "may_13":
                query = {"data.annotation" : annotation, "batchId" : {"$in" : ["YEovynDRvSuSt3vbo", "6xLZfraumwfy5NQg9", "c5vXSD8eEvepgWvQy", "8Hc6xRH8QamrMccwH", "KiimzucPkxvq63Mcx"]}}
            elif annotation == "may_17":
                query = {"data.annotation" : annotation, "batchId" : {"$in" : ["fLnbJ6PwpgjE8HfEj", "fggbthAMiwBE5cdYX", "vF3QdkRZCJJFJgiw3",
                                                                               "kuiTWvSdnE6guJztQ", "jSKtKy77ehfnwrZCX",
                                                                               "7R9RG24vw8774CfHP", "oJah6oZEh2psEq6y9"]}}
            else:
                query = {"data.annotation" : annotation}
            result = GAME_COLLECTION.find(query)
            annotation_dict = {}

            # Iterate over each game
            for doc in result:
                _id = doc["_id"]
                curr_dict = get_game(_id)
                if curr_dict is None:
                    continue
                annotation_dict[_id] = curr_dict

            game_dict[annotation] = annotation_dict

        player_dict = get_players(game_dict)
        with open(filepath, 'wb') as f:
            pickle.dump((game_dict, player_dict), f)

    return game_dict, player_dict

def filter_games(filter, all_games):
    # First filter based on annotation
    filtered_games = game_annotation_filter(filter["annotation"], all_games) 
    
    # Next filter based on all other parameters
    final_filtered_games = {}
    for game_id, game_dict in filtered_games.items():
        # Filter based on end reason
        if game_dict["status"] not in filter["status"]:
            continue

        # Filter based on context
        if game_dict["treatment"] not in filter["treatment"]:
            continue

        # Filter based on accuracy
        game_acc = game_dict["accuracy"] * 100
        if game_acc < filter["accuracy"][0]:
            continue
        if game_acc > filter["accuracy"][1]:
            continue

        # Filter based on idle rounds
        if game_dict["idleRounds"] > int(filter["idle_count"]):
            continue

        final_filtered_games[game_id] = game_dict

    return final_filtered_games

def game_annotation_filter(annotation, all_games):
    filtered_games = {}
    for curr_annotation, anno_dict in all_games.items():
        if annotation != "all" and curr_annotation != annotation:
            continue

        for game_id, game_dict in anno_dict.items():
            if game_id == "JjZrDALqAg5qHp7tg":
                continue
            filtered_games[game_id] = game_dict

    return filtered_games

def get_players(all_games):
    players_dict = {}

    # Iterate over each game
    for anno, anno_dict in all_games.items():
        anno_player_dict = {}
        for game_id, game_dict in anno_dict.items():
            construct_player_dict(anno_player_dict, 0, game_id, game_dict)
            construct_player_dict(anno_player_dict, 1, game_id, game_dict)
            
        players_dict[anno] = anno_player_dict

    return players_dict

def construct_player_dict(anno_player_dict, player_idx, game_id, game_dict):
    # Get the player and their partner
    player = game_dict["players"][player_idx]
    partner = game_dict["players"][1 - player_idx]
    if player == "human_model":
        return

    # Prelims
    if player not in anno_player_dict:
        anno_player_dict[player] = []
    player_dict = {"gameId" : game_id}

    # Get game performance values
    set_player_game_performance(player_dict, player, game_dict)

    # Get player doc specific info
    set_player_doc_values(player_dict, player, game_id, game_dict, player_idx)

    # Get partner ratings
    set_player_ratings(player_dict, partner, game_id)

    anno_player_dict[player].append(player_dict)

def set_player_game_performance(player_dict, player, game_dict):
    speaker_correctnesses = []
    listener_correctnesses = []
    msg_lens = []
    speaker_times = []
    listener_times = []

    for curr_round in game_dict["roundDicts"]:
        # Add speaker data
        if curr_round["speaker"] == player:
            if curr_round["speakerTime"] != -1:
                speaker_times.append(curr_round["speakerTime"])
                msg_lens.append(curr_round["msgLen"])
            if curr_round["selection"] != "no_clicks":
                speaker_correctnesses.append(1 if curr_round["selection"] == curr_round["target"] else 0)
        else:
            if curr_round["listenerTime"] != -1:
                listener_times.append(curr_round["listenerTime"])
                listener_correctnesses.append(1 if curr_round["selection"] == curr_round["target"] else 0)

    player_dict["accuracy"] = game_dict["accuracy"]
    player_dict["speakerAcc"] = np.mean(speaker_correctnesses)
    player_dict["listenerAcc"] = np.mean(listener_correctnesses)
    player_dict["msgLen"] = np.mean(msg_lens)
    player_dict["speakerTime"] = np.mean(speaker_times)
    player_dict["listenerTime"] = np.mean(listener_times)
    player_dict["treatment"] = game_dict["treatment"]
    player_dict["status"] = game_dict["status"]

def set_player_doc_values(player_dict, player, game_id, game_dict, player_idx):
    # First record the info we already have
    player_dict["feedback"] = game_dict["feedback"][player_idx]
    player_dict["demographicInfo"] = game_dict["demographicInfo"][player_idx]

    # Get information from the doc
    try:
        player_doc = PLAYERS_COLLECTION.find({"urlParams.workerId" : player,
                                              "gameId" : game_id})[0]
        player_dict["bonus"] = 0.6 + player_doc["data"]["bonus"]

        overall_time = float(game_dict["time_msg"].split()[0]) + 1e-8
        if "lobbyStartAt" in player_doc["data"]:
            lobby_start = pd.to_datetime(player_doc["data"]["lobbyStartAt"])
            lobby_end = pd.to_datetime(player_doc["data"]["lobbyEndAt"])
            lobby_time = (lobby_end - lobby_start).total_seconds() / 60
            overall_time += lobby_time

        player_dict["hourlyPay"] = 60 * player_dict["bonus"] / overall_time
        player_dict["exitReason"] = player_doc["exitReason"] if "exitReason" in player_doc else ""
    except:
        print(player)

def set_player_ratings(player_dict, partner, game_id):
    try:
        partner_doc = PLAYERS_COLLECTION.find({"urlParams.workerId" : partner,
                                               "gameId" : game_id})[0]
        if partner_doc["data"]["surveyResponses"] == "didNotSubmit":
            player_dict["satisfied"] = -1
            player_dict["comprehension"] = -1
            player_dict["grammatical"] = -1
            player_dict["clear"] = -1
            player_dict["non-ambiguity"] = -1
        else:
            responses = partner_doc["data"]["surveyResponses"]
            player_dict["satisfied"] = SURVEY_RESPONSES.index(responses["satisfied"]) + 1
            player_dict["comprehension"] = int(responses["comprehension"])
            player_dict["grammatical"] = int(responses["grammatical"])
            player_dict["clear"] = int(responses["clear"])
            player_dict["non-ambiguity"] = int(responses["ambiguous"])
    except:
        player_dict["satisfied"] = -1
        player_dict["comprehension"] = -1
        player_dict["grammatical"] = -1
        player_dict["clear"] = -1
        player_dict["non-ambiguity"] = -1

        
def filter_players(filter, all_players):
    set_of_all_players = set()
    for anno, player_dict in all_players.items():
        for player in player_dict.keys():
            set_of_all_players.add(player)

    player_dict = {}
    for player in set_of_all_players:
        print(player)

        player_games = filter_player_games(filter, player, all_players)
        if len(player_games) == 0:
            continue

        curr_player_dict = {
            "total" : len(player_games),
            "accuracy" : np.mean([game["accuracy"] for game in player_games if game["status"] == "complete"]),
            "speakerAcc" : np.mean([game["speakerAcc"] for game in player_games if game["status"] == "complete"]),
            "listenerAcc" : np.mean([game["listenerAcc"] for game in player_games if game["status"] == "complete"]),
            "msgLen" : np.mean([game["msgLen"] for game in player_games if game["status"] == "complete"]),
            "satisfied" : np.mean([game["satisfied"] for game in player_games if game["status"] == "complete"]),
            "hourlyPay" : np.mean([game["hourlyPay"] for game in player_games if game["status"] == "complete"])
        }
        player_dict[player] = curr_player_dict

    return player_dict
        

def filter_player_games(filter, player, all_players):
    games = []
    for anno, player_dict in all_players.items():
        if filter["annotation"] != "all" and anno != filter["annotation"]:
            continue
        if player not in player_dict:
            continue
        player_games = player_dict[player]

        # Filter games out
        chosen_games = []
        for game in player_games:
            # Filter based on game info
            if game["treatment"] not in filter["treatment"]:
                continue
            if game["status"] not in filter["status"]:
                continue
            if game["accuracy"] * 100 < filter["accuracy"][0]:
                continue
            if game["accuracy"] * 100> filter["accuracy"][1]:
                continue

            # Filter based on rating
            if game["satisfied"] < filter["satisfied"][0]:
                continue
            if game["satisfied"] > filter["satisfied"][1]:
                continue
                
            chosen_games.append(game)

        games.extend(chosen_games)

    return games
    

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


def display_game_summary(filter, all_games):
    """
    Displays the games according the filter, which is a dictionary
    Returns the game id of a game we want to see

    [treatment, status, accuracy, idleRounds, timeSpent, bonus, playerA, playerB]
    """
    # create columns:

    cols = st.columns(spec=[1.5, 1, 1, 1, 1.3, 1.3, 1, 1.3, 1.3, 1.3], gap="medium")
    header_lst = [
        "Game Ids",
        "Treatment",
        "Status",
        "Accuracy",
        "Idle Rounds",
        "Message Len",
        "Time",
        "Hourly Pay",
        "Player A",
        "Player B"
    ]
    make_cols(cols, header_lst, ["Game Ids", "Player A", "Player B"], 18)
    games = filter_games(filter, all_games)

    accs = []
    times = []
    bonuses = []

    for game_id, game_dict in games.items():
        # new columns for alignment purposes
        cols = st.columns(spec=[1.5, 1, 1, 1, 1.3, 1.3, 1, 1.3, 1.3, 1.3], gap="medium")

        # Write remaining columns
        cols[0].button(game_id, on_click=view_game, args=[game_id, game_dict])
        col_write(cols[1], game_dict["treatment"])
        col_write(cols[2], game_dict["status"])
        col_write(cols[3], f"{game_dict['accuracy']*100:.2f}")
        col_write(cols[4], game_dict["idleRounds"])
        col_write(cols[5], f"{game_dict['msgLen']:.2f}")
        col_write(cols[6], game_dict["time_msg"])
        col_write(cols[7], f"${game_dict['hourlyPay']:.2f}/hr")
        col_write(cols[8], game_dict["players"][0][:6])
        col_write(cols[9], game_dict["players"][1][:6])

        accs.append(game_dict['accuracy'] * 100)
        bonuses.append(game_dict['hourlyPay'])
        times.append(float(game_dict['time_msg'].split()[0]))

    # Add average values here
    cols = st.columns(spec=[1.5, 1, 1, 1, 1.3, 1.3, 1, 1.3, 1.3, 1.3], gap="medium")
    col_write(cols[0], "Average", txt_size=18, orient="left")
    col_write(cols[3], f"{np.mean(accs):.2f}")
    col_write(cols[6], f"{np.mean(times):.2f} mins")
    col_write(cols[7], f"${np.mean(bonuses):.2f}/hr")

    st.write(f"{len(games)} games played")

def display_player_summary(filter, all_players):
    """
    Displays summary of players
    """
    st.title("Player Summary")
    cols = st.columns(spec=[2, 1, 1, 1, 1, 1, 1, 1], gap="medium")
    header_lst = [
        "Player Ids",
        "Total Games",
        "Accuracy",
        "Speaker Accuracy",
        "Listener Accuracy",
        "Message Length",
        "Avg rating",
        "Hourly Pay"
    ]

    make_cols(cols, header_lst, ["Player Ids"], 22)
    players = filter_players(filter, all_players)

    avg_hourly = []
    for player, player_dict in players.items():
        cols = st.columns(spec=[2, 1, 1, 1, 1, 1, 1, 1], gap="medium")

        cols[0].button(
            player[:25], on_click=set_player, args=[player, ["player_sum"]], key=player + "player_sum"
        )

        col_write(cols[1], player_dict["total"])
        col_write(cols[2], f'{player_dict["accuracy"] * 100:.2f}')
        col_write(cols[3], f'{player_dict["speakerAcc"] * 100:.2f}')
        col_write(cols[4], f'{player_dict["listenerAcc"] * 100:.2f}')
        col_write(cols[5], f'{player_dict["msgLen"]:.2f}')
        col_write(cols[6], f'{player_dict["satisfied"]:.2f}')
        col_write(cols[7], f'{player_dict["hourlyPay"]:.2f}')
        avg_hourly.append(player_dict["hourlyPay"])

    col_write(cols[7], f"Average Hourly: {np.mean(avg_hourly):.2f}")

def display_player(player, filter, all_players, all_games):
    """
    Displays player stats.
    This includes a list of games they played (and associated annotation),
    their average message length, their rating from their peers
    """
    if filter["player_id"] == "":  # from the sidebar
        player = st.session_state.player_id
        filter["player_id"] = player
    else:
        player = filter["player_id"]
    games = filter_player_games(filter, player, all_players)

    if len(games) == 0:
        display_no_player()
        return

    make_buttons()

    st.subheader("Player: " + player)
    display_gen(games, player, all_games)

def display_gen(games, player, all_games):
    """
    Displays the general game stats for a player, including game length, number
    of turns, how the game ended, the player's role, message length, and
    hourly play for each game in [games]
    """
    cols = st.columns([1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], gap="medium")
    titles = [
        "Games",
        "Acc",
        "spkAcc",
        "lstAcc",
        "msgLen",
        "spkTime",
        "lstTime",
        "Time",
        "hourlyPay",
        "satis",
        "comp",
        "grmr",
        "clear",
        "Non-amb"
    ]
    make_cols(cols, titles, ["Games Played"], 18)

    for game in games:
        cols = st.columns([1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], gap="medium")

        _id = game["gameId"]
        game_dict = get_game_from_all(all_games, _id)
        cols[0].button(
            _id, 
            on_click=view_game,
            args=[_id, game_dict, ["spec_player", player]],
            key=(_id + "playergen"),
        )

        col_write(cols[1], f'{game["accuracy"] * 100:.2f}')
        col_write(cols[2], f'{game["speakerAcc"] * 100:.2f}')
        col_write(cols[3], f'{game["listenerAcc"] * 100:.2f}')
        col_write(cols[4], f'{game["msgLen"]:.2f}')
        col_write(cols[5], f'{game["speakerTime"]:.2f}')
        col_write(cols[6], f'{game["listenerTime"]:.2f}')
        col_write(cols[7], f'{game_dict["time_msg"]}')
        col_write(cols[8], f'{game["hourlyPay"]:.2f}')
        col_write(cols[9], f'{game["satisfied"]:.2f}')
        col_write(cols[10], f'{game["comprehension"]:.2f}')
        col_write(cols[11], f'{game["grammatical"]:.2f}')
        col_write(cols[12], f'{game["clear"]:.2f}')
        col_write(cols[13], f'{game["non-ambiguity"]:.2f}')

def display_game(id, game_info):
    """
    Displays game information for game id:
    game_info has the following elements:
    [game_length, turns, reason, speaker, listener, msg_len]

    QUERIES THE DATABASE
    """
    make_buttons()
    st.header("Game ID: " + id)

    s, l = st.columns(spec=[1, 1], gap="medium")
    s.markdown(f"**Treatment:** {game_info['treatment']}")
    l.markdown(f"**Status:** {game_info['status']}")

    s.markdown(f"**Accuracy:** {100 * game_info['accuracy']:.2f}")
    l.markdown(f"**Idle Rounds:** {game_info['idleRounds']}")

    s.markdown(f"**Time:** {game_info['time_msg']}")
    l.markdown(f"**Hourly Pay:** ${game_info['hourlyPay']:.2f}/hr")

    s.markdown("**Player A:**")
    s.button(
        (game_info["players"][0]), on_click=set_player, args=[game_info["players"][0], ["spec_game", id, game_info]]
    )

    l.markdown("**Player B:**")
    l.button(
        (game_info["players"][1]), on_click=set_player, args=[game_info["players"][1], ["spec_game", id, game_info]]
    )

    st.divider()
    for curr_round in game_info['roundDicts']:
        if curr_round["isAttnCheck"]:
            st.markdown(f"### Attention Check Round")
        else:
            st.markdown(f"### Round {curr_round['index'] + 1}")

        s, l = st.columns(spec=[1, 1], gap="medium")
        s.markdown(f"**Speaker:** {curr_round['speaker']}")
        l.markdown(f"**Listener:** {curr_round['listener']}")
        
        # Let's first note if there was a mix-up
        if "reportedGameId" in curr_round:
            if curr_round["reportedGameId"] != id and curr_round["round_id"] != curr_round["reportedRoundId"]:
                st.write(
                    f'<span style="font-size: 20px; line-height:2; color:red">Model responded to different game-round pair',
                    unsafe_allow_html=True,
                )

        if curr_round["idleReason"] == "speaker":
            st.write(
                f'<span style="font-size: 20px; line-height:2; color:red">{curr_round["chat"]}',
                unsafe_allow_html=True,
            )
        else:
            chat = curr_round['chat']
            st.write(
                f'<span style="font-size: 20px; line-height:1">{chat}',
                unsafe_allow_html=True,
            )

        targets = svg2png([curr_round["target"]])
        context = svg2png(curr_round['context'])

        if curr_round["selection"] == "no_clicks" or curr_round["selection"] is None:
            clicks = svg2png([])
        else:
            clicks = svg2png([curr_round["selection"]])

        display_context(context, targets, clicks)

    st.divider()

    # Display player feedback
    st.header("Feedbacks:")

    st.markdown(f"#### Player A: {game_info['players'][0]}")
    st.write(
        f'<span style="font-size: 20px; line-height:1">{game_info["feedback"][0]}',
        unsafe_allow_html=True,
    )

    st.markdown(f"#### Player B: {game_info['players'][1]}")
    st.write(
        f'<span style="font-size: 20px; line-height:1">{game_info["feedback"][1]}',
        unsafe_allow_html=True,
    )
    
    st.divider()
    
    st.header("Demographics:")
    st.markdown(f"#### Player A: {game_info['players'][0]}")
    st.write(
        f'<span style="font-size: 20px; line-height:1"> English: {game_info["demographicInfo"][0]["english"]}',
        unsafe_allow_html=True,
    )
    st.write(
        f'<span style="font-size: 20px; line-height:1"> English: {game_info["demographicInfo"][0]["languages"]}',
        unsafe_allow_html=True,
    )
    st.write(
        f'<span style="font-size: 20px; line-height:1"> Where learn: {game_info["demographicInfo"][0]["whereLearn"]}',
        unsafe_allow_html=True,
    )

    st.markdown(f"#### Player B: {game_info['players'][1]}")
    st.write(
        f'<span style="font-size: 20px; line-height:1"> English: {game_info["demographicInfo"][1]["english"]}',
        unsafe_allow_html=True,
    )
    st.write(
        f'<span style="font-size: 20px; line-height:1"> English: {game_info["demographicInfo"][1]["languages"]}',
        unsafe_allow_html=True,
    )
    st.write(
        f'<span style="font-size: 20px; line-height:1"> Where learn: {game_info["demographicInfo"][1]["whereLearn"]}',
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
    filter["annotation"] = st.sidebar.radio("Please select an annotation", ANNOTATIONS)

    # Can either view a certain game or a certain player
    filter["collection"] = st.sidebar.selectbox(
        "Collection to view", ("Game", "Player")
    )

    disable_status = st.session_state.player_id == "" and filter["player_id"] == ""

    if filter["annotation"] != "":
        if filter["collection"] == "Game":
            st.session_state.home = "game"
            filter["game_id"] = st.sidebar.text_input(
                "Game Id", "", max_chars=32, key="id"
            )
            if filter["game_id"] != "":
                st.session_state.display = "spec_game"
            else:
                st.session_state.display = "game_sum"
            disable_status = False

            filter["idle_count"] = st.sidebar.radio(
                "What's the maximum number of idle rounds?",
                [0, 1, 2, 3]
            )
        else:
            st.session_state.home = "player"
            filter["player_id"] = st.sidebar.text_input(
                "Player Id", "", max_chars=32, key="id"
            )
            if not disable_status:
                st.session_state.display = "spec_player"
            else:
                st.session_state.display = "player_sum"

    filter["satisfied"] = st.sidebar.slider(
        "Select satisfaction range",
        -1.0,
        6.0,
        (-1.0, 6.0),
        disabled=not disable_status,
    )

    # game end reason
    filter["status"] = st.sidebar.multiselect(
        "End reason: ",
        ["complete", "premature"],
        ["complete", "premature"],
    )

    # game end reason
    filter["treatment"] = st.sidebar.multiselect(
        "Context treatment: ",
        ["human_model", "human_human"],
    )

    # message length
    filter["accuracy"] = st.sidebar.slider(
        "Game accuracy", 0.0, 100.0, (0.0, 100.0)
    )


    return filter

def main():
    """
    Runs every time a click is made
    """
    add_new_states()

    all_games, all_players = get_all_cached()

    filter = {"annotation": "", "game_id" : "", "player_id" : ""}
    new_filter = sidebar(filter)

    if new_filter["annotation"] == "" or st.session_state.display == "intro":
        display_title()
        return None

    if new_filter["collection"] == "Game" and new_filter["game_id"] != "":
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
    elif st.session_state.player_id != "":
        st.session_state.display = "spec_player"
    elif new_filter["player_id"] != "":
        st.session_state.player_id = new_filter["player_id"]
        st.session_state.history.append(["player_sum"])

    # display as necessary
    if st.session_state.display == "intro" or new_filter["annotation"] == "":
        display_title()
    if st.session_state.display == "game_sum" and new_filter["annotation"] != "":
        display_game_summary(new_filter, all_games)
    elif st.session_state.display == "spec_game":
        display_game(st.session_state.game_id, st.session_state.game_info)
    elif st.session_state.display == "player_sum" and new_filter["annotation"] != "":
        display_player_summary(new_filter, all_players)
    elif st.session_state.display == "spec_player":
        display_player(st.session_state.player_id, new_filter, all_players, all_games)

    return None


if __name__ == "__main__":
    main()
