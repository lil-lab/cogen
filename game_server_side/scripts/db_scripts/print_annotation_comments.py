import numpy as np
from pymongo import MongoClient
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str)
    return parser.parse_args()

def get_mongodb_contents():
    mongo_url = "fill_in"
    myclient = MongoClient(mongo_url)
    mydb = myclient["TangramsCompGen"]

    games_col = mydb["games"]
    players_col = mydb["players"]
    return games_col, players_col

if __name__ == "__main__":
    # Get the args
    args = get_args()

    # Iterate over each game
    games_col, players_col = get_mongodb_contents()
    annotation = args.annotation
    if annotation == "may_1":
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
    games = games_col.find(query)
    for game in games:
        player_ids = game["playerIds"]
        for player_id in player_ids:
            player_dict = players_col.find({"_id" : player_id})[0]
            if "bot" in player_dict:
                continue

            hashed_id = player_dict["urlParams"]["workerId"]

            # Print comment
            if player_dict["data"]["surveyResponses"] != "didNotSubmit":
                feedback = player_dict["data"]["surveyResponses"]["feedback"]
                if feedback != "":
                    print(f"Game: {game['_id']}, player: {hashed_id}")
                    print(feedback)
                    print()
            if player_dict["data"]["errorSurveyResponses"] != "didNotSubmit":
                feedback = player_dict["data"]["errorSurveyResponses"]["feedback"]    
                if feedback != "":
                    print(f"Game: {game['_id']}, player: {hashed_id}")
                    print(feedback)
                    print()
