import Empirica from "meteor/empirica:core";
import "./callbacks.js";
import "./bots.js";
import { tangramsRelativeDir } from "./private.json";
import _ from "lodash";
import path from "path";
import { Mutex } from 'async-mutex';
const { MongoClient } = require("mongodb");

const fs = require("fs"); // read files
const mutex = new Mutex();
const MONGO_URI = Meteor.settings["galaxy.meteor.com"]["env"]["MONGO_URL"];
const DB_NAME = "TangramsCompGen";
const COLLECTION_NAME = "GameConfigFiles";
const BASE_CONFIGURATION_PATH = "/games/";
const NUMBER_OF_JSONS = 19;
// var game_index = 0;


function getOrderedImages(images, tangramsOreder){
  let new_images = Array(images.length);
  for (i = 0; i < images.length; i++) {
    new_images[tangramsOreder[i]] = images[i];
  }
  return new_images;
}

// gameInit is where the structure of a game is defined.  Just before
// every game starts, once all the players needed are ready, this
// function is called with the treatment and the list of players.  You
// must then add rounds and stages to the game, depending on the
// treatment and the players. You can also get/set initial values on
// your game, players, rounds and stages (with get/set methods), that
// will be able to use later in the game.
Empirica.gameInit((game, treatment) => {
    console.log(
	"Game with a treatment: ",
	treatment,
	" will start, with workers",
	_.map(game.players, "id")
    );

    // Sample whether on the blue team or red team
    game.set("teamColor", treatment.teamColor);

    // Define the bot treatment for the game
    game.set("full_IP", treatment.full_IP);
    game.set("no_ji_IP", treatment.no_ji_IP);
    game.set("no_ds_IP", treatment.no_ds_IP);
    game.set("baseline_IP", treatment.baseline_IP);
    game.set("annotation", treatment.annotation)
    const botGame = treatment.botsCount == '1';
    let botIdx;
    if (botGame) {
	for (let i = 0; i < 2; i++) {
	    if (typeof game.players[i].bot != 'undefined') {
		botIdx = i;
	    }
	}
    }

    // I use this to play the sound on the UI when the game starts
    game.set("justStarted", true);

    // Sample the game configuration at random
    let basePath = path.join(__meteor_bootstrap__.serverDir, "../web.browser/app"); // directory for folders in /public after built
    let configName = BASE_CONFIGURATION_PATH + treatment.experimentName;
    if (treatment.numConfigs == 0) {
	configName += "/game_json_" + (treatment.gameNum).toString() + ".json";	
    } else {
	let sampledIndex = Math.floor(Math.random() * treatment.numConfigs);
	configName += "/game_json_" + sampledIndex.toString() + ".json";
    }

    console.log("use " + configName);
    let configPath = basePath + configName;
    game.set("configFile", "public" + configName)

    // Load the game config json and the idx to tangram json
    let rawdata = fs.readFileSync(configPath);
    let gameFile = JSON.parse(rawdata);
    let gameConfig = gameFile["blocks"];

    let idx2tPath = basePath + BASE_CONFIGURATION_PATH + "idx_to_tangram.json";
    let idx2tRawdata = fs.readFileSync(idx2tPath);
    let idx2t = JSON.parse(idx2tRawdata);

    // Iterate over each block
    let trialNum = 1;
    let totalTrials = 0
    for (let i = 0; i < gameConfig.length; i++) {
	totalTrials += gameConfig[i]["tgt"].length;
    }

    for (let i = 0; i < gameConfig.length; i++) {
	// Create data for each tangram
	const imageIndices = gameConfig[i]["img"];
	let imageDicts = Array(imageIndices.length);
	for (let j = 0; j < imageIndices.length; j++) {
	    const imageIndex = imageIndices[j];
	    const imageFile = idx2t[imageIndex.toString()];
	    const imagePath = basePath + tangramsRelativeDir + imageFile;
	    imageDicts[j] = {};
	    imageDicts[j]["path"] = imageFile;
	    imageDicts[j]["data"] = fs.readFileSync(imagePath, "utf8");
	}

	// Get the roles for the block
	let roleArray = Array(2);
	const currRoles = gameConfig[i]["roles"];
	for (let j = 0; j < 2; j++) {
	    let currRole = (currRoles[j] == 0) ? "speaker" : "listener";
	    roleArray[j] = currRole;
	}

	const targetIndices = gameConfig[i]["tgt"];
	for (let j = 0; j < targetIndices.length; j++){
	    const round = game.addRound();

	    if (botGame) {
		if ("anno" in gameConfig[i]) {
		    round.set("isAttnCheck", true);
		    round.set("attnCheckAnno", gameConfig[i]["anno"]);
		}

		if ("bot_precomputed_utterances" in gameConfig[i]) {
		    round.set("precomputedBot", true);
		    round.set("botPrecomputedUtterance", gameConfig[i]["bot_precomputed_utterances"][j]);
		}

		round.set("botTreatment", gameConfig[i]["bot_treatment"]);
	    } else if ("anno" in gameConfig[i]) {
		round.set("isAttnCheck", true);
	    }

	    round.set("chat", []);
	    round.set("clickedTangram", "no_clicks");
	    round.set("idleReason", "not_idle");
	    round.set("idleStageSubmitted", false);

	    round.set("numTrials", totalTrials);
	    round.set("trialNum", trialNum);
	    trialNum++;

	    round.set("tangrams", [
		getOrderedImages(imageDicts, gameConfig[i]["order"][0]),
		getOrderedImages(imageDicts, gameConfig[i]["order"][1]),
	    ]);
	  
	    const targetIndex = targetIndices[j];
	    const targetPath = idx2t[targetIndex.toString()];
	    round.set("target", targetPath);
	    round.set("roles", roleArray);

	    if (botGame) { 
		round.set('botActed', false);
		round.set('botActionTime', -1);
		console.log(roleArray);

		playerIdx = (botIdx + 1) % 2
		if (roleArray[0] == "listener") {
		    round.set("listener", game.players[playerIdx]);
		    round.set("speaker", game.players[botIdx]);
		} else {
		    round.set("speaker", game.players[playerIdx]);
		    round.set("listener", game.players[botIdx]);
		}
	    } else {
		if (roleArray[0] == "listener") {
		    round.set("listener", game.players[0]);
		    round.set("speaker", game.players[1]);
		} else {
		    round.set("speaker", game.players[0]);
		    round.set("listener", game.players[1]);
		}
	    }

	    round.set("listenerObservesMessageSec", -1);	    

	    round.addStage({
		name: "description",
		displayName: "Description",
		durationInSeconds: '65',
	    });
	}
    }
});
