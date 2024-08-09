import moment from "moment";
import { ValidatedMethod } from "meteor/mdg:validated-method";
import SimpleSchema from "simpl-schema";
const crypto = require('crypto');

import { Batches } from "../batches/batches.js";
import { GameLobbies } from "../game-lobbies/game-lobbies";
import { IdSchema } from "../default-schemas.js";
import { LobbyConfigs } from "../lobby-configs/lobby-configs.js";
import { Games } from "../games/games.js";
import { Players } from "./players";
import { exitStatuses } from "./players.js";
import { sleep, weightedRandom } from "../../lib/utils.js";
import shared from "../../shared.js";
import gameLobbyLock from "../../gameLobby-lock.js";

const numHumansInQueue = (lobby) => {
    let numHumans = 0;
    const queuedPlayerList = lobby.queuedPlayerIds;
    for (let j = 0; j < queuedPlayerList.length; j += 1) {
	currPlayer = Players.findOne({_id : queuedPlayerList[j]});
	if (currPlayer.bot === undefined) {
	    numHumans += 1;
	}
    }
    
    return numHumans;
}

const firstHalfFullLobby = (weightedLobbyPool) => {
    firstLobbyDict = weightedLobbyPool[0];
    if (firstLobbyDict.weight > 0) {
	return firstLobbyDict.value;
    } else {
	// Sample randomly
	const randomIndex = Math.floor(Math.random() * weightedLobbyPool.length);
	return weightedLobbyPool[randomIndex].value;
    }
}

const partlyRandomizedGreedyMatchmaking = (weightedLobbyPool) => {
    const numHalfFullLobbies = getNumHalfFullLobbies(weightedLobbyPool);
    if (numHalfFullLobbies >= 2) {
	const waitTimes = getPlayerWaitTimes(weightedLobbyPool, numHalfFullLobbies);
	console.log(waitTimes);
	const randomIndex = sampleFromWaitTimes(weightedLobbyPool, waitTimes);
	return weightedLobbyPool[randomIndex].value;
    } else {
	// Sample randomly
	const randomIndex = Math.floor(Math.random() * weightedLobbyPool.length);
	return weightedLobbyPool[randomIndex].value;
    }
}

const getPlayerWaitTimes = (weightedLobbyPool, numHalfFull) => {
    const waitTimes = [];
    for (let i = 0; i < numHalfFull; i += 1) {
	// Get the player
	currLobby = weightedLobbyPool[i].value;
	if (currLobby.playerIds.length === 1) {
	    currPlayer = currLobby.playerIds[0];
	    currPlayerDict = Players.findOne({_id : currPlayer});

	    // Get the time
	    const now = moment();
	    const startTimeAt = moment(currPlayerDict.timeoutStartedAt);
	    const diff = now.diff(startTimeAt, 'seconds')
	    waitTimes.push(diff)
	} else {
	    // Edge case: The player we wanted to sample got kicked out or someone else joined
	    waitTimes.push(0);
	}
    }
    return waitTimes;
}

const sampleFromWaitTimes = (weightedLobbyPool, waitTimes) => {
    // Get the sum
    let total = 0;
    for (let j = 0; j < waitTimes.length; j += 1) {
	total += waitTimes[j];
    }

    // Sample from the distribution
    if (total !== 0) {
	let probTotal = 0;
	const uniformSample = Math.random();
	console.log(uniformSample);

	for (let j = 0; j < waitTimes.length; j += 1) {
	    probTotal += waitTimes[j] / total;
	    console.log(probTotal);

	    if (uniformSample <= probTotal) {
		return j;
	    }
	}
	console.log("Matchmaking should never fail here");
    } else {
	// Edge case: All lobbies got filled up!
	const randomIndex = Math.floor(Math.random() * weightedLobbyPool.length);
	return randomIndex;
    }
}


const getNumHalfFullLobbies = (weightedLobbyPool) => {
    let count = 0;
    for (let j = 0; j < weightedLobbyPool.length; j += 1) {
	if (weightedLobbyPool[j].weight > 0) {
	    count += 1
	} else {
	    return count;
	}
    }
    return count;
}

const isPlayerInAnotherLobby = (player, ipAddr) => {
    // Get the hashed worker ID
    const hashedId = player.urlParams["workerId"];

    // Get all running lobbies
    const allLobbies = GameLobbies.find({
	status: "running",
	timedOutAt: {$exists: false}
    }).fetch();

    // Get all hashed worker IDs within those running lobbies
    for (let i = 0; i < allLobbies.length; i += 1) {
	currLobby = allLobbies[i];

	// Get all ids in the lobby
	const currIdSet = new Set();
	queuedPlayerList = currLobby.queuedPlayerIds;
	for (let j = 0; j < queuedPlayerList.length; j += 1) {
	    currIdSet.add(queuedPlayerList[j])
	}
	playerList = currLobby.playerIds;
	for (let j = 0; j < playerList.length; j += 1) {
	    currIdSet.add(playerList[j]);
	}

	// Iterate over each id in the lobby
	for (const currId of currIdSet) {
	    currPlayer = Players.findOne({_id : currId});
	    currHashedId = currPlayer.urlParams["workerId"];
	    currHashedIp = currPlayer.data["hashedIP"];

	    if (currHashedId === hashedId && currPlayer.exitStatus === undefined) {
		return {
		    matchReason : "sameWorker",
		    matchedHash : currHashedId,
		    matchedLobby : currLobby._id
		}
	    }

	    if (Meteor.settings.preventMatchingIP) {
		if (currHashedIp === ipAddr && currPlayer.exitStatus === undefined) {
		    return {
			matchReason : "sameIP",
			matchedHash : currHashedId,
			matchedLobby : currLobby._id
		    }
		}
	    }
	}
    }

    return {
	matchReason : "noMatch",
    }
}

const didPlayerCompleteDemographics = (player) => {
    const hashedId = player.urlParams["workerId"];
    const matchingWorkers = Players.find({
	'urlParams.workerId': hashedId,
	'data.completedDemographics': true
    }).fetch();

    return !(!matchingWorkers || matchingWorkers.length === 0);
}

export const createPlayer = new ValidatedMethod({
    name: "Players.methods.create",

    validate: new SimpleSchema({
	id: {
	    type: String
	},
	urlParams: {
	    type: Object,
	    blackbox: true,
	    defaultValue: {}
	},
    }).validator(),

    run(player) {
	// Find the first running batch (in order of running started time)
	const batch = Batches.findOne(
	    { status: "running", full: false },
	    { sort: { runningAt: 1 } }
	);

	if (!batch) {
	    // The UI should update and realize there is no batch available
	    // This should be a rare case where a fraction of a second of
	    // desynchornisation when the last available batch just finished.
	    // If this is the case, since the user exist in the DB at this point
	    // but has no lobby assigned, and the UI will soon determine there
	    // is no available game, the UI will switch to "No experiments
	    // available", nothing else to do.
	    return;
	}

	// TODO: MAYBE, add verification that the user is not current connected
	// elsewhere and this is not a flagrant impersonation. Note that is
	// extremely difficult to guaranty. Could also add verification of user's
	// id with email verication for example. For now the assumption is that
	// there is no immediate reason or long-term motiviation for people to hack
	// each other's player account.

	const existing = Players.findOne({ id: player.id });

	// If the player already has a game lobby assigned, no need to
	// re-initialize them
	if (existing && existing.gameLobbyId) {
	    return existing._id;
	}

	
	if (existing) {
	    player = existing;
	} else {
	    // Because of a bug in SimpleSchema around blackbox: true, skipping
	    // validation here. Validation did happen at the method level though.
	    player._id = Players.insert(player, {
		filter: false,
		validate: false
	    });
	}

	// Check if the player is currently in another lobby; If so, exit
	let ipAddr = crypto.createHash('md5').update(this.connection.clientAddress).digest('hex');	
	const matchingDict = isPlayerInAnotherLobby(player, ipAddr);
	if (matchingDict.matchReason !== "noMatch") {
	    Players.update(player._id,
			   {$set : {['data.matchingWorker'] : matchingDict.matchReason,
				    ['data.matchedWorkerId'] : matchingDict.matchedHash,
				    ['data.matchedLobby'] : matchingDict.matchedLobby,
				    exitStatus: "matchingPlayerKickedOut",
				    exitAt: new Date(),
				    exitReason: "We detected you joining a game while already playing another"}});
	    return player._id;
	}

	// Looking for all lobbies for batch (for which that game has not started yet)
	const lobbies = GameLobbies.find({
	    batchId: batch._id,
	    status: "running",
	    timedOutAt: { $exists: false },
	    gameId: { $exists: false }
	}).fetch();

	if (lobbies.length === 0) {
	    // This is the same case as when there are no batches available.
	    return;
	}

	// Let's first try to find lobbies for which their queue isn't full yet
	let lobbyPool = lobbies.filter(
	    l => l.availableCount > l.queuedPlayerIds.length
	);

	// If no lobbies still have "availability", search for lobbies outside of current batch
	if (lobbyPool.length === 0) {
	    const newLobbies = GameLobbies.find({
		status: "running",
		timedOutAt: { $exists: false },
		gameId: { $exists: false }
	    }).fetch();

	    lobbyPool = newLobbies.filter(
		l => l.availableCount > l.queuedPlayerIds.length
	    );

	    // Give up if these are also all in pre-lobby phase
	    if (lobbyPool.length === 0) {
		return;
	    }
	}

	// Map each lobby to the number of non-bots in its queue
	const weightedLobbyPool = lobbyPool.map(lobby => {
	    return {
		value: lobby,
		weight: numHumansInQueue(lobby)
	    };
	});

	// Sort the lobbies by the number of players in them: 
	weightedLobbyPool.sort((a, b) => b.weight - a.weight);
	const lobby = firstHalfFullLobby(weightedLobbyPool);
	//const lobby = partlyRandomizedGreedyMatchmaking(weightedLobbyPool);
	if (!lobby) {
	    return;
	}

	// Adding the player to specified lobby queue
	GameLobbies.update(lobby._id, {
	    $addToSet: {
		queuedPlayerIds: player._id
	    }
	});

	const gameLobbyId = lobby._id;
	const $set = { gameLobbyId };

	// Check if there will be instructions
	let skipInstructions = lobby.debugMode;

	// If there are no instruction, mark the player as ready immediately
	if (skipInstructions) {
	    $set.readyAt = new Date();
	}

	Players.update(player._id, { $set });

	//const completedDemographics = didPlayerCompleteDemographics(player);
	const completedDemographics = true;
	let prelimUpdate = { ['data.completedDemographics'] : completedDemographics,
			     ['data.lobbyStartAt'] : new Date()};
	Players.update(player._id,
		       {$set : prelimUpdate},
		       {
			   autoConvert: false,
			   filter: false,
			   validate: false,
			   trimStrings: false,
			   removeEmptyStrings: false
		       });
	console.log("Completed demographics: ", completedDemographics);

	// If there are no instruction, player is ready, notify the lobby
	if (skipInstructions) {
	    GameLobbies.update(gameLobbyId, {
		$addToSet: { playerIds: player._id }
	    });
	}

	return player._id;
    }
});

export const playerReady = new ValidatedMethod({
  name: "Players.methods.ready",

  validate: IdSchema.validator(),

  async run({ _id }) {
    if (!Meteor.isServer) {
      return;
    }

    try {
      // Lobby might be locked if game is currently being created.
      // We retry until lobby is unlocked.
      while (!assignToLobby(_id)) {
        await sleep(1000);
      }
    } catch (error) {
      console.error("Players.methods.ready", error);
    }
  }
});

function assignToLobby(_id) {
  const player = Players.findOne(_id);

  if (!player) {
    throw `unknown ready player: ${_id}`;
  }
  const { readyAt, gameLobbyId } = player;

  if (readyAt) {
    // Already ready
    return true;
  }

  const lobby = GameLobbies.findOne(gameLobbyId);

  if (!lobby) {
    throw `unknown lobby for ready player: ${_id}`;
  }

  // GameLobby is locked.
  if (gameLobbyLock[gameLobbyId]) {
    return false;
  }

  // Game is Full, bail the player
  if (lobby.playerIds.length === lobby.availableCount) {
    // User already ready, something happened out of order
    if (lobby.playerIds.includes(_id)) {
      return true;
    }

    // Mark the player's participation attemp as failed if
    // not already marked exited
    Players.update(
      {
        _id,
        exitAt: { $exists: false }
      },
      {
        $set: {
          exitAt: new Date(),
          exitStatus: "gameFull"
        }
      }
    );

    return true;
  }

  // Try to update the GameLobby with the playerIds we just queried.
  GameLobbies.update(
    { _id: gameLobbyId, playerIds: lobby.playerIds },
    {
      $addToSet: { playerIds: _id }
    }
  );

  // If the playerId insert succeeded (playerId WAS added to playerIds),
  // mark the user record as ready and potentially start the individual
  // lobby timer.
  const lobbyUpdated = GameLobbies.findOne(gameLobbyId);
  if (lobbyUpdated.playerIds.includes(_id)) {
    // If it did work, mark player as ready
    $set = { readyAt: new Date() };

    // If it's an individual lobby timeout, mark the first timer as started.
    const lobbyConfig = LobbyConfigs.findOne(lobbyUpdated.lobbyConfigId);
    if (lobbyConfig.timeoutType === "individual") {
      $set.timeoutStartedAt = new Date();
      $set.timeoutWaitCount = 1;
    }

    Players.update(_id, { $set });
    return true;
  }

  // If the playerId insert failed (playerId NOT added to playerIds), the
  // playerIds has changed since it was queried and the lobby might not
  // have any available slots left, loop and retry.
  return false;
}

export const updatePlayerData = new ValidatedMethod({
  name: "Players.methods.updateData",

  validate: new SimpleSchema({
    playerId: {
      type: String,
      regEx: SimpleSchema.RegEx.Id
    },
    key: {
      type: String
    },
    value: {
      type: String
    },
    append: {
      type: Boolean,
      optional: true
    },
    noCallback: {
      type: Boolean,
      optional: true
    }
  }).validator(),

  run({ playerId, key, value, append, noCallback }) {
    const player = Players.findOne(playerId);
    if (!player) {
      throw new Error("player not found");
    }
    // TODO check can update this record player

    const val = JSON.parse(value);
    let update = { [`data.${key}`]: val };
    const modifier = append ? { $push: update } : { $set: update };

    Players.update(playerId, modifier, {
      autoConvert: false,
      filter: false,
      validate: false,
      trimStrings: false,
      removeEmptyStrings: false
    });

    if (Meteor.isServer && !noCallback) {
      shared.callOnChange({
        playerId,
        player,
        key,
        value: val,
        prevValue: player.data && player.data[key],
        append
      });
    }
  }
});

export const markPlayerExitStepDone = new ValidatedMethod({
  name: "Players.methods.markExitStepDone",

  validate: new SimpleSchema({
    playerId: {
      type: String,
      regEx: SimpleSchema.RegEx.Id
    },
    stepName: {
      type: String
    }
  }).validator(),

  run({ playerId, stepName }) {
    const player = Players.findOne(playerId);
    if (!player) {
      throw new Error("player not found");
    }
    // TODO check can update this record player

    Players.update(playerId, { $addToSet: { exitStepsDone: stepName } });
  }
});

export const extendPlayerTimeoutWait = new ValidatedMethod({
  name: "Players.methods.extendTimeoutWait",

  validate: new SimpleSchema({
    playerId: {
      type: String,
      regEx: SimpleSchema.RegEx.Id
    }
  }).validator(),

  run({ playerId }) {
    const player = Players.findOne(playerId);
    if (!player) {
      throw new Error("player not found");
    }

    Players.update(playerId, {
      $inc: { timeoutWaitCount: 1 },
      $set: { timeoutStartedAt: new Date() }
    });
  }
});

export const endPlayerTimeoutWait = new ValidatedMethod({
  name: "Players.methods.endTimeoutWait",

  validate: new SimpleSchema({
    playerId: {
      type: String,
      regEx: SimpleSchema.RegEx.Id
    }
  }).validator(),

  run({ playerId }) {
    const player = Players.findOne(playerId);
    if (!player) {
      throw new Error("player not found");
    }

    Players.update(playerId, {
      $set: {
        exitStatus: "playerEndedLobbyWait",
        exitAt: new Date()
      }
    });

    GameLobbies.update(player.gameLobbyId, {
      $pull: {
          playerIds: playerId,
	  queuedPlayerIds: playerId
      }
    });

    GameLobbies.update(player.gameLobbyId, {
      $addToSet: {
	  quitPlayerIds: playerId
      }
    });

  }
});

export const earlyExitPlayer = new ValidatedMethod({
  name: "Players.methods.admin.earlyExitPlayer",

  validate: new SimpleSchema({
    exitReason: {
      label: "Reason for Exit",
      type: String,
      regEx: /[a-zA-Z0-9_]+/
    },
    playerId: {
      type: String,
      regEx: SimpleSchema.RegEx.Id
    },
    gameId: {
      type: String,
      regEx: SimpleSchema.RegEx.Id
    }
  }).validator(),

  run({ exitReason, playerId, gameId }) {
    if (!Meteor.isServer) {
      return;
    }

    const game = Games.findOne(gameId);

    if (!game) {
      throw new Error("game not found");
    }

    if (game && game.finishedAt) {
      if (Meteor.isDevelopment) {
        console.log("\n\ngame already ended!");
      }

      return;
    }

    const currentPlayer = Players.findOne(playerId);

    if (currentPlayer && currentPlayer.exitAt) {
      if (Meteor.isDevelopment) {
          console.log("\nplayer already exited!");
      }

      return;
    }

    Players.update(playerId, {
      $set: {
        exitAt: new Date(),
        exitStatus: "custom",
        exitReason
      }
    });

    const players = Players.find({ gameId }).fetch();
    const onlinePlayers = players.filter(player => !player.exitAt);

    if (!onlinePlayers || (onlinePlayers && onlinePlayers.length === 0)) {
      Games.update(gameId, {
        $set: {
          finishedAt: new Date(),
          status: "custom",
          endReason: "finished_early"
        }
      });

      GameLobbies.update(
        { gameId },
        {
          $set: {
            status: "custom",
            endReason: "finished_early"
          }
        }
      );
    }
  }
});

export const earlyExitPlayerLobby = new ValidatedMethod({
  name: "Players.methods.admin.earlyExitPlayerLobby",

  validate: new SimpleSchema({
    exitReason: {
      label: "Reason for Exit",
      type: String,
      regEx: /[a-zA-Z0-9_]+/
    },
    playerId: {
      type: String,
      regEx: SimpleSchema.RegEx.Id
    },
    gameLobbyId: {
      type: String,
      regEx: SimpleSchema.RegEx.Id
    }
  }).validator(),

  run({ exitReason, playerId, gameLobbyId }) {
    if (!Meteor.isServer) {
      return;
    }

    const gameLobby = GameLobbies.findOne(gameLobbyId);

    if (!gameLobby) {
      throw new Error("gameLobby not found");
    }

    const currentPlayer = Players.findOne(playerId);

    if (currentPlayer && currentPlayer.exitAt) {
      if (Meteor.isDevelopment) {
          console.log("\nplayer already exited!");
      }

      return;
    }

    Players.update(playerId, {
      $set: {
        exitAt: new Date(),
        exitStatus: "custom",
        exitReason
      }
    });

      GameLobbies.update(gameLobby["_id"], {
	  $pull: {
	      playerIds: playerId,
	      queuedPlayerIds: playerId
	  }
      });

      GameLobbies.update(gameLobby["_id"], {
	  $addToSet: {
	      quitPlayerIds: playerId
	  }
      });
      
  }
});

export const retireSinglePlayer = new ValidatedMethod({
  name: "Players.methods.admin.retireSingle",

  validate: new SimpleSchema({
    playerId: {
      type: String,
      regEx: SimpleSchema.RegEx.Id
    }
  }).validator(),

  run({ playerId }) {
    if (!playerId) {
      throw new Error("empty playerId");
    }

    if (!this.userId) {
      throw new Error("unauthorized");
    }

    const player = Players.findOne({
      _id: playerId,
      retiredAt: { $exists: false }
    });

    if (!player) {
      throw new Error("Player not found");
    }

    const timestamp = new Date().toISOString();

    Players.update(playerId, {
      $set: {
        id: `${player.id} (Retired custom at ${timestamp})`,
        retiredAt: new Date(),
        retiredReason: "custom"
      }
    });

    return player;
  }
});

export const retireGameFullPlayers = new ValidatedMethod({
  name: "Players.methods.admin.retireGameFull",

  validate: new SimpleSchema({
    retiredReason: {
      label: "Retired Reason",
      type: String,
      optional: true,
      allowedValues: exitStatuses
    }
  }).validator(),

  run({ retiredReason }) {
    if (!this.userId) {
      throw new Error("unauthorized");
    }

    const players = Players.find({
      exitStatus: retiredReason,
      retiredAt: { $exists: false }
    }).fetch();

    const timestamp = new Date().toISOString();

    for (let i = 0; i < players.length; i++) {
      const player = players[i];

      Players.update(player._id, {
        $set: {
          id: `${player.id} (Retired ${retiredReason} at ${timestamp})`,
          retiredAt: new Date(),
          retiredReason
        }
      });
    }

    return players.length;
  }
});

export const playerWasRetired = new ValidatedMethod({
  name: "Players.methods.playerWasRetired",

  validate: IdSchema.validator(),

  run({ _id }) {
    return Boolean(
      Players.findOne({
        _id,
        exitStatus: { $exists: true },
        retiredAt: { $exists: true }
      })
    );
  }
});

export const updatePlayerStatus = new ValidatedMethod({
  name: "Players.methods.updateStatus",

  validate: new SimpleSchema({
    playerId: {
      type: String,
      regEx: SimpleSchema.RegEx.Id
    },

    idle: {
      type: Boolean
    },

    lastActivityAt: {
      type: Date
    }
  }).validator(),

  run({ playerId, idle, lastActivityAt }) {
    if (Meteor.isServer) {
      const playerIdConn = shared.playerIdForConn(this.connection);
      if (!playerIdConn) {
        return;
      }
      if (playerId !== playerIdConn) {
        console.error(
          "Attempting to update player status from wrong connection"
        );
        return;
      }
    }

    Players.update(playerId, {
      $set: {
        idle,
        lastActivityAt
      }
    });
  }
});
