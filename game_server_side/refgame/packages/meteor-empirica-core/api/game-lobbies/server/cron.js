import moment from "moment";

import { GameLobbies } from "../game-lobbies.js";
import { LobbyConfigs } from "../../lobby-configs/lobby-configs";
import { Players } from "../../players/players.js";
import { createGameFromLobby } from "../../games/create.js";
import Cron from "../../../startup/server/cron.js";

const checkLobbyTimeout = (log, lobby, lobbyConfig) => {
  // Timeout hasn't started yet
  if (!lobby.timeoutStartedAt) {
    return;
  }

  const now = moment();
  const startTimeAt = moment(lobby.timeoutStartedAt);
  const endTimeAt = startTimeAt.add(lobbyConfig.timeoutInSeconds, "seconds");
  const ended = now.isSameOrAfter(endTimeAt);

  if (!ended) {
    return;
  }

  switch (lobbyConfig.timeoutStrategy) {
    case "fail":
      GameLobbies.update(lobby._id, {
        $set: { timedOutAt: new Date(), status: "failed" }
      });
      Players.update(
        { _id: { $in: lobby.queuedPlayerIds } },
        {
          $set: {
            exitStatus: "gameLobbyTimedOut",
            exitAt: new Date()
          }
        },
        { multi: true }
      );
      break;
    case "ignore":
      createGameFromLobby(lobby);
      break;

    // case "bots": {

    //   break;
    // }

    default:
      log.error(
        `unknown LobbyConfig.timeoutStrategy: ${lobbyConfig.timeoutStrategy}`
      );
  }
};

const lobbyWaitTime = (currLobby, lobby) => {
    if (currLobby.playerIds.length == 0 || currLobby._id === lobby._id) {
	return -1
    } else {
	currPlayer = currLobby.playerIds[0];
	currPlayerDict = Players.findOne({_id : currPlayer});

	const now = moment();
	const startTimeAt = moment(currPlayerDict.timeoutStartedAt);
	const diff = now.diff(startTimeAt, 'seconds')
	return diff;
    }
}

const numBotsInQueue = (lobby) => {
    let numBots = 0;
    const queuedPlayerList = lobby.queuedPlayerIds;
    for (let j = 0; j < queuedPlayerList.length; j += 1) {
	currPlayer = Players.findOne({_id : queuedPlayerList[j]});
	if (currPlayer.bot !== undefined) {
	    numBots += 1;
	}
    }
    return numBots;
}

const getNumBotLobbies = (weightedLobbyPool) => {
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

const sampleBotLobby = (weightedLobbyPool, lobbyId) => {
    botLobbyIndices = getBotLobbyIndices(weightedLobbyPool, lobbyId);
    numBotLobbies = botLobbyIndices.length;
    console.log("Num bot lobbies: ", numBotLobbies)
    if (numBotLobbies > 0) {
	const botIndex = Math.floor(Math.random() * numBotLobbies);
	const randomIndex = botLobbyIndices[botIndex];
	return weightedLobbyPool[randomIndex]
    } else {
	console.log("Whoops! There were no bot lobbies available")
	return null;
    }
}

const getBotLobbyIndices = (weightedLobbyPool, lobbyId) => {
    let lobbyIndices = [];
    for (let j = 0; j < weightedLobbyPool.length; j += 1) {
	currLobby = weightedLobbyPool[j];
	queueLength = currLobby.queuedPlayerIds.length;
	if (queueLength > 0 && currLobby._id !== lobbyId) {
	    lobbyIndices.push(j);
	}
    }
    return lobbyIndices;
}

const checkIndividualTimeout = (log, lobby, lobbyConfig) => {
  const now = moment();
  Players.find({ _id: { $in: lobby.queuedPlayerIds } }).forEach(player => {
  let startTimeAt;
  let inLobby;
    if ("timeoutStartedAt" in player) {
	startTimeAt = moment(player.timeoutStartedAt);
	inLobby = true;
    } else {
	startTimeAt = moment(player.data["lobbyStartAt"]);
	inLobby = false;
    }
    const waitTime = now.diff(startTimeAt, 'seconds');
    const endTimeAt = startTimeAt.add(lobbyConfig.timeoutInSeconds, "seconds");
    const ended = now.isSameOrAfter(endTimeAt);

      if (inLobby && waitTime >= 180 && (waitTime % 30) <= 2) {
	  // Check if there is an available bot lobby
	  const lobbies = GameLobbies.find({
	      status: "running",
	      timedOutAt: { $exists: false },
	      gameId: { $exists: false }
	  }).fetch();
	  const candidateLobby = sampleBotLobby(lobbies, lobby._id);

	  if (!candidateLobby) {
	      return
	  } else {
	      GameLobbies.update(lobby._id, {
		  $pull: {
		      playerIds: player._id,
		      queuedPlayerIds: player._id
		  }
	      });

	      GameLobbies.update(candidateLobby._id, {
		  $addToSet: {
		      playerIds: player._id,
		      queuedPlayerIds: player._id
		  }
	      });

	      Players.update(player._id, {
		  $set : {
		      gameLobbyId: candidateLobby._id
		  }
	      })
	      return;
	  }
      } else if (!inLobby && waitTime >= 60 && (waitTime % 30 <= 2)) {
	  Players.update(player._id, {
	      $set: {
		  exitStatus: "playerLobbyTimedOut",
		  exitAt: new Date(),
		  exitReason: "Thanks for waiting, and sorry that there weren't enough other players for your game to being in a timely fashion!",
		  ["data.lobbyEndAt"] : new Date()
	      }
	  });

	  GameLobbies.update(lobby._id, {
	      $pull: {
		  playerIds: player._id,
		  queuedPlayerIds: player._id
	      }
	  });

	  GameLobbies.update(lobby._id, {
	      $addToSet: {
		  quitPlayerIds: player._id
	      }
	  });
      } else if (!ended || player.timeoutWaitCount <= lobbyConfig.extendCount) {
	  return;
      }

    Players.update(player._id, {
      $set: {
          exitStatus: "playerLobbyTimedOut",
          exitAt: new Date(),
	  exitReason: "Thanks for waiting, and sorry that there weren't enough other players for your game to being in a timely fashion!",
	  ["data.lobbyEndAt"] : new Date()
      }
    });

    GameLobbies.update(lobby._id, {
      $pull: {
          playerIds: player._id,
	  queuedPlayerIds: player._id
      }
    });

      GameLobbies.update(lobby._id, {
	  $addToSet: {
	      quitPlayerIds: player._id
	  }
      });

  });
};

Cron.add({
  name: "Check lobby timeouts",
  interval: 1000,
  task: function(log) {
    const query = {
      status: "running",
      gameId: { $exists: false },
      timedOutAt: { $exists: false }
    };

    GameLobbies.find(query).forEach(lobby => {
      const lobbyConfig = LobbyConfigs.findOne(lobby.lobbyConfigId);

      switch (lobbyConfig.timeoutType) {
        case "lobby":
          checkLobbyTimeout(log, lobby, lobbyConfig);
          break;
        case "individual":
          checkIndividualTimeout(log, lobby, lobbyConfig);
          break;
        default:
          log.error(
            `unknown LobbyConfig.timeoutType: ${lobbyConfig.timeoutType}`
          );
      }
    });
  }
});
