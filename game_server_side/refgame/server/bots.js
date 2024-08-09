import Empirica from "meteor/empirica:core";
import {HTTP } from "meteor/http"


function sendMessage(round, bot, message) {
    round.append("chat", {
	text: message, // "I am a bot sending a message"
	playerId: bot._id,
	role: bot.get('role'),
	type: "message",
	time: new Date()
    });
}

function chooseTarget(target_path, game, round, bot) {
    const partner = _.find(game.players, p => p._id === bot.get('partner'));
    round.set("clickedTangram", target_path);
    round.set("clickedTime", new Date());
    Meteor.setTimeout(() => bot.stage.submit(), 3000);
    Meteor.setTimeout(() => partner.stage.submit(), 3000);
}

function getDescription(data, game, round, bot, botIP, remainingTime) {
    HTTP.call( 'POST', botIP + '/generate_description', {
	data: data
    }, function( error, response ) {
	if ( error ) {
	    console.log('error getting stims');
	    console.log( error );
	} else {
	    const content = JSON.parse(response.content);
	    round.set('botActionTime', content.timePassed);
	    round.set('reportedGameId', content.gameId)
	    round.set('reportedRoundId', content.roundId)
	    sendMessage(round, bot, content.description)
	    console.debug("Speaker bot prediction finished: Round ", round._id, "game", game._id, " starting ", 60 - remainingTime, " seconds into the round");
	}
    });
}

function predictTarget(data, game, round, bot, botIP, remainingTime) {
    //    HTTP.call( 'POST', 'http://' + botIP + ':8080/predict_target', {
    HTTP.call( 'POST', botIP + '/predict_target', {
	data: data
    }, function( error, response ) {
	if ( error ) {
	    console.log('error getting stims');
	    console.log( error );
	} else {
	    const content = JSON.parse(response.content);
	    round.set('botActionTime', content.timePassed);
	    round.set('reportedGameId', content.gameId)
	    round.set('reportedRoundId', content.roundId)
	    if (content.gameId != game._id && content.roundId != round._id) {
		content.path = round.get('target');
		round.set('responseFailure', true);
	    }
	    console.debug("Listener bot prediction finished: Round ", round._id, "game", game._id, " starting ", 60 - remainingTime, " seconds into the round");
	    chooseTarget(content.path, game, round, bot)
	}
    });
}

Empirica.bot("bob", {
    // Called during each stage at tick interval (~1s at the moment)
    onStageTick(bot, game, round, stage, secondsRemaining) {

	//console.debug("Round ", round._id, "game", game._id, " call started with tick ", secondsRemaining);	

    // if the bot is a speaker, generate a description
    if (bot.get('role') === "speaker") {
	const speakerMsgs = _.filter(round.get("chat"), msg => {
	    return msg.role == "speaker"
	});
	if ((speakerMsgs.length === 0) && !(round.get('botActed'))) {
	    round.set('botActed', true);
	    round.set('botRequestStartTimestamp', new Date())
	    round.set('botRequestStartSecs', 60 - secondsRemaining)
	    console.debug("Speaker bot prediction: Round ", round._id, "game", game._id, " starting ", 60 - secondsRemaining, " seconds into the round");

	    const tangramURLs = round.get('tangrams')[0];
	    const target = round.get('target');
	    const bot_treatment = round.get('botTreatment')
	    const botIP = game.get(bot_treatment + "_IP")
	    const gameId = game._id
	    const roundId = round._id

            const data = {'image_paths': tangramURLs, 'target': target,
		    'bot_treatment' : bot_treatment, 'round_id' : roundId,
		    'attnCheckAnno' : "", 'game_id' : gameId} 

	    if (round.get('isAttnCheck')) {
		data["attnCheckAnno"] = round.get("attnCheckAnno");
	    }
	    if (round.get('precomputedBot')) {
		data["attnCheckAnno"] = round.get("botPrecomputedUtterance");
	    }


            getDescription(data, game, round, bot, botIP, secondsRemaining);
	}
    }

    // if the bot is a listener, predict the target
    if (bot.get('role') === "listener") {
	const speakerMsgs = _.filter(round.get("chat"), msg => {
            return msg.role == 'speaker' & msg.playerId == bot.get('partner')
	})

	if ((speakerMsgs.length > 0) && !(round.get('botActed'))) {
	    round.set('botActed', true);
	    round.set('botRequestStartTimestamp', new Date())
	    round.set('botRequestStartSecs', 60 - secondsRemaining)
	    console.debug("Listener bot prediction: Round ", round._id, "game", game._id, " starting ", 60 - secondsRemaining, " seconds into the round");

            const lastMsg = speakerMsgs[speakerMsgs.length-1].text;
	    const tangramURLs = round.get('tangrams')[1];
	    const target = round.get('target');
	    const bot_treatment = round.get('botTreatment');
	    const botIP = game.get(bot_treatment + "_IP")
	    const gameId = game._id
	    const roundId = round._id

            const data = {'image_paths': tangramURLs, 'description': lastMsg,
			  'target' : target, 'bot_treatment' : bot_treatment,
			  'round_id' : roundId, 'game_id' : gameId};

            predictTarget(data, game, round, bot, botIP, secondsRemaining);
      }
    }

	//console.debug("Round ", round._id, "game", game._id, " call ended with tick ", secondsRemaining);	
  }
});
