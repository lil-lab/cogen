import React from "react";

import SocialInteractions from "./SocialInteractions.jsx";
import Task from "./Task.jsx";
const crypto = require('crypto');

const roundSound = new Audio("experiment/round-sound.mp3");

export default class Round extends React.Component {

    // Round sound effects
    componentDidMount() {
	const { game } = this.props;
	if (game.get("justStarted")) {
	    const gameSound = new Audio("experiment/start-game.mp3");
	    gameSound.autoplay = true;
	    gameSound.play();
	    game.set("justStarted", false);
	} else {
	    roundSound.play();
	}
    }

    constructor() {
	super();
	this.state = {timeRemaining: -1};
    }

    getTimeRemaining = (time) => {
	this.setState({timeRemaining: time});
    }

    render() {
	const {round, stage, player, game } = this.props;  

	// Record player IP address in hashed form
	if (!player.get("savedIP")) {
	    player.set("savedIP", true);
	    fetch('https://api.ipify.org?format=json')
		.then(response => response.json())
		.then(data => {
		    player.set("hashedIP", crypto.createHash('md5').update(data.ip).digest('hex'));
		})
		.catch(error => {
		    console.log("Game start");
		})
	}

	// end the trial if the speaker didn't send messages.
	let timeRemainRound = this.state.timeRemaining - 5
	if (this.state.timeRemaining !== -1 && timeRemainRound <= 12 && player.get("role") === "listener") {
	    const speakerMsgs = _.filter(round.get("chat"), (msg) => {
		return msg.role == "speaker";
	    });

	    if (speakerMsgs.length === 0 && !round.get("idleStageSubmitted")) {
		round.set("idleStageSubmitted", true);
		const partner = _.find(
		    game.players,
		    (p) => p._id === player.get("partner")
		);
		Meteor.setTimeout(() => player.stage.submit(), 500);
		Meteor.setTimeout(() => partner.stage.submit(), 500);
	    }
	}

	if (this.state.timeRemaining !== -1  && timeRemainRound < 0 && player.get("role") === "speaker") {
	    if (round.get("clickedTangram") === "no_clicks" && !round.get("idleStageSubmitted")) {
		round.set("idleStageSubmitted", true);
		const partner = _.find(
		    game.players,
		    (p) => p._id === player.get("partner")
		);
		Meteor.setTimeout(() => player.stage.submit(), 500);
		Meteor.setTimeout(() => partner.stage.submit(), 500);
	    }
	}

    
	return (
	    <div className="round">
		<SocialInteractions game={game} round={round} stage={stage} player={player} getTimeRemaining={this.getTimeRemaining}/>
		<Task game={game} round={round} stage={stage} player={player} timeRemaining={timeRemainRound}/>
	    </div>
	);
    }
}
