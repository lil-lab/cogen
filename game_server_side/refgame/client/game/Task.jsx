import React from "react";

import Tangram from "./Tangram.jsx";

export default class Task extends React.Component {

    constructor(props) {
	super(props);

	// We want each participant to see tangrams in a random but stable order
	// so we shuffle at the beginning and save in state
	this.state = {
	    activeButton: false,
	};
    }

    render() {
	const { game, round, stage, player, timeRemaining } = this.props;
	const target = round.get("target");
	if (player.get("role") == "speaker") {
	    var tangramURLs = round.get("tangrams")[0];
	} else {
	    var tangramURLs = round.get("tangrams")[1];
	}

	var correct;
	if (round.get("clickedTangram") != "no_clicks") {
	    correct = round.get("clickedTangram") === target;
	} else {
	    correct = false;
	}

	let tangramsToRender;
	if (tangramURLs) {
	    tangramsToRender = tangramURLs.map((tangram, i) => (
		<Tangram
		    key={tangram["path"]}
		    tangram={tangram}
		    tangram_num={i}
		    round={round}
		    stage={stage}
		    game={game}
		    player={player}
		    target={target}
		    timeRemaining={timeRemaining}
		/>
	    ));
	}

	const speakerMsgs = _.filter(round.get("chat"), (msg) => {
	    return msg.role == "speaker";
	});

	let feedback = "";
	if (round.get("clickedTangram") === "no_clicks") {
	    if (timeRemaining !== -6 && timeRemaining <= 0) {
		if (player.get('role') === 'speaker') {
		    feedback = "Sorry, the listener didn't make a prediction in time. You earned no correctness bonus this round."
		} else {
		    feedback = "Time's up! You did not make a guess this round."
		}
	    } else if (
		timeRemaining !== -1 &&
		    timeRemaining < 15 &&
		    speakerMsgs.length === 0
	    ) {
		if (player.get('role') === 'speaker') {
		    feedback = "Time's up! You did not send a message this round"
		} else {
		    feedback = "Sorry, the speaker didn't sent a description in time. You earned no correctness bonus this round.";
		}
	    } else if (player.get('numIdleRounds') > 0) {
		feedback = "You idled during the last turn you had a chance to act. If you idle for two consecutive turns, the game will end and you will not receive pay.";
	    }
	    
	} else {
	    if (correct) {
		feedback = "Correct! You earned a $0.075 correctness bonus!";
	    } else {
		feedback =
		    "Oops, that wasn't the target! You earned no correctness bonus this round.";
	    }
	}

	return (
	    <div className="task" style={{ display: "inline-block" }}>
		<div className="board">
		    <h1 className="roleIndicator">
			{" "}
			You are the{" "}
			<span
			    style={{
				color: player.get("role") === "speaker" ? "red" : "blue",
				fontWeight: "bold",
			    }}
			>
			    {player.get("role")}
			</span>
			.
		    </h1>
		    <div className="all-tangrams">
			<div className="tangrams">
			    <div style={{ marginLeft: "80px" }}>
				{tangramsToRender[0]}
				{tangramsToRender[1]}
				{tangramsToRender[2]}
			    </div>
			    <div style={{}}>
				{tangramsToRender[3]}
				{tangramsToRender[4]}
				{tangramsToRender[5]}
				{tangramsToRender[6]}
			    </div>
			    <div style={{ marginLeft: "80px" }}>
				{tangramsToRender[7]}
				{tangramsToRender[8]}
				{tangramsToRender[9]}
			    </div>
			</div>
		    </div>

		    <h3 className="feedbackIndicator">{feedback}</h3>
		</div>
	    </div>
	);
    }
}
