import React from "react";

import {StageTimeWrapper} from "meteor/empirica:core";
import Timer from "./Timer.jsx";

class timer extends React.Component {
    render() {
	let { remainingSeconds, player, setTimeRemaining } = this.props;

	setTimeRemaining(remainingSeconds) // "actual" time remaining (out of 60s)
	let timeRemainRound = remainingSeconds - 5

	if (player.get("role") == 'speaker') {
	    timeRemainRound -= 15 // speaker timer is 15s less than listener
	}

	let minutes = ("0" + Math.floor(timeRemainRound / 60)).slice(-2);
	let seconds = ("0" + (timeRemainRound - minutes * 60)).slice(-2);
	if (timeRemainRound < 0){
	    // speaker timer goes to 00:00; listener still has 15s for selection
	    minutes = "00"
	    seconds = "00"
	}
  
	const classes = ["timer", "bp3-card"];
	if (timeRemainRound <= 5) {
	    classes.push("lessThan5");
	} else if (timeRemainRound <= 10) {
	    classes.push("lessThan10");
	}

	return (
	    <div className={classes.join(" ")}>
		<h5 className='bp3-heading'>Timer</h5>
		<span className="seconds">{minutes}:{seconds}</span>
	    </div>
	);
    }
}

export default (Timer = StageTimeWrapper(timer));
