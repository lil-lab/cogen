import React from "react";

import { Centered } from "meteor/empirica:core";

import {
  Button,
  Classes,
  FormGroup,
  RadioGroup,
  TextArea,
  Intent,
  Radio,
} from "@blueprintjs/core";

export default class ErrorSurvey extends React.Component {
    static stepName = "ExitSurvey";
    state = {
	feedback: "",
    };

    handleChange = (event) => {
	const el = event.currentTarget;
	this.setState({ [el.name]: el.value });
    };

    handleSubmit = (event) => {
	event.preventDefault();
	
	const { player, game } = this.props;
	player.set('errorSurveyResponses', this.state)

	console.log(this.state);
	console.log(this.props.onSubmit);
	this.props.onSubmit(this.state);
    };

    surveyForm = (player) => {
	const {feedback} = this.state;

	let msg;
	switch (player.exitStatus) {
	case "gameFull":
            msg = "All games you are eligible for have filled up too fast...";
            break;
	case "gameLobbyTimedOut":
            msg = "There were NOT enough players for the game to start..";
            break;
	case "playerEndedLobbyWait":
            msg =
		"You decided to stop waiting, we are sorry it was too long a wait.";
            break;
	default:
            msg = "Unfortunately this game was cancelled...";
            break;
	}

	if (player.exitReason) msg = player.exitReason;
	return (
	    <div>
		<h1>Your game has ended early</h1>
		<p>{msg}</p>
		<form onSubmit={this.handleSubmit}>
		    <h2>
			Error and bug reporting question
		    </h2>

		    <div className="pt-form-group">
			<div className="pt-form-content">
			    <FormGroup
				className={"form-group"}
				inline={false}
				label={
				    "1. Did you notice any problems or have any other comments about the study?"
				}
				labelFor={"feedback"}
			    >
				<TextArea
				    id="feedback"
				    name="feedback"
				    large={true}
				    intent={Intent.PRIMARY}
				    onChange={this.handleChange}
				    value={feedback}
				    fill={true}
				/>
			    </FormGroup>
			</div>
		    </div>
		    <br />
	      
		    <button type="submit" className="pt-button pt-intent-primary">
			Submit
			<span className="pt-icon-standard pt-icon-key-enter pt-align-right" />
		    </button>
		</form>
	    </div>
	);
    };

    componentWillMount() {}

    idleError = (player) => {
	const msg = player.exitReason;
	return (
	    <>
	    <h1>Your game has ended early</h1>
	    <p>{msg}</p>

	    <p>
		As described in the HIT's first page and in the consent form, you will not receive
		pay if you exit early or idle for two consecutive rounds. Please email us
		at{" "}
		<a href="mailto: lillabcornell@gmail.com">lillabcornell@gmail.com</a> or join{" "}
		<a href="fill_in" target="_blank">
		    our Discord server
		</a>{" "} if you have questions or believe there is an error.
	    </p>
	    </>
	);
    }

    kickedOutMessage = (player) => {
	const msg = player.exitReason;
	return (
	    <>
	    <h1>{msg}</h1>

	    <p>
		We require players to complete our HITs one at a time. You cannot enter a game if you
		are already in the lobby for one or are already playing with someone else. Please 
		email us at{" "}
		<a href="mailto: lillabcornell@gmail.com">lillabcornell@gmail.com</a> or join{" "}
		<a href="fill_in" target="_blank">
		    our Discord server
		</a>{" "} if you have questions or believe there is an error.
	    </p>
	    </>
	);
    }

    render() {
	const { player, game } = this.props;

	if (player.exitStatus === "matchingPlayerKickedOut") {
	    return (
		<Centered>
		    <div className="kicked-out">{this.kickedOutMessage(player)}</div>
		</Centered>
	    );
	} else if (player.get("blame")) {
	    return (
		<Centered>
		    <div className="idle-error">{this.idleError(player)}</div>
		</Centered>
	    );
	} else {
	    return (
		<Centered>
		    <div className="exit-survey">{this.surveyForm(player)}</div>
		</Centered>
	    );
	}
    }
}
