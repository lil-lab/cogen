import React, { Component } from 'react';
import { Button } from "@blueprintjs/core";
import { Centered } from "meteor/empirica:core";
const crypto = require('crypto')

export default class NewPlayer extends Component {
    constructor(props){
	super(props);

	// Get the URL information
	const searchParams = new URL(document.location).searchParams;
	const workerId = searchParams.get("workerId");
	const hashedId = crypto.createHash('md5').update(workerId).digest('hex');
	const assignmentId = searchParams.get("assignmentId");
	const workerAssignment = hashedId + "_" + assignmentId;
	this.state = {id : workerAssignment};
    }

    handleSubmit = event => {
	event.preventDefault();

	const {handleNewPlayer} = this.props;
	const {id} = this.state;

	handleNewPlayer(id);
    };

    render() {
	// Get the URL parameters and set state
	//this.setState({ worker : workerId, assignment : assignmentId})

	return (
	    <Centered>
		<div className="welcome">
		    <h1 className={"bp3-heading"}> Welcome </h1>

		    <p> Welcome to our experiment involving tangram reference games!</p>

		    <p> You will first be presented with a review of the tutorial given in the qualification task and then be placed into a lobby where you will be paired with another player.</p>

		    <p>Click on the "Next" button below to proceed.</p>

		    <button
			type="button"
			className="bp3-button bp3-intent-primary"
			onClick={this.handleSubmit}
		    >
			Next
			<span className="bp3-icon-standard bp3-icon-double-chevron-right bp3-align-right" />
		    </button>
		    
		</div>
	    </Centered>
	);
    }
}
