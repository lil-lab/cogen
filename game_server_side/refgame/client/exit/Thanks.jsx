import React from "react";

import { Centered } from "meteor/empirica:core";
import { Button } from "@blueprintjs/core";

const submitHIT = () => {
    const searchParams = new URL(document.location).searchParams;
    
    // create the form element and point it to the correct endpoint
    const form = document.createElement('form')
    form.action = (new URL('mturk/externalSubmit', searchParams.get('turkSubmitTo'))).href
    form.method = 'post'
 
    // attach the assignmentId
    const inputAssignmentId = document.createElement('input')
    inputAssignmentId.name = 'assignmentId'
    inputAssignmentId.value = searchParams.get('assignmentId')
    inputAssignmentId.hidden = true
    form.appendChild(inputAssignmentId)
 
    const inputCoordinates = document.createElement('input')
    inputCoordinates.name = 'coordinates'
    inputCoordinates.value = 'hello'
    inputCoordinates.hidden = true
    form.appendChild(inputCoordinates)

    // attach the form to the HTML document and trigger submission
    document.body.appendChild(form)
    form.submit()
}


export default class Thanks extends React.Component {
  static stepName = "Thanks";

  componentWillMount() {}

    exitMessage = (player, game) => {
	return (
	    <div>
		{" "}
		<h1> Experiment Completed </h1>
		<br />
		<p>
		    <strong>
			Your final bonus is ${+player.get("bonus").toFixed(2) || 0.00}.
		    </strong>{" "}
		</p>
		<p>
		    Thank you again for participating! Please submit the HIT using the button below
		    to be able to receive compensation. Email us at{" "}
		    <a href="mailto: lillabcornell@gmail.com">lillabcornell@gmail.com</a> if
	            you have any questions or concerns. You can also join{" "}
	            <a href="fill_in" target="_blank">
			our Discord server
		    </a>
	            {" "}for bug reports and early announcements of HITs.
		</p>
	    </div>
	);
    };

    render() {
	const { player, game } = this.props;
	if (!game) {
	    return <h1> Error generating code! Please contact requester. </h1>;
	}
	return (
	    <Centered>
		<div className="game finished">
		    {this.exitMessage(player, game)}
		    <hr />
		    <div className="pt-non-ideal-state-description"></div>

		    <button
			type="button"
			className="bp3-button bp3-intent-primary"
			onClick={submitHIT}
		    >
			Submit HIT
			<span className="bp3-icon-standard bp3-icon-double-chevron-right bp3-align-right" />
		    </button>
		</div>
	    </Centered>
	);
    }
}
