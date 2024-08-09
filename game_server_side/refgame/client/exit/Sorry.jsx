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

export default class Sorry extends React.Component {
  static stepName = "Sorry";

  render() {
      const { player, game, hasNext, onSubmit } = this.props;
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
          msg = "Unfortunately the Game was cancelled...";
          break;
      }

      if (player.exitReason) msg = player.exitReason;

      let bonus = player.get("bonus");
      if (bonus === undefined) {
	  bonus = 0.0;
      }

      if (player.get("blame")) {
	  return (
	      <Centered>
		  <div className="score">
		      <h1>Sorry!</h1>
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
		  </div>
	      </Centered>
	  );
      } else {
	  return (
	      <Centered>
		  <div className="score">
		      <h1>Sorry!</h1>
		      <p>{msg}</p>

		      <p>
			  <strong>
			      Your bonus until this point in the game was ${+bonus.toFixed(2) || 0.00}.
			  </strong>{" "}
			  Please submit the HIT with the button below to receive the pay for your time. 
			  Email us at{" "}
			  <a href="mailto: lillabcornell@gmail.com">lillabcornell@gmail.com</a> if
			  you have any questions or concerns. You can also join{" "}
			  <a href="fill_in" target="_blank">
			      our Discord server
			  </a>
			  {" "}for bug reports and early announcements of HITs.
		      </p>

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
}
