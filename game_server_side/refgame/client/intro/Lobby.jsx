import React from "react";
import { Alert, Intent, NonIdealState } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { Centered, shared } from "meteor/empirica:core";

export default class Lobby extends React.Component {
    componentWillMount() {}
    state = {
        remainingTime : (1000 * 60 * 4),
	times_waited : 0,
    }

  render() {
    const { gameLobby, treatment, player } = this.props;
    const total = treatment.factor("playerCount").value;
    const exisiting = gameLobby.playerIds.length;
    const timeElapsed = Date.now() - player.readyAt;

    if (exisiting >= total) {
      return (
          <div className="core">
          <div className="game-lobby">
            <NonIdealState
              icon={IconNames.PLAY}
              title="Game loading..."
              description="Your game will be starting shortly, get ready!"
            />
          </div>
        </div>
      );
    } else {
      return (
       <div className="core">
        <div className="game-lobby">
          <NonIdealState
            icon={IconNames.TIME}
            title="Lobby"
            description={
              <>
                <p>Please wait for the game to be ready...</p>
                <p>
                  {exisiting} / {total} players ready.
                </p>
                  <p> If it takes longer than {(this.state.remainingTime / 1000/ 60).toFixed(0)} minutes to have enough players, you will have the option to leave with compensation. </p>
                  <p> You've been waiting {(timeElapsed / 1000 / 60).toFixed(2)} minutes.</p>
              </>
            }
          />
        </div>
      </div>
    );
    }
  }
}
