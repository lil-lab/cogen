import { render } from "react-dom";

import "@blueprintjs/icons/lib/css/blueprint-icons.css";
import "@blueprintjs/core/lib/css/blueprint.css";

import Empirica from "meteor/empirica:core";

import Consent from "./intro/Consent.jsx";
import Lobby from "./intro/Lobby.jsx";
import NewPlayer from "./intro/NewPlayer.jsx";

import Round from "./game/Round.jsx";
import Thanks from "./exit/Thanks.jsx";
import Sorry from "./exit/Sorry";
import ExitSurvey from "./exit/ExitSurvey";
import ErrorSurvey from "./exit/ErrorSurvey";
import customBreadcrumb from "./game/Breadcrumb.jsx";

// Set the Consent Component you want to present players (optional).
Empirica.consent(Consent);

// Set a custom Lobby
Empirica.lobby(Lobby)

// Introduction pages to show before they play the game (optional).
Empirica.introSteps((game, player, treatment) => {
  const steps = []; 
  return steps;
});

// The Round component containing the game UI logic.
// This is where you will be doing the most development.
// See client/game/Round.jsx to learn more.

Empirica.round(Round);

// End of Game pages. These may vary depending on player or game information.
// For example we can show the score of the user, or we can show them a
// different message if they actually could not participate the game (timed
// out), etc.
// The last step will be the last page shown to user and will be shown to the
// user if they come back to the website.
// If you don't return anything, or do not define this function, a default
// exit screen will be shown.
Empirica.exitSteps((game, player) => {
    if (player.exitStatus !== "finished") {
	return [ErrorSurvey, Sorry];
    }
    return [ExitSurvey, Thanks];
});

// Empirica.breadcrumb would probably go here
Empirica.breadcrumb(customBreadcrumb);

// Start the app render tree.
// NB: This must be called after any other Empirica calls (Empirica.round(),
// Empirica.introSteps(), ...).
// It is required and usually does not need changing.
Meteor.startup(() => {
  render(Empirica.routes(), document.getElementById("app"));
});
