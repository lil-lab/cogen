import React from "react";
import { createPlayer } from "../../api/players/methods";
import { setPlayerId } from "../containers/IdentifiedContainer.jsx";
import Loading from "./Loading.jsx";
import NewPlayerForm from "./NewPlayerForm.jsx";
import { AlertToaster } from "./Toasters.jsx";

const { playerIdParam } = Meteor.settings.public;
const crypto = require('crypto');

export const ConsentButtonContext = React.createContext(null);

export default class NewPlayer extends React.Component {
  state = { consented: false };

  componentDidUpdate(prevProps, prevState) {
    if (this.state.consented && !prevState.consented) {
      this.timeout = setTimeout(() => {
        this.timeout = null;
        const idField = window.document.querySelector("#id");
        if (idField) {
          idField.focus();
        }
      }, 100);
    }
  }

  componentWillMount() {
    if (!this.props.Consent) {
      this.playerFromIdParam();
    }
  }

  componentWillUnmount() {
    if (this.timeout) {
      clearTimeout(this.timeout);
    }
  }

  handleConsent = () => {
    this.setState({ consented: true });
    this.playerFromIdParam();
  };

    handleNewPlayer = id => {
	if (!id || !id.trim()) {
	    AlertToaster.show({ message: "Player Identifier cannot be empty!" });
	    return;
	}


	const urlParams = {};
	const searchParams = new URL(document.location).searchParams;
	for (var pair of searchParams.entries()) {
	    if (pair[0] != "workerId") {
		console.log(pair[0]);
		urlParams[pair[0]] = pair[1];
	    } 
	}

	createPlayer.call({ id, urlParams }, (err, _id) => {
	    if (err) {
		console.error(err);
		AlertToaster.show({ message: String(err) });
		return;
	    }

	    setPlayerId(_id);
	});
    };

  playerFromIdParam() {
      const { playerIdParam } = Meteor.settings.public;

      if (playerIdParam) {
	  const searchParams = new URL(document.location).searchParams;
	  const workerId = searchParams.get("workerId");
	  const assignmentId = searchParams.get("assignmentId");

	  if (workerId && assignmentId) {
	      const hashedId = crypto.createHash('md5').update(workerId).digest('hex');
	      const id = hashedId + "_" + assignmentId;

              const urlParams = {};
              for (var pair of searchParams.entries()) {
		  if (pair[0] != "workerId") {
		      urlParams[pair[0]] = pair[1];
		  } else {
		      urlParams[pair[0]] = hashedId;
		  }
              }


              this.setState({ attemptingAutoLogin: true });
              createPlayer.call({ id, urlParams }, (err, _id) => {
		  if (err) {
		      this.setState({ attemptingAutoLogin: false });
		      console.error(err);
		      AlertToaster.show({ message: String(err) });
		      return;
		  }
		  
		  setPlayerId(_id);
              });
	  }
      }
  }

  render() {
    const { Consent, CustomNewPlayer } = this.props;
    const { consented, attemptingAutoLogin } = this.state;
    const Form = CustomNewPlayer || NewPlayerForm;

    if (attemptingAutoLogin) {
      return <Loading />;
    }

    if (!consented && Consent) {
      return (
        <ConsentButtonContext.Provider value={this.handleConsent}>
          <Consent {...this.props} onConsent={this.handleConsent} />
        </ConsentButtonContext.Provider>
      );
    }

    return <Form {...this.props} handleNewPlayer={this.handleNewPlayer} />;
  }
}
