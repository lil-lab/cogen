import React from "react";
import EventLog from "./EventLog";
import ChatLog from "./ChatLog";
import Timer from "./Timer";

export default class SocialInteractions extends React.Component {
    renderPlayer(player, self = false) {
	var key, color, avatar, playerName;
	if (!player) {
	    key = undefined;
	    color = undefined;
	    avatar = undefined;
	    playerName = undefined;
	}
	else {
	    key = player._id;
	    color = player.get("nameColor");
	    avatar = player.get("avatar");
	    playerName = player.get("name");
	}

	return (
	    <div className="player" key={key}>
		<span className="image"></span>
		<img src={avatar} />
		<span className="name" style={{ color: color }}>
		    {playerName}
		    {self ? " (You)" : " (Partner)"}
		</span>
	    </div>
	);
    }

    constructor() {
	super();
	this.state = {timeRemaining: -1};
    }

    setTimeRemaining = (time) => {
	this.props.getTimeRemaining(time); // pass time remaining up to Round component
	this.setState({timeRemaining: time});
    }

    render() {
	const { game, round, stage, player } = this.props;
	const partnerId = player.get('partner')
	const partner = _.filter(game.players, p => p._id === partnerId)[0];
	let timeRemainRound = this.state.timeRemaining - 5

	const messages = round.get("chat")
              .filter(({playerId}) => playerId === partnerId || playerId === player._id)
              .map(({ text, playerId }) => ({
		  text,
		  subject: game.players.find(p => p._id === playerId)
              }));

	const events = stage.get("log").map(({ subjectId, ...rest }) => ({
	    subject: subjectId && game.players.find(p => p._id === subjectId),
	    ...rest
	}));

	if (messages.length !== 0) {
	    if (player.get("role") === "listener" && round.get('listenerObservesMessageSec') === -1) {
		round.set('listenerObservesMessageSec', 60 - timeRemainRound);
	    }
	}

	return (
	    <div className="social-interactions" style={{width: "30%", display: "inline-block"}}>
		<div className="status">
		    <div className="players bp3-card">
			{this.renderPlayer(player, true)}
			{this.renderPlayer(partner)}
		    </div>

		    <Timer stage={stage} player={player} setTimeRemaining={this.setTimeRemaining} />
        
		    <div className="total-score bp3-card">
			<h5 className='bp3-heading'>Bonus</h5>

			<h2 className='bp3-heading'>${(player.get("bonus") || 0).toFixed(2)}</h2>
		    </div>
		</div>
		<ChatLog messages={messages} round={round} stage={stage} player={player} timeRemaining={timeRemainRound}/>
		<EventLog events={events} round={round} game={game} stage={stage} player={player} />
	    </div>
	);
    }
}
