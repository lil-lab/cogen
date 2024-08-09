import React from "react";
import Author from "./Author";
import Timer from "./Timer";

export default class ChatLog extends React.Component {
    state = { comment: "" };

    handleEmoji = (e) => {
	e.preventDefault();
	const text = e.currentTarget.value;
	console.log(text);
	const { round, player, stage, timeRemaining } = this.props;
	console.log(stage);
	console.log(timeRemaining);
	round.append("chat", {
	    text,
	    playerId: player._id,
	    target: round.get("target"),
	    role: player.get("role"),
	    type: "message",
	    time: new Date(),
	    secUntilSend: 60 - timeRemaining,
	});
    };

    handleChange = (e) => {
	const el = e.currentTarget;
	this.setState({ [el.name]: el.value });

	const { round, timeRemaining } = this.props;
    };

    handleSubmit = (e) => {
	e.preventDefault();
	const text = this.state.comment.trim();
	const { round, player, stage, timeRemaining } = this.props;

	if (text !== "") {
	    console.log("message sent after seconds: ", 60 - timeRemaining);

	    round.append("chat", {
		text,
		playerId: player._id,
		target: round.get("target"),
		role: player.get("role"),
		time: new Date(),
		secUntilSend: 60 - timeRemaining,
	    });
	    this.setState({ comment: "" });
	}
    };

    render() {
	const { comment } = this.state;
	const { messages, player, round, stage, timeRemaining } = this.props;

	var placeholder = "Enter chat message";

	var disableAttribute = null;
	if (player.get("role") == "listener") {
	    disableAttribute = "disabled";
	    placeholder = "You are the listener. You can't send a message";
	}

	if (player.get("role") == "speaker") {
	    if (messages.length == 0) {
		placeholder = "You can send only one message";
	    } else {
		disableAttribute = "disabled";
		placeholder = "You have already sent one message";
	    }
	}

	if (
	    timeRemaining !== -1 &&
		timeRemaining <= 15 &&
		player.get("role") === "speaker"
	) {
	    disableAttribute = "disabled";
	    placeholder =
		"The next 15s is the selection stage. You can't send messages.";
	}

	return (
	    <div className="chat bp3-card">
		<Messages messages={messages} player={player} />
		<form onSubmit={this.handleSubmit}>
		    <div className="bp3-control-group">
			<input
			    name="comment"
			    type="text"
			    className="bp3-input bp3-fill"
			    placeholder={placeholder}
			    value={comment}
			    onChange={this.handleChange}
			    autoComplete="off"
			    disabled={disableAttribute}
			/>
			<button
			    type="submit"
			    className="bp3-button bp3-intent-primary"
			    disabled={disableAttribute}
			>
			    Send
			</button>
		    </div>
		</form>
	    </div>
	);
    }
}

const chatSound = new Audio("experiment/unsure.mp3");
class Messages extends React.Component {

    componentDidMount() {
	this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    }

    componentDidUpdate(prevProps) {
	if (prevProps.messages.length < this.props.messages.length) {
	    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
	    chatSound.play();
	}
    }

    render() {
	const { messages, player } = this.props;

	return (
	    <div className="messages" ref={(el) => (this.messagesEl = el)}>
		{messages.length === 0 ? (
		    <div className="empty">No messages yet...</div>
		) : null}
		{messages.map((message, i) => (
		    <Message
			key={i}
			message={message}
			self={message.subject ? player._id === message.subject._id : null}
		    />
		))}
	    </div>
	);
    }
}

class Message extends React.Component {
    render() {
	const { text, subject } = this.props.message;
	const { self } = this.props;
	return (
	    <div className="message">
		<Author player={subject} self={self} />
		{text}
	    </div>
	);
    }
}
