import React from "react";

function htmlToElements(html) {
    var template = document.createElement("template");
    template.innerHTML = html;
    return template.content.childNodes;
}

export default class Tangram extends React.Component {

    handleClick = (e) => {
	const { game, tangram, tangram_num, stage, player, round, timeRemaining } =
	      this.props;
	const speakerMsgs = _.filter(round.get("chat"), (msg) => {
	    return (msg.role == "speaker") & (msg.playerId == player.get("partner"));
	});
	const partner = _.find(
	    game.players,
	    (p) => p._id === player.get("partner")
	);

	// only register click for listener and only after the speaker has sent a message
	if (
	    (speakerMsgs.length > 0) &&
		(round.get("clickedTangram") === "no_clicks") &&
		(player.get("role") == "listener")
	) {
	    round.set("clickedTangram", tangram["path"]);
	    if (stage.name == "description") {
		round.set("secUntilClick", 60 - timeRemaining);
	    }

	    Meteor.setTimeout(() => player.stage.submit(), 1500);
	    Meteor.setTimeout(() => partner.stage.submit(), 1500);
	}
    };

    render() {
	const {
	    game,
	    tangram,
	    tangram_num,
	    round,
	    stage,
	    player,
	    target,
	    timeRemaining,
	} = this.props;
	const tangram_path = tangram["path"];
	const tangram_data = tangram["data"];
	var colors = undefined;
	if ("coloring-reassigment" in tangram) {
	    colors = tangram["coloring-reassigment"];
	}
	const tangramPositions = {0: {"row": 1, "column": 1},
				  1: {"row": 1, "column": 2},
				  2: {"row": 1, "column": 3},
				  3: {"row": 1, "column": 1},
				  4: {"row": 1, "column": 2},
				  5: {"row": 1, "column": 3},
				  6: {"row": 1, "column": 4},
				  7: {"row": 1, "column": 1},
				  8: {"row": 1, "column": 2},
				  9: {"row": 1, "column": 3},
				 }
	const currentPosition = tangramPositions[tangram_num]

	const mystyle = {
	    backgroundSize: "cover",
	    width:  "auto",
	    height: "auto",
	    display: "inline-block",
	    margin: "15px",
	};

	// Highlight target object for speaker at selection stage
	// Show it to both players at feedback stage.
	if (
	    (target == tangram_path) & (player.get("role") == "speaker")
	) {
	    _.extend(mystyle, {
		outline: "10px solid #000",
		outlineOffset: "4px",
		zIndex: "9",
	    });
	}

	// Highlight clicked object in green if correct; red if incorrect
	let clickedPath;
	if (round.get("clickedTangram") != "no_clicks"){
	    clickedPath = round.get("clickedTangram");
	    const color = clickedPath === target ? "green" : "red";
	    if (
		((target === tangram_path) & (player.get("role") == "speaker")) ||
		    ((tangram_path === clickedPath) & (player.get("role") == "listener"))
	    ) {
		_.extend(mystyle, {
		    outline: `10px solid ${color}`,
		    zIndex: "9",
		});
	    }
	}

	var elements = htmlToElements(tangram_data);
	for (let i = 0; i < elements.length; i++) {
	    if (elements[i].nodeName == "svg") {
		var svg = elements[i];
	    }
	}
	var childrenArray = Array.prototype.slice.call(svg.childNodes);

	var bodyElement = document.evaluate(
	    "/html/body",
	    document,
	    null,
	    XPathResult.FIRST_ORDERED_NODE_TYPE,
	    null
	).singleNodeValue;

	var numRows = 3 ;
	var minSize, tangramWidth, tangramHeight;
	minSize = Math.min(bodyElement.offsetWidth, bodyElement.offsetHeight);
	tangramWidth =  minSize / 2 / numRows;
	tangramHeight = minSize / 2 / numRows;

	if ((4 * tangramWidth >= 0.5 * bodyElement.offsetWidth) || (3 * tangramHeight >= bodyElement.offsetHeight)) {
	    tangramWidth =  minSize / 4 / numRows;
	    tangramHeight = minSize / 4 / numRows;
	}


	return (
	    <div id={tangram_path} onClick={this.handleClick} style={mystyle}>
		<svg
		    baseProfile="full"
		    viewBox={svg.getAttribute("viewBox")}
		    width={tangramWidth}
		    height={tangramHeight}
		    xmlns="http://www.w3.org/2000/svg"
		>
		    {childrenArray.map((node, index) => {
			if (node.nodeName == "polygon") {
			    if (
				colors === undefined ||
				    !(node.getAttribute("id") in colors)
			    ) {
				var colorFill = "#1C1C1C"; 
			    } else {
				var colorFill = colors[node.getAttribute("id")];
			    }
			    var id = tangram_path + "_" + node.getAttribute("id");
			    return (
				<polygon
				    key={id}
				    id={id}
				    fill={colorFill}
				    points={node.getAttribute("points")}
				    stroke={colorFill} 
				    strokeWidth={"2"} 
				    transform={node.getAttribute("transform")}
				/>
			    );
			}
		    })}
		</svg>
	    </div>
	);
    }
}
