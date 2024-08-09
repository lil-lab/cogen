import React from "react";

import { Centered, ConsentButton } from "meteor/empirica:core";
import BrowserDetection from "react-browser-detection";

export default class Consent extends React.Component {
  static renderConsent() {
      const searchParams = new URL(document.location).searchParams;
      const workerId = searchParams.get("workerId");
      const assignmentId = searchParams.get("assignmentId");

      return (
	  <Centered>
	      <div className="consent bp3-ui-text">
		  <h1 className="bp3-heading">REFERENCE GAME TASK</h1>
		  <p><strong>
			 For bug reports, early announcements about future HITs, and feedback join our DISCORD
			 server:{" "}
			 <a href="fill_in" target="_blank">
			     fill_in
			 </a>. You can also contact us at{" "}
			 <a href="mailto: lillabcornell@gmail.com">
			     lillabcornell@gmail.com
			 </a>.
		     </strong></p>

		  <h3 className="bp3-heading">TASK OVERVIEW</h3>
		  <p>
		      First time workers will be required to complete a qualification quiz. This task itself
		      includes three stages (once you complete the short qualifier):
		  </p>
		  <ul>
		      <li>
			  You will enter a lobby to be  matched with your partner. A sound will play after you
			  get matched. If you do not get matched with a partner in 4 minutes, you will have the
			  option of submitting the HIT.
		      </li>
		      <li>
			  You will play <strong>41 rounds of reference games with a partner (human or AI)</strong>,
			  where you will be assigned either a speaker or a listener role. 
		      </li>
		      <li>
			  You will complete a survey, and submit the HIT.
		      </li>
		  </ul>
		  
		  <h3 className="bp3-heading">
		      <a href="https://lil-lab.github.io/tangrams-refgame-dev/" target="_blank">
			  VIDEO DEMONSTRATION AND GAME RULES
		      </a>
		  </h3>

		  <h3 className="bp3-heading">PAYMENT INFORMATION</h3>
		  <p>
		      A complete game usually takes between <strong>20-30 minutes</strong>. Each round is at most{" "}
		      <strong>60 seconds.</strong>
		  </p>

		  <p>
		      You will receive a base pay of <strong>$0.60</strong> for completing the HIT. On top of the base
		      pay, you will receive a pay of <strong>$0.05</strong> for each round you complete (distributed as bonus).
		      If you complete a round with a correct guess, you will receive a further <strong>$0.075</strong> bonus
		      for that round. The total pay will vary based on performance in the game but should be over <strong>$14/hr</strong> on average.
		      {" "}Note that we will be storing encrypted worker IDs to be able to assign bonuses. 
		  </p>

		  <p>
		      You will skip a round if you do not do anything within the allotted time. You will not receive any
		      bonus for such rounds. If you skip two consecutive rounds, we will assume you have abandoned the game
		      and terminate the game. The HIT will be considered as incomplete, and you will not be able to submit it.
		  </p>

		  <h3 className="bp3-heading">CONSENT FORM</h3>
		  <p>
		      The Principal Investigator (PI) associated with this study is Yoav Artzi (Department of Computer Science at Cornell University).
		      Before you choose to accept this HIT, please review{" "}
		      <a href="https://bit.ly/tangram-consent-form" target="_blank">
			  our consent form.
		      </a>
		  </p>
		  {
		      ((workerId && assignmentId)) ? this.IAgreeButton() : ""
		  }
	      </div>
	  </Centered>
      );
  }

  static IAgreeButton = () => {
    return (
      <>
        <p>By clicking "I agree", you acknowledge that you are 18 years or older,
          have read this consent form, agree to its contents, and agree to take
          part in this research. If you do not wish to consent, close this page
          and return the task.</p>
        <ConsentButton text="I AGREE" />
      </>
    );
  };

  renderNoFirefox = () => {
    console.log("this is fire fox");
    return (
      <div className="consent">
        <h1
          className="bp3-heading"
          style={{ textAlign: "center", color: "red" }}
        >
          DO NOT USE FIREFOX!!
        </h1>
        <p style={{ textAlign: "center" }}>
          Please, don't use firefox! It breaks our game and ruins the experience
          for your potential teammates!
        </p>
      </div>
    );
  };

  render() {
    const browserHandler = {
      default: (browser) =>
        browser === "firefox"
          ? this.renderNoFirefox()
          : Consent.renderConsent(),
    };

    return (
      <Centered>
        <BrowserDetection>{browserHandler}</BrowserDetection>
      </Centered>
    );
  }
}
