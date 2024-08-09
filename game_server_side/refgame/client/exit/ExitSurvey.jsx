import React from "react";

import { Centered } from "meteor/empirica:core";

import {
  Button,
  Classes,
  FormGroup,
  RadioGroup,
  TextArea,
  Intent,
  Radio,
} from "@blueprintjs/core";

export default class ExitSurvey extends React.Component {
    static stepName = "ExitSurvey";
    state = {
	satisfied: "",
	comprehension: "",
	grammatical: "",
	clear: "",
	ambiguous: "",
	english: "",
	languages: "",
	whereLearn: "",
	feedback: "",
    };

    handleChange = (event) => {
	const el = event.currentTarget;
	this.setState({ [el.name]: el.value });
    };

    demographicHandleSubmit = (event) => {
	event.preventDefault();
	
	// Check if player responded to each required category
	const requiredCategories = [
	    "satisfied",
	    "comprehension",
	    "grammatical",
	    "clear",
	    "ambiguous",
	    "english",
	    "languages",
	    "whereLearn"
	]
	let allowSubmit = true;
	for (let i = 0; i < requiredCategories.length; i += 1) {
	    if (this.state[requiredCategories[i]] === "") {
		allowSubmit = false;
	    }
	}

	if (allowSubmit) {
	    const { player, game } = this.props;
	    player.set('surveyResponses', this.state)
	    player.set("completedDemographics", true);

	    console.log(this.state);
	    console.log(this.props.onSubmit);
	    this.props.onSubmit(this.state);
	} else {
	    console.log("Please answer the unanswered questions");
	}
    };

    nonDemographicHandleSubmit = (event) => {
	event.preventDefault();
	
	// Check if player responded to each required category
	const requiredCategories = [
	    "satisfied",
	    "comprehension",
	    "grammatical",
	    "clear",
	    "ambiguous",
	]
	let allowSubmit = true;
	for (let i = 0; i < requiredCategories.length; i += 1) {
	    if (this.state[requiredCategories[i]] === "") {
		console.log(requiredCategories[i]);
		allowSubmit = false;
	    }
	}

	if (allowSubmit) {
	    const { player, game } = this.props;
	    player.set('surveyResponses', this.state)
	    player.set("completedDemographics", true);

	    console.log(this.state);
	    console.log(this.props.onSubmit);
	    this.props.onSubmit(this.state);
	} else {
	    console.log("Please answer the unanswered questions");
	}
    };


  fullSurvey = (submitFunction) => {
    const {
	satisfied,
	comprehension,
	grammatical,
	clear,
	ambiguous,
	english,
	languages,
	whereLearn,
	feedback,
    } = this.state;

    return (
      <div>
          <h1>Finally, please answer the following short survey.</h1>

          <form onSubmit={submitFunction}>
	      <h2>
		  Game performance questions
	      </h2>
              <h3>
		  You must answer questions in this section to be able to proceed.
              </h3>

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <RadioGroup
			  name="satisfied"
			  label="1. How satisfied are you with your partner's performance in the game?"
			  onChange={this.handleChange}
			  selectedValue={satisfied}
		      >
			  <Radio
			      label="Very satisfied"
			      value="verySatisfied"
			      className={"pt-inline"}
			  />
			  <Radio
			      label="Satisfied"
			      value="satisfied"
			      className={"pt-inline"}
			  />
			  <Radio
			      label="Somewhat satisfied"
			      value="somewhatSatisfied"
			      className={"pt-inline"}
			  />

			  <Radio
			      label="Somewhat dissatisfied"
			      value="somewhatDissatisfied"
			      className={"pt-inline"}
			  />
			  <Radio
			      label="Dissatisfied"
			      value="dissatisfied"
			      className={"pt-inline"}
			  />
			  <Radio
			      label="Very dissatisfied"
			      value="veryDissatisfied"
			      className={"pt-inline"}
			  />
		      </RadioGroup>
		  </div>
              </div>
              <br />
	      
              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <RadioGroup
			  name="comprehension"
			  label="2. On a scale of 1-6 (where 6 is the best), how well did your partner understand your descriptions?"
			  onChange={this.handleChange}
			  selectedValue={comprehension}
		      >
			  <Radio
			      label="6 - understood almost all descriptions"
			      value="6"
			      className={"pt-inline"}
			  />
			  <Radio label="5" value="5" className={"pt-inline"} />
			  <Radio label="4" value="4" className={"pt-inline"} />
			  <Radio
			      label="3"
			      value="3"
			      className={"pt-inline"}
			  />
			  <Radio label="2" value="2" className={"pt-inline"} />
			  <Radio
			      label="1 - did not understand any of my descriptions"
			      value="1"
			      className={"pt-inline"}
			  />
		      </RadioGroup>
		  </div>
              </div>
              <br />

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <RadioGroup
			  name="grammatical"
			  label="3. On a scale of 1-6 (where 6 is the best), how grammatical were your partner's descriptions?"
			  onChange={this.handleChange}
			  selectedValue={grammatical}
		      >
			  <Radio
			      label="6 - there were no grammatical issues"
			      value="6"
			      className={"pt-inline"}
			  />
			  <Radio label="5" value="5" className={"pt-inline"} />
			  <Radio label="4" value="4" className={"pt-inline"} />
			  <Radio
			      label="3"
			      value="3"
			      className={"pt-inline"}
			  />
			  <Radio label="2" value="2" className={"pt-inline"} />
			  <Radio
			      label="1 - almost all descriptions were ungrammatical"
			      value="1"
			      className={"pt-inline"}
			  />
		      </RadioGroup>
		  </div>
              </div>
              <br />

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <RadioGroup
			  name="clear"
			  label="4. On a scale of 1-6 (where 6 is the easiest to understand), how easy to understand were your partner's descriptions?"
			  onChange={this.handleChange}
			  selectedValue={clear}
		      >
			  <Radio
			      label="6 - all descriptions were easy to understand"
			      value="6"
			      className={"pt-inline"}
			  />
			  <Radio label="5" value="5" className={"pt-inline"} />
			  <Radio label="4" value="4" className={"pt-inline"} />
			  <Radio
			      label="3"
			      value="3"
			      className={"pt-inline"}
			  />
			  <Radio label="2" value="2" className={"pt-inline"} />
			  <Radio
			      label="1 - no description was easy to understand"
			      value="1"
			      className={"pt-inline"}
			  />
		      </RadioGroup>
		  </div>
              </div>
              <br />

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <RadioGroup
			  name="ambiguous"
			  label="5. On a scale of 1-6 (where 6 is the easiest), how easy was it to distinguish the target image from other images in the context based on your partner's descriptions?"
			  onChange={this.handleChange}
			  selectedValue={ambiguous}
		      >
			  <Radio
			      label="6 - my partner's descriptions always made the target clear"
			      value="6"
			      className={"pt-inline"}
			  />
			  <Radio label="5" value="5" className={"pt-inline"} />
			  <Radio label="4" value="4" className={"pt-inline"} />
			  <Radio
			      label="3"
			      value="3"
			      className={"pt-inline"}
			  />
			  <Radio label="2" value="2" className={"pt-inline"} />
			  <Radio
			      label="1 - I was never able to choose the target based on my partner's descriptions"
			      value="1"
			      className={"pt-inline"}
			  />
		      </RadioGroup>
		  </div>
              </div>
              <br />

	      <h2>
		  Demographic questions
	      </h2>
              <h3>
		  You must answer questions in this section to be able to proceed. You
		  will not be asked these demographic questions again in future HITs from
		  this study.
              </h3>

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <RadioGroup
			  name="english"
			  label="6. Is English your native language?"
			  onChange={this.handleChange}
			  selectedValue={english}
		      >
			  <Radio label="Yes" value="yes" className={"pt-inline"} />
			  <Radio label="No" value="no" className={"pt-inline"} />
		      </RadioGroup>
		  </div>
              </div>
              <br />

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <FormGroup
			  className={"form-group"}
			  inline={false}
			  label={`7. What languages do you know? How would you rate your proficiency in each language? (1=basic knowledge, 5=native speaker level) 
              Please format the response in the form of "Language(Proficiency)", e.g.: German(5), French(4).`}
			  labelFor={"languages"}
		      >
			  <TextArea
			      id="languages"
			      large={true}
			      intent={Intent.PRIMARY}
			      onChange={this.handleChange}
			      value={languages}
			      fill={true}
			      name="languages"
			  />
		      </FormGroup>
		  </div>
              </div>
              <br />

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <FormGroup
			  className={"form-group"}
			  inline={false}
			  label={
			      "8. Where did you learn English? (If you are a native speaker, please indicate the country where you learned to speak.)"
			  }
			  labelFor={"whereLearn"}
		      >
			  <TextArea
			      id="whereLearn"
			      large={true}
			      intent={Intent.PRIMARY}
			      onChange={this.handleChange}
			      value={whereLearn}
			      fill={true}
			      name="whereLearn"
			  />
		      </FormGroup>
		  </div>
              </div>
              <br />


	      <h2>
		  Feedback for the study
	      </h2>

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <FormGroup
			  className={"form-group"}
			  inline={false}
			  label={
			      "9. Did you notice any problems or have any other comments about the study?"
			  }
			  labelFor={"feedback"}
		      >
			  <TextArea
			      id="feedback"
			      name="feedback"
			      large={true}
			      intent={Intent.PRIMARY}
			      onChange={this.handleChange}
			      value={feedback}
			      fill={true}
			  />
		      </FormGroup>
		  </div>
              </div>
              <br />

          <button type="submit" className="pt-button pt-intent-primary">
            Submit
            <span className="pt-icon-standard pt-icon-key-enter pt-align-right" />
          </button>
        </form>
      </div>
    );
  };

  partialSurvey = (submitFunction) => {
    const {
	satisfied,
	comprehension,
	grammatical,
	clear,
	ambiguous,
	feedback,
    } = this.state;

    return (
      <div>
          <h1>Finally, please answer the following short survey.</h1>

          <form onSubmit={submitFunction}>
	      <h2>
		  Game performance questions
	      </h2>
              <h3>
		  You must answer questions in this section to be able to proceed.
              </h3>

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <RadioGroup
			  name="satisfied"
			  label="1. How satisfied are you with your partner's performance in the game?"
			  onChange={this.handleChange}
			  selectedValue={satisfied}
		      >
			  <Radio
			      label="Very satisfied"
			      value="verySatisfied"
			      className={"pt-inline"}
			  />
			  <Radio
			      label="Satisfied"
			      value="satisfied"
			      className={"pt-inline"}
			  />
			  <Radio
			      label="Somewhat satisfied"
			      value="somewhatSatisfied"
			      className={"pt-inline"}
			  />

			  <Radio
			      label="Somewhat dissatisfied"
			      value="somewhatDissatisfied"
			      className={"pt-inline"}
			  />
			  <Radio
			      label="Dissatisfied"
			      value="dissatisfied"
			      className={"pt-inline"}
			  />
			  <Radio
			      label="Very dissatisfied"
			      value="veryDissatisfied"
			      className={"pt-inline"}
			  />
		      </RadioGroup>
		  </div>
              </div>
              <br />
	      
              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <RadioGroup
			  name="comprehension"
			  label="2. On a scale of 1-6 (where 6 is the best), how well did your partner understand your descriptions?"
			  onChange={this.handleChange}
			  selectedValue={comprehension}
		      >
			  <Radio
			      label="6 - understood almost all descriptions"
			      value="6"
			      className={"pt-inline"}
			  />
			  <Radio label="5" value="5" className={"pt-inline"} />
			  <Radio label="4" value="4" className={"pt-inline"} />
			  <Radio
			      label="3"
			      value="3"
			      className={"pt-inline"}
			  />
			  <Radio label="2" value="2" className={"pt-inline"} />
			  <Radio
			      label="1 - did not understand any of my descriptions"
			      value="1"
			      className={"pt-inline"}
			  />
		      </RadioGroup>
		  </div>
              </div>
              <br />

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <RadioGroup
			  name="grammatical"
			  label="3. On a scale of 1-6 (where 6 is the best), how grammatical were your partner's descriptions?"
			  onChange={this.handleChange}
			  selectedValue={grammatical}
		      >
			  <Radio
			      label="6 - there were no grammatical issues"
			      value="6"
			      className={"pt-inline"}
			  />
			  <Radio label="5" value="5" className={"pt-inline"} />
			  <Radio label="4" value="4" className={"pt-inline"} />
			  <Radio
			      label="3"
			      value="3"
			      className={"pt-inline"}
			  />
			  <Radio label="2" value="2" className={"pt-inline"} />
			  <Radio
			      label="1 - almost all descriptions were ungrammatical"
			      value="1"
			      className={"pt-inline"}
			  />
		      </RadioGroup>
		  </div>
              </div>
              <br />

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <RadioGroup
			  name="clear"
			  label="4. On a scale of 1-6 (where 6 is the easiest to understand), how easy to understand were your partner's descriptions?"
			  onChange={this.handleChange}
			  selectedValue={clear}
		      >
			  <Radio
			      label="6 - all descriptions were easy to understand"
			      value="6"
			      className={"pt-inline"}
			  />
			  <Radio label="5" value="5" className={"pt-inline"} />
			  <Radio label="4" value="4" className={"pt-inline"} />
			  <Radio
			      label="3"
			      value="3"
			      className={"pt-inline"}
			  />
			  <Radio label="2" value="2" className={"pt-inline"} />
			  <Radio
			      label="1 - no description was easy to understand"
			      value="1"
			      className={"pt-inline"}
			  />
		      </RadioGroup>
		  </div>
              </div>
              <br />

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <RadioGroup
			  name="ambiguous"
			  label="5. On a scale of 1-6 (where 6 is the easiest), how easy was it to distinguish the target image from other images in the context based on your partner's descriptions?"
			  onChange={this.handleChange}
			  selectedValue={ambiguous}
		      >
			  <Radio
			      label="6 - my partner's descriptions always made the target clear"
			      value="6"
			      className={"pt-inline"}
			  />
			  <Radio label="5" value="5" className={"pt-inline"} />
			  <Radio label="4" value="4" className={"pt-inline"} />
			  <Radio
			      label="3"
			      value="3"
			      className={"pt-inline"}
			  />
			  <Radio label="2" value="2" className={"pt-inline"} />
			  <Radio
			      label="1 - I was never able to choose the target based on my partner's descriptions"
			      value="1"
			      className={"pt-inline"}
			  />
		      </RadioGroup>
		  </div>
              </div>
              <br />

	      <h2>
		  Feedback for the study
	      </h2>

              <div className="pt-form-group">
		  <div className="pt-form-content">
		      <FormGroup
			  className={"form-group"}
			  inline={false}
			  label={
			      "6. Did you notice any problems or have any other comments about the study?"
			  }
			  labelFor={"feedback"}
		      >
			  <TextArea
			      id="feedback"
			      name="feedback"
			      large={true}
			      intent={Intent.PRIMARY}
			      onChange={this.handleChange}
			      value={feedback}
			      fill={true}
			  />
		      </FormGroup>
		  </div>
              </div>
              <br />
	      
          <button type="submit" className="pt-button pt-intent-primary">
            Submit
            <span className="pt-icon-standard pt-icon-key-enter pt-align-right" />
          </button>
        </form>
      </div>
    );
  };

  componentWillMount() {}

  render() {
      const { player, game } = this.props;
      const showDemographics = !player.get("completedDemographics");
      const submitFunction = (showDemographics) ? this.demographicHandleSubmit : this.nonDemographicHandleSubmit;
      const formContent = (showDemographics) ? this.fullSurvey(submitFunction) : this.partialSurvey(submitFunction);

      return (
	  <Centered>
              <div className="exit-survey">{formContent}</div>
	  </Centered>
      );
  }
}
