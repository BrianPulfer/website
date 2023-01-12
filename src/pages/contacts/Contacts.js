import React from "react";

import { Button, Card, CardGroup, CardImg, Row } from "react-bootstrap";
import "./Contacts.css";
import BPTitle from "../../components/title/BPTitle";
import BPParagraph from "../../components/paragraph/BPParagraph";

import Mail from "./img/gmail.png";
import Linkedin from "./img/linkedin.png";
import Twitter from "./img/twitter.png";
import GitHub from "./img/github.png";
import trackPage from "../../utilities/ga/ga";

class Contacts extends React.Component {
  render() {
    trackPage();
    return (
      <div className={"contacts-div"}>
        <BPTitle size={"600%"} text={"Contacts"} />
        <BPParagraph
          size={"200%"}
          text={"Here are some ways you can can in touch with me."}
        />

        <CardGroup className={"contacts row offset-xl-3"}>
          <Card id={"mail"} className={"contact col col-xl-2"}>
            <a href={"mailto:brianpulfer95@gmail.com"}>
              <CardImg variant={"top"} src={Mail} />
            </a>
            <Card.Body>
              <Card.Title>E-Mail</Card.Title>
              <Card.Text>
                Feel free to send me e-mails if you are interest in learning
                more about who I am and what i do.
              </Card.Text>
              <Button
                className={"contact-btn"}
                href={"mailto:brianpulfer95@gmail.com"}
              >
                Send me an e-mail
              </Button>
            </Card.Body>
          </Card>
          <Card id={"linkedin"} className={"contact col col-xl-2"}>
            <a href={"https://www.linkedin.com/in/BrianPulfer/"}>
              <CardImg variant={"top"} src={Linkedin} />
            </a>
            <Card.Body>
              <Card.Title>LinkedIn</Card.Title>
              <Card.Text>
                Everything about my professional carreer can be found on my
                LinkedIn profile.
              </Card.Text>
              <Button
                className={"contact-btn"}
                href={"https://www.linkedin.com/in/BrianPulfer/"}
              >
                Find me on LinkedIn
              </Button>
            </Card.Body>
          </Card>
          <Card id={"twitter"} className={"contact col col-xl-2"}>
            <a href={"https://twitter.com/PulferBrian21"}>
              <CardImg variant={"top"} src={Twitter} />
            </a>
            <Card.Body>
              <Card.Title>Twitter</Card.Title>
              <Card.Text>
                Usually I only tweet about my own major achievements and deep
                thoughts.
              </Card.Text>
              <Button
                className={"contact-btn"}
                href={"https://twitter.com/PulferBrian21"}
              >
                Tweet @Brian
              </Button>
            </Card.Body>
          </Card>
          <Card id={"github"} className={"contact col col-xl-2"}>
            <a href={"https://github.com/BrianPulfer"}>
              <CardImg variant={"top"} src={GitHub} />
            </a>
            <Card.Body>
              <Card.Title>GitHub</Card.Title>
              <Card.Text>
                Checkout some of my professional work and toy-projects, as well
                as my contributions.
              </Card.Text>
              <Button
                className={"contact-btn"}
                href={"https://github.com/BrianPulfer"}
              >
                Find me on GitHub
              </Button>
            </Card.Body>
          </Card>
        </CardGroup>
      </div>
    );
  }
}

export default Contacts;
