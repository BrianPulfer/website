import React from "react";
import { Button, Image, Row, Col, ListGroup } from "react-bootstrap";

import BPTitle from "../../components/title/BPTitle";
import BPParagraph from "../../components/paragraph/BPParagraph";

import Avatar from "./img/avatar.png";
import "./Me.css";

import trackPage from "../../utilities/ga/ga";

class Me extends React.Component {
  render() {
    trackPage();

    const titleSize = "500%";
    const paragraphSize = "180%";

    let mainImage = (
      <Image id={"me-img"} src={Avatar} className={"mx-auto"} fluid />
    );

    return (
      <div className={"me-div"}>
        <div id={"Image"}>
          <Row>
            <Col className={"text-center"}>{mainImage}</Col>
          </Row>
        </div>

        <div id={"Overview"}>
          <BPTitle
            classes={"h1 bp-title-me"}
            size={titleSize}
            color={"black"}
            text={"Overview"}
            underline={true}
          />
          <BPParagraph
            size={paragraphSize}
            text={"Hey there, this is Brian! 👨🏽‍💻"}
          />
          <BPParagraph
            size={paragraphSize}
            text={
              "I am a Machine Learning practitioner and enthusiast. I am fascinated by the potential of these beautiful and elegant algorithms and their potential impact on our world. " +
              "Someday in the future, I'd love to use Machine Learning to fight the climate crisis, as I believe is one of the biggest priorities and challenges of my generation. " +
              "Before that, I am constantly working on becoming a world expert in the field of ML and learn interesting 'hacks' in Computer Science in general."
            }
          />
          <BPParagraph
            size={paragraphSize}
            text={
              "I am currently a Ph.D. student in Machine Learning, with a focus on anomaly detection and adversarial attacks, by the University of Geneva, Switzerland."
            }
          />
          <BPParagraph
            size={paragraphSize}
            text={
              "This is my personal portfolio, where I publish updates on my career, projects, publications and more. I hope you enjoy it!"
            }
          />

        </div>

        <div id={"News"}>
          <BPTitle
              classes={"h1 bp-title-me"}
              size={titleSize}
              color={"black"}
              text={"News 📰"}
              underline={true}
            />

          <ListGroup>
            <ListGroup.Item className={"news"}>
              <BPTitle text={"01.11.2021"} size={paragraphSize} side={true}/>
              <BPParagraph text={"I Joined the Stochastic Information Processing (SIP) group of the University of Geneva in quality of Ph.D. Student in Machine Learning."} size={paragraphSize}/>
            </ListGroup.Item>
          </ListGroup>
        </div>

        <div id={"Download"} className={"last"}>
          <BPTitle
            classes={"h1 bp-title-me"}
            size={titleSize}
            color={"black"}
            text={"My CV"}
            underline={true}
          />
          <BPParagraph
            size={paragraphSize}
            text={
              "You can download my CV from the following button."
            }
          />

          <Row>
            <a
              className={"col-12 text-center"}
              href={
                process.env.PUBLIC_URL + "/resources/cv/Brian Pulfer CV.pdf"
              }
              download
            >
              <Button>
                <i className={"fa fa-download"}></i>
                Download my CV
              </Button>
            </a>
          </Row>
        </div>
      </div>
    );
  }
}

export default Me;
