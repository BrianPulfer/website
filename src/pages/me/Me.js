import React from "react";
import { Button, Image, Row, Col } from "react-bootstrap";

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

        <div id={"Text"}>
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
              "This is my personal portfolio, where I publish updates on my career, projects, publications and more. Should anything be unclear, don't hesitate to contact me through the '/contact' section."
            }
          />

          <BPTitle
            classes={"h1 bp-title-me"}
            size={titleSize}
            color={"black"}
            text={"About Me"}
            underline={true}
          />
          <BPParagraph
            size={paragraphSize}
            text={
              "I was born on September 21st, 1995 in my hometown Lugano, Switzerland 🇨🇭, where I have been living in ever since."
            }
          />
          <BPParagraph
            size={paragraphSize}
            text={
              "As everyone, I am interested in music, books and video games. I frequent the gym and do jogging regularly, although football is my favourite sport. " +
              "I also enjoy meditating, In my free time, I like to learn about new things. That's why I usually take on challenges like hackathons and toy-projects."
            }
          />

          <BPParagraph
            size={paragraphSize}
            text={
              "People say of me that I am determined, funny, precise and smart."
            }
          />
        </div>
        <div id={"Download"} className={"last"}>
          <BPTitle
            classes={"h1 bp-title-me"}
            size={titleSize}
            color={"black"}
            text={"My Curriculum Vitae"}
            underline={true}
          />
          <BPParagraph
            size={paragraphSize}
            text={
              "You can download my curriculum vitae from the following button."
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
