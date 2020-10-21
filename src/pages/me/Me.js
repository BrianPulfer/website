import React from 'react';
import {Button, Image, Row, Col} from 'react-bootstrap';

import BPTitle from "../../components/title/BPTitle";
import BPParagraph from "../../components/paragraph/BPParagraph";

import MeHome from './img/MeHome.jpg';

import './Me.css';

class Me extends React.Component{
  render() {
      const titleSize = "500%"
      const paragraphSize = "180%";

    return (
        <div>
            <div id={"Image"}>
                <Row>
                    <Col className={"text-center"}>
                        <Image id={"me-img"} src={MeHome} className={"mx-auto"} fluid />
                    </Col>
                </Row>
            </div>

            <div id={"Text"}>
                <BPTitle classes={"h1 bp-title-me"} size={titleSize} color={"black"} text={"About Me"} underline={true}/>
                <BPParagraph size={paragraphSize}
                    text={
                        "My name is Brian Pulfer, I am a 25 years old software engineer currently studying Artificial Intelligence in a master course offered by the University of Southern Switzerland (USI)." +
                        " I was born on September 21st, 1995 in my hometown Lugano (Southern part of Switzerland) where I have been living in ever since. I also served the Swiss Army from March 2015 to January 2016, completing all of my obligations as a Swiss citizen."
                    }
                />
                <BPParagraph size={paragraphSize}
                    text={
                        " My biggest interests are music, books and video games. Football is my biggest passion and I feel that sports are the biggest source of happiness, this is why I also frequent the gym regularly." +
                        " I also worry about living my life in the best possible way, that's why I like reading about philosophy and try to carry a mentally healthy lifestyle through meditation."
                    }
                />

                <BPParagraph size={paragraphSize}
                    text={
                        " People say of me that I am determined, funny, precise and smart." +
                        " My main interests regarding computer science are Artificial Intelligence, smart algorithms in general, virtual reality, web development and mobile development."
                    }
                />
            </div>
            <div id={"Download"} className={"last"}>
                <BPTitle classes={'h1 bp-title-me'} size={titleSize} color={"black"} text={"My Curriculum Vitae"} underline={true}/>
                <BPParagraph size={paragraphSize}
                             text={"You can download my curriculum vitae from the following button."}
                />

                <Row>
                    <a className={"col-12 text-center"} href={process.env.PUBLIC_URL+'/resources/cv/Brian Pulfer CV.pdf'} download>
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
