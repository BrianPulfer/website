import React from "react";

import BPTitle from "../../components/title/BPTitle";
import BPParagraph from "../../components/paragraph/BPParagraph";

import './Career.css';
import {Image, Row, Col} from "react-bootstrap";

import CPC from './img/cpc.jpg'
import Army from './img/army.png'
import SUPSI from './img/supsi.png'
import USI from './img/usi.png'

class Career extends React.Component{

    render() {
        return (
            <div>
                <div className={"career-step"}>
                    <Row>
                        <Col className={"text-center"}>
                            <Image src={USI} fluid/>
                        </Col>
                    </Row>
                    <BPTitle className={"bp-title-career"} text={"Master in Artificial Intelligence (2019 - 2021)"}/>
                    <BPParagraph size={"150%"}
                                 text={
                                     "I am currently a Master student in Artificial Intelligence by the University of Southern Switzerland (USI - Università della Svizzera Italiana)."
                                 }
                    />
                </div>

                <div className={"career-step"}>
                    <Row>
                        <Col className={"text-center"}>
                            <Image src={SUPSI} fluid/>
                        </Col>
                    </Row>
                    <BPTitle className={"bp-title-career"} text={"Bachelor in Computer Science (2016 - 2019)"}/>
                    <BPParagraph size={"150%"}
                                 text={
                                     "From September 2016 to September 2019 I have been studying computer science by the University of Applied Sciences of Southern Switzerland (SUPSI - Scuola Universitaria Professionale della Svizzera Italiana) were I obtained my degree in Computer Science." +
                                     " During these 3 years I have learned a lot both in mathematics subjects, such as Linear Algebra, Calculus, Probability, Statistics, both in computer science subjects such as algorithms and data structures, software engineering, design patterns, agile development, through many practical projects."
                                 }
                    />
                </div>

                <div className={"career-step"}>
                    <Row>
                        <Col className={"text-center"}>
                            <Image id={"army_img"} src={Army} fluid/>
                        </Col>
                    </Row>
                    <BPTitle className={"bp-title-career"} text={"Service by the Swiss Army (2015 - 2016)"}/>
                    <BPParagraph size={"150%"}
                                 text={
                                     "From march 2015 to january 2016 I have been serving the Swiss Army (10 months) concluding my obligations as a swiss citizen and obtaining the grade of soldier." +
                                     "In November 2019, I was finally completely discharged of any duty related to the army."
                                 }
                    />
                </div>

                <div className={"career-step"}>
                    <Row>
                        <Col className={"text-center"}>
                            <Image src={CPC} fluid/>
                        </Col>
                    </Row>
                    <BPTitle className={"bp-title-career"} text={"Commercial High School (2010 - 2014)"}/>
                    <BPParagraph size={"150%"}
                        text={
                            "From september 2010 to september 2014 I have studied in the Commercial High School of Lugano (CPC - Centro Professionale Commerciale) were I obtained both the AFC (Attestato federale di capacità) degree and the commercial federal maturity degree (MFC - Maturità Federale Commerciale)."
                        }
                    />
                </div>

            </div>
        );
    }
}

export default Career;