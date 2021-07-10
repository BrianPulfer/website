import React from "react";

import BPTitle from "../../components/title/BPTitle";
import BPParagraph from "../../components/paragraph/BPParagraph";

import './Career.css';
import {Image, Row, Col} from "react-bootstrap";

import CPC from './img/cpc.jpg'
import Army from './img/army.png'
import SUPSI from './img/supsi.png'
import USI from './img/usi.png'

class Career extends React.Component {

    render() {
        return (
            <div>
                <div className={"career-step"}>
                    <Row>
                        <Col className={"text-center"}>
                            <Image src={USI} fluid/>
                        </Col>
                    </Row>
                    <BPTitle className={"bp-title-career"}
                             text={"Internship as Machine Learning Engineer (July & August 2020)"}/>
                    <BPParagraph size={"150%"}
                                 text={
                                     "During the summer of 2020, I have been working as a Machine Learning Engineer by the University of Southern Switzerland (USI) on a project regarding image classification, object detection and segmentation." +
                                     " During this short but enriching experience I have been working daily with frameworks such as Pytorch, Tensorflow's Keras and OpenCV. With these frameworks, operations such as image processing, data augmentation and transfer-learning were carried out."
                                 }
                    />
                </div>


                <div className={"career-step"}>
                    <Row>
                        <Col className={"text-center"}>
                            <Image src={USI} fluid/>
                        </Col>
                    </Row>
                    <BPTitle className={"bp-title-career"} text={"Master in Artificial Intelligence (2019 - 2021)"}/>
                    <BPParagraph size={"150%"}
                                 text={
                                     "From September 2019, I have been studying Machine Learning and Deep Learning in the Master course offered by the University of Southern Switzerland (USI).\n" +
                                     "By learning about supervised/unsupervised/reinforcement learning, deep neural networks and fascinating mathematical and statistical concepts, I have grown a passion for the field." +
                                     "I graduated in June 2021 with a GPA of 9.1/10."
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
                                     "From September 2016 to September 2019 I have been studying computer science by the University of Applied Sciences of Southern Switzerland (SUPSI - Scuola Universitaria Professionale della Svizzera Italiana) where I obtained my degree in Computer Science." +
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
                                     "From september 2010 to september 2014 I have studied in the Commercial High School of Lugano (CPC - Centro Professionale Commerciale) where I obtained both the AFC (Attestato federale di capacità) degree and the commercial federal maturity degree (MFC - Maturità Federale Commerciale)."
                                 }
                    />
                </div>

            </div>
        );
    }
}

export default Career;