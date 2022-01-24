import React from 'react';

import {Col, Row} from "react-bootstrap";
import BPPublication from "../../components/publication/BPPublication";

import "./Publications.css";

class Publications extends React.Component {
    render() {
        return (
            <React.Fragment>
                <Row>
                    <Col className={'col-10 offset-1'}>
                        <div className={"publication"}>
                            <BPPublication
                                link={"https://arxiv.org/abs/2112.11255"}
                                date={"Dec 21"}
                                title={"Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems"}
                                abstract={"Safe deployment of self-driving cars (SDC) necessitates thorough simulated and in-field testing. Most testing techniques consider virtualized SDCs within a simulation environment, whereas less effort has been directed towards assessing whether such techniques transfer to and are effective with a physical real-world vehicle. In this paper, we leverage the Donkey Car open-source framework to empirically compare testing of SDCs when deployed on a physical small-scale vehicle vs its virtual simulated counterpart. In our empirical study, we investigate transferability of behavior and failure exposure between virtual and real-world environments on a vast set of corrupted and adversarial settings. While a large number of testing results do transfer between virtual and physical environments, we also identified critical shortcomings that contribute to the reality gap between the virtual and physical world, threatening the potential of existing testing solutions when applied to physical SDCs"}
                                citation={"Stocco, A., Pulfer, B., & Tonella, P. (2021). Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems. CoRR, abs/2112.11255. https://arxiv.org/abs/2112.11255"}/>
                        </div>
                    </Col>
                </Row>
            </React.Fragment>
        );
    }
}

export default Publications;