import React from "react";

import {Button, Card, CardGroup, CardImg} from "react-bootstrap";
import './Contacts.css';
import BPTitle from "../../components/title/BPTitle";
import BPParagraph from "../../components/paragraph/BPParagraph";

class Contacts extends React.Component {
    render() {
        const base_path = '/resources/img/contacts/';
        return (
            <div>
                <BPTitle size={"600%"} text={"Contacts"}/>
                <BPParagraph size={"200%"} text={"Here are some ways you can contact me to learn more about who I am."} />

                <CardGroup className={"contacts"}>
                    <Card id={"mail"} className={"contact"}>
                        <a href={'mailto:brianpulfer95@gmail.com'}>
                            <CardImg variant={'top'} src={base_path + 'gmail.png'}/>
                        </a>
                        <Card.Body>
                            <Card.Title>
                                E-Mail
                            </Card.Title>
                            <Card.Text>
                                Feel free to send me e-mails if you are interest in learning more about who I am and
                                what i do.
                            </Card.Text>
                            <Card.Link>
                                <Button className={"contact-btn"} href={'mailto:brianpulfer95@gmail.com'}>Send me an
                                    e-mail</Button>
                            </Card.Link>
                        </Card.Body>
                    </Card>
                    <Card id={"linkedin"} className={"contact"}>
                        <a href={'https://www.linkedin.com/in/brian-pulfer-91a65417b/'}>
                            <CardImg variant={'top'} src={base_path + 'linkedin.png'}/>
                        </a>
                        <Card.Body>
                            <Card.Title>
                                LinkedIn
                            </Card.Title>
                            <Card.Text>
                                Everything about my professional carreer can be found on my LinkedIn profile.
                            </Card.Text>
                            <Card.Link>
                                <Button className={"contact-btn"}
                                        href={'https://www.linkedin.com/in/brian-pulfer-91a65417b/'}>Find me on
                                    LinkedIn</Button>
                            </Card.Link>
                        </Card.Body>
                    </Card>
                    <Card id={"twitter"} className={"contact"}>
                        <a href={'https://twitter.com/PulferBrian21'}>
                            <CardImg variant={'top'} src={base_path + 'twitter.png'}/>
                        </a>
                        <Card.Body>
                            <Card.Title>
                                Twitter
                            </Card.Title>
                            <Card.Text>
                                Usually I only tweet about my own major achievements and deep thoughts.
                            </Card.Text>
                            <Card.Link>
                                <Button className={"contact-btn"} href={'https://twitter.com/PulferBrian21'}>Tweet @Brian</Button>
                            </Card.Link>
                        </Card.Body>
                    </Card>
                    <Card id={"github"} className={"contact"}>
                        <a href={'https://github.com/BrianPulfer'}>
                            <CardImg variant={'top'} src={base_path + 'github.png'}/>
                        </a>
                        <Card.Body>
                            <Card.Title>
                                GitHub
                            </Card.Title>
                            <Card.Text>
                                Checkout some of my professional work and toy-projects, as well as my contributions.
                            </Card.Text>
                            <Card.Link>
                                <Button className={"contact-btn"} href={'https://github.com/BrianPulfer'}>Find me on
                                    GitHub</Button>
                            </Card.Link>
                        </Card.Body>
                    </Card>
                    <Card id={"kaggle"} className={"contact"}>
                        <a href={'https://www.kaggle.com/brianpulfer'}>
                            <CardImg variant={'top'} src={base_path + 'kaggle.png'}/>
                        </a>
                        <Card.Body>
                            <Card.Title>
                                Kaggle
                            </Card.Title>
                            <Card.Text>
                                In my free time I like to compete through some challenges and get inspired by data.
                            </Card.Text>
                            <Card.Link>
                                <Button className={"contact-btn"} href={'https://www.kaggle.com/brianpulfer'}>My challenges</Button>
                            </Card.Link>
                        </Card.Body>
                    </Card>
                    <Card id={"tds"} className={"contact"}>
                        <a href={'https://medium.com/@brianpulfer'}>
                            <CardImg variant={'top'} src={base_path + 'tds.png'}/>
                        </a>
                        <Card.Body>
                            <Card.Title>
                                Towards Data Science
                            </Card.Title>
                            <Card.Text>
                                Some of my work / thoughts that I felt worth sharing can be found at my TDS account.
                            </Card.Text>
                            <Card.Link>
                                <Button className={"contact-btn"} href={'https://medium.com/@brianpulfer'}>Read my
                                    posts</Button>
                            </Card.Link>
                        </Card.Body>
                    </Card>
                </CardGroup>
            </div>
        );
    }
}

export default Contacts;