import React from "react";
import {Navbar, Nav} from 'react-bootstrap';

import './navbarBP.css'

const HOME_PATH = process.env.PUBLIC_URL + '/';
const ME_PATH = process.env.PUBLIC_URL + '/#/Me';
const CAREER_PATH = process.env.PUBLIC_URL + '/#/Career';
const PROJECTS_PATH = process.env.PUBLIC_URL + '/#/Projects';
const CONTACTS_PATH = process.env.PUBLIC_URL + '/#/Contacts';

class NavbarBP extends React.Component {

    constructor(props) {
        super(props);


        this.toggleExpand = this.toggleExpand.bind(this);
        this.closeNav = this.closeNav.bind(this);

        this.state = {
            navExpanded: false
        }
    }

    toggleExpand() {
        let newExpanded = !this.state.navExpanded;
        this.setState({navExpanded: newExpanded});
    }

    closeNav() {
        this.setState({navExpanded: false});
        window.scrollTo(0, 0);
    }

    render() {

        return (
            <Navbar expand="lg" onClick={this.toggleExpand} expanded={this.state.navExpanded}>
                <Navbar.Brand className="BPBrand" href={HOME_PATH}>Brian Pulfer</Navbar.Brand>
                <Navbar.Toggle aria-controls="basic-navbar-nav"/>
                <Navbar.Collapse>
                    <Nav className="ml-auto" onSelect={this.closeNav}>
                        <Nav.Link href={ME_PATH}>Me</Nav.Link>
                        <Nav.Link href={CAREER_PATH}>Career</Nav.Link>
                        <Nav.Link href={PROJECTS_PATH}>Projects</Nav.Link>
                        <Nav.Link href={CONTACTS_PATH}>Contacts</Nav.Link>
                    </Nav>
                </Navbar.Collapse>
            </Navbar>
        )
    }
}

export default NavbarBP;