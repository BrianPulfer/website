import React from "react";
import {Navbar, Nav} from 'react-bootstrap';

import './navbarBP.css'

const HOME_PATH = process.env.PUBLIC_URL + '/';
const ME_PATH = process.env.PUBLIC_URL + '/#/Me';
const CAREER_PATH = process.env.PUBLIC_URL + '/#/Career';
const PROJECTS_PATH = process.env.PUBLIC_URL + '/#/Projects';
const PUBLICATIONS_PATH = process.env.PUBLIC_URL + '/#/Publications';
const CONTACTS_PATH = process.env.PUBLIC_URL + '/#/Contacts';

const POSSIBLE_SELECTIONS = ["Me", "Career", "Projects", "Publications", "Contacts"];
const SELECTED = "selected";
const NOT_SELECTED = "not-selected";
const NO_SELECTION = () => [NOT_SELECTED, NOT_SELECTED, NOT_SELECTED, NOT_SELECTED];

class NavbarBP extends React.Component {

    constructor(props) {
        super(props);

        this.toggleExpand = this.toggleExpand.bind(this);
        this.closeNav = this.closeNav.bind(this);

        const splitHREF = window.location.href.split("/");
        const addr = splitHREF[splitHREF.length - 1];
        let initialSelection = NO_SELECTION();

        initialSelection = initialSelection.map((elem, i) => {
            if (addr === POSSIBLE_SELECTIONS[i]) {
                return SELECTED;
            }
            return NOT_SELECTED;
        });

        this.state = {
            navExpanded: false,
            selection: initialSelection
        }
    }

    toggleExpand() {
        let newExpanded = !this.state.navExpanded;
        this.setState({navExpanded: newExpanded});
    }

    closeNav(address) {
        const splitAddress = address.split("/");
        const addr = splitAddress[splitAddress.length - 1];

        let newSelection = NO_SELECTION();

        for (let i = 0; i < POSSIBLE_SELECTIONS.length; i++) {
            if (addr === POSSIBLE_SELECTIONS[i]) {
                newSelection[i] = SELECTED;
            }
        }

        this.setState({
            navExpanded: false,
            selection: newSelection
        });
        window.scrollTo(0, 0);
    }

    render() {
        return (
            <Navbar expand="lg" onClick={this.toggleExpand} expanded={this.state.navExpanded}>
                <Navbar.Brand className="BPBrand" href={HOME_PATH}>Brian Pulfer</Navbar.Brand>
                <Navbar.Toggle aria-controls="basic-navbar-nav"/>
                <Navbar.Collapse>
                    <Nav className="ml-auto" onSelect={this.closeNav}>
                        <Nav.Link href={ME_PATH} className={this.state.selection[0]}>Me</Nav.Link>
                        <Nav.Link href={CAREER_PATH} className={this.state.selection[1]}>Career</Nav.Link>
                        <Nav.Link href={PROJECTS_PATH} className={this.state.selection[2]}>Projects</Nav.Link>
                        <Nav.Link href={PUBLICATIONS_PATH} className={this.state.selection[3]}>Publications</Nav.Link>
                        <Nav.Link href={CONTACTS_PATH} className={this.state.selection[4]}>Contacts</Nav.Link>
                    </Nav>
                </Navbar.Collapse>
            </Navbar>
        )
    }
}

export default NavbarBP;