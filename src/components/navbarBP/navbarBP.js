import React from "react";
import {Navbar, Nav} from 'react-bootstrap';

import {HOME_PATH, ME_PATH, CAREER_PATH, PROJECTS_PATH, CONTACTS_PATH} from './../../utilities/paths.js'

import './navbarBP.css'


class NavbarBP extends React.Component{

    render() {
        return (
            <Navbar expand="lg" bg={"dark"}>
                <Navbar.Brand className="BPBrand" href={HOME_PATH}>Brian Pulfer</Navbar.Brand>
                <Navbar.Toggle aria-controls="basic-navbar-nav" />
                <Navbar.Collapse id="basic-navbar-nav">
                    <Nav className="ml-auto">
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