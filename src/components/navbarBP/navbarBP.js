import React from "react";
import {Navbar, Nav} from 'react-bootstrap';

import './navbarBP.css'


class NavbarBP extends React.Component{

    render() {
        const HOME_PATH = process.env.PUBLIC_URL+'/';
        const ME_PATH = process.env.PUBLIC_URL+'/#/Me';
        const CAREER_PATH = process.env.PUBLIC_URL+'/#/Career';
        const PROJECTS_PATH = process.env.PUBLIC_URL+'/#/Projects';
        const CONTACTS_PATH = process.env.PUBLIC_URL+'/#/Contacts';

        return (
            <Navbar expand="lg">
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