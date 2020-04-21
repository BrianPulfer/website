import React from "react";
import {Navbar, Nav} from 'react-bootstrap';

import './navbarBP.css'

class NavbarBP extends React.Component{

    constructor(props) {
        super(props);
    }

    componentDidMount() {
        // Called right after component renders
    }

    componentWillUnmount() {
        // Called right before component will be deleted
    }

    render() {
        const me_link = process.env.PUBLIC_URL+'/Me';
        const career_link = process.env.PUBLIC_URL+'/Career';
        const projects_link = process.env.PUBLIC_URL+'/Projects';
        const contacts_link = process.env.PUBLIC_URL+'/Contacts';

        return (
            <Navbar expand="lg" bg={"dark"}>
                <Navbar.Brand className="BPBrand" href={me_link}>Brian Pulfer</Navbar.Brand>
                <Navbar.Toggle aria-controls="basic-navbar-nav" />
                <Navbar.Collapse id="basic-navbar-nav">
                    <Nav className="ml-auto">
                        <Nav.Link href={me_link}>Me</Nav.Link>
                        <Nav.Link href={career_link}>Career</Nav.Link>
                        <Nav.Link href={projects_link}>Projects</Nav.Link>
                        <Nav.Link href={contacts_link}>Contacts</Nav.Link>
                    </Nav>
                </Navbar.Collapse>
            </Navbar>
        )
    }
}

export default NavbarBP;