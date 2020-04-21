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
        return (
            <Navbar expand="lg" bg={"dark"}>
                <Navbar.Brand className="BPBrand" href="/Me">Brian Pulfer</Navbar.Brand>
                <Navbar.Toggle aria-controls="basic-navbar-nav" />
                <Navbar.Collapse id="basic-navbar-nav">
                    <Nav className="ml-auto">
                        <Nav.Link href="/Me">Me</Nav.Link>
                        <Nav.Link href="/Career">Career</Nav.Link>
                        <Nav.Link href="/Projects">Projects</Nav.Link>
                        <Nav.Link href="/Contacts">Contacts</Nav.Link>
                    </Nav>
                </Navbar.Collapse>
            </Navbar>
        )
    }
}

export default NavbarBP;