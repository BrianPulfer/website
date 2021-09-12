import React from "react";
import {Route, HashRouter} from "react-router-dom";

import Me from "./me/Me";
import Career from "./career/Career";
import Projects from "./projects/Projects";
import Publications from "./publications/Publications";
import Contacts from "./contacts/Contacts";

import Container from "react-bootstrap/Container";
import NavbarBP from "../components/navbarBP/navbarBP";

import './App.css'

// TODO: Fix view-port for all pages except for Contacts
// TODO: Change how main picture gets displayed based on screen size
// TODO: Change navbar to be transparent and color-changing
// TODO: Add 'Leadership and Awards' section?
// TODO: Add 'Publications' section?

class App extends React.Component {
    render() {
        return (
            <React.Fragment>
                <NavbarBP/>
                <Container fluid>
                    <HashRouter>
                        <Route exact path={'/'} component={Me}/>
                        <Route path={'/Me'} component={Me}/>
                        <Route path={'/Career'} component={Career}/>
                        <Route path={'/Projects'} component={Projects}/>
                        <Route path={'/Publications'} component={Publications}/>
                        <Route path={'/Contacts'} component={Contacts}/>
                    </HashRouter>
                </Container>
            </React.Fragment>
        )
    }
}

export default App;