import React from "react";
import {Route, BrowserRouter} from "react-router-dom";

import Me from "./me/Me";
import Career from "./career/Career";
import Projects from "./projects/Projects";
import Contacts from "./contacts/Contacts";

import Container from "react-bootstrap/Container";
import NavbarBP from "../components/navbarBP/navbarBP";

import {HOME_PATH, ME_PATH, CAREER_PATH, PROJECTS_PATH, CONTACTS_PATH} from './../utilities/paths.js'

class App extends React.Component {
    render() {
        return (
            <React.Fragment>
                <NavbarBP />
                <Container fluid>
                    <BrowserRouter>
                        <Route exact path={HOME_PATH} component={Me}/>
                        <Route path={ME_PATH} component={Me}/>
                        <Route path={CAREER_PATH} component={Career}/>
                        <Route path={PROJECTS_PATH} component={Projects}/>
                        <Route path={CONTACTS_PATH} component={Contacts}/>
                    </BrowserRouter>
                </Container>
            </React.Fragment>
        )
    }
}

export default App;