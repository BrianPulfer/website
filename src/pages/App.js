import React from "react";
import {HashRouter as Router, Route} from "react-router-dom";

import Me from "./me/Me";
import Career from "./career/Career";
import Awards from "./awards/awards";
import Projects from "./projects/Projects";
import Publications from "./publications/Publications";
import Blog from "./blog/blog";
import Contacts from "./contacts/Contacts";
import NoMatch from "./nomatch/NoMatch";

import Container from "react-bootstrap/Container";
import NavbarBP from "../components/navbarBP/navbarBP";

import './App.css'


function App() {
    return (
        <React.Fragment>
            <NavbarBP/>
            <Container fluid>
                <Router>
                    <Route exact path={'/'} component={Me}/>
                    <Route exact path={'/Me'} component={Me}/>
                    <Route exact path={'/Career'} component={Career}/>
                    <Route exact path={'/Awards'} component={Awards}/>
                    <Route exact path={'/Projects'} component={Projects}/>
                    <Route exact path={'/Publications'} component={Publications}/>
                    <Route exact path={'/Blog'} component={Blog}/>
                    <Route exact path={'/Contacts'} component={Contacts}/>
                    <Route component={NoMatch} />
                </Router>
            </Container>
        </React.Fragment>
    )
}

export default App;