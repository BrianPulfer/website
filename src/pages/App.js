import React from "react";
import {Route, HashRouter} from "react-router-dom";

import Me from "./me/Me";
import Career from "./career/Career";
import Awards from "./awards/awards";
import Projects from "./projects/Projects";
import Publications from "./publications/Publications";
import Blog from "./blog/blog";
import Contacts from "./contacts/Contacts";

import Container from "react-bootstrap/Container";
import NavbarBP from "../components/navbarBP/navbarBP";

import './App.css'

// TODO: Change navbar to be transparent and color-changing

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
                        <Route path={'/Awards'} component={Awards}/>
                        <Route path={'/Projects'} component={Projects}/>
                        <Route path={'/Publications'} component={Publications}/>
                        <Route path={'/Blog'} component={Blog} />
                        <Route path={'/Contacts'} component={Contacts}/>
                    </HashRouter>
                </Container>
            </React.Fragment>
        )
    }
}

export default App;