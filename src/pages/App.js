import React from "react";
import {BrowserRouter as Router, Route} from "react-router-dom";

import Me from "./me/Me";
import Career from "./career/Career";
import Projects from "./projects/Projects";
import Contacts from "./contacts/Contacts";

import Container from "react-bootstrap/Container";
import NavbarBP from "../components/navbarBP/navbarBP";

class App extends React.Component {
    render() {
        return (
            <React.Fragment>
                <NavbarBP />
                <Container fluid>
                    <Router>
                        <Route exact path={process.env.PUBLIC_URL+"/"} component={Me}/>
                        <Route path={process.env.PUBLIC_URL+"/Me"} component={Me}/>
                        <Route path={process.env.PUBLIC_URL+"/Career"} component={Career}/>
                        <Route path={process.env.PUBLIC_URL+"/Projects"} component={Projects}/>
                        <Route path={process.env.PUBLIC_URL+"/Contacts"} component={Contacts}/>
                    </Router>
                </Container>
            </React.Fragment>
        )
    }
}

export default App;