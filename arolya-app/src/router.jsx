import {BrowserRouter, Route, Routes} from 'react-router-dom';
import React from 'react';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';

function Router() {
    return (
        <BrowserRouter>
            <main className="main">
                <Routes>        
                    <Route path="/" element = {<LandingPage/>}/>
                    <Route path="/login" element = {<LoginPage/>}/>
                    <Route component = {() => <h1>404: Page not found</h1>}/>

                </Routes>
            </main>
        </BrowserRouter>
    )
    }
export default Router;
