import React from 'react'

function LandingPage() {
  return (
    <div>
        <h1>Welcome to Arolya</h1>
        <br/>
        <p>You are not logged in</p>
        <div className='button-span'>
            <a href="/Register">
                <button > Register</button>
            </a>
            <a href="/login">
                <button> Login</button>
            </a>
        </div>
    </div>
  )
}

export default LandingPage