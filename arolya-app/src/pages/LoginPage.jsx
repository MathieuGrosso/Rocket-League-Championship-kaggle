import React,{ useState} from 'react';
import httpClient from '../httpClient';
// import axios from 'axios'


export const LoginPage = () => {
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [username, setUsername] = useState('')

    const logInUser = async () => {
        console.log(email, password, username)
            const resp = await httpClient.post('http://127.0.0.1:5000/login', {
    "email":email,
    "password":password,
    "username":username
});
    console.log(resp)
        }

      

  return (
    <div>
        <h1>Log into your account</h1>
        <br/>
        <form>
            <div>
                <label htmlfor="Email"> Email: </label>
                <input  type="text"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            id=""
            />
            </div>
            <div>
                <label htmlfor="Password"> Password: </label>
                <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            id=""
          />
            </div>
            <div>
                <label htmlfor="Username"> Username: </label>
                <input
            type="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            id=""
          />
            </div>
            <button type="button" onClick={()=>logInUser()}>Submit</button>
        </form>
    </div>
  )
}

export default LoginPage