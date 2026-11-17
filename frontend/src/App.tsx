import { useEffect, useState } from 'react'
import Home from './pages/Home'
import SessionPage from './pages/Session'
import Compare from './pages/Compare'
import Privacy from './pages/Privacy'
import Review from './pages/Review'
import ProgressPage from './pages/Progress'
import Duo from './pages/Duo'
import Capture from './pages/Capture'

function useHash() {
  const [hash, setHash] = useState(window.location.hash || '#/')
  useEffect(() => {
    const on = () => { setHash(window.location.hash || '#/'); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return hash
}

export default function App() {
  const hash = useHash()
  const parts = hash.replace(/^#\//, '').split('/')
  let page
  if (parts[0] === 's' && parts[1]) page = <SessionPage key={parts[1]} id={parts[1]} />
  else if (parts[0] === 'compare' && parts[1] && parts[2]) page = <Compare a={parts[1]} b={parts[2]} />
  else if (parts[0] === 'shared' && parts[1]) page = <SessionPage key={parts[1]} shareToken={parts[1]} />
  else if (parts[0] === 'duo' && parts[1] && parts[2]) page = <Duo a={parts[1]} b={parts[2]} />
  else if (parts[0] === 'review') page = <Review />
  else if (parts[0] === 'progress') page = <ProgressPage />
  else if (parts[0] === 'capture') page = <Capture />
  else if (parts[0] === 'privacy') page = <Privacy />
  else page = <Home />
  return (
    <>
      <header className="top no-print">
        <a className="brand" href="#/"><span className="ball" aria-hidden />Swingwise</a>
        <nav>
          <a href="#/">Sessions</a>
          <a href="#/progress">Progress</a>
          <a href="#/review">Review</a>
          <a href="#/privacy">Your data</a>
        </nav>
      </header>
      <main>{page}</main>
      <footer className="foot">
        Technique analysis and education only. Not a medical device, not clinically validated, and not a substitute for a coach or clinician.
      </footer>
    </>
  )
}
