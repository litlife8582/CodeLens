import React, { useState } from 'react';
import Header from './Header';
import './styles.css';

export default function App() {
  const [count, setCount] = useState(0);

  return (
    <div className="container">
      <Header title="Mini Counter App" />
      <main>
        <p>Current Count: {count}</p>
        <button onClick={() => setCount(count + 1)}>Increment</button>
      </main>
    </div>
  );
}