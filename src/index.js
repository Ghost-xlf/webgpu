import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import OceanWave from './OceanWave';
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    // <React.StrictMode>
      	// <WebGPU2WebGL />
        <OceanWave />
    // </React.StrictMode>
);

reportWebVitals();
