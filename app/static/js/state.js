// DOM Elements
const fileInput = document.getElementById('fileInput');
const generateBtn = document.getElementById('generateBtn');
const extractBtn = document.getElementById('extractBtn');
const cornersBtn = document.getElementById('cornersBtn');
const matchingBtn = document.getElementById('matchingBtn');
const resetBtn = document.getElementById('resetBtn');
const status = document.getElementById('status');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');
const generatedVariants = document.getElementById('generatedVariants');
const extractionResults = document.getElementById('extractionResults');
const edgeResults = document.getElementById('edgeResults');
const matchResults = document.getElementById('matchResults');

// State Variables
let uploadedFilename = null;
let selectedVariantBlob = null;
let selectedVariantSeed = null;
