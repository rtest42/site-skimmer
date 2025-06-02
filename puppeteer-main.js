import { ScrapeImage } from './puppeteer-helper.js';

// Default values if this file was ran directly
let keyword = process.argv.length >= 4 ? process.argv[2] : "shirt";
let rounds = process.argv.length >= 4 ? process.argv[3] : "5";
let width = process.argv.length >= 5 ? process.argv[4] : "1920";
let height = process.argv.length >= 6 ? process.argv[5]: "1080";
ScrapeImage(`https://pinterest.com/search/pins/?q=${keyword}&rs=typed`, parseInt(rounds), parseInt(width), parseInt(height));
