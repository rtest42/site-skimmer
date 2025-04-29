import { ScrapeImage } from './puppeteer-helper.js';

ScrapeImage(`https://pinterest.com/search/pins/?q=${process.argv[2]}&rs=typed`, parseInt(process.argv[3]));