// const { Scraper } = require('./scrapper');
import { ScrapeImage } from './puppeteer-helper.js';

async function GetImages(q, save) {
    if(save){
        await ScrapeImage(`https://pinterest.com/search/pins/?q=${q}&rs=typed`, q);
    } else {
        // await Scraper(`https://pinterest.com/search/pins/?q=${q}&rs=typed`);
    }
};

export { GetImages };
