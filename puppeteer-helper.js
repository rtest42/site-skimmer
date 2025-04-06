import puppeteer from 'puppeteer';
import fs from 'fs';

let iterations = 0;
let download_urls = new Set();

async function ScrapeImage(url, category){
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.setViewport({ width: 2560, height: 1440 });
    await page.goto(url);
    //console.log('Scraping: ' + url);
    
    await page.waitForSelector('img', { visible: true, timeout: 10000 });


    let previousHeight = 0;
    while (true) {
        previousHeight = await page.evaluate(() => document.body.scrollHeight);
        await page.evaluate(() => window.scrollBy(0, window.innerHeight));
        await new Promise(resolve => setTimeout(resolve, 1500)); // Wait for images to load
        let newHeight = await page.evaluate(() => document.body.scrollHeight);
        //console.log(newHeight);
        if (newHeight>=3000) {//=== previousHeight) {
            //console.log("height: " + newHeight);
            break; // Stop if no new content loads
        }
    }

    const html = await page.content(); // Get full page HTML
    // fs.writeFileSync(`page_${iterations}.html`, html, 'utf-8'); // Save to file
    //console.log(`Saved full HTML to page_${iterations}.html`);


    const data = await page.evaluate(() => {
        const images = document.querySelectorAll('img');

        const imgurl = Array.from(images).map(img => img.src);

        return imgurl;
    })

    
    data.forEach(element => download_urls.add(element));

    const urls = await page.evaluate(() => {
        const a = document.querySelectorAll('a');

        const aurl = Array.from(a).map(x => x.href);
    
        return aurl;
    })

    //console.log(data);
    //console.log(data.length);
    //console.log(urls);
    //console.log(urls.length);
    for (let i = 0; i < data.length; i++) {
        const page = await browser.newPage();
        await page.goto(data[i]);
        if(!fs.existsSync("./label2/test/" + category)){
            //console.log("[WARNING] Image folder not found in current directory, Creating one...");
            fs.mkdirSync("./label2/test/" + category);
            await page.screenshot({ path:`./label2/test/${category}/image${[i]}_${iterations}.png`});
        } else {
            await page.screenshot({ path:`./label2/test/${category}/image${[i]}_${iterations}.png`});
        }
        if(i == 2){
            //console.log('[WARNING] To avoid memory issues the image uploads are limited.');
            break;
        }
        
    }

    if (iterations < 1) {
        iterations++;
        const urlPins = urls.filter(str => str.includes("/pin"));
        const randomUrl = urlPins[Math.floor(Math.random() * urlPins.length)];
        await ScrapeImage(randomUrl);
    } else {
        console.log(JSON.stringify([...download_urls]));
    }

    await browser.close();
}

export { ScrapeImage };
