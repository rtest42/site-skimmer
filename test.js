import puppeteer from 'puppeteer-core';

//const puppeteer = require('puppeteer-core');
/*
(async () => {
  const browser = await puppeteer.launch({
    //channel: 'edge',  // You can also use 'chrome', 'chrome-beta', 'chromium', 'edge', etc.
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
  });
  const page = await browser.newPage();
  await page.goto('https://developer.chrome.com/');

    // Set screen size.
    await page.setViewport({width: 1080, height: 1024});

    // Type into search box.
    //await page.locator('.devsite-search-field').fill('automate beyond recorder');

    // Wait and click on first result.
    //await page.locator('.devsite-result-item-link').click();

    // Locate the full title with a unique string.
    //const textSelector = await page
    //.locator('text/Customize and automate')
    //.waitHandle();
    //const fullTitle = await textSelector?.evaluate(el => el.textContent);

    // Print the full title.
    //console.log('The title of this blog post is "%s".', fullTitle);

    await browser.close();
})();*/
// Launch the browser and open a new blank page
//const page = await browser.newPage();
/*
// Navigate the page to a URL.
await page.goto('https://developer.chrome.com/');

// Set screen size.
await page.setViewport({width: 1080, height: 1024});

// Type into search box.
await page.locator('.devsite-search-field').fill('automate beyond recorder');

// Wait and click on first result.
await page.locator('.devsite-result-item-link').click();

// Locate the full title with a unique string.
const textSelector = await page
  .locator('text/Customize and automate')
  .waitHandle();
const fullTitle = await textSelector?.evaluate(el => el.textContent);

// Print the full title.
console.log('The title of this blog post is "%s".', fullTitle);

await browser.close();
*/
// const puppeteer = require('puppeteer');

const browser = await puppeteer.launch({
  //channel: 'edge',  // You can also use 'chrome', 'chrome-beta', 'chromium', 'edge', etc.
  executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
});
const main = async (website) => {/*
    const browser = await puppeteer.launch({
        //channel: 'edge',  // You can also use 'chrome', 'chrome-beta', 'chromium', 'edge', etc.
        executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
      });*/
    const page = await browser.newPage();
    await page.goto(website); // replace with your URL

    const allImages = await page.evaluate(() => {
        return Array.from(document.images, img => img.src);
    });

    console.log(allImages);
    // Scroll to the bottom of the page
    //await autoScroll(page);

    // Get the HTML content
    //const htmlContent = await page.content();
    //console.log(htmlContent);

    await browser.close();
}

main('https://www.outerknown.com/collections/mens-pants');
/*
// Function to scroll to the bottom of the page
async function autoScroll(page) {
    await page.evaluate(async () => {
        await new Promise((resolve, reject) => {
            let totalHeight = 0;
            const distance = 100; // Scroll step
            const timer = setInterval(() => {
                window.scrollBy(0, distance);
                totalHeight += distance;

                if (totalHeight >= document.body.scrollHeight) {
                    clearInterval(timer);
                    resolve();
                }
            }, 100); // Time between each scroll (ms)
        });
    });
}
*/